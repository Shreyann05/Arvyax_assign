import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pipeline import EmotionPipeline, engineer_features, build_tfidf, combine_features, handle_edge_cases
import warnings
warnings.filterwarnings("ignore")


def evaluate_model(pipeline, df_test):
    preds = pipeline.predict(df_test)

    if "emotional_state" in df_test.columns:
        acc = accuracy_score(df_test["emotional_state"].values, preds["predicted_state"].values)
        print(f"\nEmotion State Accuracy: {acc:.4f}")
        print(classification_report(df_test["emotional_state"].values, preds["predicted_state"].values, zero_division=0))

    if "intensity" in df_test.columns:
        mae = mean_absolute_error(df_test["intensity"].astype(int).values, preds["predicted_intensity"].astype(int).values)
        print(f"Intensity MAE: {mae:.4f} (scale 1-5)\n")

    print("Decision Engine Distribution:")
    print("  what_to_do:", preds["what_to_do"].value_counts().to_dict())
    print("  when_to_do:", preds["when_to_do"].value_counts().to_dict())
    print(f"\n  Uncertain: {preds['uncertain_flag'].sum()} / {len(preds)}")
    print(f"  Avg confidence: {preds['confidence'].mean():.3f}")

    return preds


def ablation_study(df_train, df_test):
    print("\nABLATION STUDY: Text-only vs Full model")

    df_train = handle_edge_cases(df_train)
    df_test = handle_edge_cases(df_test)

    df_tr, meta_cols, encoders = engineer_features(df_train, fit_encoders=True)
    df_te, _, _ = engineer_features(df_test, fit_encoders=False, encoders=encoders)

    X_text_tr, tfidf = build_tfidf(df_tr["text_clean"], fit=True)
    X_text_te, _ = build_tfidf(df_te["text_clean"], fit=False, vectorizer=tfidf)

    scaler = StandardScaler()
    X_meta_tr = scaler.fit_transform(df_tr[meta_cols].values)
    X_meta_te = scaler.transform(df_te[meta_cols].values)

    le = LabelEncoder()
    y_tr = le.fit_transform(df_train["emotional_state"])
    y_te = df_test["emotional_state"].values

    rf_text = RandomForestClassifier(100, class_weight="balanced", random_state=42)
    rf_text.fit(X_text_tr.toarray(), y_tr)
    acc_text = accuracy_score(y_te, le.inverse_transform(rf_text.predict(X_text_te.toarray())))

    X_full_tr = combine_features(X_meta_tr, X_text_tr)
    X_full_te = combine_features(X_meta_te, X_text_te)
    rf_full = RandomForestClassifier(100, class_weight="balanced", random_state=42)
    rf_full.fit(X_full_tr, y_tr)
    acc_full = accuracy_score(y_te, le.inverse_transform(rf_full.predict(X_full_te)))

    print(f"  Text-only:    {acc_text:.4f}")
    print(f"  Text+metadata: {acc_full:.4f}")
    print(f"  Delta:        {acc_full - acc_text:+.4f}")
    print("\n  Metadata (sleep, stress, energy, time) adds important context")
    print("  especially when text is short or vague.")

    return {"text_only": acc_text, "full_model": acc_full}


def feature_importance_analysis(pipeline, df):
    print("\nFEATURE IMPORTANCE")

    df = handle_edge_cases(df)
    df, meta_cols, _ = engineer_features(df, fit_encoders=True)
    X_text, _ = build_tfidf(df["text_clean"], fit=True)
    X_meta = StandardScaler().fit_transform(df[meta_cols].values)
    X = combine_features(X_meta, X_text)
    y = LabelEncoder().fit_transform(df["emotional_state"])

    rf = RandomForestClassifier(100, class_weight="balanced", random_state=42)
    rf.fit(X, y)

    tfidf_names = [f"tfidf_{w}" for w in pipeline.tfidf.get_feature_names_out()] if pipeline.tfidf else []
    feature_names = meta_cols + tfidf_names

    importances = rf.feature_importances_
    if len(importances) != len(feature_names):
        feature_names = [f"f_{i}" for i in range(len(importances))]

    fi = pd.DataFrame({"feature": feature_names[:len(importances)],
                        "importance": importances}).sort_values("importance", ascending=False)

    print("\nTop 15 features:")
    print(fi.head(15).to_string(index=False))

    meta_imp = fi[~fi["feature"].str.startswith("tfidf_")]["importance"].sum()
    text_imp = fi[fi["feature"].str.startswith("tfidf_")]["importance"].sum()
    total = meta_imp + text_imp
    print(f"\n  Metadata: {meta_imp/total*100:.1f}%")
    print(f"  Text (TF-IDF): {text_imp/total*100:.1f}%")

    return fi


def load_data(path):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", default=None)
    parser.add_argument("--save", default="model_artifacts")
    args = parser.parse_args()

    df_train = load_data(args.train)
    print(f"Loaded training data: {df_train.shape}")
    print(f"Columns: {list(df_train.columns)}")
    print(f"Emotion classes: {df_train['emotional_state'].value_counts().to_dict()}")
    missing = df_train.isnull().sum()
    print(f"Missing values:\n{missing[missing > 0]}\n")

    ep = EmotionPipeline()
    print("Cross-Validation (5-fold)")
    cv_scores = ep.cross_validate(df_train)
    print(f"  Emotion CV:   {cv_scores['emotion_cv_mean']:.4f} +/- {cv_scores['emotion_cv_std']:.4f}")
    print(f"  Intensity CV: {cv_scores['intensity_cv_mean']:.4f} +/- {cv_scores['intensity_cv_std']:.4f}")

    df_tr, df_val = train_test_split(df_train, test_size=0.2, random_state=42,
                                      stratify=df_train["emotional_state"])

    ep = EmotionPipeline()
    ep.fit(df_train)
    ep.save(args.save)

    print("\nValidation Set Evaluation")
    evaluate_model(ep, df_val)

    feature_importance_analysis(ep, df_train)
    ablation_study(df_tr, df_val)

    if args.test:
        df_test = load_data(args.test)
        print(f"\nLoaded test data: {df_test.shape}")
        test_preds = ep.predict(df_test)
        test_preds.to_csv("predictions.csv", index=False)
        print(f"Saved predictions.csv ({len(test_preds)} rows)")
        print(test_preds.head(10).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
