import pandas as pd
import numpy as np
import re
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import scipy.sparse as sp
import joblib
import os

warnings.filterwarnings("ignore")

TIME_ORDER = {"early_morning": 0, "morning": 1, "afternoon": 2, "evening": 3, "night": 4}
MOOD_SCORE = {"calm": 1, "focused": 2, "neutral": 3, "mixed": 3, "restless": 4, "overwhelmed": 5}
FACE_SCORE = {"happy_face": 1, "calm_face": 2, "neutral_face": 3, "tired_face": 4, "tense_face": 5, "none": 3, "": 3}
QUALITY_SCORE = {"clear": 3, "vague": 2, "conflicted": 1, "": 2}


def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return "no reflection"
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_length(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())


def extract_sentiment_keywords(text):
    text = str(text).lower()
    positive_words = {"calm", "peaceful", "lighter", "settled", "clear", "focused",
                      "better", "gentle", "softened", "quiet", "breathe", "relieved",
                      "organized", "sharper", "lock", "ready", "easier"}
    negative_words = {"racing", "flooded", "heavy", "overloaded", "exhausted", "tired",
                      "pressure", "stressed", "scattered", "distracted", "fidgety",
                      "unsettled", "carrying", "overwhelmed", "restless", "tense",
                      "jumping", "switching", "pulled", "buzz", "anxious"}
    mixed_words = {"but", "though", "still", "part", "can't", "idk", "maybe",
                   "split", "both", "yet", "however"}
    words = set(text.split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)
    mix = len(words & mixed_words)
    return {"pos_kw": pos, "neg_kw": neg, "mix_kw": mix,
            "sentiment_ratio": (pos - neg) / (pos + neg + 1)}


def engineer_features(df, fit_encoders=True, encoders=None):
    df = df.copy()

    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce").fillna(
        df.get("sleep_hours", pd.Series()).median() if not fit_encoders and encoders else 6.5
    )
    df["energy_level"] = pd.to_numeric(df["energy_level"], errors="coerce").fillna(3)
    df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce").fillna(3)
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce").fillna(15)

    if fit_encoders:
        sleep_median = df["sleep_hours"].median()
        if encoders is None:
            encoders = {}
        encoders["sleep_median"] = sleep_median
    else:
        sleep_median = encoders.get("sleep_median", 6.5)
        df["sleep_hours"] = df["sleep_hours"].fillna(sleep_median)

    df["time_numeric"] = df["time_of_day"].map(TIME_ORDER).fillna(2)
    df["prev_mood_score"] = df["previous_day_mood"].fillna("neutral").map(MOOD_SCORE).fillna(3)
    df["face_score"] = df["face_emotion_hint"].fillna("none").map(FACE_SCORE).fillna(3)
    df["quality_score"] = df["reflection_quality"].fillna("vague").map(QUALITY_SCORE).fillna(2)

    cat_cols = ["ambience_type", "time_of_day", "previous_day_mood", "face_emotion_hint", "reflection_quality"]
    if fit_encoders:
        encoders["le"] = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].fillna("unknown").astype(str))
            encoders["le"][col] = le
    else:
        for col in cat_cols:
            le = encoders["le"][col]
            known = set(le.classes_)
            df[col + "_enc"] = df[col].fillna("unknown").astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

    df["text_clean"] = df["journal_text"].apply(clean_text)
    df["text_length"] = df["journal_text"].apply(text_length)

    kw_df = df["journal_text"].apply(lambda t: pd.Series(extract_sentiment_keywords(str(t))))
    df = pd.concat([df, kw_df], axis=1)

    df["stress_energy_ratio"] = df["stress_level"] / (df["energy_level"] + 0.1)
    df["sleep_stress"] = df["sleep_hours"] * (6 - df["stress_level"])
    df["is_short_text"] = (df["text_length"] < 5).astype(int)
    df["is_night"] = (df["time_of_day"].fillna("").isin(["night", "evening"])).astype(int)
    df["conflict_signal"] = ((df["face_score"] > 3) & (df["quality_score"] < 2)).astype(int)

    metadata_cols = [
        "duration_min", "sleep_hours", "energy_level", "stress_level",
        "time_numeric", "prev_mood_score", "face_score", "quality_score",
        "text_length", "pos_kw", "neg_kw", "mix_kw", "sentiment_ratio",
        "stress_energy_ratio", "sleep_stress", "is_short_text", "is_night", "conflict_signal",
        "ambience_type_enc", "time_of_day_enc", "previous_day_mood_enc",
        "face_emotion_hint_enc", "reflection_quality_enc"
    ]

    return df, metadata_cols, encoders


def build_tfidf(texts, fit=True, vectorizer=None, max_features=300):
    if fit:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X_text = vectorizer.fit_transform(texts)
    else:
        X_text = vectorizer.transform(texts)
    return X_text, vectorizer


def combine_features(X_meta, X_text):
    return np.hstack([X_meta, X_text.toarray()])


def train_emotion_model(X, y):
    base = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2,
                                   class_weight="balanced", random_state=42)
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X, y)
    return model


def train_intensity_model(X, y):
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                        subsample=0.8, random_state=42)
    model.fit(X, y)
    return model


def compute_uncertainty(proba, short_text, reflection_quality):
    max_prob = proba.max(axis=1)
    confidence = np.round(max_prob, 3)
    uncertain_flag = (
        (max_prob < 0.45) |
        (short_text == 1) |
        (reflection_quality.fillna("vague").values == "conflicted")
    ).astype(int)
    return confidence, uncertain_flag


def decide_action(row):
    state = row.get("predicted_state", "neutral")
    intensity = int(row.get("predicted_intensity", 3))
    stress = float(row.get("stress_level", 3))
    energy = float(row.get("energy_level", 3))
    tod = str(row.get("time_of_day", "morning")).lower()

    if state == "overwhelmed":
        what = "box_breathing" if intensity >= 4 else "grounding"
    elif state == "restless":
        what = "movement" if energy >= 4 else "journaling"
    elif state == "calm":
        what = "deep_work" if energy >= 3 else "light_planning"
    elif state == "focused":
        what = "deep_work"
    elif state == "mixed":
        what = "journaling" if stress >= 4 else "pause"
    elif state == "neutral":
        what = "light_planning" if energy >= 4 else "rest"
    else:
        what = "pause"

    if stress >= 5 and energy <= 2:
        what = "rest"

    if state in ("overwhelmed", "restless") and intensity >= 4:
        when = "now"
    elif state in ("calm", "focused") and tod in ("morning", "early_morning"):
        when = "now"
    elif tod in ("early_morning", "morning"):
        when = "within_15_min"
    elif tod == "afternoon":
        when = "later_today"
    elif tod == "evening":
        when = "tonight"
    elif tod == "night":
        when = "tomorrow_morning"
    else:
        when = "within_15_min"

    if state == "calm" and intensity <= 2:
        when = "later_today"

    return what, when


MESSAGES = {
    ("overwhelmed", "box_breathing"): "You're carrying a lot right now. Let's pause everything and try some box breathing — four counts in, hold, out, hold. Just that, right now.",
    ("overwhelmed", "grounding"): "Things feel heavy, but you're here. Try grounding: name 5 things you can see around you. Bring yourself back to this moment.",
    ("restless", "movement"): "Your energy needs somewhere to go. Step outside or shake it out for a few minutes — your mind will follow.",
    ("restless", "journaling"): "You seem scattered. Try writing down everything in your head — no filter, just a brain dump. It helps clear the fog.",
    ("calm", "deep_work"): "You're in a good place. This is a great window to focus on something meaningful — protect this time.",
    ("calm", "light_planning"): "You seem settled. Even a quick 10-minute plan for the day can make things feel much more manageable.",
    ("focused", "deep_work"): "Your mind is sharp right now. Dive into your hardest task first — you're primed for it.",
    ("mixed", "journaling"): "It's okay to feel more than one thing at once. Try journaling to untangle what's going on underneath.",
    ("mixed", "pause"): "Something is sitting with you. You don't have to solve it now — just give yourself permission to pause.",
    ("neutral", "light_planning"): "You feel steady. Use this quiet energy to set a gentle intention for the rest of your day.",
    ("neutral", "rest"): "Nothing urgent is pulling at you. This is a good moment to simply rest — guilt-free.",
}


def generate_message(state, what):
    key = (state, what)
    if key in MESSAGES:
        return MESSAGES[key]
    return f"You seem {state}. Consider doing some {what.replace('_', ' ')} when you feel ready."


def handle_edge_cases(df):
    df = df.copy()
    short_mask = df["journal_text"].apply(text_length) < 5
    df.loc[short_mask, "journal_text"] = df.loc[short_mask, "journal_text"].fillna("") + " felt okay, not sure"
    for col, default in [("sleep_hours", 6.5), ("energy_level", 3), ("stress_level", 3), ("duration_min", 15)]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


class EmotionPipeline:
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.tfidf = None
        self.emotion_model = None
        self.intensity_model = None
        self.emotion_le = LabelEncoder()
        self.metadata_cols = []
        self.feature_names = []

    def fit(self, df):
        df = handle_edge_cases(df)
        df, self.metadata_cols, self.encoders = engineer_features(df, fit_encoders=True)

        X_text, self.tfidf = build_tfidf(df["text_clean"], fit=True)
        X_meta = self.scaler.fit_transform(df[self.metadata_cols].values)
        X = combine_features(X_meta, X_text)

        tfidf_names = [f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()]
        self.feature_names = self.metadata_cols + tfidf_names

        y_emotion = self.emotion_le.fit_transform(df["emotional_state"])
        y_intensity = df["intensity"].astype(int).values

        self.emotion_model = train_emotion_model(X, y_emotion)
        self.intensity_model = train_intensity_model(X, y_intensity)

        print(f"Trained on {len(df)} samples")
        print(f"Emotion classes: {list(self.emotion_le.classes_)}")
        print(f"Features: {X.shape[1]}")
        return self

    def predict(self, df):
        df_orig = df.copy()
        df = handle_edge_cases(df)
        df, _, _ = engineer_features(df, fit_encoders=False, encoders=self.encoders)

        X_text, _ = build_tfidf(df["text_clean"], fit=False, vectorizer=self.tfidf)
        X_meta = self.scaler.transform(df[self.metadata_cols].values)
        X = combine_features(X_meta, X_text)

        emotion_proba = self.emotion_model.predict_proba(X)
        emotion_idx = emotion_proba.argmax(axis=1)
        predicted_state = self.emotion_le.inverse_transform(emotion_idx)
        predicted_intensity = self.intensity_model.predict(X)

        confidence, uncertain_flag = compute_uncertainty(
            emotion_proba, df["is_short_text"].values, df["reflection_quality"]
        )

        results = []
        for i, (state, intensity) in enumerate(zip(predicted_state, predicted_intensity)):
            row_data = {
                "predicted_state": state,
                "predicted_intensity": intensity,
                "stress_level": df["stress_level"].iloc[i],
                "energy_level": df["energy_level"].iloc[i],
                "time_of_day": df_orig["time_of_day"].iloc[i] if "time_of_day" in df_orig else "morning",
            }
            what, when = decide_action(pd.Series(row_data))
            message = generate_message(state, what)
            results.append({
                "id": df_orig["id"].iloc[i] if "id" in df_orig else i + 1,
                "predicted_state": state,
                "predicted_intensity": int(intensity),
                "confidence": float(confidence[i]),
                "uncertain_flag": int(uncertain_flag[i]),
                "what_to_do": what,
                "when_to_do": when,
                "supportive_message": message
            })

        return pd.DataFrame(results)

    def cross_validate(self, df):
        df = handle_edge_cases(df)
        df, self.metadata_cols, enc = engineer_features(df, fit_encoders=True)
        X_text, tvec = build_tfidf(df["text_clean"], fit=True)
        scaler = StandardScaler()
        X_meta = scaler.fit_transform(df[self.metadata_cols].values)
        X = combine_features(X_meta, X_text)

        y_emotion = LabelEncoder().fit_transform(df["emotional_state"])
        y_intensity = df["intensity"].astype(int).values

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf_base = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        emotion_cv_scores = cross_val_score(rf_base, X, y_emotion, cv=cv, scoring="accuracy")
        gb_base = GradientBoostingClassifier(n_estimators=100, random_state=42)
        intensity_cv_scores = cross_val_score(gb_base, X, y_intensity, cv=cv, scoring="accuracy")

        return {
            "emotion_cv_mean": round(emotion_cv_scores.mean(), 4),
            "emotion_cv_std": round(emotion_cv_scores.std(), 4),
            "intensity_cv_mean": round(intensity_cv_scores.mean(), 4),
            "intensity_cv_std": round(intensity_cv_scores.std(), 4),
        }

    def save(self, path="model_artifacts"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.emotion_model, f"{path}/emotion_model.pkl")
        joblib.dump(self.intensity_model, f"{path}/intensity_model.pkl")
        joblib.dump(self.tfidf, f"{path}/tfidf.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.emotion_le, f"{path}/emotion_le.pkl")
        joblib.dump(self.encoders, f"{path}/encoders.pkl")
        joblib.dump(self.metadata_cols, f"{path}/metadata_cols.pkl")
        print(f"Model saved to {path}/")

    @classmethod
    def load(cls, path="model_artifacts"):
        ep = cls()
        ep.emotion_model = joblib.load(f"{path}/emotion_model.pkl")
        ep.intensity_model = joblib.load(f"{path}/intensity_model.pkl")
        ep.tfidf = joblib.load(f"{path}/tfidf.pkl")
        ep.scaler = joblib.load(f"{path}/scaler.pkl")
        ep.emotion_le = joblib.load(f"{path}/emotion_le.pkl")
        ep.encoders = joblib.load(f"{path}/encoders.pkl")
        ep.metadata_cols = joblib.load(f"{path}/metadata_cols.pkl")
        return ep
