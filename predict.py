import argparse
import pandas as pd
from pipeline import EmotionPipeline


def load_data(path):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_artifacts")
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    ep = EmotionPipeline.load(args.model)
    print(f"Model loaded from {args.model}")

    df_test = load_data(args.test)
    print(f"Loaded test data: {df_test.shape}")

    preds = ep.predict(df_test)

    out_cols = ["id", "predicted_state", "predicted_intensity", "confidence", "uncertain_flag", "what_to_do", "when_to_do"]
    preds[out_cols].to_csv(args.output, index=False)
    print(f"Saved {args.output}")

    full_out = args.output.replace(".csv", "_full.csv")
    preds.to_csv(full_out, index=False)
    print(f"Saved {full_out} (with supportive messages)")

    print("\nSample predictions:")
    print(preds.head(10).to_string(index=False))

    print("\nSummary:")
    print(preds["predicted_state"].value_counts().to_string())
    print(f"\nUncertain: {preds['uncertain_flag'].sum()} / {len(preds)}")
    print(f"Avg confidence: {preds['confidence'].mean():.3f}")


if __name__ == "__main__":
    main()
