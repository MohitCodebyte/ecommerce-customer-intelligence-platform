import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_repeat_purchase_production import (
    engineer_features,
    build_feature_matrix,
)


def main():
    parser = argparse.ArgumentParser(
        description="Predict repeat purchase propensity"
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="Input CSV file"
    )

    parser.add_argument(
        "--output-file",
        default="data/processed/repeat_purchase_predictions.csv",
        help="Output CSV file"
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    model_path = (
        project_root
        / "artifacts_production"
        / "models"
        / "xgb_repeat_purchase_production.joblib"
    )

    imputer_path = (
        project_root
        / "artifacts_production"
        / "models"
        / "imputer.joblib"
    )

    features_path = (
        project_root
        / "artifacts_production"
        / "models"
        / "feature_columns.json"
    )

    threshold_path = (
        project_root
        / "artifacts_production"
        / "reports"
        / "threshold.json"
    )

    print("Loading input data...")

    df = pd.read_csv(args.input_file)

    print(f"Input rows: {len(df):,}")

    print("Loading production artifacts...")

    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    threshold = float(threshold_data["selected_threshold"])

    print(f"Production threshold: {threshold:.4f}")

    print("Building feature matrix...")

    X = build_feature_matrix(
        df,
        feature_columns=feature_columns,
    )

    X_imputed = imputer.transform(X)

    print("Generating predictions...")

    probabilities = np.mean([m.predict_proba(X_imputed)[:, 1] for m in model], axis=0)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = df.copy()

    result["repeat_purchase_probability"] = probabilities
    result["repeat_purchase_prediction"] = predictions

    output_path = Path(args.output_file)

    if not output_path.is_absolute():
        output_path = project_root / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)

    print(f"Total customers: {len(result):,}")
    print(
        f"Predicted repeat purchasers: "
        f"{predictions.sum():,}"
    )

    print(
        f"Predicted repeat rate: "
        f"{predictions.mean():.2%}"
    )

    print(
        f"Average probability: "
        f"{probabilities.mean():.4f}"
    )

    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

