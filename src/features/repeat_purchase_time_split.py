import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_repeat_purchase_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 90)
    print("TIME-BASED REPEAT PURCHASE SPLIT")
    print("=" * 90)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"],
        errors="coerce"
    )

    df = df.sort_values(
        "snapshot_date"
    ).reset_index(drop=True)

    print()
    print(f"Total rows : {len(df):,}")
    print(
        f"Snapshots  : "
        f"{df['snapshot_date'].dt.date.astype(str).unique().tolist()}"
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Latest snapshot is kept completely untouched as TEST.
    #
    # Older snapshots are TRAIN.
    #
    # This avoids using future snapshot information to train
    # the model and reduces customer-history leakage.
    # ---------------------------------------------------------

    unique_dates = sorted(
        df["snapshot_date"].dropna().unique()
    )

    if len(unique_dates) < 2:
        raise ValueError(
            "At least 2 snapshots are required."
        )

    test_date = unique_dates[-1]

    train_dates = unique_dates[:-1]

    train_df = df[
        df["snapshot_date"].isin(train_dates)
    ].copy()

    test_df = df[
        df["snapshot_date"].eq(test_date)
    ].copy()

    # ---------------------------------------------------------
    # Remove identifiers / dates from MODEL INPUT
    # ---------------------------------------------------------

    forbidden_columns = [
        "customer_unique_id",
        "snapshot_date",
        "future_end_date",
        "first_purchase_date",
        "last_purchase_date",
        "repeat_purchase"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in forbidden_columns
    ]

    X_train = train_df[
        feature_columns
    ].copy()

    y_train = train_df[
        "repeat_purchase"
    ].astype(int)

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        "repeat_purchase"
    ].astype(int)

    # ---------------------------------------------------------
    # PRINT SPLIT INFORMATION
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("TIME-BASED SPLIT")
    print("=" * 90)

    print()
    print(
        f"Training snapshots : "
        f"{[str(x.date()) for x in train_dates]}"
    )

    print(
        f"Test snapshot      : "
        f"{test_date.date()}"
    )

    print()
    print(
        f"Training rows      : "
        f"{len(train_df):,}"
    )

    print(
        f"Test rows          : "
        f"{len(test_df):,}"
    )

    print()
    print("TRAIN TARGET:")

    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"Train repeat rate  : "
        f"{y_train.mean() * 100:.2f}%"
    )

    print()
    print("TEST TARGET:")

    print(
        y_test.value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"Test repeat rate   : "
        f"{y_test.mean() * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # SAVE SPLITS
    # ---------------------------------------------------------

    train_output = (
        OUTPUT_DIR
        / "repeat_purchase_train.csv"
    )

    test_output = (
        OUTPUT_DIR
        / "repeat_purchase_test.csv"
    )

    train_df.to_csv(
        train_output,
        index=False
    )

    test_df.to_csv(
        test_output,
        index=False
    )

    # ---------------------------------------------------------
    # SAVE FEATURE LIST
    # ---------------------------------------------------------

    feature_report = {
        "target": "repeat_purchase",

        "feature_columns": feature_columns,

        "forbidden_columns": forbidden_columns,

        "train_snapshots": [
            str(x.date())
            for x in train_dates
        ],

        "test_snapshot":
            str(test_date.date()),

        "train_rows":
            len(train_df),

        "test_rows":
            len(test_df),

        "train_repeat_rate":
            float(y_train.mean()),

        "test_repeat_rate":
            float(y_test.mean()),

        "leakage_safe_split": True,

        "split_method":
            "Time-based: older snapshots for training, latest snapshot for untouched testing."
    }

    report_path = (
        REPORT_DIR
        / "repeat_purchase_time_split.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_report,
            file,
            indent=2
        )

    print()
    print("=" * 90)
    print("FILES SAVED")
    print("=" * 90)

    print(train_output)
    print(test_output)
    print(report_path)

    print()
    print("=" * 90)
    print("TIME-BASED SPLIT COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()

