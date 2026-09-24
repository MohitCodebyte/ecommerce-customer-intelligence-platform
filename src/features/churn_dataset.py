import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_ORDERS = BASE_DIR / "data" / "raw" / "olist_orders_dataset.csv"
RAW_CUSTOMERS = BASE_DIR / "data" / "raw" / "olist_customers_dataset.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "customer_churn_dataset.csv"
REPORT_FILE = REPORT_DIR / "churn_dataset_summary.json"


def main():

    print("=" * 70)
    print("LEAKAGE-FREE CUSTOMER CHURN DATASET BUILDER")
    print("=" * 70)

    if not RAW_ORDERS.exists():
        raise FileNotFoundError(
            f"Orders file not found:\n{RAW_ORDERS}"
        )

    if not RAW_CUSTOMERS.exists():
        raise FileNotFoundError(
            f"Customers file not found:\n{RAW_CUSTOMERS}"
        )

    orders = pd.read_csv(RAW_ORDERS)
    customers = pd.read_csv(RAW_CUSTOMERS)

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce"
    )

    orders["order_delivered_customer_date"] = pd.to_datetime(
        orders["order_delivered_customer_date"],
        errors="coerce"
    )

    # Only completed/delivered orders are used for customer behavior.
    completed = orders[
        orders["order_status"].eq("delivered")
    ].copy()

    completed = completed.dropna(
        subset=[
            "customer_id",
            "order_purchase_timestamp"
        ]
    )

    # Map customer_id -> stable customer_unique_id.
    customer_map = customers[
        ["customer_id", "customer_unique_id"]
    ].drop_duplicates()

    completed = completed.merge(
        customer_map,
        on="customer_id",
        how="left"
    )

    completed = completed.dropna(
        subset=["customer_unique_id"]
    )

    completed = completed.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp"
        ]
    )

    dataset_end = completed[
        "order_purchase_timestamp"
    ].max()

    # Historical snapshot.
    # We intentionally create features from the past and
    # observe behavior AFTER the snapshot for the label.
    snapshot_date = dataset_end - pd.Timedelta(days=180)

    observation_end = snapshot_date
    future_end = snapshot_date + pd.Timedelta(days=180)

    historical = completed[
        completed["order_purchase_timestamp"]
        <= observation_end
    ].copy()

    future = completed[
        (
            completed["order_purchase_timestamp"]
            > observation_end
        )
        &
        (
            completed["order_purchase_timestamp"]
            <= future_end
        )
    ].copy()

    print()
    print(f"Dataset end       : {dataset_end.date()}")
    print(f"Snapshot date     : {snapshot_date.date()}")
    print(f"Future period end : {future_end.date()}")
    print()
    print(
        f"Historical orders : {len(historical):,}"
    )
    print(
        f"Future orders     : {len(future):,}"
    )

    # ------------------------------------------------------------
    # HISTORICAL CUSTOMER FEATURES
    # ------------------------------------------------------------

    grouped = historical.groupby(
        "customer_unique_id"
    )

    feature_df = grouped[
        "order_purchase_timestamp"
    ].agg(
        first_purchase_date="min",
        last_purchase_date="max",
        total_orders="count"
    ).reset_index()

    feature_df["recency_days"] = (
        observation_end
        - feature_df["last_purchase_date"]
    ).dt.days

    feature_df["customer_lifetime_days"] = (
        feature_df["last_purchase_date"]
        - feature_df["first_purchase_date"]
    ).dt.days

    feature_df["customer_lifetime_days"] = (
        feature_df["customer_lifetime_days"]
        .clip(lower=0)
    )

    feature_df["repeat_customer"] = (
        feature_df["total_orders"] > 1
    ).astype(int)

    feature_df["purchase_frequency"] = (
        feature_df["total_orders"]
        /
        (
            feature_df["customer_lifetime_days"]
            .clip(lower=1)
            / 30.0
        )
    )

    # Purchase gaps are calculated ONLY from historical purchases.
    gap_rows = []

    for customer_id, group in grouped:

        dates = (
            group["order_purchase_timestamp"]
            .sort_values()
            .drop_duplicates()
        )

        if len(dates) < 2:
            continue

        gaps = (
            dates.diff()
            .dropna()
            .dt.total_seconds()
            / 86400.0
        )

        gap_rows.append(
            {
                "customer_unique_id": customer_id,
                "mean_purchase_gap_days": gaps.mean(),
                "median_purchase_gap_days": gaps.median(),
                "max_purchase_gap_days": gaps.max(),
                "purchase_gap_count": len(gaps)
            }
        )

    gap_df = pd.DataFrame(gap_rows)

    feature_df = feature_df.merge(
        gap_df,
        on="customer_unique_id",
        how="left"
    )

    gap_columns = [
        "mean_purchase_gap_days",
        "median_purchase_gap_days",
        "max_purchase_gap_days",
        "purchase_gap_count"
    ]

    for column in gap_columns:
        feature_df[column] = (
            feature_df[column]
            .fillna(0)
        )

    # ------------------------------------------------------------
    # FUTURE DATA IS USED ONLY TO CREATE THE TARGET
    # ------------------------------------------------------------

    future_counts = (
        future.groupby("customer_unique_id")
        .size()
        .rename("future_purchase_count")
        .reset_index()
    )

    feature_df = feature_df.merge(
        future_counts,
        on="customer_unique_id",
        how="left"
    )

    feature_df["future_purchase_count"] = (
        feature_df["future_purchase_count"]
        .fillna(0)
        .astype(int)
    )

    # A customer is considered retained/non-churned
    # if they make at least one completed purchase
    # during the future observation period.
    feature_df["churn_label"] = (
        feature_df["future_purchase_count"] == 0
    ).astype(int)

    feature_df["label_observation_days"] = 180

    # Every customer has a complete 180-day future window
    # because the snapshot is exactly 180 days before dataset end.
    feature_df["label_eligible"] = 1

    # ------------------------------------------------------------
    # CRITICAL:
    # Remove future information from model dataset.
    #
    # future_purchase_count is ONLY used above to construct
    # churn_label. It MUST NOT survive as a feature.
    # ------------------------------------------------------------

    feature_columns = [
        "customer_unique_id",
        "first_purchase_date",
        "last_purchase_date",
        "total_orders",
        "recency_days",
        "customer_lifetime_days",
        "repeat_customer",
        "purchase_frequency",
        "mean_purchase_gap_days",
        "median_purchase_gap_days",
        "max_purchase_gap_days",
        "purchase_gap_count",
        "churn_label"
    ]

    final_df = feature_df[
        feature_columns
    ].copy()

    # Sort for reproducibility.
    final_df = final_df.sort_values(
        "customer_unique_id"
    ).reset_index(drop=True)

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary = {
        "dataset_type": "historical_snapshot_churn_dataset",
        "snapshot_date": str(snapshot_date.date()),
        "future_observation_days": 180,
        "customers": int(len(final_df)),
        "churned": int(
            final_df["churn_label"].sum()
        ),
        "non_churned": int(
            (final_df["churn_label"] == 0).sum()
        ),
        "churn_rate": float(
            final_df["churn_label"].mean()
        ),
        "feature_columns": [
            c for c in feature_columns
            if c != "churn_label"
        ],
        "target_column": "churn_label",
        "future_purchase_count_used_only_for_label": True,
        "leakage_columns_removed": [
            "future_purchase_count",
            "label_observation_days",
            "label_eligible"
        ]
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("DATASET CREATED")
    print("=" * 70)

    print(
        f"Customers       : {len(final_df):,}"
    )

    print(
        f"Churned         : "
        f"{summary['churned']:,}"
    )

    print(
        f"Non-churned     : "
        f"{summary['non_churned']:,}"
    )

    print(
        f"Churn rate      : "
        f"{summary['churn_rate'] * 100:.2f}%"
    )

    print()
    print("FINAL COLUMNS:")
    for column in final_df.columns:
        print(f"  - {column}")

    print()
    print(f"Dataset saved:")
    print(OUTPUT_FILE)

    print()
    print(f"Summary saved:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
