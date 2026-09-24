import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

ORDERS_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "olist_orders_dataset.csv"
)

CUSTOMERS_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "olist_customers_dataset.csv"
)

ORDER_ITEMS_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "olist_order_items_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 90)
    print("REPEAT PURCHASE PREDICTION DATASET BUILDER")
    print("=" * 90)

    orders = pd.read_csv(ORDERS_FILE)

    customers = pd.read_csv(CUSTOMERS_FILE)

    items = pd.read_csv(ORDER_ITEMS_FILE)

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # COMPLETED ORDERS
    # ---------------------------------------------------------

    completed = orders[
        orders["order_status"].eq("delivered")
        & orders["order_purchase_timestamp"].notna()
    ].copy()

    completed = completed.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id"
            ]
        ],
        on="customer_id",
        how="left"
    )

    completed = completed.dropna(
        subset=[
            "customer_unique_id",
            "order_purchase_timestamp"
        ]
    )

    # ---------------------------------------------------------
    # ORDER VALUE
    # ---------------------------------------------------------

    items["price"] = pd.to_numeric(
        items["price"],
        errors="coerce"
    )

    items["freight_value"] = pd.to_numeric(
        items["freight_value"],
        errors="coerce"
    )

    order_values = (
        items
        .groupby("order_id")
        .agg(
            order_product_value=("price", "sum"),
            order_freight_value=("freight_value", "sum")
        )
        .reset_index()
    )

    order_values["order_value"] = (
        order_values["order_product_value"]
        + order_values["order_freight_value"]
    )

    completed = completed.merge(
        order_values[
            [
                "order_id",
                "order_value"
            ]
        ],
        on="order_id",
        how="left"
    )

    completed["order_value"] = (
        completed["order_value"]
        .fillna(0)
    )

    completed = completed.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp"
        ]
    )

    dataset_end = (
        completed["order_purchase_timestamp"]
        .max()
    )

    print()
    print(
        f"Completed orders : {len(completed):,}"
    )

    print(
        f"Customers        : "
        f"{completed['customer_unique_id'].nunique():,}"
    )

    print(
        f"Dataset end      : "
        f"{dataset_end.date()}"
    )

    # ---------------------------------------------------------
    # USE MULTIPLE HISTORICAL SNAPSHOTS
    # ---------------------------------------------------------
    #
    # We create samples from several historical dates.
    # This gives the model more realistic examples of
    # customers who did and did not purchase again.
    #
    # Future period = 180 days
    #
    # IMPORTANT:
    # Future purchase is ONLY used for target creation.
    # It is never included in model features.
    # ---------------------------------------------------------

    snapshot_dates = [
        dataset_end - pd.Timedelta(days=180),
        dataset_end - pd.Timedelta(days=240),
        dataset_end - pd.Timedelta(days=300),
        dataset_end - pd.Timedelta(days=365),
        dataset_end - pd.Timedelta(days=425),
        dataset_end - pd.Timedelta(days=485),
    ]

    FUTURE_DAYS = 180

    all_samples = []

    for snapshot_date in snapshot_dates:

        future_end = (
            snapshot_date
            + pd.Timedelta(days=FUTURE_DAYS)
        )

        historical = completed[
            completed["order_purchase_timestamp"]
            <= snapshot_date
        ].copy()

        future = completed[
            (
                completed["order_purchase_timestamp"]
                > snapshot_date
            )
            &
            (
                completed["order_purchase_timestamp"]
                <= future_end
            )
        ].copy()

        if historical.empty:
            continue

        # -----------------------------------------------------
        # CUSTOMER HISTORICAL FEATURES
        # -----------------------------------------------------

        grouped = (
            historical
            .groupby("customer_unique_id")
        )

        features = grouped.agg(
            first_purchase_date=(
                "order_purchase_timestamp",
                "min"
            ),

            last_purchase_date=(
                "order_purchase_timestamp",
                "max"
            ),

            total_orders=(
                "order_id",
                "nunique"
            ),

            total_revenue=(
                "order_value",
                "sum"
            ),

            average_order_value=(
                "order_value",
                "mean"
            )
        ).reset_index()

        features["recency_days"] = (
            snapshot_date
            - features["last_purchase_date"]
        ).dt.total_seconds() / 86400

        features["customer_lifetime_days"] = (
            features["last_purchase_date"]
            - features["first_purchase_date"]
        ).dt.total_seconds() / 86400

        features["customer_lifetime_days"] = (
            features["customer_lifetime_days"]
            .clip(lower=1)
        )

        features["repeat_customer"] = (
            features["total_orders"] > 1
        ).astype(int)

        features["purchase_frequency"] = (
            features["total_orders"]
            / (
                features["customer_lifetime_days"]
                / 30
            )
        )

        # -----------------------------------------------------
        # PURCHASE GAPS
        # -----------------------------------------------------

        gaps = (
            historical
            .sort_values(
                [
                    "customer_unique_id",
                    "order_purchase_timestamp"
                ]
            )
            .groupby("customer_unique_id")
            ["order_purchase_timestamp"]
            .diff()
            .dt.total_seconds()
            / 86400
        )

        gap_df = pd.DataFrame(
            {
                "customer_unique_id":
                    historical["customer_unique_id"],
                "gap_days":
                    gaps.values
            }
        )

        gap_stats = (
            gap_df
            .dropna()
            .groupby("customer_unique_id")
            ["gap_days"]
            .agg(
                mean_purchase_gap_days="mean",
                median_purchase_gap_days="median",
                max_purchase_gap_days="max",
                purchase_gap_count="count"
            )
            .reset_index()
        )

        features = features.merge(
            gap_stats,
            on="customer_unique_id",
            how="left"
        )

        # -----------------------------------------------------
        # FUTURE TARGET
        # -----------------------------------------------------

        future_purchase = (
            future
            .groupby("customer_unique_id")
            ["order_id"]
            .nunique()
            .rename("future_purchase_count")
        )

        features = features.set_index(
            "customer_unique_id"
        )

        features = features.join(
            future_purchase,
            how="left"
        )

        features["future_purchase_count"] = (
            features["future_purchase_count"]
            .fillna(0)
            .astype(int)
        )

        features["repeat_purchase"] = (
            features["future_purchase_count"] > 0
        ).astype(int)

        features = features.reset_index()

        # -----------------------------------------------------
        # SNAPSHOT INFORMATION
        # -----------------------------------------------------

        features["snapshot_date"] = (
            snapshot_date
        )

        features["future_end_date"] = (
            future_end
        )

        features["observation_window_days"] = (
            FUTURE_DAYS
        )

        # -----------------------------------------------------
        # REMOVE TARGET-LEAKING COLUMN
        # -----------------------------------------------------

        features = features.drop(
            columns=[
                "future_purchase_count"
            ]
        )

        all_samples.append(
            features
        )

        print()
        print(
            f"Snapshot: {snapshot_date.date()}"
        )

        print(
            f"Customers: {len(features):,}"
        )

        print(
            f"Repeat purchase: "
            f"{features['repeat_purchase'].sum():,}"
        )

        print(
            f"No repeat purchase: "
            f"{(features['repeat_purchase'] == 0).sum():,}"
        )

        print(
            f"Repeat rate: "
            f"{features['repeat_purchase'].mean() * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # COMBINE SNAPSHOTS
    # ---------------------------------------------------------

    if not all_samples:
        raise RuntimeError(
            "No historical snapshots were generated."
        )

    final_df = pd.concat(
        all_samples,
        ignore_index=True
    )

    # ---------------------------------------------------------
    # CLEAN NUMERIC VALUES
    # ---------------------------------------------------------

    numeric_columns = [
        "recency_days",
        "customer_lifetime_days",
        "total_orders",
        "total_revenue",
        "average_order_value",
        "purchase_frequency",
        "mean_purchase_gap_days",
        "median_purchase_gap_days",
        "max_purchase_gap_days",
        "purchase_gap_count"
    ]

    for column in numeric_columns:

        final_df[column] = pd.to_numeric(
            final_df[column],
            errors="coerce"
        )

    # ---------------------------------------------------------
    # FINAL COLUMN ORDER
    # ---------------------------------------------------------

    final_columns = [
        "customer_unique_id",
        "snapshot_date",
        "future_end_date",

        "first_purchase_date",
        "last_purchase_date",

        "total_orders",
        "total_revenue",
        "average_order_value",

        "recency_days",
        "customer_lifetime_days",
        "repeat_customer",
        "purchase_frequency",

        "mean_purchase_gap_days",
        "median_purchase_gap_days",
        "max_purchase_gap_days",
        "purchase_gap_count",

        "observation_window_days",

        "repeat_purchase"
    ]

    final_df = final_df[
        final_columns
    ]

    # ---------------------------------------------------------
    # REMOVE INVALID ROWS
    # ---------------------------------------------------------

    final_df = final_df.dropna(
        subset=[
            "repeat_purchase",
            "recency_days",
            "total_orders"
        ]
    )

    final_df["repeat_purchase"] = (
        final_df["repeat_purchase"]
        .astype(int)
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "customer_repeat_purchase_dataset.csv"
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    target_counts = (
        final_df["repeat_purchase"]
        .value_counts()
        .sort_index()
    )

    repeat_count = int(
        target_counts.get(1, 0)
    )

    no_repeat_count = int(
        target_counts.get(0, 0)
    )

    total_rows = len(
        final_df
    )

    repeat_rate = (
        repeat_count / total_rows
        if total_rows
        else 0
    )

    print()
    print("=" * 90)
    print("FINAL DATASET")
    print("=" * 90)

    print(
        f"Rows              : {total_rows:,}"
    )

    print(
        f"Features + target : {len(final_df.columns)}"
    )

    print(
        f"Repeat purchase   : {repeat_count:,}"
    )

    print(
        f"No repeat         : {no_repeat_count:,}"
    )

    print(
        f"Repeat rate       : {repeat_rate * 100:.2f}%"
    )

    print()
    print("Target distribution:")
    print(
        final_df["repeat_purchase"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    report = {

        "dataset": "Olist Brazilian E-Commerce Public Dataset",

        "future_observation_window_days":
            FUTURE_DAYS,

        "snapshots_used":
            [
                str(x.date())
                for x in snapshot_dates
            ],

        "rows":
            total_rows,

        "repeat_purchase_count":
            repeat_count,

        "no_repeat_purchase_count":
            no_repeat_count,

        "repeat_purchase_rate":
            repeat_rate,

        "target_definition":
            "1 = customer made at least one completed purchase during the 180-day future observation window; 0 = no completed purchase during that window.",

        "leakage_rule":
            "Future purchase information is used only to construct the target and is excluded from model features.",

        "model_features": [
            column
            for column in final_columns
            if column not in [
                "customer_unique_id",
                "snapshot_date",
                "future_end_date",
                "first_purchase_date",
                "last_purchase_date",
                "repeat_purchase"
            ]
        ]
    }

    report_path = (
        REPORT_DIR
        / "repeat_purchase_dataset_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print()
    print("=" * 90)
    print("FILES SAVED")
    print("=" * 90)

    print(output_path)
    print(report_path)

    print()
    print("=" * 90)
    print("REPEAT PURCHASE DATASET COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()

