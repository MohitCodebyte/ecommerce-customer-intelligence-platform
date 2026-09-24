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

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


WINDOWS = [90, 120, 180, 240, 300, 365]


def main():

    print("=" * 90)
    print("CHURN TARGET REDESIGN ANALYSIS")
    print("=" * 90)

    if not ORDERS_FILE.exists():
        raise FileNotFoundError(f"Orders file not found:\n{ORDERS_FILE}")

    if not CUSTOMERS_FILE.exists():
        raise FileNotFoundError(
            f"Customers file not found:\n{CUSTOMERS_FILE}"
        )

    orders = pd.read_csv(ORDERS_FILE)
    customers = pd.read_csv(CUSTOMERS_FILE)

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # COMPLETED ORDERS ONLY
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

    completed = completed.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp"
        ]
    )

    dataset_end = completed["order_purchase_timestamp"].max()

    print()
    print(f"Completed orders : {len(completed):,}")
    print(
        f"Unique customers : "
        f"{completed['customer_unique_id'].nunique():,}"
    )
    print(
        f"Dataset end      : "
        f"{dataset_end.date()}"
    )

    # ---------------------------------------------------------
    # CUSTOMER PURCHASE HISTORY
    # ---------------------------------------------------------

    customer_history = (
        completed
        .groupby("customer_unique_id")
        ["order_purchase_timestamp"]
        .agg(
            first_purchase="min",
            last_purchase="max",
            total_orders="count"
        )
        .reset_index()
    )

    customer_history["repeat_customer"] = (
        customer_history["total_orders"] > 1
    )

    # ---------------------------------------------------------
    # WINDOW ANALYSIS
    # ---------------------------------------------------------

    results = []

    for window in WINDOWS:

        snapshot_date = (
            dataset_end
            - pd.Timedelta(days=window)
        )

        future_end = dataset_end

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

        # Customers observable at snapshot
        customers_at_snapshot = (
            historical["customer_unique_id"]
            .drop_duplicates()
        )

        history_counts = (
            historical
            .groupby("customer_unique_id")
            .size()
            .rename("historical_orders")
        )

        future_counts = (
            future
            .groupby("customer_unique_id")
            .size()
            .rename("future_orders")
        )

        analysis = pd.DataFrame(
            {
                "customer_unique_id":
                    customers_at_snapshot
            }
        )

        analysis = analysis.set_index(
            "customer_unique_id"
        )

        analysis = analysis.join(
            history_counts,
            how="left"
        )

        analysis = analysis.join(
            future_counts,
            how="left"
        )

        analysis["historical_orders"] = (
            analysis["historical_orders"]
            .fillna(0)
            .astype(int)
        )

        analysis["future_orders"] = (
            analysis["future_orders"]
            .fillna(0)
            .astype(int)
        )

        analysis["repeat_customer"] = (
            analysis["historical_orders"] > 1
        )

        analysis["returned"] = (
            analysis["future_orders"] > 0
        )

        analysis["churn"] = (
            analysis["future_orders"] == 0
        )

        total = len(analysis)

        churn_count = int(
            analysis["churn"].sum()
        )

        non_churn_count = int(
            analysis["returned"].sum()
        )

        repeat_analysis = analysis[
            analysis["repeat_customer"]
        ]

        repeat_total = len(
            repeat_analysis
        )

        repeat_returned = int(
            repeat_analysis["returned"].sum()
        )

        repeat_churned = int(
            repeat_analysis["churn"].sum()
        )

        single_analysis = analysis[
            ~analysis["repeat_customer"]
        ]

        single_total = len(
            single_analysis
        )

        single_returned = int(
            single_analysis["returned"].sum()
        )

        single_churned = int(
            single_analysis["churn"].sum()
        )

        results.append(
            {
                "window_days": window,

                "snapshot_date":
                    str(snapshot_date.date()),

                "future_end":
                    str(future_end.date()),

                "customers_at_snapshot":
                    total,

                "churn_count":
                    churn_count,

                "non_churn_count":
                    non_churn_count,

                "churn_rate":
                    round(
                        churn_count / total,
                        6
                    )
                    if total else None,

                "non_churn_rate":
                    round(
                        non_churn_count / total,
                        6
                    )
                    if total else None,

                "repeat_customers":
                    repeat_total,

                "repeat_returned":
                    repeat_returned,

                "repeat_churned":
                    repeat_churned,

                "repeat_return_rate":
                    round(
                        repeat_returned / repeat_total,
                        6
                    )
                    if repeat_total else None,

                "single_customers":
                    single_total,

                "single_returned":
                    single_returned,

                "single_churned":
                    single_churned,

                "single_return_rate":
                    round(
                        single_returned / single_total,
                        6
                    )
                    if single_total else None,

                "class_ratio_churn_to_non_churn":
                    round(
                        churn_count / non_churn_count,
                        2
                    )
                    if non_churn_count else None
            }
        )

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("WINDOW COMPARISON")
    print("=" * 90)

    display_columns = [
        "window_days",
        "customers_at_snapshot",
        "churn_count",
        "non_churn_count",
        "churn_rate",
        "repeat_customers",
        "repeat_returned",
        "repeat_return_rate",
        "single_customers",
        "single_returned",
        "single_return_rate",
        "class_ratio_churn_to_non_churn"
    ]

    print(
        results_df[
            display_columns
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # REPEAT-CUSTOMER-ONLY ANALYSIS
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("REPEAT CUSTOMER TARGET ANALYSIS")
    print("=" * 90)

    repeat_results = []

    for window in WINDOWS:

        snapshot_date = (
            dataset_end
            - pd.Timedelta(days=window)
        )

        historical = completed[
            completed["order_purchase_timestamp"]
            <= snapshot_date
        ]

        future = completed[
            (
                completed["order_purchase_timestamp"]
                > snapshot_date
            )
            &
            (
                completed["order_purchase_timestamp"]
                <= dataset_end
            )
        ]

        historical_counts = (
            historical
            .groupby("customer_unique_id")
            .size()
        )

        repeat_ids = historical_counts[
            historical_counts > 1
        ].index

        future_counts = (
            future
            .groupby("customer_unique_id")
            .size()
        )

        repeat_future = (
            future_counts
            .reindex(repeat_ids)
            .fillna(0)
        )

        returned = (
            repeat_future > 0
        )

        total_repeat = len(
            repeat_future
        )

        returned_count = int(
            returned.sum()
        )

        churned_count = (
            total_repeat
            - returned_count
        )

        repeat_results.append(
            {
                "window_days": window,

                "repeat_customers":
                    total_repeat,

                "returned":
                    returned_count,

                "churned":
                    churned_count,

                "return_rate":
                    round(
                        returned_count
                        / total_repeat,
                        6
                    )
                    if total_repeat else None,

                "churn_rate":
                    round(
                        churned_count
                        / total_repeat,
                        6
                    )
                    if total_repeat else None
            }
        )

    repeat_df = pd.DataFrame(
        repeat_results
    )

    print(
        repeat_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------------------

    report = {
        "dataset_end":
            str(dataset_end.date()),

        "windows":
            results,

        "repeat_customer_analysis":
            repeat_results,

        "interpretation": {
            "purpose":
                "Compare historical inactivity windows before selecting a churn target.",

            "important_rule":
                "Future purchase behavior is used only to construct the target, not as a model feature.",

            "single_customer_warning":
                "Single-purchase customers may represent non-repeat behavior rather than established churn.",

            "next_step":
                "Select a defensible target definition and build a leakage-safe historical churn dataset."
        }
    }

    report_path = (
        REPORT_DIR
        / "churn_target_redesign.json"
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

    csv_path = (
        REPORT_DIR
        / "churn_target_window_comparison.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    repeat_csv_path = (
        REPORT_DIR
        / "repeat_customer_target_analysis.csv"
    )

    repeat_df.to_csv(
        repeat_csv_path,
        index=False
    )

    print()
    print("=" * 90)
    print("REPORTS SAVED")
    print("=" * 90)

    print(report_path)
    print(csv_path)
    print(repeat_csv_path)

    print()
    print("=" * 90)
    print("TARGET REDESIGN ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()

