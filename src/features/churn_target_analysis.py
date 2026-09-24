from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CUSTOMERS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_customers_dataset.csv"
)

ORDERS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_orders_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "churn_target_analysis.json"
)


# ============================================================
# WINDOWS TO ANALYZE
# ============================================================

ANALYSIS_WINDOWS = [
    90,
    120,
    180,
    240
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 80)
    print("CHURN TARGET ANALYSIS")
    print("=" * 80)

    print("\nLoading customers...")
    customers = pd.read_csv(
        CUSTOMERS_FILE
    )

    print("Loading orders...")
    orders = pd.read_csv(
        ORDERS_FILE
    )

    return customers, orders


# ============================================================
# PREPARE ORDERS
# ============================================================

def prepare_orders(
    customers,
    orders
):

    orders[
        "order_purchase_timestamp"
    ] = pd.to_datetime(
        orders[
            "order_purchase_timestamp"
        ],
        errors="coerce"
    )

    # Completed purchases only
    orders = orders[
        orders[
            "order_status"
        ].eq("delivered")
    ].copy()

    orders = orders[
        orders[
            "order_purchase_timestamp"
        ].notna()
    ].copy()

    # Map customer_id -> customer_unique_id
    customer_map = customers[
        [
            "customer_id",
            "customer_unique_id"
        ]
    ].drop_duplicates()

    orders = orders.merge(
        customer_map,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    orders = orders[
        orders[
            "customer_unique_id"
        ].notna()
    ].copy()

    orders = orders.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp"
        ]
    )

    return orders


# ============================================================
# CREATE SNAPSHOT
# ============================================================

def create_snapshot(
    orders,
    window
):

    dataset_end = orders[
        "order_purchase_timestamp"
    ].max()

    snapshot_date = (
        dataset_end
        - pd.Timedelta(
            days=window
        )
    )

    historical_orders = orders[
        orders[
            "order_purchase_timestamp"
        ] <= snapshot_date
    ].copy()

    future_orders = orders[
        (
            orders[
                "order_purchase_timestamp"
            ] > snapshot_date
        )
        &
        (
            orders[
                "order_purchase_timestamp"
            ] <= dataset_end
        )
    ].copy()

    return (
        snapshot_date,
        dataset_end,
        historical_orders,
        future_orders
    )


# ============================================================
# BUILD CUSTOMER SEGMENTS
# ============================================================

def build_customer_segments(
    historical_orders,
    future_orders
):

    customer_history = (
        historical_orders
        .groupby(
            "customer_unique_id"
        )
        .agg(
            historical_orders=(
                "order_id",
                "nunique"
            ),
            last_purchase_date=(
                "order_purchase_timestamp",
                "max"
            )
        )
        .reset_index()
    )

    customer_history[
        "customer_type"
    ] = "single_purchase"

    customer_history.loc[
        customer_history[
            "historical_orders"
        ] > 1,
        "customer_type"
    ] = "repeat_customer"

    future_customers = set(
        future_orders[
            "customer_unique_id"
        ].unique()
    )

    customer_history[
        "future_purchase"
    ] = (
        customer_history[
            "customer_unique_id"
        ]
        .isin(
            future_customers
        )
        .astype(int)
    )

    customer_history[
        "churn_label"
    ] = (
        customer_history[
            "future_purchase"
        ] == 0
    ).astype(int)

    return customer_history


# ============================================================
# ANALYZE CUSTOMER GROUP
# ============================================================

def analyze_group(
    data,
    group_name
):

    total = len(data)

    future_purchase_count = int(
        data[
            "future_purchase"
        ].sum()
    )

    no_future_purchase_count = (
        total
        - future_purchase_count
    )

    future_purchase_rate = (
        future_purchase_count
        / total
        * 100
        if total > 0
        else 0
    )

    churn_rate = (
        no_future_purchase_count
        / total
        * 100
        if total > 0
        else 0
    )

    return {
        "group": group_name,
        "customers": int(total),
        "future_purchase_customers": (
            future_purchase_count
        ),
        "no_future_purchase_customers": (
            no_future_purchase_count
        ),
        "future_purchase_rate_percent": round(
            future_purchase_rate,
            2
        ),
        "churn_rate_percent": round(
            churn_rate,
            2
        )
    }


# ============================================================
# PRINT GROUP ANALYSIS
# ============================================================

def print_group_results(
    results
):

    print("\n" + "=" * 80)
    print("CUSTOMER GROUP ANALYSIS")
    print("=" * 80)

    print(
        f"\n{'Group':<25}"
        f"{'Customers':<15}"
        f"{'Future Purchase':<20}"
        f"{'Future Rate':<15}"
        f"{'Churn Rate':<15}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['group']:<25}"
            f"{result['customers']:<15}"
            f"{result['future_purchase_customers']:<20}"
            f"{str(result['future_purchase_rate_percent']) + '%':<15}"
            f"{str(result['churn_rate_percent']) + '%':<15}"
        )


# ============================================================
# RUN WINDOW ANALYSIS
# ============================================================

def run_window_analysis(
    orders,
    window
):

    (
        snapshot_date,
        dataset_end,
        historical_orders,
        future_orders
    ) = create_snapshot(
        orders,
        window
    )

    customer_data = build_customer_segments(
        historical_orders,
        future_orders
    )

    results = []

    # All customers
    results.append(
        analyze_group(
            customer_data,
            "All customers"
        )
    )

    # Single purchase
    single_purchase = customer_data[
        customer_data[
            "customer_type"
        ].eq("single_purchase")
    ]

    results.append(
        analyze_group(
            single_purchase,
            "Single-purchase"
        )
    )

    # Repeat customers
    repeat_customers = customer_data[
        customer_data[
            "customer_type"
        ].eq("repeat_customer")
    ]

    results.append(
        analyze_group(
            repeat_customers,
            "Repeat customers"
        )
    )

    return {
        "window_days": window,
        "snapshot_date": (
            snapshot_date.strftime(
                "%Y-%m-%d"
            )
        ),
        "dataset_end_date": (
            dataset_end.strftime(
                "%Y-%m-%d"
            )
        ),
        "results": results
    }


# ============================================================
# PRINT WINDOW SUMMARY
# ============================================================

def print_window_summary(
    window_result
):

    print("\n")
    print("=" * 80)

    print(
        f"WINDOW: "
        f"{window_result['window_days']} DAYS"
    )

    print("=" * 80)

    print(
        f"Snapshot date : "
        f"{window_result['snapshot_date']}"
    )

    print(
        f"Dataset end   : "
        f"{window_result['dataset_end_date']}"
    )

    print_group_results(
        window_result[
            "results"
        ]
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    window_results
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "dataset": (
            "Olist Brazilian E-Commerce Public Dataset"
        ),
        "order_filter": "delivered",
        "customer_identity": (
            "customer_unique_id"
        ),
        "analysis_windows": (
            window_results
        )
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    customers, orders = load_data()

    orders = prepare_orders(
        customers,
        orders
    )

    print(
        f"\nCompleted orders : "
        f"{len(orders):,}"
    )

    print(
        f"Unique customers : "
        f"{orders['customer_unique_id'].nunique():,}"
    )

    all_results = []

    for window in ANALYSIS_WINDOWS:

        result = run_window_analysis(
            orders,
            window
        )

        all_results.append(
            result
        )

        print_window_summary(
            result
        )

    save_report(
        all_results
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
