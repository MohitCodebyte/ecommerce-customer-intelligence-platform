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
OUTPUT_FILE = OUTPUT_DIR / "churn_validation.json"


# ============================================================
# VALIDATION WINDOWS
# ============================================================

VALIDATION_WINDOWS = [90, 120, 180, 240, 300, 365]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 80)
    print("CHURN BEHAVIOR VALIDATION")
    print("=" * 80)

    print("\nLoading customers...")
    customers = pd.read_csv(CUSTOMERS_FILE)

    print("Loading orders...")
    orders = pd.read_csv(ORDERS_FILE)

    return customers, orders


# ============================================================
# PREPARE ORDER DATA
# ============================================================

def prepare_orders(customers, orders):

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce"
    )

    # Only completed purchases
    orders = orders[
        orders["order_status"].eq("delivered")
    ].copy()

    orders = orders[
        orders["order_purchase_timestamp"].notna()
    ].copy()

    # customer_id -> customer_unique_id
    customer_map = customers[
        ["customer_id", "customer_unique_id"]
    ].drop_duplicates()

    orders = orders.merge(
        customer_map,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    orders = orders[
        orders["customer_unique_id"].notna()
    ].copy()

    orders = orders.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp"
        ]
    )

    return orders


# ============================================================
# BUILD CUSTOMER PURCHASE HISTORY
# ============================================================

def build_customer_history(orders):

    history = (
        orders
        .groupby("customer_unique_id")
        ["order_purchase_timestamp"]
        .apply(list)
        .reset_index(name="purchase_dates")
    )

    # Only customers with at least 2 purchases
    history["purchase_count"] = history[
        "purchase_dates"
    ].apply(len)

    repeat_customers = history[
        history["purchase_count"] >= 2
    ].copy()

    return repeat_customers


# ============================================================
# CALCULATE PURCHASE GAP BEHAVIOR
# ============================================================

def calculate_gaps(history):

    records = []

    for _, row in history.iterrows():

        customer_id = row["customer_unique_id"]
        dates = row["purchase_dates"]

        dates = sorted(dates)

        for i in range(len(dates) - 1):

            previous_purchase = dates[i]
            next_purchase = dates[i + 1]

            gap_days = (
                next_purchase - previous_purchase
            ).total_seconds() / 86400

            records.append(
                {
                    "customer_unique_id": customer_id,
                    "previous_purchase_date": (
                        previous_purchase.strftime("%Y-%m-%d")
                    ),
                    "next_purchase_date": (
                        next_purchase.strftime("%Y-%m-%d")
                    ),
                    "gap_days": round(gap_days, 2)
                }
            )

    return pd.DataFrame(records)


# ============================================================
# VALIDATE CHURN WINDOWS
# ============================================================

def validate_windows(gaps):

    results = []

    total_gaps = len(gaps)

    for window in VALIDATION_WINDOWS:

        # Customers who returned after being inactive
        # for at least this many days.
        qualifying = gaps[
            gaps["gap_days"] >= window
        ].copy()

        qualifying_count = len(qualifying)

        percentage = (
            qualifying_count / total_gaps * 100
            if total_gaps > 0
            else 0
        )

        results.append(
            {
                "window_days": window,
                "return_gaps_at_or_above_window": qualifying_count,
                "percentage_of_repeat_purchase_gaps": round(
                    percentage,
                    2
                )
            }
        )

    return results


# ============================================================
# GAP DISTRIBUTION
# ============================================================

def gap_statistics(gaps):

    if gaps.empty:
        return {}

    return {
        "gap_records": int(len(gaps)),
        "mean_gap_days": round(
            float(gaps["gap_days"].mean()),
            2
        ),
        "median_gap_days": round(
            float(gaps["gap_days"].median()),
            2
        ),
        "75th_percentile_days": round(
            float(gaps["gap_days"].quantile(0.75)),
            2
        ),
        "90th_percentile_days": round(
            float(gaps["gap_days"].quantile(0.90)),
            2
        ),
        "95th_percentile_days": round(
            float(gaps["gap_days"].quantile(0.95)),
            2
        ),
        "99th_percentile_days": round(
            float(gaps["gap_days"].quantile(0.99)),
            2
        )
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    history,
    gaps,
    validation_results
):

    print("\n" + "=" * 80)
    print("CUSTOMER RETURN BEHAVIOR")
    print("=" * 80)

    print(
        f"\nRepeat customers analyzed : "
        f"{len(history):,}"
    )

    print(
        f"Repeat purchase gaps      : "
        f"{len(gaps):,}"
    )

    print("\n" + "-" * 80)

    print(
        f"{'Window':<15}"
        f"{'Long-gap returns':<25}"
        f"{'% of repeat gaps':<20}"
    )

    print("-" * 60)

    for result in validation_results:

        print(
            f"{str(result['window_days']) + ' days':<15}"
            f"{result['return_gaps_at_or_above_window']:<25}"
            f"{str(result['percentage_of_repeat_purchase_gaps']) + '%':<20}"
        )

    print("\n" + "=" * 80)
    print("PURCHASE GAP STATISTICS")
    print("=" * 80)

    stats = gap_statistics(gaps)

    for key, value in stats.items():

        print(
            f"{key:<30}: {value}"
        )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    history,
    gaps,
    validation_results
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
        "customer_identity": "customer_unique_id",
        "repeat_customer_definition": (
            "customer with at least 2 delivered orders"
        ),
        "repeat_customers_analyzed": int(
            len(history)
        ),
        "purchase_gap_records": int(
            len(gaps)
        ),
        "gap_statistics": gap_statistics(
            gaps
        ),
        "validation_windows": validation_results
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
    print("VALIDATION COMPLETED")
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

    history = build_customer_history(
        orders
    )

    print(
        f"Repeat customers : "
        f"{len(history):,}"
    )

    gaps = calculate_gaps(
        history
    )

    if gaps.empty:

        print(
            "\nNo repeat purchase gaps found."
        )

        return

    validation_results = validate_windows(
        gaps
    )

    print_results(
        history,
        gaps,
        validation_results
    )

    save_report(
        history,
        gaps,
        validation_results
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
