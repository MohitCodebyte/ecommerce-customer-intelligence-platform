from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CUSTOMERS_FILE = PROJECT_ROOT / "data" / "raw" / "olist_customers_dataset.csv"
ORDERS_FILE = PROJECT_ROOT / "data" / "raw" / "olist_orders_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_FILE = OUTPUT_DIR / "churn_window_analysis.json"


# ============================================================
# CHURN WINDOWS TO TEST
# ============================================================

CHURN_WINDOWS = [90, 120, 180, 240]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 80)
    print("CHURN WINDOW ANALYSIS")
    print("=" * 80)

    print("\nLoading customers...")
    customers = pd.read_csv(CUSTOMERS_FILE)

    print("Loading orders...")
    orders = pd.read_csv(ORDERS_FILE)

    return customers, orders


# ============================================================
# PREPARE CUSTOMER PURCHASE DATA
# ============================================================

def prepare_data(customers, orders):

    # Convert purchase timestamp
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce"
    )

    # Keep only completed/delivered orders
    orders = orders[
        orders["order_status"].eq("delivered")
    ].copy()

    # Remove rows where purchase date is unavailable
    orders = orders[
        orders["order_purchase_timestamp"].notna()
    ].copy()

    # Map order-level customer_id to persistent customer_unique_id
    customer_map = customers[
        ["customer_id", "customer_unique_id"]
    ].drop_duplicates()

    orders = orders.merge(
        customer_map,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    # Remove rows where customer identity is unavailable
    orders = orders[
        orders["customer_unique_id"].notna()
    ].copy()

    # Sort purchases chronologically
    orders = orders.sort_values(
        ["customer_unique_id", "order_purchase_timestamp"]
    )

    return orders


# ============================================================
# CREATE CUSTOMER LEVEL DATA
# ============================================================

def create_customer_history(orders):

    customer_history = (
        orders
        .groupby("customer_unique_id")
        .agg(
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
            )
        )
        .reset_index()
    )

    return customer_history


# ============================================================
# ANALYZE CHURN WINDOWS
# ============================================================

def analyze_windows(customer_history, analysis_date):

    results = []

    total_customers = len(customer_history)

    for window in CHURN_WINDOWS:

        cutoff_date = analysis_date - pd.Timedelta(
            days=window
        )

        churned = customer_history[
            customer_history["last_purchase_date"] < cutoff_date
        ].copy()

        active = customer_history[
            customer_history["last_purchase_date"] >= cutoff_date
        ].copy()

        churned_count = len(churned)
        active_count = len(active)

        churn_rate = (
            churned_count / total_customers * 100
            if total_customers > 0
            else 0
        )

        results.append(
            {
                "churn_window_days": window,
                "analysis_date": analysis_date.strftime("%Y-%m-%d"),
                "total_customers": total_customers,
                "churned_customers": churned_count,
                "active_customers": active_count,
                "churn_rate_percent": round(
                    churn_rate,
                    2
                )
            }
        )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print("\n" + "=" * 80)
    print("CHURN WINDOW COMPARISON")
    print("=" * 80)

    print(
        f"\n{'Window':<15}"
        f"{'Churned':<15}"
        f"{'Active':<15}"
        f"{'Churn Rate':<15}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{str(result['churn_window_days']) + ' days':<15}"
            f"{result['churned_customers']:<15}"
            f"{result['active_customers']:<15}"
            f"{str(result['churn_rate_percent']) + '%':<15}"
        )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(results, analysis_date):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "analysis": {
            "analysis_date": analysis_date.strftime("%Y-%m-%d"),
            "dataset": "Olist Brazilian E-Commerce Public Dataset",
            "order_filter": "delivered",
            "customer_identity": "customer_unique_id",
            "churn_windows_tested": CHURN_WINDOWS
        },
        "results": results
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

    print(f"\nSaved: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    # Load
    customers, orders = load_data()

    # Prepare
    orders = prepare_data(
        customers,
        orders
    )

    print(
        f"\nCompleted orders used       : {len(orders):,}"
    )

    # Customer history
    customer_history = create_customer_history(
        orders
    )

    print(
        f"Unique customers analyzed  : "
        f"{len(customer_history):,}"
    )

    # IMPORTANT:
    # Use the latest purchase date available in the dataset
    # as the analysis/reference date.
    analysis_date = orders[
        "order_purchase_timestamp"
    ].max()

    print(
        f"Analysis/reference date     : "
        f"{analysis_date.strftime('%Y-%m-%d')}"
    )

    # Analyze windows
    results = analyze_windows(
        customer_history,
        analysis_date
    )

    # Display
    print_results(results)

    # Save
    save_report(
        results,
        analysis_date
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()