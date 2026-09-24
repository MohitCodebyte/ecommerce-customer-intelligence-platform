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
OUTPUT_FILE = OUTPUT_DIR / "churn_observation_analysis.json"


# ============================================================
# OBSERVATION WINDOWS
# ============================================================

OBSERVATION_WINDOWS = [90, 120, 180, 240, 300, 365]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 80)
    print("CHURN OBSERVATION ANALYSIS")
    print("=" * 80)

    print("\nLoading customers...")
    customers = pd.read_csv(CUSTOMERS_FILE)

    print("Loading orders...")
    orders = pd.read_csv(ORDERS_FILE)

    return customers, orders


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(customers, orders):

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce"
    )

    orders = orders[
        orders["order_status"].eq("delivered")
    ].copy()

    orders = orders[
        orders["order_purchase_timestamp"].notna()
    ].copy()

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
# CUSTOMER OBSERVATION DATA
# ============================================================

def create_customer_observation_data(orders):

    dataset_end_date = orders[
        "order_purchase_timestamp"
    ].max()

    customer_data = (
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

    customer_data["dataset_end_date"] = dataset_end_date

    customer_data["days_observed_after_last_purchase"] = (
        customer_data["dataset_end_date"]
        - customer_data["last_purchase_date"]
    ).dt.days

    customer_data["repeat_customer"] = (
        customer_data["total_orders"] > 1
    )

    return customer_data


# ============================================================
# OBSERVATION WINDOW ANALYSIS
# ============================================================

def analyze_observation_windows(customer_data):

    results = []

    total_customers = len(customer_data)

    for window in OBSERVATION_WINDOWS:

        eligible = customer_data[
            customer_data[
                "days_observed_after_last_purchase"
            ] >= window
        ].copy()

        not_eligible = customer_data[
            customer_data[
                "days_observed_after_last_purchase"
            ] < window
        ].copy()

        eligible_count = len(eligible)
        not_eligible_count = len(not_eligible)

        eligible_percent = (
            eligible_count / total_customers * 100
            if total_customers > 0
            else 0
        )

        repeat_eligible = eligible[
            eligible["repeat_customer"]
        ]

        repeat_eligible_count = len(
            repeat_eligible
        )

        results.append(
            {
                "observation_window_days": window,
                "total_customers": total_customers,
                "eligible_customers": eligible_count,
                "not_eligible_customers": not_eligible_count,
                "eligible_percent": round(
                    eligible_percent,
                    2
                ),
                "eligible_repeat_customers": (
                    repeat_eligible_count
                )
            }
        )

    return results


# ============================================================
# CUSTOMER OBSERVATION SUMMARY
# ============================================================

def observation_summary(customer_data):

    summary = {
        "total_customers": int(
            len(customer_data)
        ),
        "repeat_customers": int(
            customer_data["repeat_customer"].sum()
        ),
        "single_purchase_customers": int(
            (~customer_data["repeat_customer"]).sum()
        ),
        "dataset_start_date": (
            customer_data["first_purchase_date"]
            .min()
            .strftime("%Y-%m-%d")
        ),
        "dataset_end_date": (
            customer_data["dataset_end_date"]
            .max()
            .strftime("%Y-%m-%d")
        ),
        "median_days_observed_after_last_purchase": round(
            float(
                customer_data[
                    "days_observed_after_last_purchase"
                ].median()
            ),
            2
        ),
        "mean_days_observed_after_last_purchase": round(
            float(
                customer_data[
                    "days_observed_after_last_purchase"
                ].mean()
            ),
            2
        )
    }

    return summary


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    customer_data,
    results
):

    summary = observation_summary(
        customer_data
    )

    print("\n" + "=" * 80)
    print("CUSTOMER OBSERVATION SUMMARY")
    print("=" * 80)

    print(
        f"\nTotal customers              : "
        f"{summary['total_customers']:,}"
    )

    print(
        f"Repeat customers             : "
        f"{summary['repeat_customers']:,}"
    )

    print(
        f"Single-purchase customers    : "
        f"{summary['single_purchase_customers']:,}"
    )

    print(
        f"Dataset start                : "
        f"{summary['dataset_start_date']}"
    )

    print(
        f"Dataset end                  : "
        f"{summary['dataset_end_date']}"
    )

    print(
        f"Median observation period    : "
        f"{summary['median_days_observed_after_last_purchase']} days"
    )

    print(
        f"Mean observation period      : "
        f"{summary['mean_days_observed_after_last_purchase']} days"
    )

    print("\n" + "=" * 80)
    print("OBSERVATION WINDOW COMPARISON")
    print("=" * 80)

    print(
        f"\n{'Window':<15}"
        f"{'Eligible':<15}"
        f"{'Excluded':<15}"
        f"{'Eligible %':<15}"
        f"{'Repeat Eligible':<20}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{str(result['observation_window_days']) + ' days':<15}"
            f"{result['eligible_customers']:<15}"
            f"{result['not_eligible_customers']:<15}"
            f"{str(result['eligible_percent']) + '%':<15}"
            f"{result['eligible_repeat_customers']:<20}"
        )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    customer_data,
    results
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
        "observation_definition": (
            "Customer is eligible for a churn label "
            "only when the dataset contains at least "
            "the selected number of days after the "
            "customer's last purchase."
        ),
        "summary": observation_summary(
            customer_data
        ),
        "observation_windows": results
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

    orders = prepare_data(
        customers,
        orders
    )

    print(
        f"\nCompleted orders : "
        f"{len(orders):,}"
    )

    customer_data = (
        create_customer_observation_data(
            orders
        )
    )

    print(
        f"Unique customers : "
        f"{len(customer_data):,}"
    )

    results = analyze_observation_windows(
        customer_data
    )

    print_results(
        customer_data,
        results
    )

    save_report(
        customer_data,
        results
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
