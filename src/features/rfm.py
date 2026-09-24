from pathlib import Path
import json
import pandas as pd
import numpy as np


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

PAYMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_order_payments_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_FILE = OUTPUT_DIR / "customer_rfm.csv"
REPORT_FILE = REPORT_DIR / "rfm_analysis.json"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 80)
    print("CUSTOMER RFM ANALYSIS")
    print("=" * 80)

    print("\nLoading customers...")
    customers = pd.read_csv(CUSTOMERS_FILE)

    print("Loading orders...")
    orders = pd.read_csv(ORDERS_FILE)

    print("Loading payments...")
    payments = pd.read_csv(PAYMENTS_FILE)

    return customers, orders, payments


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(customers, orders, payments):

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce"
    )

    # Only completed orders
    orders = orders[
        orders["order_status"].eq("delivered")
    ].copy()

    orders = orders[
        orders["order_purchase_timestamp"].notna()
    ].copy()

    # customer_id -> customer_unique_id
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
        orders["customer_unique_id"].notna()
    ].copy()

    # Payment value per order.
    # Payments can contain multiple rows for one order,
    # therefore aggregate before merging.
    payment_summary = (
        payments
        .groupby("order_id")
        .agg(
            order_payment_value=(
                "payment_value",
                "sum"
            )
        )
        .reset_index()
    )

    orders = orders.merge(
        payment_summary,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    orders["order_payment_value"] = (
        orders["order_payment_value"]
        .fillna(0)
    )

    return orders


# ============================================================
# BUILD RFM DATASET
# ============================================================

def calculate_rfm(orders):

    analysis_date = (
        orders["order_purchase_timestamp"].max()
        + pd.Timedelta(days=1)
    )

    customer_rfm = (
        orders
        .groupby("customer_unique_id")
        .agg(
            last_purchase_date=(
                "order_purchase_timestamp",
                "max"
            ),
            frequency=(
                "order_id",
                "nunique"
            ),
            monetary=(
                "order_payment_value",
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Recency
    # --------------------------------------------------------

    customer_rfm["recency"] = (
        analysis_date
        - customer_rfm["last_purchase_date"]
    ).dt.days

    # --------------------------------------------------------
    # Monetary average
    # --------------------------------------------------------

    customer_rfm["average_order_value"] = (
        customer_rfm["monetary"]
        / customer_rfm["frequency"]
    )

    # --------------------------------------------------------
    # Repeat customer
    # --------------------------------------------------------

    customer_rfm["repeat_customer"] = (
        customer_rfm["frequency"] > 1
    ).astype(int)

    # --------------------------------------------------------
    # Basic cleanup
    # --------------------------------------------------------

    customer_rfm["monetary"] = (
        customer_rfm["monetary"]
        .round(2)
    )

    customer_rfm["average_order_value"] = (
        customer_rfm["average_order_value"]
        .round(2)
    )

    customer_rfm = customer_rfm[
        [
            "customer_unique_id",
            "last_purchase_date",
            "recency",
            "frequency",
            "monetary",
            "average_order_value",
            "repeat_customer"
        ]
    ].copy()

    return customer_rfm, analysis_date


# ============================================================
# RFM SCORES
# ============================================================

def create_rfm_scores(customer_rfm):

    # Recency:
    # lower is better -> reverse scoring
    customer_rfm["R_score"] = pd.qcut(
        customer_rfm["recency"],
        q=5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop"
    )

    # Frequency:
    # higher is better
    customer_rfm["F_score"] = pd.qcut(
        customer_rfm["frequency"].rank(
            method="first"
        ),
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
    )

    # Monetary:
    # higher is better
    customer_rfm["M_score"] = pd.qcut(
        customer_rfm["monetary"].rank(
            method="first"
        ),
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
    )

    customer_rfm["R_score"] = (
        customer_rfm["R_score"].astype(int)
    )

    customer_rfm["F_score"] = (
        customer_rfm["F_score"].astype(int)
    )

    customer_rfm["M_score"] = (
        customer_rfm["M_score"].astype(int)
    )

    customer_rfm["RFM_score"] = (
        customer_rfm["R_score"].astype(str)
        + customer_rfm["F_score"].astype(str)
        + customer_rfm["M_score"].astype(str)
    )

    customer_rfm["RFM_total"] = (
        customer_rfm["R_score"]
        + customer_rfm["F_score"]
        + customer_rfm["M_score"]
    )

    return customer_rfm


# ============================================================
# BUSINESS SEGMENTS
# ============================================================

def assign_segments(customer_rfm):

    def segment(row):

        r = row["R_score"]
        f = row["F_score"]
        m = row["M_score"]

        # High-value recent customers
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"

        # Recent + frequent customers
        if r >= 4 and f >= 3:
            return "Loyal Customers"

        # Recent but low frequency
        if r >= 4 and f <= 2:
            return "New / Promising"

        # Good frequency but losing recency
        if r == 3 and f >= 3:
            return "Potential Loyalists"

        # Previously valuable but becoming inactive
        if r <= 2 and f >= 4 and m >= 4:
            return "At Risk High Value"

        # High monetary value but lower recency
        if r <= 2 and m >= 4:
            return "At Risk Valuable"

        # Low recent activity
        if r <= 2 and f <= 2:
            return "Hibernating"

        return "Needs Attention"

    customer_rfm["customer_segment"] = (
        customer_rfm.apply(
            segment,
            axis=1
        )
    )

    return customer_rfm


# ============================================================
# SEGMENT SUMMARY
# ============================================================

def create_segment_summary(customer_rfm):

    summary = (
        customer_rfm
        .groupby("customer_segment")
        .agg(
            customers=(
                "customer_unique_id",
                "count"
            ),
            avg_recency=(
                "recency",
                "mean"
            ),
            avg_frequency=(
                "frequency",
                "mean"
            ),
            avg_monetary=(
                "monetary",
                "mean"
            ),
            total_monetary=(
                "monetary",
                "sum"
            )
        )
        .reset_index()
    )

    summary["avg_recency"] = (
        summary["avg_recency"]
        .round(2)
    )

    summary["avg_frequency"] = (
        summary["avg_frequency"]
        .round(2)
    )

    summary["avg_monetary"] = (
        summary["avg_monetary"]
        .round(2)
    )

    summary["total_monetary"] = (
        summary["total_monetary"]
        .round(2)
    )

    return summary


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    customer_rfm,
    segment_summary,
    analysis_date
):

    print("\n" + "=" * 80)
    print("RFM SUMMARY")
    print("=" * 80)

    print(
        f"\nAnalysis date : "
        f"{analysis_date.strftime('%Y-%m-%d')}"
    )

    print(
        f"Customers     : "
        f"{len(customer_rfm):,}"
    )

    print(
        f"Repeat customers : "
        f"{customer_rfm['repeat_customer'].sum():,}"
    )

    print("\n" + "-" * 80)
    print("CUSTOMER SEGMENTS")
    print("-" * 80)

    print(
        f"\n{'Segment':<25}"
        f"{'Customers':<15}"
        f"{'Avg Recency':<15}"
        f"{'Avg Frequency':<17}"
        f"{'Avg Monetary':<15}"
    )

    print("-" * 90)

    for _, row in segment_summary.iterrows():

        print(
            f"{row['customer_segment']:<25}"
            f"{int(row['customers']):<15}"
            f"{row['avg_recency']:<15}"
            f"{row['avg_frequency']:<17}"
            f"{row['avg_monetary']:<15}"
        )


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_outputs(
    customer_rfm,
    segment_summary,
    analysis_date
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Customer-level RFM
    customer_rfm.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Report
    report = {
        "dataset": (
            "Olist Brazilian E-Commerce Public Dataset"
        ),
        "order_filter": "delivered",
        "customer_identity": (
            "customer_unique_id"
        ),
        "analysis_date": (
            analysis_date.strftime(
                "%Y-%m-%d"
            )
        ),
        "total_customers": int(
            len(customer_rfm)
        ),
        "repeat_customers": int(
            customer_rfm[
                "repeat_customer"
            ].sum()
        ),
        "segments": (
            segment_summary
            .to_dict(
                orient="records"
            )
        )
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\n" + "=" * 80)
    print("RFM ANALYSIS COMPLETED")
    print("=" * 80)

    print(
        f"\nSaved customer RFM : {OUTPUT_FILE}"
    )

    print(
        f"Saved report       : {REPORT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    customers, orders, payments = load_data()

    orders = prepare_data(
        customers,
        orders,
        payments
    )

    print(
        f"\nCompleted orders used : "
        f"{orders['order_id'].nunique():,}"
    )

    print(
        f"Customers available   : "
        f"{orders['customer_unique_id'].nunique():,}"
    )

    customer_rfm, analysis_date = (
        calculate_rfm(
            orders
        )
    )

    customer_rfm = create_rfm_scores(
        customer_rfm
    )

    customer_rfm = assign_segments(
        customer_rfm
    )

    segment_summary = (
        create_segment_summary(
            customer_rfm
        )
    )

    print_results(
        customer_rfm,
        segment_summary,
        analysis_date
    )

    save_outputs(
        customer_rfm,
        segment_summary,
        analysis_date
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
