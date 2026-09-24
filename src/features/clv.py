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

RFM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_rfm.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "customer_clv.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "clv_analysis.json"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 80)
    print("CUSTOMER LIFETIME VALUE ANALYSIS")
    print("=" * 80)

    print("\nLoading customers...")
    customers = pd.read_csv(
        CUSTOMERS_FILE
    )

    print("Loading orders...")
    orders = pd.read_csv(
        ORDERS_FILE
    )

    print("Loading payments...")
    payments = pd.read_csv(
        PAYMENTS_FILE
    )

    print("Loading RFM data...")
    rfm = pd.read_csv(
        RFM_FILE
    )

    return (
        customers,
        orders,
        payments,
        rfm
    )


# ============================================================
# PREPARE ORDER DATA
# ============================================================

def prepare_orders(
    customers,
    orders,
    payments
):

    orders[
        "order_purchase_timestamp"
    ] = pd.to_datetime(
        orders[
            "order_purchase_timestamp"
        ],
        errors="coerce"
    )

    # Completed orders only
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
        orders[
            "customer_unique_id"
        ].notna()
    ].copy()

    # Aggregate payment values per order.
    payment_summary = (
        payments
        .groupby("order_id")
        .agg(
            order_value=(
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

    orders["order_value"] = (
        orders["order_value"]
        .fillna(0)
    )

    return orders


# ============================================================
# CALCULATE HISTORICAL CUSTOMER VALUE
# ============================================================

def calculate_customer_value(orders):

    analysis_date = (
        orders[
            "order_purchase_timestamp"
        ].max()
        + pd.Timedelta(days=1)
    )

    customer_value = (
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
            ),
            total_revenue=(
                "order_value",
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Customer lifetime
    # --------------------------------------------------------

    customer_value[
        "customer_lifetime_days"
    ] = (
        customer_value[
            "last_purchase_date"
        ]
        - customer_value[
            "first_purchase_date"
        ]
    ).dt.days

    # Avoid zero lifetime for one-time buyers.
    customer_value[
        "customer_lifetime_months"
    ] = (
        customer_value[
            "customer_lifetime_days"
        ].clip(lower=1)
        / 30.44
    )

    # --------------------------------------------------------
    # Average Order Value
    # --------------------------------------------------------

    customer_value[
        "average_order_value"
    ] = (
        customer_value[
            "total_revenue"
        ]
        / customer_value[
            "total_orders"
        ]
    )

    # --------------------------------------------------------
    # Historical purchase frequency
    # --------------------------------------------------------

    customer_value[
        "orders_per_month"
    ] = (
        customer_value[
            "total_orders"
        ]
        / customer_value[
            "customer_lifetime_months"
        ]
    )

    # --------------------------------------------------------
    # Revenue per month
    # --------------------------------------------------------

    customer_value[
        "revenue_per_month"
    ] = (
        customer_value[
            "total_revenue"
        ]
        / customer_value[
            "customer_lifetime_months"
        ]
    )

    # --------------------------------------------------------
    # Recency
    # --------------------------------------------------------

    customer_value[
        "recency_days"
    ] = (
        analysis_date
        - customer_value[
            "last_purchase_date"
        ]
    ).dt.days

    # --------------------------------------------------------
    # Repeat customer
    # --------------------------------------------------------

    customer_value[
        "repeat_customer"
    ] = (
        customer_value[
            "total_orders"
        ] > 1
    ).astype(int)

    return (
        customer_value,
        analysis_date
    )


# ============================================================
# SIMPLE CLV ESTIMATE
# ============================================================

def calculate_clv(
    customer_value
):

    # --------------------------------------------------------
    # Historical CLV
    # --------------------------------------------------------
    #
    # This is not a prediction of future revenue.
    #
    # It represents historical customer value:
    #
    # Total revenue generated by customer.
    #
    # --------------------------------------------------------

    customer_value[
        "historical_clv"
    ] = (
        customer_value[
            "total_revenue"
        ]
    )

    # --------------------------------------------------------
    # Annualized revenue estimate
    # --------------------------------------------------------
    #
    # Useful only for customers with meaningful lifetime.
    # It is an analytical normalization, not guaranteed future
    # revenue.
    #
    # --------------------------------------------------------

    customer_value[
        "annualized_revenue"
    ] = (
        customer_value[
            "revenue_per_month"
        ]
        * 12
    )

    # --------------------------------------------------------
    # Value tier
    # --------------------------------------------------------

    customer_value[
        "clv_value_tier"
    ] = pd.qcut(
        customer_value[
            "historical_clv"
        ].rank(
            method="first"
        ),
        q=5,
        labels=[
            "Low",
            "Below Average",
            "Medium",
            "High",
            "Very High"
        ]
    )

    return customer_value


# ============================================================
# MERGE WITH RFM
# ============================================================

def merge_rfm(
    customer_value,
    rfm
):

    rfm_columns = [
        "customer_unique_id",
        "R_score",
        "F_score",
        "M_score",
        "RFM_score",
        "RFM_total",
        "customer_segment"
    ]

    rfm_subset = rfm[
        rfm_columns
    ].copy()

    customer_value = customer_value.merge(
        rfm_subset,
        on="customer_unique_id",
        how="left",
        validate="one_to_one"
    )

    return customer_value


# ============================================================
# CREATE CLV SUMMARY
# ============================================================

def create_summary(
    customer_value,
    analysis_date
):

    summary = {
        "analysis_date": (
            analysis_date.strftime(
                "%Y-%m-%d"
            )
        ),

        "customers": int(
            len(customer_value)
        ),

        "repeat_customers": int(
            customer_value[
                "repeat_customer"
            ].sum()
        ),

        "total_historical_revenue": round(
            float(
                customer_value[
                    "total_revenue"
                ].sum()
            ),
            2
        ),

        "average_customer_value": round(
            float(
                customer_value[
                    "historical_clv"
                ].mean()
            ),
            2
        ),

        "median_customer_value": round(
            float(
                customer_value[
                    "historical_clv"
                ].median()
            ),
            2
        ),

        "average_order_value": round(
            float(
                customer_value[
                    "average_order_value"
                ].mean()
            ),
            2
        )
    }

    tier_summary = (
        customer_value
        .groupby(
            "clv_value_tier",
            observed=True
        )
        .agg(
            customers=(
                "customer_unique_id",
                "count"
            ),
            total_revenue=(
                "historical_clv",
                "sum"
            ),
            average_value=(
                "historical_clv",
                "mean"
            )
        )
        .reset_index()
    )

    tier_summary[
        "total_revenue"
    ] = tier_summary[
        "total_revenue"
    ].round(2)

    tier_summary[
        "average_value"
    ] = tier_summary[
        "average_value"
    ].round(2)

    summary[
        "clv_tiers"
    ] = tier_summary.to_dict(
        orient="records"
    )

    return summary


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    customer_value,
    summary
):

    print("\n" + "=" * 80)
    print("CLV SUMMARY")
    print("=" * 80)

    print(
        f"\nAnalysis date          : "
        f"{summary['analysis_date']}"
    )

    print(
        f"Customers              : "
        f"{summary['customers']:,}"
    )

    print(
        f"Repeat customers       : "
        f"{summary['repeat_customers']:,}"
    )

    print(
        f"Historical revenue     : "
        f"₹{summary['total_historical_revenue']:,.2f}"
    )

    print(
        f"Average customer value : "
        f"₹{summary['average_customer_value']:,.2f}"
    )

    print(
        f"Median customer value  : "
        f"₹{summary['median_customer_value']:,.2f}"
    )

    print(
        f"Average order value    : "
        f"₹{summary['average_order_value']:,.2f}"
    )

    print("\n" + "-" * 80)
    print("CLV VALUE TIERS")
    print("-" * 80)

    print(
        f"\n{'Tier':<20}"
        f"{'Customers':<15}"
        f"{'Total Revenue':<20}"
        f"{'Average Value':<20}"
    )

    print("-" * 80)

    for tier in summary[
        "clv_tiers"
    ]:

        print(
            f"{str(tier['clv_value_tier']):<20}"
            f"{tier['customers']:<15}"
            f"₹{tier['total_revenue']:<19,.2f}"
            f"₹{tier['average_value']:<19,.2f}"
        )


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_outputs(
    customer_value,
    summary
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    customer_value.to_csv(
        OUTPUT_FILE,
        index=False
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 80)
    print("CLV ANALYSIS COMPLETED")
    print("=" * 80)

    print(
        f"\nSaved customer CLV : {OUTPUT_FILE}"
    )

    print(
        f"Saved report        : {REPORT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        customers,
        orders,
        payments,
        rfm
    ) = load_data()

    orders = prepare_orders(
        customers,
        orders,
        payments
    )

    print(
        f"\nCompleted orders used : "
        f"{orders['order_id'].nunique():,}"
    )

    customer_value, analysis_date = (
        calculate_customer_value(
            orders
        )
    )

    customer_value = calculate_clv(
        customer_value
    )

    customer_value = merge_rfm(
        customer_value,
        rfm
    )

    summary = create_summary(
        customer_value,
        analysis_date
    )

    print_results(
        customer_value,
        summary
    )

    save_outputs(
        customer_value,
        summary
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
