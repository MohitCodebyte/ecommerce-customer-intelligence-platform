"""
Customer Analytics Engine
-------------------------

Creates dashboard-ready business analytics from:

- Customer Intelligence Master Dataset
- Customer Recommendations

Outputs:
    reports/customer_analytics.json
    data/processed/customer_segment_summary.csv
    data/processed/customer_clv_summary.csv
    data/processed/customer_priority_summary.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PROCESSED.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS.mkdir(
    parents=True,
    exist_ok=True
)

MASTER_FILE = (
    PROCESSED
    / "customer_intelligence_master.csv"
)

RECOMMENDATION_FILE = (
    PROCESSED
    / "customer_recommendations.csv"
)

ANALYTICS_FILE = (
    REPORTS
    / "customer_analytics.json"
)

SEGMENT_FILE = (
    PROCESSED
    / "customer_segment_summary.csv"
)

CLV_FILE = (
    PROCESSED
    / "customer_clv_summary.csv"
)

PRIORITY_FILE = (
    PROCESSED
    / "customer_priority_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def money(value):
    if pd.isna(value):
        return 0.0

    return float(value)


def pct(value):
    return round(float(value), 2)


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("CUSTOMER ANALYTICS ENGINE")
print("=" * 80)


# ============================================================
# LOAD MASTER
# ============================================================

if not MASTER_FILE.exists():
    raise FileNotFoundError(
        f"Master dataset not found:\n{MASTER_FILE}"
    )

print()
print("Loading Customer Intelligence Master Dataset...")

df = pd.read_csv(
    MASTER_FILE
)

print(
    "Customers loaded : "
    + f"{len(df):,}"
)


# ============================================================
# LOAD RECOMMENDATIONS
# ============================================================

if not RECOMMENDATION_FILE.exists():
    raise FileNotFoundError(
        f"Recommendation dataset not found:\n"
        f"{RECOMMENDATION_FILE}"
    )

print(
    "Loading recommendation data..."
)

recommendations = pd.read_csv(
    RECOMMENDATION_FILE
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "historical_clv",
    "total_revenue",
    "total_orders",
    "average_order_value",
    "completed_orders",
    "completed_revenue",
    "recency",
    "frequency",
    "monetary",
    "orders_per_month",
    "revenue_per_month",
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# BASIC KPI
# ============================================================

print()
print("Calculating core KPIs...")


total_customers = len(df)

repeat_customers = int(
    (
        df["customer_type"]
        == "Repeat Customer"
    ).sum()
)

single_purchase_customers = int(
    (
        df["customer_type"]
        == "Single Purchase Customer"
    ).sum()
)

total_revenue = money(
    df["historical_clv"].sum()
)

average_customer_value = money(
    df["historical_clv"].mean()
)

median_customer_value = money(
    df["historical_clv"].median()
)

total_orders = int(
    df["completed_orders"].sum()
)

average_order_value = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)

repeat_customer_rate = (
    repeat_customers
    / total_customers
    * 100
    if total_customers > 0
    else 0
)

single_purchase_rate = (
    single_purchase_customers
    / total_customers
    * 100
    if total_customers > 0
    else 0
)


# ============================================================
# RFM SEGMENT ANALYSIS
# ============================================================

print("Calculating RFM segment analytics...")

segment_summary = (
    df
    .groupby(
        "rfm_segment",
        dropna=False
    )
    .agg(
        customers=(
            "customer_unique_id",
            "nunique"
        ),
        revenue=(
            "historical_clv",
            "sum"
        ),
        average_customer_value=(
            "historical_clv",
            "mean"
        ),
        average_recency=(
            "recency",
            "mean"
        ),
        average_frequency=(
            "frequency",
            "mean"
        ),
        average_monetary=(
            "monetary",
            "mean"
        ),
    )
    .reset_index()
)

segment_summary["customer_share_pct"] = (
    segment_summary["customers"]
    / total_customers
    * 100
)

segment_summary["revenue_share_pct"] = (
    segment_summary["revenue"]
    / total_revenue
    * 100
)

segment_summary = segment_summary.sort_values(
    "revenue",
    ascending=False
)

segment_summary.to_csv(
    SEGMENT_FILE,
    index=False
)


# ============================================================
# CLV TIER ANALYSIS
# ============================================================

print("Calculating CLV tier analytics...")

clv_summary = (
    df
    .groupby(
        "clv_value_tier",
        dropna=False
    )
    .agg(
        customers=(
            "customer_unique_id",
            "nunique"
        ),
        revenue=(
            "historical_clv",
            "sum"
        ),
        average_customer_value=(
            "historical_clv",
            "mean"
        ),
        median_customer_value=(
            "historical_clv",
            "median"
        ),
    )
    .reset_index()
)

clv_summary["customer_share_pct"] = (
    clv_summary["customers"]
    / total_customers
    * 100
)

clv_summary["revenue_share_pct"] = (
    clv_summary["revenue"]
    / total_revenue
    * 100
)

clv_summary = clv_summary.sort_values(
    "revenue",
    ascending=False
)

clv_summary.to_csv(
    CLV_FILE,
    index=False
)


# ============================================================
# CUSTOMER TYPE ANALYSIS
# ============================================================

customer_type_summary = (
    df
    .groupby(
        "customer_type",
        dropna=False
    )
    .agg(
        customers=(
            "customer_unique_id",
            "nunique"
        ),
        revenue=(
            "historical_clv",
            "sum"
        ),
        average_customer_value=(
            "historical_clv",
            "mean"
        ),
        average_orders=(
            "completed_orders",
            "mean"
        ),
    )
    .reset_index()
)

customer_type_summary["customer_share_pct"] = (
    customer_type_summary["customers"]
    / total_customers
    * 100
)

customer_type_summary["revenue_share_pct"] = (
    customer_type_summary["revenue"]
    / total_revenue
    * 100
)


# ============================================================
# RETENTION ANALYSIS
# ============================================================

print("Calculating retention workload...")

retention_summary = (
    df
    .groupby(
        "retention_status",
        dropna=False
    )
    .agg(
        customers=(
            "customer_unique_id",
            "nunique"
        ),
        revenue=(
            "historical_clv",
            "sum"
        ),
        average_customer_value=(
            "historical_clv",
            "mean"
        ),
    )
    .reset_index()
)

retention_summary["customer_share_pct"] = (
    retention_summary["customers"]
    / total_customers
    * 100
)

retention_summary["revenue_share_pct"] = (
    retention_summary["revenue"]
    / total_revenue
    * 100
)

retention_summary = retention_summary.sort_values(
    "revenue",
    ascending=False
)


# ============================================================
# BUSINESS PRIORITY
# ============================================================

print("Calculating business priority analytics...")

if "business_priority" in recommendations.columns:

    priority_summary = (
        recommendations
        .groupby(
            "business_priority",
            dropna=False
        )
        .agg(
            customers=(
                "customer_unique_id",
                "nunique"
            )
        )
        .reset_index()
    )

else:

    priority_summary = pd.DataFrame(
        columns=[
            "business_priority",
            "customers",
        ]
    )


if "recommendation_priority" in recommendations.columns:

    recommendation_priority_summary = (
        recommendations
        .groupby(
            "recommendation_priority",
            dropna=False
        )
        .agg(
            customers=(
                "customer_unique_id",
                "nunique"
            )
        )
        .reset_index()
    )

else:

    recommendation_priority_summary = pd.DataFrame(
        columns=[
            "recommendation_priority",
            "customers",
        ]
    )


# ============================================================
# SAVE PRIORITY SUMMARY
# ============================================================

if not priority_summary.empty:

    priority_summary.to_csv(
        PRIORITY_FILE,
        index=False
    )

else:

    pd.DataFrame(
        {
            "business_priority": [],
            "customers": [],
        }
    ).to_csv(
        PRIORITY_FILE,
        index=False
    )


# ============================================================
# TOP CUSTOMERS
# ============================================================

top_customers = (
    df[
        [
            "customer_unique_id",
            "customer_city",
            "customer_state",
            "rfm_segment",
            "clv_value_tier",
            "historical_clv",
            "customer_type",
            "retention_status",
        ]
    ]
    .sort_values(
        "historical_clv",
        ascending=False
    )
    .head(20)
)

top_customers_records = (
    top_customers
    .replace(
        {
            np.nan: None
        }
    )
    .to_dict(
        orient="records"
    )
)


# ============================================================
# HIGH VALUE CUSTOMERS
# ============================================================

very_high_customers = df[
    df["clv_value_tier"]
    == "Very High"
]

very_high_revenue = money(
    very_high_customers[
        "historical_clv"
    ].sum()
)

very_high_customer_count = len(
    very_high_customers
)

very_high_revenue_share = (
    very_high_revenue
    / total_revenue
    * 100
    if total_revenue > 0
    else 0
)


# ============================================================
# CRITICAL RETENTION
# ============================================================

critical_customers = df[
    df["retention_status"]
    == "High Priority Retention"
]

critical_customer_count = len(
    critical_customers
)

critical_customer_revenue = money(
    critical_customers[
        "historical_clv"
    ].sum()
)


# ============================================================
# IMMEDIATE BUSINESS PRIORITY
# ============================================================

if "business_priority" in recommendations.columns:

    immediate = recommendations[
        recommendations[
            "business_priority"
        ]
        == "Immediate"
    ]

    immediate_count = int(
        immediate[
            "customer_unique_id"
        ]
        .nunique()
    )

else:

    immediate_count = 0


# ============================================================
# TOP SEGMENT BY REVENUE
# ============================================================

if not segment_summary.empty:

    top_segment_row = (
        segment_summary.iloc[0]
    )

    top_revenue_segment = str(
        top_segment_row[
            "rfm_segment"
        ]
    )

    top_revenue_segment_value = money(
        top_segment_row[
            "revenue"
        ]
    )

else:

    top_revenue_segment = None
    top_revenue_segment_value = 0


# ============================================================
# TOP CLV TIER BY REVENUE
# ============================================================

if not clv_summary.empty:

    top_clv_row = (
        clv_summary.iloc[0]
    )

    top_revenue_clv_tier = str(
        top_clv_row[
            "clv_value_tier"
        ]
    )

    top_revenue_clv_value = money(
        top_clv_row[
            "revenue"
        ]
    )

else:

    top_revenue_clv_tier = None
    top_revenue_clv_value = 0


# ============================================================
# DATA QUALITY
# ============================================================

missing_cells = int(
    df.isna().sum().sum()
)

customers_with_missing_value = int(
    df[
        "historical_clv"
    ]
    .isna()
    .sum()
)


# ============================================================
# FINAL ANALYTICS JSON
# ============================================================

analytics = {

    "analysis_date":
        "2018-08-30",

    "dataset": {
        "customers":
            total_customers,

        "completed_orders":
            total_orders,

        "historical_revenue":
            total_revenue,
    },

    "core_kpis": {

        "total_customers":
            total_customers,

        "repeat_customers":
            repeat_customers,

        "single_purchase_customers":
            single_purchase_customers,

        "repeat_customer_rate_pct":
            pct(repeat_customer_rate),

        "single_purchase_rate_pct":
            pct(single_purchase_rate),

        "total_completed_orders":
            total_orders,

        "average_order_value":
            money(average_order_value),

        "average_customer_value":
            money(average_customer_value),

        "median_customer_value":
            money(median_customer_value),

        "total_historical_revenue":
            total_revenue,
    },

    "customer_type": {

        "repeat_customers":
            repeat_customers,

        "single_purchase_customers":
            single_purchase_customers,

        "repeat_customer_rate_pct":
            pct(repeat_customer_rate),

        "single_purchase_rate_pct":
            pct(single_purchase_rate),

        "summary":
            customer_type_summary
            .to_dict(
                orient="records"
            ),
    },

    "rfm": {

        "segments":
            segment_summary
            .to_dict(
                orient="records"
            ),

        "top_revenue_segment":
            top_revenue_segment,

        "top_revenue_segment_value":
            top_revenue_segment_value,
    },

    "clv": {

        "very_high_customers":
            int(very_high_customer_count),

        "very_high_revenue":
            very_high_revenue,

        "very_high_revenue_share_pct":
            pct(very_high_revenue_share),

        "top_revenue_clv_tier":
            top_revenue_clv_tier,

        "top_revenue_clv_value":
            top_revenue_clv_value,

        "tiers":
            clv_summary
            .to_dict(
                orient="records"
            ),
    },

    "retention": {

        "retention_summary":
            retention_summary
            .to_dict(
                orient="records"
            ),

        "high_priority_retention_customers":
            critical_customer_count,

        "high_priority_retention_revenue":
            critical_customer_revenue,

        "immediate_business_priority_customers":
            immediate_count,
    },

    "recommendations": {

        "business_priority":
            priority_summary
            .to_dict(
                orient="records"
            ),

        "recommendation_priority":
            recommendation_priority_summary
            .to_dict(
                orient="records"
            ),
    },

    "top_customers":
        top_customers_records,

    "data_quality": {

        "missing_cells":
            missing_cells,

        "customers_missing_clv":
            customers_with_missing_value,
    },

    "output_files": {

        "analytics":
            str(ANALYTICS_FILE),

        "segment_summary":
            str(SEGMENT_FILE),

        "clv_summary":
            str(CLV_FILE),

        "priority_summary":
            str(PRIORITY_FILE),
    },
}


# ============================================================
# SAVE JSON
# ============================================================

print()
print("Saving analytics...")

with open(
    ANALYTICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        analytics,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 80)
print("CUSTOMER ANALYTICS ENGINE COMPLETED")
print("=" * 80)

print()
print("CORE KPIs")
print("-" * 80)

print(
    "Total customers          : "
    + f"{total_customers:,}"
)

print(
    "Repeat customers         : "
    + f"{repeat_customers:,}"
)

print(
    "Single purchase          : "
    + f"{single_purchase_customers:,}"
)

print(
    "Repeat customer rate     : "
    + f"{repeat_customer_rate:.2f}%"
)

print(
    "Total completed orders   : "
    + f"{total_orders:,}"
)

print(
    "Historical revenue       : "
    + f"₹{total_revenue:,.2f}"
)

print(
    "Average customer value   : "
    + f"₹{average_customer_value:,.2f}"
)

print(
    "Median customer value    : "
    + f"₹{median_customer_value:,.2f}"
)

print(
    "Average order value      : "
    + f"₹{average_order_value:,.2f}"
)


print()
print("CLV")
print("-" * 80)

print(
    "Very High customers      : "
    + f"{very_high_customer_count:,}"
)

print(
    "Very High revenue        : "
    + f"₹{very_high_revenue:,.2f}"
)

print(
    "Very High revenue share  : "
    + f"{very_high_revenue_share:.2f}%"
)


print()
print("RETENTION")
print("-" * 80)

print(
    "High priority customers  : "
    + f"{critical_customer_count:,}"
)

print(
    "High priority revenue    : "
    + f"₹{critical_customer_revenue:,.2f}"
)

print(
    "Immediate priority       : "
    + f"{immediate_count:,}"
)


print()
print("TOP REVENUE SEGMENT")
print("-" * 80)

print(
    "Segment                  : "
    + str(top_revenue_segment)
)

print(
    "Revenue                  : "
    + f"₹{top_revenue_segment_value:,.2f}"
)


print()
print("OUTPUT FILES")
print("-" * 80)

print(
    "Analytics JSON            : "
    + str(ANALYTICS_FILE)
)

print(
    "Segment summary           : "
    + str(SEGMENT_FILE)
)

print(
    "CLV summary               : "
    + str(CLV_FILE)
)

print(
    "Priority summary          : "
    + str(PRIORITY_FILE)
)


print()
print("Done.")
