"""
Business Recommendation Engine
--------------------------------

Creates actionable business recommendations
from the Customer Intelligence Master Dataset.

Input:
data/processed/customer_intelligence_master.csv

Outputs:
data/processed/customer_recommendations.csv
reports/recommendation_summary.json
"""


from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "customer_intelligence_master.csv"
)

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

OUTPUT_FILE = (
    PROCESSED
    / "customer_recommendations.csv"
)

REPORT_FILE = (
    REPORTS
    / "recommendation_summary.json"
)


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 80)
print("BUSINESS RECOMMENDATION ENGINE")
print("=" * 80)

print()
print("Loading Customer Intelligence Master Dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Master dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Customers loaded       : "
    + f"{len(df):,}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "customer_unique_id",
    "rfm_segment",
    "clv_value_tier",
    "historical_clv",
    "recency",
    "frequency",
    "monetary",
    "customer_type",
    "retention_status",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================================
# CLEAN TYPES
# ============================================================

df["recency"] = pd.to_numeric(
    df["recency"],
    errors="coerce"
)

df["frequency"] = pd.to_numeric(
    df["frequency"],
    errors="coerce"
)

df["monetary"] = pd.to_numeric(
    df["monetary"],
    errors="coerce"
)

df["historical_clv"] = pd.to_numeric(
    df["historical_clv"],
    errors="coerce"
)

df["rfm_score"] = pd.to_numeric(
    df["rfm_score"],
    errors="coerce"
)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendation(row):

    segment = str(
        row["rfm_segment"]
    )

    value_tier = str(
        row["clv_value_tier"]
    )

    customer_type = str(
        row["customer_type"]
    )

    recency = row["recency"]

    frequency = row["frequency"]

    # --------------------------------------------------------
    # CHAMPIONS
    # --------------------------------------------------------

    if segment == "Champions":

        return {
            "priority": "High",
            "action": "Protect and reward",
            "campaign": "VIP Loyalty Campaign",
            "objective": "Retain high-engagement customers",
            "offer": "Early access, loyalty rewards or exclusive benefits",
            "channel": "Email + CRM + Loyalty",
            "reason": (
                "Customer shows strong recent activity "
                "and high RFM performance."
            ),
        }


    # --------------------------------------------------------
    # LOYAL CUSTOMERS
    # --------------------------------------------------------

    if segment == "Loyal Customers":

        return {
            "priority": "High",
            "action": "Strengthen loyalty",
            "campaign": "Loyal Customer Program",
            "objective": "Increase repeat purchasing",
            "offer": "Loyalty points, bundles or member benefits",
            "channel": "Email + CRM",
            "reason": (
                "Customer demonstrates relatively strong "
                "recency and purchase engagement."
            ),
        }


    # --------------------------------------------------------
    # POTENTIAL LOYALISTS
    # --------------------------------------------------------

    if segment == "Potential Loyalists":

        if customer_type == "Single Purchase Customer":

            return {
                "priority": "High",
                "action": "Drive second purchase",
                "campaign": "Second Purchase Campaign",
                "objective": "Convert first-time buyers into repeat customers",
                "offer": "Personalized second-order incentive",
                "channel": "Email + Remarketing",
                "reason": (
                    "Customer is relatively recent but "
                    "has purchased only once."
                ),
            }

        return {
            "priority": "Medium",
            "action": "Increase purchase frequency",
            "campaign": "Repeat Purchase Campaign",
            "objective": "Move customer toward loyal status",
            "offer": "Cross-sell, bundle or personalized recommendation",
            "channel": "Email + CRM",
            "reason": (
                "Customer shows promising engagement "
                "but has room to increase frequency."
            ),
        }


    # --------------------------------------------------------
    # NEW / PROMISING
    # --------------------------------------------------------

    if segment == "New / Promising":

        return {
            "priority": "Medium",
            "action": "Encourage second purchase",
            "campaign": "New Customer Activation",
            "objective": "Establish repeat purchasing behavior",
            "offer": "Next-purchase incentive or product recommendation",
            "channel": "Email + Remarketing",
            "reason": (
                "Customer is relatively recent but "
                "purchase frequency is still developing."
            ),
        }


    # --------------------------------------------------------
    # AT RISK HIGH VALUE
    # --------------------------------------------------------

    if segment == "At Risk High Value":

        return {
            "priority": "Critical",
            "action": "Immediate retention intervention",
            "campaign": "High Value Win-Back",
            "objective": "Re-engage valuable inactive customers",
            "offer": "Personalized incentive based on customer value",
            "channel": "CRM + Email + Retargeting",
            "reason": (
                "Customer has poor recent activity "
                "but historically high monetary value."
            ),
        }


    # --------------------------------------------------------
    # AT RISK VALUABLE
    # --------------------------------------------------------

    if segment == "At Risk Valuable":

        return {
            "priority": "High",
            "action": "Re-engage customer",
            "campaign": "Value Customer Win-Back",
            "objective": "Recover declining customer activity",
            "offer": "Relevant product offer or limited incentive",
            "channel": "Email + Remarketing",
            "reason": (
                "Customer has declining recency "
                "with meaningful historical value."
            ),
        }


    # --------------------------------------------------------
    # NEEDS ATTENTION
    # --------------------------------------------------------

    if segment == "Needs Attention":

        return {
            "priority": "Medium",
            "action": "Re-engage",
            "campaign": "Customer Re-Engagement",
            "objective": "Restore purchase activity",
            "offer": "Personalized product recommendation",
            "channel": "Email + Remarketing",
            "reason": (
                "Customer activity has weakened "
                "and requires re-engagement."
            ),
        }


    # --------------------------------------------------------
    # HIBERNATING
    # --------------------------------------------------------

    if segment == "Hibernating":

        return {
            "priority": "Low",
            "action": "Win back selectively",
            "campaign": "Dormant Customer Win-Back",
            "objective": "Test whether inactive customers can be recovered",
            "offer": "Targeted reactivation incentive",
            "channel": "Email",
            "reason": (
                "Customer shows low recent engagement "
                "and low purchase activity."
            ),
        }


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return {
        "priority": "Review",
        "action": "Manual review",
        "campaign": "Customer Review",
        "objective": "Determine appropriate next action",
        "offer": "No automatic offer",
        "channel": "CRM",
        "reason": "Customer profile requires review.",
    }


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

print()
print("Generating business recommendations...")

recommendations = (
    df.apply(
        generate_recommendation,
        axis=1,
        result_type="expand"
    )
)


# ============================================================
# MERGE
# ============================================================

recommendation_columns = [
    "priority",
    "action",
    "campaign",
    "objective",
    "offer",
    "channel",
    "reason",
]

for column in recommendation_columns:

    df[
        "recommendation_"
        + column
    ] = recommendations[column].values


# ============================================================
# RETENTION PRIORITY SCORE
# ============================================================

priority_score_map = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Review": 0,
}

df["retention_priority_score"] = (
    df["recommendation_priority"]
    .map(priority_score_map)
    .fillna(0)
)


# ============================================================
# BUSINESS ACTION FLAG
# ============================================================

df["requires_retention_action"] = (
    df["retention_priority_score"] >= 2
).astype(int)


# ============================================================
# VALUE + RETENTION PRIORITY
# ============================================================

def calculate_business_priority(row):

    value = str(
        row["clv_value_tier"]
    )

    priority = str(
        row["recommendation_priority"]
    )

    if (
        value == "Very High"
        and priority == "Critical"
    ):
        return "Immediate"

    if (
        value in ["Very High", "High"]
        and priority in ["Critical", "High"]
    ):
        return "High"

    if priority == "Medium":
        return "Medium"

    return "Low"


df["business_priority"] = (
    df.apply(
        calculate_business_priority,
        axis=1
    )
)


# ============================================================
# SORT
# ============================================================

business_priority_rank = {
    "Immediate": 1,
    "High": 2,
    "Medium": 3,
    "Low": 4,
}

df["_business_priority_rank"] = (
    df["business_priority"]
    .map(business_priority_rank)
    .fillna(99)
)

df = df.sort_values(
    [
        "_business_priority_rank",
        "historical_clv",
    ],
    ascending=[
        True,
        False,
    ]
)

df = df.drop(
    columns=[
        "_business_priority_rank"
    ]
)

df = df.reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARIES
# ============================================================

priority_summary = (
    df["recommendation_priority"]
    .value_counts()
    .to_dict()
)

action_summary = (
    df["recommendation_action"]
    .value_counts()
    .to_dict()
)

campaign_summary = (
    df["recommendation_campaign"]
    .value_counts()
    .to_dict()
)

business_priority_summary = (
    df["business_priority"]
    .value_counts()
    .to_dict()
)


# ============================================================
# HIGH VALUE AT RISK
# ============================================================

high_value_at_risk = df[
    (
        df["clv_value_tier"]
        == "Very High"
    )
    &
    (
        df["recommendation_priority"]
        == "Critical"
    )
]

high_value_at_risk_count = len(
    high_value_at_risk
)

high_value_at_risk_revenue = float(
    high_value_at_risk[
        "historical_clv"
    ].sum()
)


# ============================================================
# REPORT
# ============================================================

summary = {

    "customers": int(
        len(df)
    ),

    "priority_distribution":
        priority_summary,

    "action_distribution":
        action_summary,

    "campaign_distribution":
        campaign_summary,

    "business_priority_distribution":
        business_priority_summary,

    "high_value_critical_customers": {
        "customers":
            int(high_value_at_risk_count),

        "historical_revenue":
            high_value_at_risk_revenue,
    },

    "customers_requiring_retention_action":
        int(
            df[
                "requires_retention_action"
            ].sum()
        ),

    "output_file":
        str(OUTPUT_FILE),

    "report_file":
        str(REPORT_FILE),
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 80)
print("BUSINESS RECOMMENDATION ENGINE COMPLETED")
print("=" * 80)

print()
print(
    "Customers analyzed       : "
    + f"{len(df):,}"
)

print(
    "Retention action needed  : "
    + f"{int(df['requires_retention_action'].sum()):,}"
)

print()
print("RECOMMENDATION PRIORITY")
print("-" * 80)

for key, value in (
    df[
        "recommendation_priority"
    ]
    .value_counts()
    .items()
):

    print(
        f"{key:<30} {value:>10,}"
    )


print()
print("BUSINESS PRIORITY")
print("-" * 80)

for key, value in (
    df[
        "business_priority"
    ]
    .value_counts()
    .items()
):

    print(
        f"{key:<30} {value:>10,}"
    )


print()
print("TOP ACTIONS")
print("-" * 80)

for key, value in (
    df[
        "recommendation_action"
    ]
    .value_counts()
    .head(10)
    .items()
):

    print(
        f"{key:<40} {value:>10,}"
    )


print()
print("HIGH VALUE CRITICAL CUSTOMERS")
print("-" * 80)

print(
    "Customers : "
    + f"{high_value_at_risk_count:,}"
)

print(
    "Revenue   : "
    + f"₹{high_value_at_risk_revenue:,.2f}"
)


print()
print("OUTPUT FILES")
print("-" * 80)

print(
    "Recommendations : "
    + str(OUTPUT_FILE)
)

print(
    "Summary report  : "
    + str(REPORT_FILE)
)


print()
print("SAMPLE RECOMMENDATIONS")
print("-" * 80)

sample_columns = [
    "customer_unique_id",
    "rfm_segment",
    "clv_value_tier",
    "historical_clv",
    "recommendation_priority",
    "recommendation_action",
    "recommendation_campaign",
    "recommendation_offer",
    "recommendation_channel",
    "business_priority",
]

sample_columns = [
    column
    for column in sample_columns
    if column in df.columns
]

print(
    df[
        sample_columns
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("Done.")
