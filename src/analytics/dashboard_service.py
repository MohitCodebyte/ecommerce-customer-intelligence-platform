from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(filename):

    path = PROCESSED_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def build_dashboard_data():

    master = read_csv(
        "customer_intelligence_master.csv"
    )

    recommendations = read_csv(
        "customer_recommendations.csv"
    )

    if master.empty:
        raise FileNotFoundError(
            "customer_intelligence_master.csv is missing or empty."
        )


    # ========================================================
    # CUSTOMER METRICS
    # ========================================================

    total_customers = len(master)


    if "repeat_customer" in master.columns:

        repeat_customers = int(
            master["repeat_customer"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    elif "customer_type" in master.columns:

        repeat_customers = int(
            master["customer_type"]
            .astype(str)
            .str.lower()
            .eq("repeat")
            .sum()
        )

    else:

        repeat_customers = int(
            (master["frequency"] > 1).sum()
        )


    single_customers = (
        total_customers - repeat_customers
    )


    repeat_rate = (
        round(
            repeat_customers /
            total_customers * 100,
            2
        )
        if total_customers
        else 0
    )


    # ========================================================
    # REVENUE
    # ========================================================

    if "historical_clv" in master.columns:

        historical_revenue = float(
            master["historical_clv"]
            .fillna(0)
            .sum()
        )

    elif "total_revenue" in master.columns:

        historical_revenue = float(
            master["total_revenue"]
            .fillna(0)
            .sum()
        )

    else:

        historical_revenue = 0


    average_customer_value = (
        round(
            historical_revenue /
            total_customers,
            2
        )
        if total_customers
        else 0
    )


    # ========================================================
    # CLV TIER
    # Actual project column: clv_value_tier
    # ========================================================

    very_high_customers = 0
    very_high_revenue = 0

    clv_column = None

    if "clv_value_tier" in master.columns:
        clv_column = "clv_value_tier"

    elif "clv_tier" in master.columns:
        clv_column = "clv_tier"


    if clv_column:

        clv_values = (
            master[clv_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        very_high_mask = (
            clv_values == "very high"
        )

        very_high_customers = int(
            very_high_mask.sum()
        )

        if "historical_clv" in master.columns:

            very_high_revenue = float(
                master.loc[
                    very_high_mask,
                    "historical_clv"
                ]
                .fillna(0)
                .sum()
            )


    very_high_revenue_share = (
        round(
            very_high_revenue /
            historical_revenue * 100,
            2
        )
        if historical_revenue
        else 0
    )


    # ========================================================
    # RETENTION PRIORITY
    # Actual project column: retention_status
    # ========================================================

    high_priority = 0
    immediate_priority = 0


    if "retention_status" in master.columns:

        retention = (
            master["retention_status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        high_priority = int(
            retention
            .eq("high priority retention")
            .sum()
        )

        immediate_priority = int(
            retention
            .eq("immediate priority")
            .sum()
        )


    # ========================================================
    # VALUE PRIORITY
    # ========================================================

    value_priority = {}

    if "value_priority" in master.columns:

        value_priority = (
            master["value_priority"]
            .value_counts()
            .to_dict()
        )


    # ========================================================
    # RFM SEGMENTS
    # ========================================================

    segments = []

    if "rfm_segment" in master.columns:

        segment_counts = (
            master["rfm_segment"]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = [
            "segment",
            "customers"
        ]

        segments = (
            segment_counts
            .to_dict(orient="records")
        )


    # ========================================================
    # CLV DISTRIBUTION
    # ========================================================

    clv_tiers = []

    if clv_column:

        clv_counts = (
            master[clv_column]
            .value_counts()
            .reset_index()
        )

        clv_counts.columns = [
            "tier",
            "customers"
        ]

        clv_tiers = (
            clv_counts
            .to_dict(orient="records")
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendation_count = len(
        recommendations
    )


    recommendation_priority = {}

    if not recommendations.empty:

        priority_column = None

        for candidate in [
            "priority",
            "recommendation_priority"
        ]:

            if candidate in recommendations.columns:

                priority_column = candidate
                break


        if priority_column:

            recommendation_priority = (
                recommendations[
                    priority_column
                ]
                .value_counts()
                .to_dict()
            )


    # ========================================================
    # DASHBOARD OBJECT
    # ========================================================

    dashboard = {

        "generated_from":
            "Olist Development Dataset",

        "data_source_note":
            "Olist is used only as the development and "
            "benchmark dataset. Company deployments should "
            "use uploaded company data.",


        "customers": {

            "total":
                total_customers,

            "repeat":
                repeat_customers,

            "single_purchase":
                single_customers,

            "repeat_rate":
                repeat_rate
        },


        "revenue": {

            "historical_revenue":
                round(
                    historical_revenue,
                    2
                ),

            "average_customer_value":
                average_customer_value
        },


        "clv": {

            "tier_column":
                clv_column,

            "very_high_customers":
                very_high_customers,

            "very_high_revenue":
                round(
                    very_high_revenue,
                    2
                ),

            "very_high_revenue_share":
                very_high_revenue_share
        },


        "retention": {

            "high_priority":
                high_priority,

            "immediate_priority":
                immediate_priority
        },


        "value_priority":
            value_priority,


        "segments":
            segments,


        "clv_tiers":
            clv_tiers,


        "recommendations": {

            "total":
                recommendation_count,

            "priority_distribution":
                recommendation_priority
        }
    }


    # ========================================================
    # SAVE
    # ========================================================

    output = (
        REPORT_DIR /
        "dashboard_data.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dashboard,
            file,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("DASHBOARD DATA SERVICE")
    print("=" * 80)

    print(
        f"Total customers       : "
        f"{total_customers:,}"
    )

    print(
        f"Repeat customers      : "
        f"{repeat_customers:,}"
    )

    print(
        f"Repeat rate           : "
        f"{repeat_rate}%"
    )

    print(
        f"Historical revenue    : "
        f"₹{historical_revenue:,.2f}"
    )

    print(
        f"Very high customers   : "
        f"{very_high_customers:,}"
    )

    print(
        f"Very high revenue     : "
        f"₹{very_high_revenue:,.2f}"
    )

    print(
        f"High priority         : "
        f"{high_priority:,}"
    )

    print(
        f"Immediate priority    : "
        f"{immediate_priority:,}"
    )

    print(
        f"Recommendations       : "
        f"{recommendation_count:,}"
    )

    print()
    print(
        f"Dashboard data saved  : "
        f"{output}"
    )
    print()


if __name__ == "__main__":

    build_dashboard_data()
