from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


CUSTOMERS_FILE = RAW / "olist_customers_dataset.csv"
ORDERS_FILE = RAW / "olist_orders_dataset.csv"
PAYMENTS_FILE = RAW / "olist_order_payments_dataset.csv"

RFM_FILE = PROCESSED / "customer_rfm.csv"
CLV_FILE = PROCESSED / "customer_clv.csv"
PURCHASE_FILE = PROCESSED / "customer_purchase_behavior.csv"

OUTPUT_FILE = PROCESSED / "customer_intelligence_master.csv"
REPORT_FILE = REPORTS / "customer_intelligence_summary.json"


# ============================================================
# HELPERS
# ============================================================

def load_csv(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found: {path}"
        )

    print(f"Loading {name}...")
    return pd.read_csv(path)


def normalize_columns(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return df


def require_column(df, column, source):
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' not found in {source}. "
            f"Available columns: {list(df.columns)}"
        )


# ============================================================
# RFM SEGMENT
# ============================================================

def derive_rfm_segment(row):

    r = row["r_score"]
    f = row["f_score"]
    m = row["m_score"]

    if pd.isna(r) or pd.isna(f) or pd.isna(m):
        return "Unclassified"

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    if r >= 4 and f >= 3:
        return "Loyal Customers"

    if r >= 3 and f >= 2:
        return "Potential Loyalists"

    if r >= 4 and f <= 2:
        return "New / Promising"

    if r <= 2 and m >= 4:
        return "At Risk High Value"

    if r <= 2 and m >= 3:
        return "At Risk Valuable"

    if r <= 2 and f <= 2 and m <= 2:
        return "Hibernating"

    return "Needs Attention"


# ============================================================
# RETENTION STATUS
# ============================================================

def get_retention_status(segment):

    mapping = {
        "Champions": "Protect & Retain",
        "Loyal Customers": "Strengthen Loyalty",
        "Potential Loyalists": "Convert to Loyal",
        "New / Promising": "Encourage Second Purchase",
        "Needs Attention": "Re-engage",
        "At Risk High Value": "High Priority Retention",
        "At Risk Valuable": "Priority Retention",
        "Hibernating": "Win-back / Low Engagement",
        "Unclassified": "Review",
    }

    return mapping.get(segment, "Review")


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("CUSTOMER INTELLIGENCE MASTER DATASET")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

customers = normalize_columns(
    load_csv(CUSTOMERS_FILE, "customers")
)

orders = normalize_columns(
    load_csv(ORDERS_FILE, "orders")
)

payments = normalize_columns(
    load_csv(PAYMENTS_FILE, "payments")
)

rfm = normalize_columns(
    load_csv(RFM_FILE, "RFM")
)

clv = normalize_columns(
    load_csv(CLV_FILE, "CLV")
)

purchase = normalize_columns(
    load_csv(PURCHASE_FILE, "purchase behavior")
)


# ============================================================
# VALIDATE
# ============================================================

for column in [
    "customer_id",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
]:
    require_column(
        customers,
        column,
        "customers"
    )

for column in [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
]:
    require_column(
        orders,
        column,
        "orders"
    )

for column in [
    "order_id",
    "payment_value",
    "payment_sequential",
]:
    require_column(
        payments,
        column,
        "payments"
    )

for column in [
    "customer_unique_id",
    "recency",
    "frequency",
    "monetary",
    "r_score",
    "f_score",
    "m_score",
    "rfm_score",
]:
    require_column(
        rfm,
        column,
        "customer_rfm.csv"
    )

for column in [
    "customer_unique_id",
    "historical_clv",
    "clv_value_tier",
]:
    require_column(
        clv,
        column,
        "customer_clv.csv"
    )

require_column(
    purchase,
    "customer_unique_id",
    "customer_purchase_behavior.csv"
)


# ============================================================
# CUSTOMER PROFILE
# ============================================================

print()
print("Preparing customer profile...")

customer_profile = customers[
    [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ]
].copy()

customer_profile = customer_profile.rename(
    columns={
        "customer_zip_code_prefix": "customer_zip_prefix"
    }
)

customer_profile = (
    customer_profile
    .drop_duplicates(
        "customer_unique_id"
    )
)


# ============================================================
# COMPLETED ORDERS
# ============================================================

print("Preparing completed orders...")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

delivered_orders = orders[
    orders["order_status"] == "delivered"
].copy()


# ============================================================
# CUSTOMER MAPPING
# ============================================================

customer_map = (
    customers[
        [
            "customer_id",
            "customer_unique_id",
        ]
    ]
    .drop_duplicates("customer_id")
)

delivered_orders = delivered_orders.merge(
    customer_map,
    on="customer_id",
    how="left",
    validate="many_to_one"
)


# ============================================================
# PAYMENT AGGREGATION
# ============================================================

print("Aggregating payments...")

payment_summary = (
    payments
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        order_payment_value=(
            "payment_value",
            "sum"
        ),
        payment_count=(
            "payment_sequential",
            "count"
        ),
    )
)

delivered_orders = delivered_orders.merge(
    payment_summary,
    on="order_id",
    how="left",
    validate="one_to_one"
)

delivered_orders["order_payment_value"] = (
    delivered_orders["order_payment_value"]
    .fillna(0)
)


# ============================================================
# ORDER BEHAVIOR
# ============================================================

print("Calculating customer order behavior...")

order_behavior = (
    delivered_orders
    .groupby(
        "customer_unique_id",
        as_index=False
    )
    .agg(
        completed_orders=(
            "order_id",
            "nunique"
        ),
        completed_revenue=(
            "order_payment_value",
            "sum"
        ),
        first_purchase_date=(
            "order_purchase_timestamp",
            "min"
        ),
        last_purchase_date=(
            "order_purchase_timestamp",
            "max"
        ),
    )
)

order_behavior["average_completed_order_value"] = (
    order_behavior["completed_revenue"]
    / order_behavior["completed_orders"]
)


# ============================================================
# RFM
# ============================================================

print("Preparing RFM data...")

rfm_selected = rfm[
    [
        "customer_unique_id",
        "recency",
        "frequency",
        "monetary",
        "r_score",
        "f_score",
        "m_score",
        "rfm_score",
    ]
].drop_duplicates(
    "customer_unique_id"
).copy()


# ============================================================
# CREATE RFM SEGMENTS
# ============================================================

print("Creating RFM customer segments...")

rfm_selected["rfm_segment"] = (
    rfm_selected.apply(
        derive_rfm_segment,
        axis=1
    )
)


# ============================================================
# CLV
# ============================================================

print("Preparing CLV data...")

clv_columns = [
    "customer_unique_id",
    "total_orders",
    "total_revenue",
    "customer_lifetime_days",
    "customer_lifetime_months",
    "average_order_value",
    "orders_per_month",
    "revenue_per_month",
    "historical_clv",
    "annualized_revenue",
    "clv_value_tier",
]

clv_columns = [
    column
    for column in clv_columns
    if column in clv.columns
]

clv_selected = (
    clv[clv_columns]
    .drop_duplicates(
        "customer_unique_id"
    )
    .copy()
)


# ============================================================
# PURCHASE BEHAVIOR
# ============================================================

print("Preparing purchase behavior...")

purchase_selected = (
    purchase
    .drop_duplicates(
        "customer_unique_id"
    )
    .copy()
)

purchase_selected = purchase_selected.rename(
    columns={
        column: "purchase_" + column
        for column in purchase_selected.columns
        if column != "customer_unique_id"
    }
)


# ============================================================
# MERGE
# ============================================================

print()
print("Merging customer intelligence layers...")

master = customer_profile.merge(
    rfm_selected,
    on="customer_unique_id",
    how="inner",
    validate="one_to_one"
)

print(
    "After RFM merge       : "
    + f"{len(master):,} customers"
)

master = master.merge(
    clv_selected,
    on="customer_unique_id",
    how="left",
    validate="one_to_one"
)

print(
    "After CLV merge       : "
    + f"{len(master):,} customers"
)

master = master.merge(
    purchase_selected,
    on="customer_unique_id",
    how="left",
    validate="one_to_one"
)

print(
    "After purchase merge  : "
    + f"{len(master):,} customers"
)

master = master.merge(
    order_behavior,
    on="customer_unique_id",
    how="left",
    validate="one_to_one"
)

print(
    "After order merge     : "
    + f"{len(master):,} customers"
)


# ============================================================
# CUSTOMER TYPE
# ============================================================

print()
print("Creating customer intelligence attributes...")

master["completed_orders"] = (
    master["completed_orders"]
    .fillna(0)
)

master["completed_revenue"] = (
    master["completed_revenue"]
    .fillna(0)
)

master["customer_type"] = np.where(
    master["completed_orders"] > 1,
    "Repeat Customer",
    "Single Purchase Customer"
)

master["repeat_customer"] = (
    master["completed_orders"] > 1
).astype(int)


# ============================================================
# VALUE PRIORITY
# ============================================================

value_priority_map = {
    "Very High": "Critical Value",
    "High": "High Value",
    "Medium": "Medium Value",
    "Below Average": "Developing Value",
    "Low": "Low Value",
}

master["value_priority"] = (
    master["clv_value_tier"]
    .map(value_priority_map)
    .fillna("Unclassified")
)


# ============================================================
# RETENTION STATUS
# ============================================================

master["retention_status"] = (
    master["rfm_segment"]
    .apply(get_retention_status)
)


# ============================================================
# DATA QUALITY
# ============================================================

master["has_purchase_history"] = (
    master["completed_orders"] > 0
).astype(int)

master["data_quality_flag"] = np.where(
    master["customer_unique_id"].notna()
    & master["rfm_segment"].notna()
    & master["historical_clv"].notna(),
    "Valid",
    "Review"
)


# ============================================================
# SORT
# ============================================================

priority_rank = {
    "Critical Value": 1,
    "High Value": 2,
    "Medium Value": 3,
    "Developing Value": 4,
    "Low Value": 5,
    "Unclassified": 6,
}

master["_priority_rank"] = (
    master["value_priority"]
    .map(priority_rank)
    .fillna(99)
)

master = master.sort_values(
    [
        "_priority_rank",
        "historical_clv",
    ],
    ascending=[
        True,
        False,
    ]
)

master = master.drop(
    columns=["_priority_rank"]
)

master = master.reset_index(
    drop=True
)


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

numeric_columns = master.select_dtypes(
    include=["number"]
).columns

master[numeric_columns] = (
    master[numeric_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# ============================================================
# SAVE MASTER DATASET
# ============================================================

master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARIES
# ============================================================

rfm_summary = (
    master["rfm_segment"]
    .value_counts()
    .to_dict()
)

clv_summary = (
    master["clv_value_tier"]
    .value_counts()
    .to_dict()
)

customer_type_summary = (
    master["customer_type"]
    .value_counts()
    .to_dict()
)

retention_summary = (
    master["retention_status"]
    .value_counts()
    .to_dict()
)

value_priority_summary = (
    master["value_priority"]
    .value_counts()
    .to_dict()
)


valid_count = int(
    (
        master["data_quality_flag"]
        == "Valid"
    ).sum()
)

review_count = int(
    (
        master["data_quality_flag"]
        == "Review"
    ).sum()
)


summary = {
    "analysis_date": "2018-08-30",
    "customers": int(len(master)),
    "columns": int(len(master.columns)),
    "repeat_customers": int(
        (
            master["completed_orders"] > 1
        ).sum()
    ),
    "single_purchase_customers": int(
        (
            master["completed_orders"] == 1
        ).sum()
    ),
    "historical_revenue": float(
        master["historical_clv"].sum()
    ),
    "average_customer_value": float(
        master["historical_clv"].mean()
    ),
    "median_customer_value": float(
        master["historical_clv"].median()
    ),
    "rfm_segments": rfm_summary,
    "clv_value_tiers": clv_summary,
    "customer_types": customer_type_summary,
    "value_priorities": value_priority_summary,
    "retention_status": retention_summary,
    "data_quality": {
        "valid": valid_count,
        "review": review_count,
    },
    "output_file": str(OUTPUT_FILE),
    "report_file": str(REPORT_FILE),
}


# ============================================================
# SAVE REPORT
# ============================================================

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
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("CUSTOMER INTELLIGENCE MASTER DATASET COMPLETED")
print("=" * 80)

customer_count = len(master)
column_count = len(master.columns)
repeat_count = int(
    (master["completed_orders"] > 1).sum()
)
single_count = int(
    (master["completed_orders"] == 1).sum()
)
revenue = master["historical_clv"].sum()

print(
    "Customers              : "
    + f"{customer_count:,}"
)

print(
    "Columns                : "
    + f"{column_count:,}"
)

print(
    "Repeat customers       : "
    + f"{repeat_count:,}"
)

print(
    "Single purchase        : "
    + f"{single_count:,}"
)

print(
    "Historical revenue     : "
    + f"₹{revenue:,.2f}"
)


print()
print("RFM SEGMENTS")
print("-" * 80)

for segment, count in (
    master["rfm_segment"]
    .value_counts()
    .items()
):

    print(
        f"{segment:<32} {count:>10,}"
    )


print()
print("CLV VALUE TIERS")
print("-" * 80)

for tier, count in (
    master["clv_value_tier"]
    .value_counts()
    .items()
):

    print(
        f"{tier:<32} {count:>10,}"
    )


print()
print("RETENTION STATUS")
print("-" * 80)

for status, count in (
    master["retention_status"]
    .value_counts()
    .items()
):

    print(
        f"{status:<32} {count:>10,}"
    )


print()
print("DATA QUALITY")
print("-" * 80)

print(
    "Valid customers        : "
    + f"{valid_count:,}"
)

print(
    "Customers for review   : "
    + f"{review_count:,}"
)


print()
print("OUTPUT FILES")
print("-" * 80)

print(
    "Master dataset : "
    + str(OUTPUT_FILE)
)

print(
    "Summary report : "
    + str(REPORT_FILE)
)


print()
print("SAMPLE CUSTOMER RECORDS")
print("-" * 80)

sample_columns = [
    "customer_unique_id",
    "customer_city",
    "customer_state",
    "recency",
    "frequency",
    "monetary",
    "r_score",
    "f_score",
    "m_score",
    "rfm_score",
    "rfm_segment",
    "historical_clv",
    "clv_value_tier",
    "customer_type",
    "value_priority",
    "retention_status",
]

sample_columns = [
    column
    for column in sample_columns
    if column in master.columns
]

print(
    master[
        sample_columns
    ]
    .head(5)
    .to_string(index=False)
)

print()
print("Done.")
