from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# LOAD DATA
# ============================================================

customers = pd.read_csv(
    RAW_DIR / "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    RAW_DIR / "olist_orders_dataset.csv"
)


# ============================================================
# DATE CONVERSION
# ============================================================

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)


# ============================================================
# JOIN CUSTOMER UNIQUE ID
# ============================================================

orders = orders.merge(
    customers[
        [
            "customer_id",
            "customer_unique_id"
        ]
    ],
    on="customer_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# ONLY COMPLETED / DELIVERED ORDERS
# ============================================================

completed_orders = orders[
    orders["order_status"] == "delivered"
].copy()


# ============================================================
# SORT
# ============================================================

completed_orders = completed_orders.sort_values(
    [
        "customer_unique_id",
        "order_purchase_timestamp"
    ]
)


# ============================================================
# CUSTOMER PURCHASE BEHAVIOR
# ============================================================

customer_behavior = (
    completed_orders
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


# ============================================================
# REPEAT PURCHASES
# ============================================================

customer_behavior["repeat_customer"] = (
    customer_behavior["total_orders"] > 1
).astype(int)


# ============================================================
# PURCHASE GAP
# ============================================================

completed_orders["previous_purchase_date"] = (
    completed_orders
    .groupby("customer_unique_id")
    ["order_purchase_timestamp"]
    .shift(1)
)


completed_orders["purchase_gap_days"] = (
    completed_orders["order_purchase_timestamp"]
    - completed_orders["previous_purchase_date"]
).dt.total_seconds() / 86400


# ============================================================
# GAP STATISTICS
# ============================================================

gap_statistics = (
    completed_orders
    .dropna(subset=["purchase_gap_days"])
    .groupby("customer_unique_id")
    ["purchase_gap_days"]
    .agg(
        mean_purchase_gap_days="mean",
        median_purchase_gap_days="median",
        max_purchase_gap_days="max",
        purchase_gap_count="count"
    )
    .reset_index()
)


# ============================================================
# MERGE
# ============================================================

customer_behavior = customer_behavior.merge(
    gap_statistics,
    on="customer_unique_id",
    how="left"
)


# ============================================================
# OVERALL PURCHASE GAP ANALYSIS
# ============================================================

all_gaps = completed_orders[
    "purchase_gap_days"
].dropna()


print()
print("=" * 80)
print("       CUSTOMER PURCHASE BEHAVIOR ANALYSIS")
print("=" * 80)

print()

print(
    f"Customers with completed orders : "
    f"{customer_behavior['customer_unique_id'].nunique():,}"
)

print(
    f"Customers with repeat orders    : "
    f"{customer_behavior['repeat_customer'].sum():,}"
)

print(
    f"Repeat customer rate            : "
    f"{customer_behavior['repeat_customer'].mean() * 100:.2f}%"
)

print()

print("-" * 80)
print("PURCHASE GAP STATISTICS")
print("-" * 80)

print(
    f"Gap records                     : "
    f"{len(all_gaps):,}"
)

print(
    f"Mean gap                        : "
    f"{all_gaps.mean():.2f} days"
)

print(
    f"Median gap                      : "
    f"{all_gaps.median():.2f} days"
)

print(
    f"75th percentile                 : "
    f"{all_gaps.quantile(0.75):.2f} days"
)

print(
    f"90th percentile                 : "
    f"{all_gaps.quantile(0.90):.2f} days"
)

print(
    f"95th percentile                 : "
    f"{all_gaps.quantile(0.95):.2f} days"
)

print(
    f"99th percentile                 : "
    f"{all_gaps.quantile(0.99):.2f} days"
)

print()

print("-" * 80)
print("PURCHASE GAP DISTRIBUTION")
print("-" * 80)

bins = [
    0,
    30,
    60,
    90,
    120,
    180,
    365,
    float("inf")
]

labels = [
    "0-30 days",
    "31-60 days",
    "61-90 days",
    "91-120 days",
    "121-180 days",
    "181-365 days",
    "365+ days"
]

gap_distribution = pd.cut(
    all_gaps,
    bins=bins,
    labels=labels,
    include_lowest=True
).value_counts().sort_index()

for label, count in gap_distribution.items():

    percentage = (
        count / len(all_gaps) * 100
    )

    print(
        f"{str(label):<15}"
        f"| {count:>8,}"
        f" | {percentage:>6.2f}%"
    )


# ============================================================
# SAVE INTERMEDIATE ANALYSIS
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    OUTPUT_DIR /
    "customer_purchase_behavior.csv"
)

customer_behavior.to_csv(
    output_file,
    index=False
)


print()
print("=" * 80)
print("ANALYSIS COMPLETED")
print("=" * 80)

print()
print(
    f"Saved: {output_file}"
)

print()
