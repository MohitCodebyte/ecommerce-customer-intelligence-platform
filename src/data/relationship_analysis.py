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

def load_data():

    customers = pd.read_csv(
        RAW_DIR / "olist_customers_dataset.csv"
    )

    orders = pd.read_csv(
        RAW_DIR / "olist_orders_dataset.csv"
    )

    items = pd.read_csv(
        RAW_DIR / "olist_order_items_dataset.csv"
    )

    payments = pd.read_csv(
        RAW_DIR / "olist_order_payments_dataset.csv"
    )

    reviews = pd.read_csv(
        RAW_DIR / "olist_order_reviews_dataset.csv"
    )

    products = pd.read_csv(
        RAW_DIR / "olist_products_dataset.csv"
    )

    sellers = pd.read_csv(
        RAW_DIR / "olist_sellers_dataset.csv"
    )

    return (
        customers,
        orders,
        items,
        payments,
        reviews,
        products,
        sellers,
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    (
        customers,
        orders,
        items,
        payments,
        reviews,
        products,
        sellers,
    ) = load_data()

    print()
    print("=" * 90)
    print("        E-COMMERCE BUSINESS RELATIONSHIP ANALYSIS")
    print("=" * 90)

    # ========================================================
    # CUSTOMER ANALYSIS
    # ========================================================

    print()
    print("-" * 90)
    print("CUSTOMER ANALYSIS")
    print("-" * 90)

    print(
        f"Customer records       : {len(customers):,}"
    )

    print(
        f"Unique customer IDs    : "
        f"{customers['customer_id'].nunique():,}"
    )

    print(
        f"Unique customers       : "
        f"{customers['customer_unique_id'].nunique():,}"
    )

    repeat_customer_records = (
        customers["customer_unique_id"]
        .value_counts()
    )

    print(
        f"Customers with >1 customer_id : "
        f"{(repeat_customer_records > 1).sum():,}"
    )

    # ========================================================
    # ORDER STATUS
    # ========================================================

    print()
    print("-" * 90)
    print("ORDER STATUS DISTRIBUTION")
    print("-" * 90)

    status_counts = (
        orders["order_status"]
        .value_counts()
    )

    status_percent = (
        orders["order_status"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    for status in status_counts.index:

        print(
            f"{status:<15}"
            f"| {status_counts[status]:>8,}"
            f" | {status_percent[status]:>6.2f}%"
        )

    # ========================================================
    # ORDER → CUSTOMER
    # ========================================================

    print()
    print("-" * 90)
    print("ORDER → CUSTOMER RELATIONSHIP")
    print("-" * 90)

    orders_per_customer = (
        orders.groupby("customer_id")
        .size()
    )

    print(
        f"Average orders per customer_id : "
        f"{orders_per_customer.mean():.2f}"
    )

    print(
        f"Maximum orders per customer_id : "
        f"{orders_per_customer.max()}"
    )

    # ========================================================
    # ITEMS PER ORDER
    # ========================================================

    print()
    print("-" * 90)
    print("ORDER ITEMS")
    print("-" * 90)

    items_per_order = (
        items.groupby("order_id")
        .size()
    )

    print(
        f"Orders with items       : "
        f"{items_per_order.shape[0]:,}"
    )

    print(
        f"Average items/order     : "
        f"{items_per_order.mean():.2f}"
    )

    print(
        f"Maximum items/order     : "
        f"{items_per_order.max()}"
    )

    # ========================================================
    # ORDER VALUE
    # ========================================================

    print()
    print("-" * 90)
    print("ORDER VALUE")
    print("-" * 90)

    order_values = (
        items.groupby("order_id")
        .agg(
            product_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
        )
    )

    order_values["order_value"] = (
        order_values["product_value"]
        + order_values["freight_value"]
    )

    print(
        f"Total product value    : "
        f"{order_values['product_value'].sum():,.2f}"
    )

    print(
        f"Total freight value    : "
        f"{order_values['freight_value'].sum():,.2f}"
    )

    print(
        f"Total order value      : "
        f"{order_values['order_value'].sum():,.2f}"
    )

    print(
        f"Average order value    : "
        f"{order_values['order_value'].mean():,.2f}"
    )

    print(
        f"Median order value     : "
        f"{order_values['order_value'].median():,.2f}"
    )

    # ========================================================
    # PAYMENT ANALYSIS
    # ========================================================

    print()
    print("-" * 90)
    print("PAYMENT ANALYSIS")
    print("-" * 90)

    payment_types = (
        payments["payment_type"]
        .value_counts()
    )

    for payment_type, count in payment_types.items():

        print(
            f"{payment_type:<15}"
            f"| {count:>8,}"
        )

    print()

    print(
        f"Total payment value    : "
        f"{payments['payment_value'].sum():,.2f}"
    )

    print(
        f"Average payment value  : "
        f"{payments['payment_value'].mean():,.2f}"
    )

    # ========================================================
    # REVIEW ANALYSIS
    # ========================================================

    print()
    print("-" * 90)
    print("REVIEW ANALYSIS")
    print("-" * 90)

    print(
        f"Average review score   : "
        f"{reviews['review_score'].mean():.2f}"
    )

    print(
        f"Median review score    : "
        f"{reviews['review_score'].median():.2f}"
    )

    print()
    print("Review score distribution:")

    review_counts = (
        reviews["review_score"]
        .value_counts()
        .sort_index()
    )

    for score, count in review_counts.items():

        print(
            f"Score {score}: {count:,}"
        )

    # ========================================================
    # PRODUCT ANALYSIS
    # ========================================================

    print()
    print("-" * 90)
    print("PRODUCT ANALYSIS")
    print("-" * 90)

    print(
        f"Products               : "
        f"{products['product_id'].nunique():,}"
    )

    print(
        f"Product categories     : "
        f"{products['product_category_name'].nunique():,}"
    )

    # ========================================================
    # SELLER ANALYSIS
    # ========================================================

    print()
    print("-" * 90)
    print("SELLER ANALYSIS")
    print("-" * 90)

    print(
        f"Sellers                : "
        f"{sellers['seller_id'].nunique():,}"
    )

    seller_orders = (
        items.groupby("seller_id")
        ["order_id"]
        .nunique()
    )

    print(
        f"Average orders/seller  : "
        f"{seller_orders.mean():.2f}"
    )

    print(
        f"Maximum orders/seller  : "
        f"{seller_orders.max():,}"
    )

    # ========================================================
    # DELIVERY ANALYSIS
    # ========================================================

    print()
    print("-" * 90)
    print("DELIVERY ANALYSIS")
    print("-" * 90)

    orders["order_delivered_customer_date"] = (
        pd.to_datetime(
            orders["order_delivered_customer_date"],
            errors="coerce"
        )
    )

    orders["order_estimated_delivery_date"] = (
        pd.to_datetime(
            orders["order_estimated_delivery_date"],
            errors="coerce"
        )
    )

    delivered = orders[
        orders["order_delivered_customer_date"]
        .notna()
    ].copy()

    delivered["delivery_difference_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    print(
        f"Orders with delivery date : "
        f"{len(delivered):,}"
    )

    print(
        f"Average delivery difference: "
        f"{delivered['delivery_difference_days'].mean():.2f} days"
    )

    print(
        f"Late deliveries           : "
        f"{(delivered['delivery_difference_days'] > 0).sum():,}"
    )

    print(
        f"On-time / early deliveries: "
        f"{(delivered['delivery_difference_days'] <= 0).sum():,}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 90)
    print("RELATIONSHIP ANALYSIS COMPLETED")
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
