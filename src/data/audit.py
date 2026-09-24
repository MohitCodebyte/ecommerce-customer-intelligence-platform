from pathlib import Path
import json
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = REPORT_DIR / "data_audit.json"


# ============================================================
# OLIST DATASETS
# ============================================================

FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


# ============================================================
# HELPER FUNCTION
# ============================================================

def safe_value(value):
    """
    Convert pandas/numpy values into JSON-safe values.
    """

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


# ============================================================
# AUDIT ONE CSV FILE
# ============================================================

def audit_file(dataset_name, filename):

    file_path = RAW_DIR / filename

    result = {
        "dataset": dataset_name,
        "file": filename,
        "exists": file_path.exists(),
    }

    # --------------------------------------------------------
    # FILE EXISTENCE
    # --------------------------------------------------------

    if not file_path.exists():

        result["error"] = "File not found"

        return result

    try:

        df = pd.read_csv(file_path)

        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        result["rows"] = int(df.shape[0])

        result["columns_count"] = int(df.shape[1])

        result["columns"] = list(df.columns)

        # ----------------------------------------------------
        # DATA TYPES
        # ----------------------------------------------------

        result["dtypes"] = {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        }

        # ----------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------

        missing = df.isna().sum()

        result["missing_values"] = {
            column: int(count)
            for column, count in missing.items()
            if count > 0
        }

        result["total_missing_cells"] = int(
            df.isna().sum().sum()
        )

        # ----------------------------------------------------
        # DUPLICATE ROWS
        # ----------------------------------------------------

        result["duplicate_rows"] = int(
            df.duplicated().sum()
        )

        # ----------------------------------------------------
        # UNIQUE VALUES
        # ----------------------------------------------------

        result["unique_counts"] = {
            column: int(
                df[column].nunique(dropna=True)
            )
            for column in df.columns
        }

        # ----------------------------------------------------
        # NUMERIC COLUMN ANALYSIS
        # ----------------------------------------------------

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        numeric_summary = {}

        for column in numeric_columns:

            series = df[column].dropna()

            if series.empty:
                continue

            numeric_summary[column] = {

                "min": safe_value(
                    series.min()
                ),

                "max": safe_value(
                    series.max()
                ),

                "mean": safe_value(
                    series.mean()
                ),

                "median": safe_value(
                    series.median()
                ),

                "negative_count": int(
                    (series < 0).sum()
                ),

                "zero_count": int(
                    (series == 0).sum()
                ),
            }

        result["numeric_summary"] = numeric_summary

        # ----------------------------------------------------
        # DATE COLUMN ANALYSIS
        # ----------------------------------------------------

        date_columns = []

        for column in df.columns:

            column_lower = column.lower()

            if (
                "date" in column_lower
                or "timestamp" in column_lower
            ):
                date_columns.append(column)

        date_summary = {}

        for column in date_columns:

            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            valid_dates = converted.dropna()

            date_summary[column] = {

                "valid_dates": int(
                    valid_dates.shape[0]
                ),

                "invalid_dates": int(
                    converted.isna().sum()
                ),

                "min_date": (
                    str(valid_dates.min())
                    if not valid_dates.empty
                    else None
                ),

                "max_date": (
                    str(valid_dates.max())
                    if not valid_dates.empty
                    else None
                ),
            }

        result["date_summary"] = date_summary

        # ----------------------------------------------------
        # CATEGORICAL / TEXT COLUMN ANALYSIS
        # ----------------------------------------------------

        categorical_columns = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        categorical_summary = {}

        for column in categorical_columns:

            unique_values = (
                df[column]
                .dropna()
                .unique()
            )

            sample_values = [
                str(value)
                for value in unique_values[:20]
            ]

            categorical_summary[column] = {

                "unique_count": int(
                    df[column].nunique(
                        dropna=True
                    )
                ),

                "sample_values": sample_values,
            }

        result["categorical_summary"] = (
            categorical_summary
        )

        return result

    except Exception as error:

        result["error"] = str(error)

        return result


# ============================================================
# MAIN AUDIT
# ============================================================

def run_audit():

    print()
    print("=" * 70)
    print("        E-COMMERCE DATASET AUDIT")
    print("=" * 70)
    print()

    report = {

        "project":
            "E-Commerce Customer Intelligence & Churn Prediction",

        "raw_data_directory":
            str(RAW_DIR),

        "datasets": {},
    }

    # ========================================================
    # AUDIT ALL DATASETS
    # ========================================================

    for dataset_name, filename in FILES.items():

        print(
            f"Auditing: {filename}"
        )

        result = audit_file(
            dataset_name,
            filename
        )

        report["datasets"][dataset_name] = result

        if result.get("exists"):

            print(
                f"  Rows       : "
                f"{result.get('rows'):,}"
            )

            print(
                f"  Columns    : "
                f"{result.get('columns_count')}"
            )

            print(
                f"  Missing    : "
                f"{result.get('total_missing_cells'):,}"
            )

            print(
                f"  Duplicates : "
                f"{result.get('duplicate_rows'):,}"
            )

        else:

            print(
                "  FILE NOT FOUND"
            )

        print()

    # ========================================================
    # RELATIONSHIP AUDIT
    # ========================================================

    print("=" * 70)
    print("        RELATIONSHIP / KEY AUDIT")
    print("=" * 70)
    print()

    relationships = {}

    try:

        customers = pd.read_csv(
            RAW_DIR / FILES["customers"]
        )

        orders = pd.read_csv(
            RAW_DIR / FILES["orders"]
        )

        order_items = pd.read_csv(
            RAW_DIR / FILES["order_items"]
        )

        products = pd.read_csv(
            RAW_DIR / FILES["products"]
        )

        sellers = pd.read_csv(
            RAW_DIR / FILES["sellers"]
        )

        payments = pd.read_csv(
            RAW_DIR / FILES["payments"]
        )

        reviews = pd.read_csv(
            RAW_DIR / FILES["reviews"]
        )

        # ----------------------------------------------------
        # CUSTOMER → ORDERS
        # ----------------------------------------------------

        customer_ids = set(
            customers["customer_id"]
            .dropna()
        )

        order_customer_ids = set(
            orders["customer_id"]
            .dropna()
        )

        orphan_order_customers = (
            order_customer_ids - customer_ids
        )

        relationships[
            "orders_to_customers"
        ] = {

            "orders_customer_ids":
                len(order_customer_ids),

            "customer_ids":
                len(customer_ids),

            "orphan_customer_ids":
                len(orphan_order_customers),
        }

        # ----------------------------------------------------
        # ORDERS → ORDER ITEMS
        # ----------------------------------------------------

        order_ids = set(
            orders["order_id"]
            .dropna()
        )

        item_order_ids = set(
            order_items["order_id"]
            .dropna()
        )

        orphan_item_orders = (
            item_order_ids - order_ids
        )

        relationships[
            "items_to_orders"
        ] = {

            "order_ids":
                len(order_ids),

            "item_order_ids":
                len(item_order_ids),

            "orphan_order_ids":
                len(orphan_item_orders),
        }

        # ----------------------------------------------------
        # ORDERS → PAYMENTS
        # ----------------------------------------------------

        payment_order_ids = set(
            payments["order_id"]
            .dropna()
        )

        orphan_payment_orders = (
            payment_order_ids - order_ids
        )

        relationships[
            "payments_to_orders"
        ] = {

            "payment_order_ids":
                len(payment_order_ids),

            "orphan_order_ids":
                len(orphan_payment_orders),
        }

        # ----------------------------------------------------
        # ORDERS → REVIEWS
        # ----------------------------------------------------

        review_order_ids = set(
            reviews["order_id"]
            .dropna()
        )

        orphan_review_orders = (
            review_order_ids - order_ids
        )

        relationships[
            "reviews_to_orders"
        ] = {

            "review_order_ids":
                len(review_order_ids),

            "orphan_order_ids":
                len(orphan_review_orders),
        }

        # ----------------------------------------------------
        # ORDER ITEMS → PRODUCTS
        # ----------------------------------------------------

        product_ids = set(
            products["product_id"]
            .dropna()
        )

        item_product_ids = set(
            order_items["product_id"]
            .dropna()
        )

        orphan_products = (
            item_product_ids - product_ids
        )

        relationships[
            "items_to_products"
        ] = {

            "product_ids":
                len(product_ids),

            "item_product_ids":
                len(item_product_ids),

            "orphan_product_ids":
                len(orphan_products),
        }

        # ----------------------------------------------------
        # ORDER ITEMS → SELLERS
        # ----------------------------------------------------

        seller_ids = set(
            sellers["seller_id"]
            .dropna()
        )

        item_seller_ids = set(
            order_items["seller_id"]
            .dropna()
        )

        orphan_sellers = (
            item_seller_ids - seller_ids
        )

        relationships[
            "items_to_sellers"
        ] = {

            "seller_ids":
                len(seller_ids),

            "item_seller_ids":
                len(item_seller_ids),

            "orphan_seller_ids":
                len(orphan_sellers),
        }

        # ----------------------------------------------------
        # PRINT RELATIONSHIP RESULTS
        # ----------------------------------------------------

        print(
            "Customer → Orders orphan IDs : "
            f"{len(orphan_order_customers)}"
        )

        print(
            "Orders → Items orphan IDs    : "
            f"{len(orphan_item_orders)}"
        )

        print(
            "Orders → Payments orphan IDs : "
            f"{len(orphan_payment_orders)}"
        )

        print(
            "Orders → Reviews orphan IDs  : "
            f"{len(orphan_review_orders)}"
        )

        print(
            "Items → Products orphan IDs  : "
            f"{len(orphan_products)}"
        )

        print(
            "Items → Sellers orphan IDs   : "
            f"{len(orphan_sellers)}"
        )

    except Exception as error:

        relationships["error"] = str(error)

        print(
            f"Relationship audit error: {error}"
        )

    report["relationships"] = relationships

    # ========================================================
    # SAVE JSON REPORT
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()
    print("=" * 70)
    print("AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()

    print(
        f"Report saved at:"
    )

    print(
        OUTPUT_FILE
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_audit()