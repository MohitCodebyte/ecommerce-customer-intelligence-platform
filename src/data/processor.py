from pathlib import Path
import pandas as pd
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = ROOT / "data" / "uploads"
TEMPLATE_DIR = ROOT / "data" / "templates"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

for folder in [UPLOAD_DIR, TEMPLATE_DIR, PROCESSED_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# COMPANY DATA STANDARDIZATION ENGINE
# ============================================================

ALIASES = {
    "customer_id": [
        "customer_id", "customerid", "customer id",
        "client_id", "clientid", "user_id", "userid"
    ],
    "customer_unique_id": [
        "customer_unique_id", "customer_uniqueid",
        "customer_unique", "unique_customer_id",
        "unique_customer", "client_unique_id"
    ],
    "order_id": [
        "order_id", "orderid", "order id",
        "transaction_id", "transactionid",
        "invoice_id", "invoiceid"
    ],
    "order_date": [
        "order_date", "orderdate", "order date",
        "purchase_date", "purchase_date_time",
        "transaction_date", "transactiondate",
        "date", "created_at", "order_datetime"
    ],
    "order_value": [
        "order_value", "ordervalue", "order value",
        "amount", "total_amount", "totalamount",
        "sales", "revenue", "transaction_amount",
        "purchase_amount", "price", "total"
    ],
    "product_id": [
        "product_id", "productid", "product id",
        "sku", "item_id", "itemid"
    ],
    "category": [
        "category", "product_category",
        "product_category_name", "category_name",
        "product_type"
    ],
    "quantity": [
        "quantity", "qty", "units",
        "items", "item_quantity"
    ],
    "city": [
        "city", "customer_city", "location_city"
    ],
    "state": [
        "state", "customer_state", "province",
        "region", "state_code"
    ],
    "payment_type": [
        "payment_type", "payment_method",
        "payment_method_type", "payment"
    ],
    "review_score": [
        "review_score", "reviewscore",
        "rating", "review_rating", "customer_rating"
    ]
}


def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def find_canonical_column(columns, aliases):
    normalized = {
        normalize_column_name(col): col
        for col in columns
    }

    for alias in aliases:
        alias_normalized = normalize_column_name(alias)

        if alias_normalized in normalized:
            return normalized[alias_normalized]

    return None


def detect_and_rename(df):
    rename_map = {}
    mapping = {}

    for canonical, aliases in ALIASES.items():
        actual = find_canonical_column(df.columns, aliases)

        if actual is not None:
            rename_map[actual] = canonical
            mapping[canonical] = actual

    standardized = df.rename(columns=rename_map).copy()

    return standardized, mapping


def load_file(path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def generate_templates():

    customer_template = pd.DataFrame(columns=[
        "customer_id",
        "customer_unique_id",
        "city",
        "state"
    ])

    transaction_template = pd.DataFrame(columns=[
        "order_id",
        "customer_id",
        "customer_unique_id",
        "order_date",
        "order_value",
        "product_id",
        "category",
        "quantity",
        "payment_type",
        "review_score"
    ])

    customer_template.to_csv(
        TEMPLATE_DIR / "company_customer_template.csv",
        index=False
    )

    transaction_template.to_csv(
        TEMPLATE_DIR / "company_transaction_template.csv",
        index=False
    )

    with pd.ExcelWriter(
        TEMPLATE_DIR / "company_data_template.xlsx",
        engine="openpyxl"
    ) as writer:

        customer_template.to_excel(
            writer,
            sheet_name="customers",
            index=False
        )

        transaction_template.to_excel(
            writer,
            sheet_name="transactions",
            index=False
        )


def process_latest_file():

    files = sorted(
        list(UPLOAD_DIR.glob("*.csv")) +
        list(UPLOAD_DIR.glob("*.xlsx")) +
        list(UPLOAD_DIR.glob("*.xls")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not files:
        print()
        print("No company file found.")
        print()
        print("However, company templates have been generated.")
        print()
        print(f"CSV templates:")
        print(f"  {TEMPLATE_DIR / 'company_customer_template.csv'}")
        print(f"  {TEMPLATE_DIR / 'company_transaction_template.csv'}")
        print()
        print(f"Excel template:")
        print(f"  {TEMPLATE_DIR / 'company_data_template.xlsx'}")
        print()

        return

    source = files[0]

    print()
    print(f"Processing: {source.name}")

    df = load_file(source)

    original_rows = len(df)
    original_columns = list(df.columns)

    standardized, mapping = detect_and_rename(df)

    errors = []
    warnings = []

    # --------------------------------------------------------
    # Determine data type
    # --------------------------------------------------------

    has_customer = "customer_id" in standardized.columns
    has_order = "order_id" in standardized.columns
    has_date = "order_date" in standardized.columns
    has_value = "order_value" in standardized.columns

    if has_order and has_customer and has_date and has_value:
        data_type = "transaction"

    elif has_customer:
        data_type = "customer"

    else:
        data_type = "unknown"
        errors.append(
            "Unable to identify required customer or transaction schema."
        )

    # --------------------------------------------------------
    # Standardize customer identity
    # --------------------------------------------------------

    if "customer_unique_id" not in standardized.columns:
        if "customer_id" in standardized.columns:
            standardized["customer_unique_id"] = (
                standardized["customer_id"].astype(str)
            )
            warnings.append(
                "customer_unique_id was not provided; "
                "customer_id was used as customer_unique_id."
            )

    # --------------------------------------------------------
    # Date processing
    # --------------------------------------------------------

    if "order_date" in standardized.columns:

        standardized["order_date"] = pd.to_datetime(
            standardized["order_date"],
            errors="coerce"
        )

        invalid_dates = int(
            standardized["order_date"].isna().sum()
        )

        if invalid_dates > 0:
            errors.append(
                f"{invalid_dates} rows contain invalid/missing order dates."
            )

    # --------------------------------------------------------
    # Numeric processing
    # --------------------------------------------------------

    if "order_value" in standardized.columns:

        standardized["order_value"] = pd.to_numeric(
            standardized["order_value"],
            errors="coerce"
        )

        invalid_values = int(
            standardized["order_value"].isna().sum()
        )

        if invalid_values > 0:
            errors.append(
                f"{invalid_values} rows contain invalid/missing order values."
            )

        negative_values = int(
            (standardized["order_value"] < 0).sum()
        )

        if negative_values > 0:
            warnings.append(
                f"{negative_values} rows contain negative order values."
            )

    if "quantity" in standardized.columns:

        standardized["quantity"] = pd.to_numeric(
            standardized["quantity"],
            errors="coerce"
        )

    if "review_score" in standardized.columns:

        standardized["review_score"] = pd.to_numeric(
            standardized["review_score"],
            errors="coerce"
        )

        invalid_reviews = int(
            (
                (standardized["review_score"] < 1) |
                (standardized["review_score"] > 5)
            ).sum()
        )

        if invalid_reviews > 0:
            warnings.append(
                f"{invalid_reviews} rows contain review scores outside 1-5."
            )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    duplicate_rows = int(
        standardized.duplicated().sum()
    )

    if duplicate_rows > 0:
        warnings.append(
            f"{duplicate_rows} duplicate rows detected."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_summary = {}

    for column in standardized.columns:

        missing = int(
            standardized[column].isna().sum()
        )

        if missing > 0:
            missing_summary[column] = missing

    # --------------------------------------------------------
    # Save standardized data
    # --------------------------------------------------------

    output_name = (
        "standardized_company_transactions.csv"
        if data_type == "transaction"
        else "standardized_company_customers.csv"
    )

    output_path = PROCESSED_DIR / output_name

    standardized.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Validation status
    # --------------------------------------------------------

    status = "PASS" if not errors else "FAIL"

    report = {
        "generated_at": datetime.now().isoformat(),
        "source_file": source.name,
        "detected_data_type": data_type,
        "validation_status": status,
        "original_rows": original_rows,
        "original_columns": original_columns,
        "standardized_rows": len(standardized),
        "standardized_columns": list(standardized.columns),
        "column_mapping": mapping,
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_summary,
        "errors": errors,
        "warnings": warnings,
        "output_file": str(output_path)
    }

    report_path = REPORT_DIR / "company_data_standardization.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("COMPANY DATA STANDARDIZATION RESULT")
    print("=" * 80)

    print(f"Source file       : {source.name}")
    print(f"Data type         : {data_type}")
    print(f"Rows              : {len(standardized):,}")
    print(f"Columns           : {len(standardized.columns)}")
    print(f"Duplicate rows    : {duplicate_rows:,}")
    print(f"Validation status : {status}")

    if warnings:
        print()
        print("WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print()
        print("ERRORS:")
        for error in errors:
            print(f"  - {error}")

    print()
    print(f"Standardized file : {output_path}")
    print(f"Validation report : {report_path}")
    print()


if __name__ == "__main__":

    print()
    print("=" * 80)
    print("COMPANY DATA STANDARDIZATION ENGINE")
    print("=" * 80)

    generate_templates()
    process_latest_file()
