"""
Company Data Validation Engine
------------------------------

Validates uploaded/company datasets before
they enter the Customer Intelligence pipeline.

Supported:
- CSV
- XLSX
- XLS

This module does NOT modify the original file.

It produces:
- validation status
- required columns
- missing columns
- extra columns
- data quality statistics
- duplicate statistics
- business-rule warnings

The Olist dataset remains the development/reference dataset.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
UPLOADS = ROOT / "data" / "uploads"
REPORTS = ROOT / "reports"

UPLOADS.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = (
    REPORTS
    / "company_data_validation.json"
)


# ============================================================
# SUPPORTED SCHEMAS
# ============================================================

# Minimum customer-level schema.
#
# A company does NOT need to use Olist's exact column names.
# The application will eventually map uploaded columns
# to these business concepts.

CUSTOMER_SCHEMA = {

    "customer_id": {
        "required": True,
        "aliases": [
            "customer_id",
            "customerid",
            "customer id",
            "client_id",
            "clientid",
            "client id",
        ],
    },

    "customer_unique_id": {
        "required": False,
        "aliases": [
            "customer_unique_id",
            "customer_uniqueid",
            "unique_customer_id",
            "unique_customerid",
            "customer_uuid",
            "customer_code",
        ],
    },

    "customer_city": {
        "required": False,
        "aliases": [
            "customer_city",
            "city",
            "customer city",
        ],
    },

    "customer_state": {
        "required": False,
        "aliases": [
            "customer_state",
            "state",
            "customer state",
            "region",
        ],
    },

}


ORDER_SCHEMA = {

    "order_id": {
        "required": True,
        "aliases": [
            "order_id",
            "orderid",
            "order id",
            "invoice_id",
            "invoiceid",
            "transaction_id",
            "transactionid",
        ],
    },

    "customer_id": {
        "required": True,
        "aliases": [
            "customer_id",
            "customerid",
            "customer id",
            "client_id",
            "clientid",
        ],
    },

    "order_date": {
        "required": True,
        "aliases": [
            "order_date",
            "orderdate",
            "order date",
            "purchase_date",
            "purchase_date_time",
            "purchase_datetime",
            "transaction_date",
            "transactiondate",
            "date",
        ],
    },

    "order_value": {
        "required": True,
        "aliases": [
            "order_value",
            "ordervalue",
            "order value",
            "revenue",
            "sales",
            "sales_amount",
            "amount",
            "total_amount",
            "total_value",
        ],
    },

}


# ============================================================
# HELPERS
# ============================================================

def normalize_name(name):

    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def load_file(path):

    suffix = path.suffix.lower()

    if suffix == ".csv":

        return pd.read_csv(
            path,
            low_memory=False
        )

    if suffix in [".xlsx", ".xls"]:

        return pd.read_excel(
            path
        )

    raise ValueError(
        "Unsupported file format: "
        + suffix
    )


def find_column(columns, aliases):

    normalized_columns = {
        normalize_name(column): column
        for column in columns
    }

    normalized_aliases = [
        normalize_name(alias)
        for alias in aliases
    ]

    for alias in normalized_aliases:

        if alias in normalized_columns:

            return normalized_columns[
                alias
            ]

    return None


def validate_schema(
    df,
    schema
):

    result = {}

    mapped = {}

    missing = []

    for business_field, config in schema.items():

        matched_column = find_column(
            df.columns,
            config["aliases"]
        )

        if matched_column:

            mapped[
                business_field
            ] = matched_column

        elif config["required"]:

            missing.append(
                business_field
            )

    result["mapped_columns"] = mapped

    result["missing_required_columns"] = (
        missing
    )

    result["valid"] = (
        len(missing) == 0
    )

    return result


def column_quality(df):

    quality = {}

    for column in df.columns:

        series = df[column]

        missing_count = int(
            series.isna().sum()
        )

        empty_count = int(
            (
                series
                .astype(str)
                .str.strip()
                .isin(["", "nan", "None"])
            ).sum()
        )

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        quality[column] = {

            "dtype":
                str(series.dtype),

            "rows":
                int(len(series)),

            "missing":
                missing_count,

            "missing_pct":
                round(
                    missing_count
                    / len(series)
                    * 100,
                    2
                )
                if len(series)
                else 0,

            "empty_like_values":
                empty_count,

            "unique_values":
                unique_count,

        }

    return quality


# ============================================================
# FIND INPUT FILE
# ============================================================

print()
print("=" * 80)
print("COMPANY DATA VALIDATION ENGINE")
print("=" * 80)

print()
print(
    "Looking for CSV/XLSX files in:"
)

print(
    UPLOADS
)


files = []

for pattern in [
    "*.csv",
    "*.xlsx",
    "*.xls",
]:

    files.extend(
        UPLOADS.glob(pattern)
    )


if not files:

    print()
    print(
        "No company file found in data/uploads."
    )

    print()
    print(
        "Put a CSV or Excel file inside:"
    )

    print(
        UPLOADS
    )

    print()
    print(
        "Example:"
    )

    print(
        "data/uploads/company_customers.csv"
    )

    raise SystemExit(0)


# Use newest file if multiple files exist.
input_file = max(
    files,
    key=lambda path: path.stat().st_mtime
)


print()
print(
    "Selected file:"
)

print(
    input_file
)


# ============================================================
# LOAD
# ============================================================

print()
print("Loading file...")

df = load_file(
    input_file
)

print(
    "Rows    : "
    + f"{len(df):,}"
)

print(
    "Columns : "
    + f"{len(df.columns):,}"
)


# ============================================================
# BASIC FILE VALIDATION
# ============================================================

file_size_mb = (
    input_file.stat().st_size
    / (
        1024
        * 1024
    )
)


# ============================================================
# CUSTOMER SCHEMA
# ============================================================

print()
print("Validating customer schema...")

customer_validation = validate_schema(
    df,
    CUSTOMER_SCHEMA
)


# ============================================================
# ORDER SCHEMA
# ============================================================

print()
print("Validating transaction schema...")

order_validation = validate_schema(
    df,
    ORDER_SCHEMA
)


# ============================================================
# DETERMINE DATA TYPE
# ============================================================

if order_validation["valid"]:

    detected_data_type = (
        "transaction"
    )

elif customer_validation["valid"]:

    detected_data_type = (
        "customer"
    )

else:

    detected_data_type = (
        "unknown"
    )


# ============================================================
# DUPLICATES
# ============================================================

duplicate_rows = int(
    df.duplicated().sum()
)

duplicate_pct = (
    duplicate_rows
    / len(df)
    * 100
    if len(df)
    else 0
)


# ============================================================
# QUALITY
# ============================================================

print()
print("Running data quality checks...")

quality = column_quality(
    df
)


# ============================================================
# GLOBAL MISSINGNESS
# ============================================================

total_cells = (
    len(df)
    * len(df.columns)
)

total_missing = int(
    df.isna().sum().sum()
)

missing_pct = (
    total_missing
    / total_cells
    * 100
    if total_cells
    else 0
)


# ============================================================
# BUSINESS WARNINGS
# ============================================================

warnings = []


if len(df) == 0:

    warnings.append(
        "Uploaded file contains zero rows."
    )


if len(df.columns) == 0:

    warnings.append(
        "Uploaded file contains zero columns."
    )


if duplicate_rows > 0:

    warnings.append(
        f"File contains {duplicate_rows:,} "
        "duplicate rows."
    )


if missing_pct > 20:

    warnings.append(
        "Overall missingness exceeds 20%."
    )


if detected_data_type == "unknown":

    warnings.append(
        "Required customer or transaction "
        "schema could not be detected."
    )


# ============================================================
# NUMERIC BUSINESS CHECKS
# ============================================================

business_checks = {}


if order_validation["valid"]:

    mapped = (
        order_validation[
            "mapped_columns"
        ]
    )

    date_column = mapped[
        "order_date"
    ]

    value_column = mapped[
        "order_value"
    ]

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    values = pd.to_numeric(
        df[value_column],
        errors="coerce"
    )

    invalid_dates = int(
        dates.isna().sum()
    )

    invalid_values = int(
        values.isna().sum()
    )

    negative_values = int(
        (
            values < 0
        ).sum()
    )

    zero_values = int(
        (
            values == 0
        ).sum()
    )

    business_checks = {

        "invalid_order_dates":
            invalid_dates,

        "invalid_order_values":
            invalid_values,

        "negative_order_values":
            negative_values,

        "zero_order_values":
            zero_values,

        "minimum_order_value":
            float(
                values.min()
            )
            if values.notna().any()
            else None,

        "maximum_order_value":
            float(
                values.max()
            )
            if values.notna().any()
            else None,

        "median_order_value":
            float(
                values.median()
            )
            if values.notna().any()
            else None,
    }

    if invalid_dates > 0:

        warnings.append(
            f"{invalid_dates:,} rows have "
            "invalid order dates."
        )

    if invalid_values > 0:

        warnings.append(
            f"{invalid_values:,} rows have "
            "invalid order values."
        )

    if negative_values > 0:

        warnings.append(
            f"{negative_values:,} negative "
            "order values detected."
        )


# ============================================================
# VALIDATION STATUS
# ============================================================

critical_errors = []

if detected_data_type == "unknown":

    critical_errors.append(
        "Unable to detect a supported business schema."
    )

if len(df) == 0:

    critical_errors.append(
        "File is empty."
    )


validation_status = (
    "PASS"
    if not critical_errors
    else "FAIL"
)


# ============================================================
# FINAL REPORT
# ============================================================

report = {

    "file": {

        "name":
            input_file.name,

        "path":
            str(input_file),

        "extension":
            input_file.suffix.lower(),

        "size_mb":
            round(
                file_size_mb,
                2
            ),

        "rows":
            int(len(df)),

        "columns":
            int(len(df.columns)),
    },

    "detected_data_type":
        detected_data_type,

    "validation_status":
        validation_status,

    "critical_errors":
        critical_errors,

    "customer_schema":
        customer_validation,

    "transaction_schema":
        order_validation,

    "duplicates": {

        "duplicate_rows":
            duplicate_rows,

        "duplicate_pct":
            round(
                duplicate_pct,
                2
            ),
    },

    "missingness": {

        "total_missing_cells":
            total_missing,

        "missing_pct":
            round(
                missing_pct,
                2
            ),
    },

    "business_checks":
        business_checks,

    "warnings":
        warnings,

    "column_quality":
        quality,
}


# ============================================================
# SAVE
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()
print("=" * 80)
print("VALIDATION RESULT")
print("=" * 80)

print()
print(
    "File                 : "
    + input_file.name
)

print(
    "Detected data type   : "
    + detected_data_type
)

print(
    "Validation status    : "
    + validation_status
)

print(
    "Rows                 : "
    + f"{len(df):,}"
)

print(
    "Columns              : "
    + f"{len(df.columns):,}"
)

print(
    "Duplicate rows       : "
    + f"{duplicate_rows:,}"
)

print(
    "Missing cells        : "
    + f"{total_missing:,}"
)

print(
    "Missingness          : "
    + f"{missing_pct:.2f}%"
)


print()
print("MAPPED CUSTOMER COLUMNS")
print("-" * 80)

for key, value in (
    customer_validation[
        "mapped_columns"
    ].items()
):

    print(
        f"{key:<25} -> {value}"
    )


print()
print("MAPPED TRANSACTION COLUMNS")
print("-" * 80)

for key, value in (
    order_validation[
        "mapped_columns"
    ].items()
):

    print(
        f"{key:<25} -> {value}"
    )


print()
print("WARNINGS")
print("-" * 80)

if warnings:

    for warning in warnings:

        print(
            "WARNING: "
            + warning
        )

else:

    print(
        "No major warnings."
    )


print()
print("REPORT")
print("-" * 80)

print(
    "Saved validation report:"
)

print(
    REPORT_FILE
)

print()
print("Done.")
