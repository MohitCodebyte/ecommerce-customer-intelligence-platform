from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


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
# PRINT DATA QUALITY
# ============================================================

def analyze_dataset(name, filename):

    path = RAW_DIR / filename

    print()
    print("=" * 90)
    print(f"DATASET: {name.upper()}")
    print(f"FILE   : {filename}")
    print("=" * 90)

    if not path.exists():

        print("FILE NOT FOUND")
        return

    df = pd.read_csv(path)

    print()
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print()

    # --------------------------------------------------------
    # COLUMN INFORMATION
    # --------------------------------------------------------

    print("-" * 90)
    print("COLUMN INFORMATION")
    print("-" * 90)

    for column in df.columns:

        dtype = str(df[column].dtype)

        missing = int(
            df[column].isna().sum()
        )

        missing_percent = (
            missing / len(df) * 100
        )

        unique = int(
            df[column].nunique(
                dropna=True
            )
        )

        print(
            f"{column:<45}"
            f"| dtype={dtype:<12}"
            f"| missing={missing:>8,}"
            f" ({missing_percent:>6.2f}%)"
            f"| unique={unique:>8,}"
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    print()
    print("-" * 90)
    print("DUPLICATE INFORMATION")
    print("-" * 90)

    print(
        f"Duplicate rows: {duplicate_count:,}"
    )

    # --------------------------------------------------------
    # DATE COLUMNS
    # --------------------------------------------------------

    date_columns = [
        column
        for column in df.columns
        if (
            "date" in column.lower()
            or "timestamp" in column.lower()
        )
    ]

    if date_columns:

        print()
        print("-" * 90)
        print("DATE RANGE")
        print("-" * 90)

        for column in date_columns:

            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            valid = converted.dropna()

            if not valid.empty:

                print(
                    f"{column:<45}"
                    f"| {valid.min()} → {valid.max()}"
                )

            else:

                print(
                    f"{column:<45}"
                    f"| No valid dates"
                )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        print()
        print("-" * 90)
        print("NUMERIC RANGE")
        print("-" * 90)

        for column in numeric_columns:

            series = df[column].dropna()

            if series.empty:
                continue

            print(
                f"{column:<45}"
                f"| min={series.min()}"
                f" | max={series.max()}"
                f" | median={series.median()}"
            )

    # --------------------------------------------------------
    # CATEGORICAL SAMPLE
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if categorical_columns:

        print()
        print("-" * 90)
        print("CATEGORICAL SAMPLE VALUES")
        print("-" * 90)

        for column in categorical_columns:

            values = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
            )

            preview = values[:10]

            print(
                f"\n{column}:"
            )

            print(
                "  "
                + ", ".join(preview)
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 90)
    print("       E-COMMERCE CUSTOMER INTELLIGENCE")
    print("              DATA QUALITY ANALYSIS")
    print("#" * 90)

    for name, filename in FILES.items():

        analyze_dataset(
            name,
            filename
        )

    print()
    print("#" * 90)
    print("                 ANALYSIS COMPLETE")
    print("#" * 90)
    print()


if __name__ == "__main__":
    main()
