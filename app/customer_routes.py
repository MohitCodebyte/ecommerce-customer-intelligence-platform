from pathlib import Path
import pandas as pd

from flask import Blueprint, render_template, request


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "customer_intelligence_master.csv"
)


customer_bp = Blueprint(
    "customers",
    __name__,
    url_prefix="/customers"
)


def load_customers():

    if not DATA_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(DATA_FILE)


def clean_value(value):

    if pd.isna(value):
        return "—"

    return value


@customer_bp.route("/")
def customers():

    df = load_customers()

    if df.empty:

        return render_template(
            "error.html",
            title="Customer Data Unavailable",
            message="Customer intelligence dataset is not available."
        ), 503


    query = request.args.get(
        "q",
        ""
    ).strip()


    filtered = df


    if query:

        query_lower = query.lower()

        mask = pd.Series(
            False,
            index=df.index
        )


        search_columns = [
            "customer_id",
            "customer_unique_id",
            "customer_city",
            "customer_state",
            "rfm_segment",
            "clv_value_tier",
            "customer_type",
            "value_priority",
            "retention_status"
        ]


        for column in search_columns:

            if column in df.columns:

                mask |= (
                    df[column]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        query_lower,
                        na=False
                    )
                )


        filtered = df[mask]


    # Limit list for UI performance
    display_df = filtered.head(100)


    customers_data = []

    for _, row in display_df.iterrows():

        customers_data.append({

            "customer_id":
                clean_value(
                    row.get(
                        "customer_unique_id",
                        row.get("customer_id")
                    )
                ),

            "city":
                clean_value(
                    row.get("customer_city")
                ),

            "state":
                clean_value(
                    row.get("customer_state")
                ),

            "segment":
                clean_value(
                    row.get("rfm_segment")
                ),

            "clv_tier":
                clean_value(
                    row.get("clv_value_tier")
                ),

            "historical_clv":
                clean_value(
                    row.get("historical_clv")
                ),

            "orders":
                clean_value(
                    row.get("completed_orders")
                ),

            "value_priority":
                clean_value(
                    row.get("value_priority")
                ),

            "retention_status":
                clean_value(
                    row.get("retention_status")
                )
        })


    return render_template(
        "customers.html",
        customers=customers_data,
        query=query,
        total_results=len(filtered)
    )


@customer_bp.route("/<customer_id>")
def customer_detail(customer_id):

    df = load_customers()

    if df.empty:

        return render_template(
            "error.html",
            title="Customer Data Unavailable",
            message="Customer intelligence dataset is not available."
        ), 503


    customer = df[
        (
            df["customer_unique_id"]
            .astype(str)
            == str(customer_id)
        )
        |
        (
            df["customer_id"]
            .astype(str)
            == str(customer_id)
        )
    ]


    if customer.empty:

        return render_template(
            "error.html",
            title="Customer Not Found",
            message=f"No customer found for ID: {customer_id}"
        ), 404


    row = customer.iloc[0]


    data = {

        "customer_id":
            clean_value(
                row.get("customer_unique_id")
            ),

        "customer_internal_id":
            clean_value(
                row.get("customer_id")
            ),

        "city":
            clean_value(
                row.get("customer_city")
            ),

        "state":
            clean_value(
                row.get("customer_state")
            ),


        # RFM
        "recency":
            clean_value(row.get("recency")),

        "frequency":
            clean_value(row.get("frequency")),

        "monetary":
            clean_value(row.get("monetary")),

        "r_score":
            clean_value(row.get("r_score")),

        "f_score":
            clean_value(row.get("f_score")),

        "m_score":
            clean_value(row.get("m_score")),

        "rfm_score":
            clean_value(row.get("rfm_score")),

        "rfm_segment":
            clean_value(row.get("rfm_segment")),


        # CLV
        "historical_clv":
            clean_value(row.get("historical_clv")),

        "clv_tier":
            clean_value(row.get("clv_value_tier")),

        "annualized_revenue":
            clean_value(row.get("annualized_revenue")),


        # Purchase behavior
        "total_orders":
            clean_value(row.get("completed_orders")),

        "completed_revenue":
            clean_value(row.get("completed_revenue")),

        "average_order_value":
            clean_value(
                row.get(
                    "average_completed_order_value"
                )
            ),

        "first_purchase":
            clean_value(
                row.get("first_purchase_date")
            ),

        "last_purchase":
            clean_value(
                row.get("last_purchase_date")
            ),

        "customer_type":
            clean_value(row.get("customer_type")),

        "repeat_customer":
            clean_value(row.get("repeat_customer")),


        # Business intelligence
        "value_priority":
            clean_value(row.get("value_priority")),

        "retention_status":
            clean_value(row.get("retention_status")),

        "data_quality":
            clean_value(row.get("data_quality_flag"))
    }


    return render_template(
        "customer-detail.html",
        customer=data
    )
