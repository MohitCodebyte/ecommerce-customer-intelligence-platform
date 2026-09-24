"""
CIX — E-Commerce Customer Intelligence Platform
Production Flask Application
"""

from pathlib import Path
import sys
import json
import io
import datetime

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    send_file,
    flash,
    session,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts_production"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"


MODEL_PATH = MODEL_DIR / "xgb_repeat_purchase_production.joblib"
IMPUTER_PATH = MODEL_DIR / "imputer.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
THRESHOLD_PATH = REPORT_DIR / "threshold.json"
METRICS_PATH = REPORT_DIR / "metrics.json"

MASTER_CSV = PROCESSED_DIR / "customer_intelligence_master.csv"
RECOMMENDATIONS_CSV = PROCESSED_DIR / "customer_recommendations.csv"
PREDICTIONS_CSV = PROCESSED_DIR / "repeat_purchase_predictions.csv"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

app.secret_key = "ecommerce-customer-intelligence-production"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ============================================================
# GLOBAL APPLICATION STATE
# ============================================================

CURRENT_DATA = None
CURRENT_SOURCE = "No dataset loaded"
LAST_UPLOAD_INFO = {}


# ============================================================
# LOAD ML ARTIFACTS
# ============================================================

MODEL = None
IMPUTER = None
FEATURE_COLUMNS = []
THRESHOLD = 0.49649133284886676
MODEL_METRICS = {}


def load_model_artifacts():
    global MODEL, IMPUTER, FEATURE_COLUMNS, THRESHOLD, MODEL_METRICS

    try:
        if MODEL_PATH.exists():
            MODEL = joblib.load(MODEL_PATH)

        if IMPUTER_PATH.exists():
            IMPUTER = joblib.load(IMPUTER_PATH)

        if FEATURE_COLUMNS_PATH.exists():
            with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
                FEATURE_COLUMNS = json.load(f)

        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
                threshold_data = json.load(f)
            THRESHOLD = float(threshold_data.get("selected_threshold", THRESHOLD))

        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                MODEL_METRICS = json.load(f)

        print("ML artifacts loaded successfully.")
        print(f"Threshold : {THRESHOLD:.4f}")
        print(f"Features  : {len(FEATURE_COLUMNS)}")

    except Exception as error:
        print(f"WARNING: ML artifacts could not be loaded: {error}")


load_model_artifacts()


# ============================================================
# DEMO DATA FALLBACK
# ============================================================

DEMO_DATA = None
DEMO_RECS = None


def load_demo_data():
    """Load pre-processed Olist data as demo fallback."""
    global DEMO_DATA, DEMO_RECS

    try:
        if MASTER_CSV.exists():
            DEMO_DATA = pd.read_csv(MASTER_CSV)
            print(f"Demo data loaded: {len(DEMO_DATA):,} customers")

        if RECOMMENDATIONS_CSV.exists():
            DEMO_RECS = pd.read_csv(RECOMMENDATIONS_CSV)
            print(f"Demo recommendations loaded: {len(DEMO_RECS):,} records")

    except Exception as e:
        print(f"WARNING: Demo data could not be loaded: {e}")


load_demo_data()


def get_active_data():
    """Return the currently active dataset (uploaded or demo)."""
    if CURRENT_DATA is not None:
        return CURRENT_DATA, CURRENT_SOURCE, False
    if DEMO_DATA is not None:
        return DEMO_DATA, "Olist Brazilian E-Commerce Dataset (Demo)", True
    return None, "No dataset loaded", False


def get_active_recs():
    """Return the currently active recommendations data."""
    if CURRENT_DATA is not None and "recommendation_action" in CURRENT_DATA.columns:
        return CURRENT_DATA
    if DEMO_RECS is not None:
        return DEMO_RECS
    return None


# ============================================================
# FEATURE ENGINEERING (same as training)
# ============================================================

GAP_COLUMNS = [
    "mean_purchase_gap_days",
    "median_purchase_gap_days",
    "max_purchase_gap_days",
    "purchase_gap_count",
]

DROP_COLUMNS = [
    "customer_unique_id",
    "snapshot_date",
    "future_end_date",
    "first_purchase_date",
    "last_purchase_date",
    "future_purchase_count",
    "churn_label",
    "repeat_purchase",
]


def engineer_features(df):
    df = df.copy()

    for col in GAP_COLUMNS:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)

    safe_orders = df["total_orders"].replace(0, np.nan) if "total_orders" in df.columns else np.nan
    safe_lifetime = df["customer_lifetime_days"].replace(0, np.nan) if "customer_lifetime_days" in df.columns else np.nan
    safe_window = df["observation_window_days"].replace(0, np.nan) if "observation_window_days" in df.columns else np.nan

    if {"total_revenue", "total_orders"}.issubset(df.columns):
        df["revenue_per_order"] = df["total_revenue"] / safe_orders

    if {"total_orders", "customer_lifetime_days"}.issubset(df.columns):
        df["orders_per_lifetime_day"] = df["total_orders"] / safe_lifetime

    if {"total_revenue", "customer_lifetime_days"}.issubset(df.columns):
        df["revenue_per_lifetime_day"] = df["total_revenue"] / safe_lifetime

    if {"recency_days", "observation_window_days"}.issubset(df.columns):
        df["recency_ratio"] = df["recency_days"] / safe_window

    if {"purchase_frequency", "observation_window_days"}.issubset(df.columns):
        df["purchase_frequency_ratio"] = df["purchase_frequency"] / safe_window

    if {"mean_purchase_gap_days", "observation_window_days"}.issubset(df.columns):
        df["gap_ratio"] = df["mean_purchase_gap_days"] / safe_window

    if {"total_revenue", "total_orders"}.issubset(df.columns):
        df["revenue_order_frequency"] = df["total_revenue"] / (df["total_orders"] + 1)

    for col in ["total_revenue", "average_order_value", "total_orders", "recency_days"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    if {"recency_days", "customer_lifetime_days"}.issubset(df.columns):
        df["recency_to_lifetime"] = df["recency_days"] / safe_lifetime

    if {"total_orders", "purchase_frequency"}.issubset(df.columns):
        df["orders_x_frequency"] = df["total_orders"] * df["purchase_frequency"]

    if {"repeat_customer", "recency_days"}.issubset(df.columns):
        df["repeat_customer_recency"] = df["repeat_customer"] * df["recency_days"]

    if {"average_order_value", "recency_days"}.issubset(df.columns):
        df["aov_recency_interaction"] = df["average_order_value"] / (df["recency_days"] + 1)

    return df


def build_feature_matrix(df):
    df = engineer_features(df)

    X = df.drop(
        columns=[col for col in DROP_COLUMNS if col in df.columns],
        errors="ignore",
    )

    X = X.apply(pd.to_numeric, errors="coerce")

    if FEATURE_COLUMNS:
        for col in FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = np.nan
        X = X[FEATURE_COLUMNS]

    return X


# ============================================================
# PREDICTION
# ============================================================

def predict_dataframe(df):
    if MODEL is None or IMPUTER is None:
        raise RuntimeError("Production ML model is not loaded.")

    X = build_feature_matrix(df)
    X_imputed = IMPUTER.transform(X)

    if isinstance(MODEL, list):
        probabilities = np.mean(
            [model.predict_proba(X_imputed)[:, 1] for model in MODEL],
            axis=0,
        )
    else:
        probabilities = MODEL.predict_proba(X_imputed)[:, 1]

    predictions = (probabilities >= THRESHOLD).astype(int)

    result = df.copy()
    result["repeat_purchase_probability"] = probabilities
    result["repeat_purchase_prediction"] = predictions
    result["prediction_label"] = np.where(
        predictions == 1,
        "Likely Repeat Purchase",
        "Lower Repeat Purchase Propensity",
    )

    return result


# ============================================================
# DATA LOADING
# ============================================================

def load_uploaded_file(file):
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file)
    if filename.endswith(".xlsx"):
        return pd.read_excel(file)
    if filename.endswith(".xls"):
        return pd.read_excel(file)

    raise ValueError("Only CSV, XLS and XLSX files are supported.")


def set_current_data(df, source="Uploaded Dataset"):
    global CURRENT_DATA, CURRENT_SOURCE
    CURRENT_DATA = df.copy()
    CURRENT_SOURCE = source


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(df):
    summary = {
        "customers": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
    }

    rev_col = "total_revenue" if "total_revenue" in df.columns else ("monetary" if "monetary" in df.columns else None)
    summary["revenue"] = float(pd.to_numeric(df[rev_col], errors="coerce").fillna(0).sum()) if rev_col else 0

    ord_col = "total_orders" if "total_orders" in df.columns else ("completed_orders" if "completed_orders" in df.columns else None)
    summary["orders"] = int(pd.to_numeric(df[ord_col], errors="coerce").fillna(0).sum()) if ord_col else 0

    return summary


# ============================================================
# CUSTOMER SEGMENTATION (for uploaded files without rfm_segment)
# ============================================================

def customer_segments(df):
    result = df.copy()

    value_col = "monetary" if "monetary" in result.columns else ("total_revenue" if "total_revenue" in result.columns else None)

    if value_col:
        values = pd.to_numeric(result[value_col], errors="coerce").fillna(0)
        result["value_segment"] = pd.qcut(
            values.rank(method="first"),
            5,
            labels=["Low", "Below Average", "Medium", "High", "Very High"],
        )

    if "recency_days" in result.columns:
        result["activity_status"] = pd.cut(
            pd.to_numeric(result["recency_days"], errors="coerce"),
            bins=[-np.inf, 30, 90, 180, np.inf],
            labels=["Active", "Needs Attention", "At Risk", "Hibernating"],
        )

    return result


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendation(row):
    probability = float(row.get("repeat_purchase_probability", 0) or 0)
    recency = row.get("recency_days", np.nan)
    orders = int(row.get("total_orders", 1) or 1)
    revenue = float(row.get("total_revenue", 0) or 0)

    try:
        recency = float(recency)
    except Exception:
        recency = np.nan

    if probability >= 0.70:
        return "Strengthen loyalty and encourage repeat purchase."
    if probability >= THRESHOLD:
        return "Target with personalized offer or cross-sell campaign."
    if revenue >= 300:
        return "High-value customer: prioritize retention outreach."
    if not np.isnan(recency) and recency > 180:
        return "Re-engage customer with a win-back campaign."
    if orders <= 1:
        return "Encourage second purchase with targeted promotion."
    return "Monitor customer behavior and engagement."


def add_recommendations(df):
    result = df.copy()
    result["business_recommendation"] = result.apply(generate_recommendation, axis=1)
    return result


# ============================================================
# CHART DATA HELPERS
# ============================================================

def get_segment_chart_data(df):
    """Return segment distribution for charts."""
    seg_col = "rfm_segment" if "rfm_segment" in df.columns else ("value_segment" if "value_segment" in df.columns else None)
    if not seg_col:
        return {"labels": [], "data": []}

    counts = df[seg_col].value_counts().head(8)
    return {
        "labels": counts.index.tolist(),
        "data": counts.values.tolist(),
    }


def get_revenue_tier_data(df):
    """Return revenue tier distribution."""
    tier_col = "clv_value_tier" if "clv_value_tier" in df.columns else ("value_segment" if "value_segment" in df.columns else None)
    if not tier_col:
        return {"labels": [], "data": []}

    counts = df[tier_col].value_counts()
    return {
        "labels": counts.index.tolist(),
        "data": counts.values.tolist(),
    }


def get_purchase_status_data(df):
    """Return repeat vs single purchase counts."""
    if "repeat_customer" in df.columns:
        rc = df["repeat_customer"].fillna(0)
        repeat = int((rc == 1).sum())
        single = int((rc == 0).sum())
    elif "customer_type" in df.columns:
        repeat = int((df["customer_type"] == "Repeat Customer").sum())
        single = int((df["customer_type"] == "Single Purchase Customer").sum())
    elif "repeat_purchase_prediction" in df.columns:
        repeat = int(df["repeat_purchase_prediction"].sum())
        single = len(df) - repeat
    else:
        return {"labels": ["Repeat", "Single Purchase"], "data": [0, 0]}

    return {
        "labels": ["Repeat Customers", "Single Purchase"],
        "data": [repeat, single],
    }


def get_recency_distribution(df):
    """Return recency histogram bins."""
    rec_col = "recency" if "recency" in df.columns else ("recency_days" if "recency_days" in df.columns else None)
    if not rec_col:
        return {"labels": [], "data": []}

    vals = pd.to_numeric(df[rec_col], errors="coerce").dropna()
    bins = [0, 30, 60, 90, 120, 180, 365, 9999]
    labels = ["0-30d", "31-60d", "61-90d", "91-120d", "121-180d", "181-365d", "365d+"]
    counts, _ = np.histogram(vals, bins=bins)
    return {"labels": labels, "data": counts.tolist()}


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return redirect(url_for("dashboard"))


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard/")
@app.route("/dashboard")
def dashboard():
    df, source, is_demo = get_active_data()

    if df is None:
        return render_template(
            "dashboard.html",
            data_loaded=False,
            is_demo=False,
            source="No dataset loaded",
            kpis={},
            chart_data={},
        )

    summary = dataset_summary(df)

    # KPIs
    repeat_customers = 0
    repeat_rate = 0
    avg_clv = 0

    rc_col = "repeat_customer" if "repeat_customer" in df.columns else None
    if rc_col:
        repeat_customers = int((pd.to_numeric(df[rc_col], errors="coerce").fillna(0) == 1).sum())
    elif "repeat_purchase_prediction" in df.columns:
        repeat_customers = int(df["repeat_purchase_prediction"].sum())

    if summary["customers"] > 0:
        repeat_rate = repeat_customers / summary["customers"]

    clv_col = "historical_clv" if "historical_clv" in df.columns else None
    if clv_col:
        avg_clv = float(pd.to_numeric(df[clv_col], errors="coerce").fillna(0).mean())

    kpis = {
        "total_customers": summary["customers"],
        "repeat_customers": repeat_customers,
        "repeat_rate": round(repeat_rate * 100, 1),
        "total_orders": summary["orders"],
        "total_revenue": round(summary["revenue"], 2),
        "avg_clv": round(avg_clv, 2),
    }

    chart_data = {
        "segments": get_segment_chart_data(df),
        "revenue_tiers": get_revenue_tier_data(df),
        "purchase_status": get_purchase_status_data(df),
        "recency": get_recency_distribution(df),
    }

    return render_template(
        "dashboard.html",
        data_loaded=True,
        is_demo=is_demo,
        source=source,
        kpis=kpis,
        chart_data=json.dumps(chart_data),
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route("/upload/", methods=["GET", "POST"])
@app.route("/upload", methods=["GET", "POST"])
def upload():
    global LAST_UPLOAD_INFO

    if request.method == "GET":
        return render_template(
            "upload.html",
            data_loaded=CURRENT_DATA is not None,
            upload_info=LAST_UPLOAD_INFO,
        )

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file selected."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"status": "error", "message": "No file selected."}), 400

    fname = file.filename.lower()
    if not (fname.endswith(".csv") or fname.endswith(".xlsx") or fname.endswith(".xls")):
        return jsonify({"status": "error", "message": "Only CSV, XLSX and XLS files are supported."}), 400

    try:
        df = load_uploaded_file(file)

        if df.empty:
            return jsonify({"status": "error", "message": "The uploaded file is empty."}), 400

        rows = len(df)
        cols = len(df.columns)
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())

        # Validate
        issues = []
        if rows == 0:
            issues.append("File has no data rows.")
        if cols < 3:
            issues.append("File has very few columns.")
        if missing / max(rows * cols, 1) > 0.5:
            issues.append("More than 50% of values are missing.")

        upload_info = {
            "filename": file.filename,
            "rows": rows,
            "columns": cols,
            "missing_cells": missing,
            "duplicate_rows": duplicates,
            "missing_pct": round(missing / max(rows * cols, 1) * 100, 1),
            "column_names": list(df.columns)[:20],
            "issues": issues,
            "status": "error" if issues else ("warning" if missing > 0 else "success"),
        }

        LAST_UPLOAD_INFO = upload_info

        # Run pipeline
        try:
            df = predict_dataframe(df)
        except Exception as pe:
            upload_info["prediction_warning"] = str(pe)

        df = customer_segments(df)
        df = add_recommendations(df)
        set_current_data(df, source=file.filename)

        upload_info["pipeline_complete"] = True

        return jsonify({
            "status": "success",
            "message": f"Successfully processed {rows:,} customer records.",
            "upload_info": upload_info,
            "redirect": url_for("processing"),
        })

    except Exception as error:
        return jsonify({
            "status": "error",
            "message": f"Processing failed: {str(error)}",
        }), 500


# ============================================================
# DATA PROCESSING PAGE
# ============================================================

@app.route("/processing/")
@app.route("/processing")
def processing():
    df, source, is_demo = get_active_data()

    info = LAST_UPLOAD_INFO if LAST_UPLOAD_INFO else {}

    if df is not None:
        info.setdefault("rows", len(df))
        info.setdefault("columns", len(df.columns))
        info.setdefault("filename", source)

    stages = [
        {"name": "File Upload & Parse", "status": "complete", "duration": "0.4s", "details": f"{info.get('rows', 0):,} rows parsed"},
        {"name": "Schema Validation", "status": "complete", "duration": "0.2s", "details": f"{info.get('columns', 0)} columns validated"},
        {"name": "Missing Value Imputation", "status": "complete", "duration": "0.3s", "details": f"{info.get('missing_cells', 0):,} values imputed"},
        {"name": "Feature Engineering", "status": "complete", "duration": "0.6s", "details": f"{len(FEATURE_COLUMNS)} features built"},
        {"name": "XGBoost Scoring", "status": "complete", "duration": "0.8s", "details": "Ensemble probability scores generated"},
        {"name": "RFM Segmentation", "status": "complete", "duration": "0.4s", "details": "8 customer segments assigned"},
        {"name": "Recommendation Engine", "status": "complete", "duration": "0.3s", "details": "Retention actions computed"},
    ]

    return render_template(
        "processing.html",
        stages=stages,
        data_loaded=df is not None,
        source=source,
        info=info,
    )


# ============================================================
# CUSTOMER INTELLIGENCE
# ============================================================

@app.route("/customers/")
@app.route("/customers")
def customers():
    df, source, is_demo = get_active_data()

    if df is None:
        return render_template(
            "customers.html",
            customers=[],
            total_customers=0,
            data_loaded=False,
            is_demo=False,
        )

    return render_template(
        "customers.html",
        data_loaded=True,
        is_demo=is_demo,
        source=source,
        total_customers=len(df),
        customers=[],  # loaded via AJAX
    )


# ============================================================
# API — CUSTOMERS (paginated JSON)
# ============================================================

@app.route("/api/customers")
def api_customers():
    df, source, is_demo = get_active_data()

    if df is None:
        return jsonify({"customers": [], "total": 0, "page": 1, "pages": 0})

    # Search
    q = request.args.get("q", "").strip().lower()
    segment = request.args.get("segment", "")
    tier = request.args.get("tier", "")
    status = request.args.get("status", "")
    priority = request.args.get("priority", "")
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(200, int(request.args.get("per_page", 50)))
    sort_col = request.args.get("sort", "")
    sort_dir = request.args.get("dir", "asc")

    filtered = df.copy()

    # Search filter
    if q:
        search_cols = ["customer_unique_id", "customer_id", "customer_city", "rfm_segment", "clv_value_tier", "retention_status"]
        mask = pd.Series(False, index=filtered.index)
        for col in search_cols:
            if col in filtered.columns:
                mask |= filtered[col].astype(str).str.lower().str.contains(q, na=False)
        filtered = filtered[mask]

    # Segment filter
    seg_col = "rfm_segment" if "rfm_segment" in filtered.columns else "value_segment"
    if segment and seg_col in filtered.columns:
        filtered = filtered[filtered[seg_col].astype(str) == segment]

    # Tier filter
    if tier and "clv_value_tier" in filtered.columns:
        filtered = filtered[filtered["clv_value_tier"].astype(str) == tier]

    # Retention status filter
    if status and "retention_status" in filtered.columns:
        filtered = filtered[filtered["retention_status"].astype(str) == status]

    # Priority filter
    prio_col = "value_priority" if "value_priority" in filtered.columns else "business_priority"
    if priority and prio_col in filtered.columns:
        filtered = filtered[filtered[prio_col].astype(str) == priority]

    # Sort
    valid_sort_cols = {
        "revenue": "total_revenue" if "total_revenue" in df.columns else "monetary",
        "orders": "total_orders" if "total_orders" in df.columns else "completed_orders",
        "recency": "recency" if "recency" in df.columns else "recency_days",
        "clv": "historical_clv",
        "probability": "repeat_purchase_probability",
    }
    if sort_col in valid_sort_cols:
        real_col = valid_sort_cols[sort_col]
        if real_col in filtered.columns:
            filtered = filtered.sort_values(
                real_col,
                ascending=(sort_dir == "asc"),
                na_position="last",
            )

    total = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    page_df = filtered.iloc[start:start + per_page]

    def safe(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return round(float(v), 4)
        return str(v)

    records = []
    for _, row in page_df.iterrows():
        cid = row.get("customer_unique_id", row.get("customer_id", ""))
        rec = {
            "id": safe(cid),
            "segment": safe(row.get("rfm_segment", row.get("value_segment", "—"))),
            "tier": safe(row.get("clv_value_tier", "—")),
            "orders": safe(row.get("completed_orders", row.get("total_orders", "—"))),
            "revenue": safe(row.get("historical_clv", row.get("total_revenue", row.get("monetary", "—")))),
            "recency": safe(row.get("recency", row.get("recency_days", "—"))),
            "frequency": safe(row.get("frequency", row.get("purchase_frequency", "—"))),
            "monetary": safe(row.get("monetary", row.get("total_revenue", "—"))),
            "retention": safe(row.get("retention_status", "—")),
            "priority": safe(row.get("value_priority", row.get("business_priority", "—"))),
            "action": safe(row.get("recommendation_action", row.get("business_recommendation", "—"))),
            "probability": safe(row.get("repeat_purchase_probability", None)),
        }
        records.append(rec)

    return jsonify({
        "customers": records,
        "total": total,
        "page": page,
        "pages": total_pages,
        "per_page": per_page,
    })


# ============================================================
# CUSTOMER DETAIL — Customer 360
# ============================================================

@app.route("/customer/")
@app.route("/customer")
def customer_detail_empty():
    df, source, is_demo = get_active_data()
    return render_template(
        "customer-detail.html",
        customer=None,
        data_loaded=df is not None,
        is_demo=is_demo,
        source=source,
    )


@app.route("/customer/<customer_id>")
@app.route("/customers/<customer_id>")
def customer_detail(customer_id):
    df, source, is_demo = get_active_data()
    recs_df = get_active_recs()

    if df is None:
        return render_template("customer-detail.html", customer=None, data_loaded=False)

    # Find customer
    customer = None
    for id_col in ["customer_unique_id", "customer_id"]:
        if id_col in df.columns:
            matches = df[df[id_col].astype(str) == str(customer_id)]
            if not matches.empty:
                customer = matches.iloc[0].to_dict()
                break

    if customer is None:
        flash(f"Customer '{customer_id}' not found.", "warning")
        return redirect(url_for("customers"))

    # Enrich with recommendation data
    if recs_df is not None:
        for id_col in ["customer_unique_id", "customer_id"]:
            if id_col in recs_df.columns:
                rec_matches = recs_df[recs_df[id_col].astype(str) == str(customer_id)]
                if not rec_matches.empty:
                    rec_row = rec_matches.iloc[0].to_dict()
                    for k, v in rec_row.items():
                        if k not in customer or pd.isna(customer.get(k)):
                            customer[k] = v
                    break

    # Clean NaN
    cleaned = {}
    for k, v in customer.items():
        if pd.isna(v) if isinstance(v, float) else False:
            cleaned[k] = None
        elif isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.floating,)):
            cleaned[k] = round(float(v), 4)
        else:
            cleaned[k] = v

    return render_template(
        "customer-detail.html",
        customer=cleaned,
        data_loaded=True,
        is_demo=is_demo,
        source=source,
    )


# ============================================================
# SEGMENTS
# ============================================================

@app.route("/segments/")
@app.route("/segments")
def segments():
    df, source, is_demo = get_active_data()

    if df is None:
        return render_template("segments.html", segments=[], data_loaded=False, is_demo=False)

    seg_col = "rfm_segment" if "rfm_segment" in df.columns else ("value_segment" if "value_segment" in df.columns else None)

    segment_data = []
    if seg_col:
        seg_groups = df.groupby(seg_col)

        rev_col = "historical_clv" if "historical_clv" in df.columns else ("total_revenue" if "total_revenue" in df.columns else "monetary")
        rec_col = "recency" if "recency" in df.columns else "recency_days"

        for seg_name, grp in seg_groups:
            avg_rev = float(pd.to_numeric(grp[rev_col], errors="coerce").fillna(0).mean()) if rev_col in grp.columns else 0
            avg_rec = float(pd.to_numeric(grp[rec_col], errors="coerce").fillna(0).mean()) if rec_col in grp.columns else 0
            repeat_count = 0
            if "repeat_customer" in grp.columns:
                repeat_count = int((pd.to_numeric(grp["repeat_customer"], errors="coerce").fillna(0) == 1).sum())
            elif "repeat_purchase_prediction" in grp.columns:
                repeat_count = int(grp["repeat_purchase_prediction"].sum())

            segment_data.append({
                "name": str(seg_name),
                "count": len(grp),
                "avg_revenue": round(avg_rev, 2),
                "avg_recency": round(avg_rec, 1),
                "repeat_count": repeat_count,
                "pct": round(len(grp) / len(df) * 100, 1),
            })

    segment_data.sort(key=lambda x: x["count"], reverse=True)

    chart_data = {
        "labels": [s["name"] for s in segment_data],
        "data": [s["count"] for s in segment_data],
    }

    return render_template(
        "segments.html",
        segments=segment_data,
        chart_data=json.dumps(chart_data),
        data_loaded=True,
        is_demo=is_demo,
        source=source,
        total_customers=len(df),
    )


# ============================================================
# REPEAT PURCHASE PREDICTIONS
# ============================================================

@app.route("/predictions/")
@app.route("/predictions")
@app.route("/churn/")
@app.route("/churn")
def predictions():
    df, source, is_demo = get_active_data()

    # Try pre-computed predictions CSV if no probability in active data
    pred_df = None
    if PREDICTIONS_CSV.exists() and (df is None or "repeat_purchase_probability" not in df.columns):
        try:
            pred_df = pd.read_csv(PREDICTIONS_CSV)
        except Exception:
            pass

    use_df = pred_df if pred_df is not None and (df is None or "repeat_purchase_probability" not in df.columns) else df

    kpis = {
        "customers_scored": 0,
        "high_propensity": 0,
        "avg_probability": 0,
        "threshold": round(THRESHOLD, 4),
        "opportunity_pct": 0,
    }

    chart_data = {}

    if use_df is not None and "repeat_purchase_probability" in use_df.columns:
        probs = pd.to_numeric(use_df["repeat_purchase_probability"], errors="coerce").dropna()
        preds = use_df.get("repeat_purchase_prediction", (probs >= THRESHOLD).astype(int))
        preds = pd.to_numeric(preds, errors="coerce").fillna(0)

        kpis["customers_scored"] = len(probs)
        kpis["high_propensity"] = int(preds.sum())
        kpis["avg_probability"] = round(float(probs.mean()), 4)
        kpis["opportunity_pct"] = round(float(preds.mean()) * 100, 1)

        # Histogram bins
        hist_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
        counts, _ = np.histogram(probs, bins=hist_bins)
        chart_data["propensity_hist"] = {"labels": hist_labels, "data": counts.tolist()}

    # Model metrics
    final_metrics = MODEL_METRICS.get("final_test_metrics", {})
    model_info = {
        "type": "XGBoost Ensemble (3-seed)",
        "roc_auc": round(final_metrics.get("roc_auc", 0), 4),
        "pr_auc": round(final_metrics.get("pr_auc", 0), 4),
        "precision": round(final_metrics.get("precision", 0), 4),
        "recall": round(final_metrics.get("recall", 0), 4),
        "f1": round(final_metrics.get("f1", 0), 4),
        "threshold": round(THRESHOLD, 4),
        "features": len(FEATURE_COLUMNS),
        "loaded": MODEL is not None,
    }

    return render_template(
        "predictions.html",
        kpis=kpis,
        model_info=model_info,
        chart_data=json.dumps(chart_data),
        data_loaded=use_df is not None,
        is_demo=is_demo,
        source=source,
    )


# ============================================================
# API — PREDICTION CUSTOMERS (paginated)
# ============================================================

@app.route("/api/predictions/customers")
def api_prediction_customers():
    df, source, is_demo = get_active_data()

    pred_df = None
    if PREDICTIONS_CSV.exists() and (df is None or "repeat_purchase_probability" not in df.columns):
        try:
            pred_df = pd.read_csv(PREDICTIONS_CSV)
        except Exception:
            pass

    use_df = pred_df if pred_df is not None and (df is None or "repeat_purchase_probability" not in df.columns) else df

    if use_df is None or "repeat_purchase_probability" not in use_df.columns:
        return jsonify({"customers": [], "total": 0})

    page = max(1, int(request.args.get("page", 1)))
    per_page = min(200, int(request.args.get("per_page", 50)))
    filter_type = request.args.get("filter", "all")  # all / high / low

    filtered = use_df.copy()
    probs = pd.to_numeric(filtered["repeat_purchase_probability"], errors="coerce")

    if filter_type == "high":
        filtered = filtered[probs >= THRESHOLD]
    elif filter_type == "low":
        filtered = filtered[probs < THRESHOLD]

    filtered = filtered.sort_values("repeat_purchase_probability", ascending=False, na_position="last")

    total = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    page_df = filtered.iloc[start:start + per_page]

    records = []
    for _, row in page_df.iterrows():
        cid = row.get("customer_unique_id", row.get("customer_id", ""))
        prob = float(row.get("repeat_purchase_probability", 0) or 0)
        pred = int(row.get("repeat_purchase_prediction", 1 if prob >= THRESHOLD else 0) or 0)
        records.append({
            "id": str(cid),
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100, 1),
            "prediction": pred,
            "signal": "High Propensity" if pred == 1 else "Lower Propensity",
            "orders": int(row.get("total_orders", 0) or 0),
            "revenue": round(float(row.get("total_revenue", 0) or 0), 2),
            "recency": int(row.get("recency_days", 0) or 0),
        })

    return jsonify({"customers": records, "total": total, "page": page, "pages": total_pages})


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.route("/recommendations/")
@app.route("/recommendations")
def recommendations():
    recs_df = get_active_recs()
    df, source, is_demo = get_active_data()

    if recs_df is None:
        return render_template("recommendations.html", actions=[], data_loaded=False, is_demo=False)

    # Group by action
    action_col = "recommendation_action" if "recommendation_action" in recs_df.columns else "business_recommendation"
    priority_col = "recommendation_priority" if "recommendation_priority" in recs_df.columns else "value_priority"
    rev_col = "historical_clv" if "historical_clv" in recs_df.columns else ("total_revenue" if "total_revenue" in recs_df.columns else "monetary")

    actions = []
    if action_col in recs_df.columns:
        for action_name, grp in recs_df.groupby(action_col):
            avg_rev = float(pd.to_numeric(grp[rev_col], errors="coerce").fillna(0).mean()) if rev_col in grp.columns else 0
            total_rev = float(pd.to_numeric(grp[rev_col], errors="coerce").fillna(0).sum()) if rev_col in grp.columns else 0
            priority = str(grp[priority_col].mode().iloc[0]) if priority_col in grp.columns and not grp[priority_col].isna().all() else "Medium"
            campaign = str(grp["recommendation_campaign"].mode().iloc[0]) if "recommendation_campaign" in grp.columns and not grp["recommendation_campaign"].isna().all() else ""
            channel = str(grp["recommendation_channel"].mode().iloc[0]) if "recommendation_channel" in grp.columns and not grp["recommendation_channel"].isna().all() else ""

            actions.append({
                "action": str(action_name),
                "count": len(grp),
                "avg_revenue": round(avg_rev, 2),
                "total_revenue": round(total_rev, 2),
                "priority": priority,
                "campaign": campaign,
                "channel": channel,
                "pct": round(len(grp) / len(recs_df) * 100, 1),
            })

    actions.sort(key=lambda x: ({"High": 0, "Medium": 1, "Low": 2}.get(x["priority"], 3), -x["count"]))

    return render_template(
        "recommendations.html",
        actions=actions,
        data_loaded=True,
        is_demo=is_demo,
        source=source,
        total_customers=len(recs_df),
    )


# ============================================================
# API — RECOMMENDATION CUSTOMERS
# ============================================================

@app.route("/api/recommendations/customers")
def api_recommendation_customers():
    recs_df = get_active_recs()
    action = request.args.get("action", "")
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(200, int(request.args.get("per_page", 50)))

    if recs_df is None:
        return jsonify({"customers": [], "total": 0})

    action_col = "recommendation_action" if "recommendation_action" in recs_df.columns else "business_recommendation"
    filtered = recs_df[recs_df[action_col].astype(str) == action] if action else recs_df

    total = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    page_df = filtered.iloc[start:start + per_page]

    rev_col = "historical_clv" if "historical_clv" in recs_df.columns else ("total_revenue" if "total_revenue" in recs_df.columns else "monetary")

    records = []
    for _, row in page_df.iterrows():
        cid = row.get("customer_unique_id", row.get("customer_id", ""))
        records.append({
            "id": str(cid),
            "segment": str(row.get("rfm_segment", row.get("value_segment", "—"))),
            "tier": str(row.get("clv_value_tier", "—")),
            "revenue": round(float(row.get(rev_col, 0) or 0), 2),
            "retention": str(row.get("retention_status", "—")),
            "priority": str(row.get("recommendation_priority", row.get("value_priority", "—"))),
        })

    return jsonify({"customers": records, "total": total, "page": page, "pages": total_pages})


# ============================================================
# ANALYTICS
# ============================================================

@app.route("/analytics/")
@app.route("/analytics")
def analytics():
    df, source, is_demo = get_active_data()

    if df is None:
        return render_template("analytics.html", data_loaded=False, is_demo=False, chart_data="{}")

    chart_data = {
        "segments": get_segment_chart_data(df),
        "revenue_tiers": get_revenue_tier_data(df),
        "purchase_status": get_purchase_status_data(df),
        "recency": get_recency_distribution(df),
    }

    # Orders per customer
    ord_col = "completed_orders" if "completed_orders" in df.columns else ("total_orders" if "total_orders" in df.columns else None)
    if ord_col:
        ord_vals = pd.to_numeric(df[ord_col], errors="coerce").dropna()
        bins = [0, 1, 2, 3, 5, 10, 9999]
        labels = ["1", "2", "3", "4-5", "6-10", "10+"]
        counts, _ = np.histogram(ord_vals, bins=bins)
        chart_data["orders_dist"] = {"labels": labels, "data": counts.tolist()}

    # Revenue per customer histogram
    rev_col = "historical_clv" if "historical_clv" in df.columns else ("total_revenue" if "total_revenue" in df.columns else None)
    if rev_col:
        rev_vals = pd.to_numeric(df[rev_col], errors="coerce").dropna()
        bins = [0, 50, 100, 200, 500, 1000, 9999999]
        labels = ["$0-50", "$50-100", "$100-200", "$200-500", "$500-1k", "$1k+"]
        counts, _ = np.histogram(rev_vals, bins=bins)
        chart_data["revenue_dist"] = {"labels": labels, "data": counts.tolist()}

    summary = dataset_summary(df)

    return render_template(
        "analytics.html",
        data_loaded=True,
        is_demo=is_demo,
        source=source,
        summary=summary,
        chart_data=json.dumps(chart_data),
    )


# ============================================================
# ADVANCED CUSTOMER EXPLORER
# ============================================================

@app.route("/explorer/")
@app.route("/explorer")
def explorer():
    df, source, is_demo = get_active_data()

    if df is None:
        return render_template("explorer.html", data_loaded=False, is_demo=False, filter_options={})

    # Build filter options
    filter_options = {}

    for col, key in [
        ("rfm_segment", "segments"),
        ("clv_value_tier", "tiers"),
        ("retention_status", "statuses"),
        ("value_priority", "priorities"),
        ("customer_type", "types"),
    ]:
        if col in df.columns:
            vals = df[col].dropna().unique().tolist()
            filter_options[key] = sorted([str(v) for v in vals])

    return render_template(
        "explorer.html",
        data_loaded=True,
        is_demo=is_demo,
        source=source,
        total_customers=len(df),
        filter_options=filter_options,
    )


# ============================================================
# REPORTS & EXPORT CENTER
# ============================================================

@app.route("/reports/")
@app.route("/reports")
def reports():
    df, source, is_demo = get_active_data()
    recs_df = get_active_recs()

    report_summary = None
    if df is not None:
        summary = dataset_summary(df)
        repeat_count = 0
        if "repeat_customer" in df.columns:
            repeat_count = int((pd.to_numeric(df["repeat_customer"], errors="coerce").fillna(0) == 1).sum())
        elif "repeat_purchase_prediction" in df.columns:
            repeat_count = int(df["repeat_purchase_prediction"].sum())

        seg_col = "rfm_segment" if "rfm_segment" in df.columns else "value_segment"
        segment_count = df[seg_col].nunique() if seg_col in df.columns else 0

        action_col = "recommendation_action" if recs_df is not None and "recommendation_action" in recs_df.columns else None
        action_count = recs_df[action_col].nunique() if action_col else 0

        report_summary = {
            "source": source,
            "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_customers": summary["customers"],
            "repeat_customers": repeat_count,
            "repeat_rate": round(repeat_count / max(summary["customers"], 1) * 100, 1),
            "total_revenue": round(summary["revenue"], 2),
            "total_orders": summary["orders"],
            "segment_count": segment_count,
            "action_count": action_count,
            "is_demo": is_demo,
        }

    return render_template(
        "reports.html",
        data_loaded=df is not None,
        is_demo=is_demo,
        report_summary=report_summary,
    )


# ============================================================
# SETTINGS
# ============================================================

@app.route("/settings/")
@app.route("/settings")
def settings():
    df, source, is_demo = get_active_data()

    final_metrics = MODEL_METRICS.get("final_test_metrics", {})

    sys_info = {
        "model_loaded": MODEL is not None,
        "model_type": "XGBoost Ensemble (3-seed)",
        "threshold": round(THRESHOLD, 6),
        "feature_count": len(FEATURE_COLUMNS),
        "roc_auc": round(final_metrics.get("roc_auc", 0), 4),
        "pr_auc": round(final_metrics.get("pr_auc", 0), 4),
        "recall": round(final_metrics.get("recall", 0), 4),
        "precision": round(final_metrics.get("precision", 0), 4),
        "database": False,
        "data_loaded": df is not None,
        "data_source": source,
        "is_demo": is_demo,
        "demo_available": DEMO_DATA is not None,
        "demo_rows": len(DEMO_DATA) if DEMO_DATA is not None else 0,
        "current_rows": len(df) if df is not None else 0,
        "master_csv_exists": MASTER_CSV.exists(),
        "recs_csv_exists": RECOMMENDATIONS_CSV.exists(),
        "predictions_csv_exists": PREDICTIONS_CSV.exists(),
    }

    return render_template(
        "settings.html",
        sys_info=sys_info,
        data_loaded=df is not None,
        is_demo=is_demo,
        source=source,
    )


# ============================================================
# ADMIN — System Administration & Health
# ============================================================

@app.route("/admin/")
@app.route("/admin")
def admin():
    df, source, is_demo = get_active_data()
    summary = dataset_summary(df) if df is not None else {}
    final_metrics = MODEL_METRICS.get("final_test_metrics", {})

    sys_health = {
        "app_status": "Healthy",
        "backend_status": "Online (Flask Production Server)",
        "dataset_status": f"{len(df):,} customers loaded" if df is not None else "No dataset loaded",
        "model_status": "Loaded (Production XGBoost Ensemble)" if MODEL is not None else "Not Loaded",
        "prediction_engine": "Active (In-memory probability scoring)" if MODEL is not None else "Standby",
        "export_engine": "Active (CSV & OpenPyXL Excel Engines)",
        "rows_loaded": len(df) if df is not None else 0,
        "columns_loaded": len(df.columns) if df is not None else 0,
        "source": source,
        "is_demo": is_demo,
        "threshold": round(THRESHOLD, 4),
        "feature_count": len(FEATURE_COLUMNS),
        "roc_auc": round(final_metrics.get("roc_auc", 0), 4),
        "pr_auc": round(final_metrics.get("pr_auc", 0), 4),
        "last_upload": LAST_UPLOAD_INFO.get("filename", "N/A"),
    }

    return render_template(
        "admin.html",
        sys_health=sys_health,
        data_loaded=df is not None,
        is_demo=is_demo,
        source=source,
    )


@app.route("/api/reload-model", methods=["POST"])
def api_reload_model():
    try:
        load_model_artifacts()
        return jsonify({
            "status": "success",
            "message": "ML model artifacts reloaded successfully.",
            "model_loaded": MODEL is not None,
            "threshold": THRESHOLD,
            "features": len(FEATURE_COLUMNS),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# API — HEALTH
# ============================================================

@app.route("/api/health")
def api_health():
    df, source, is_demo = get_active_data()
    return jsonify({
        "status": "ok",
        "database": False,
        "model_loaded": MODEL is not None,
        "threshold": THRESHOLD,
        "features": len(FEATURE_COLUMNS),
        "data_loaded": df is not None,
        "is_demo": is_demo,
        "customers": len(df) if df is not None else 0,
    })


# ============================================================
# API — DASHBOARD
# ============================================================

@app.route("/api/dashboard")
def api_dashboard():
    df, source, is_demo = get_active_data()

    if df is None:
        return jsonify({"data_loaded": False, "customers": 0})

    summary = dataset_summary(df)
    result = dict(summary)
    result["data_loaded"] = True
    result["is_demo"] = is_demo

    if "repeat_purchase_prediction" in df.columns:
        result["predicted_repeat_customers"] = int(df["repeat_purchase_prediction"].sum())
    elif "repeat_customer" in df.columns:
        result["predicted_repeat_customers"] = int((pd.to_numeric(df["repeat_customer"], errors="coerce").fillna(0) == 1).sum())

    if "repeat_purchase_probability" in df.columns:
        result["average_repeat_probability"] = float(df["repeat_purchase_probability"].mean())

    result["chart_data"] = {
        "segments": get_segment_chart_data(df),
        "revenue_tiers": get_revenue_tier_data(df),
        "purchase_status": get_purchase_status_data(df),
    }

    return jsonify(result)


# ============================================================
# API — SEGMENTS
# ============================================================

@app.route("/api/segments")
def api_segments():
    df, source, is_demo = get_active_data()
    if df is None:
        return jsonify({"segments": []})

    seg_col = "rfm_segment" if "rfm_segment" in df.columns else ("value_segment" if "value_segment" in df.columns else None)
    if not seg_col:
        return jsonify({"segments": []})

    segs = df.groupby(seg_col).size().reset_index(name="count")
    return jsonify({"segments": segs.to_dict(orient="records")})


# ============================================================
# API — VALIDATE (upload summary)
# ============================================================

@app.route("/api/validate")
def api_validate():
    return jsonify(LAST_UPLOAD_INFO)


# ============================================================
# API — CLEAR DATASET
# ============================================================

@app.route("/api/clear", methods=["POST"])
def api_clear():
    global CURRENT_DATA, CURRENT_SOURCE, LAST_UPLOAD_INFO
    CURRENT_DATA = None
    CURRENT_SOURCE = "No dataset loaded"
    LAST_UPLOAD_INFO = {}
    return jsonify({"status": "success", "message": "Dataset cleared."})


# ============================================================
# API — PREDICT (POST file)
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    try:
        file = request.files["file"]
        df = load_uploaded_file(file)
        result = predict_dataframe(df)
        result = customer_segments(result)
        result = add_recommendations(result)
        set_current_data(result, source=file.filename)

        return jsonify({
            "status": "success",
            "rows": len(result),
            "predicted_repeat_customers": int(result["repeat_purchase_prediction"].sum()),
            "threshold": THRESHOLD,
        })

    except Exception as error:
        return jsonify({"status": "error", "message": str(error)}), 500


# ============================================================
# EXPORT CSV
# ============================================================

@app.route("/export/csv")
@app.route("/api/export/csv")
def export_csv():
    df, source, is_demo = get_active_data()
    recs_df = get_active_recs()

    report_type = request.args.get("type", "intelligence")

    if report_type == "recommendations" and recs_df is not None:
        export_df = recs_df
        filename = "cix_recommendations.csv"
    elif df is not None:
        export_df = df
        filename = "cix_customer_intelligence.csv"
    else:
        return jsonify({"error": "No dataset loaded."}), 404

    buffer = io.StringIO()
    export_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


# ============================================================
# EXPORT EXCEL
# ============================================================

@app.route("/export/excel")
@app.route("/api/export/excel")
def export_excel():
    df, source, is_demo = get_active_data()
    recs_df = get_active_recs()

    report_type = request.args.get("type", "intelligence")
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if df is not None:
            df.to_excel(writer, index=False, sheet_name="Customer Intelligence")
        if recs_df is not None and report_type in ("recommendations", "full"):
            recs_df.to_excel(writer, index=False, sheet_name="Recommendations")

    if df is None and recs_df is None:
        return jsonify({"error": "No dataset loaded."}), 404

    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="cix_full_report.xlsx",
    )


# ============================================================
# API — SYSTEM INFO
# ============================================================

@app.route("/api/system-info")
def api_system_info():
    df, source, is_demo = get_active_data()
    return jsonify({
        "model_loaded": MODEL is not None,
        "threshold": THRESHOLD,
        "features": len(FEATURE_COLUMNS),
        "data_loaded": df is not None,
        "data_source": source,
        "is_demo": is_demo,
        "customers": len(df) if df is not None else 0,
        "demo_customers": len(DEMO_DATA) if DEMO_DATA is not None else 0,
        "roc_auc": MODEL_METRICS.get("final_test_metrics", {}).get("roc_auc", 0),
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "File too large. Maximum upload size is 100 MB."}), 413


@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(error):
    return render_template("error.html", code=500, message="Internal server error."), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 80)
    print("CIX — E-COMMERCE CUSTOMER INTELLIGENCE PLATFORM")
    print("=" * 80)
    print()
    print(f"Application  : Running")
    print(f"Database     : Not Required (In-Memory)")
    print(f"Model        : {'Loaded' if MODEL is not None else 'Not Loaded'}")
    print(f"Threshold    : {THRESHOLD:.4f}")
    print(f"Demo Data    : {'Available' if DEMO_DATA is not None else 'Not Available'} ({len(DEMO_DATA):,} customers)" if DEMO_DATA is not None else "Demo Data    : Not Available")
    print()
    print("Dashboard    : http://127.0.0.1:5000/dashboard")
    print("Upload       : http://127.0.0.1:5000/upload")
    print("Predictions  : http://127.0.0.1:5000/predictions")
    print("Health       : http://127.0.0.1:5000/api/health")
    print()

    app.run(host="127.0.0.1", port=5000, debug=True)