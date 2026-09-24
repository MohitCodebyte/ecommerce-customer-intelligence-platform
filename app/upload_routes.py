from pathlib import Path
import json
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file

ROOT = Path(__file__).resolve().parents[2]

UPLOAD_DIR = ROOT / "data" / "uploads"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

upload_bp = Blueprint(
    "upload",
    __name__,
    url_prefix="/upload"
)


ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def load_dataframe(path):

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError("Unsupported file format.")


@upload_bp.route("/", methods=["GET", "POST"])
def upload():

    if request.method == "GET":
        return render_template("upload.html")

    uploaded_file = request.files.get("file")

    if not uploaded_file or not uploaded_file.filename:
        flash("Please select a CSV or Excel file.", "error")
        return redirect(url_for("upload.upload"))

    if not allowed_file(uploaded_file.filename):
        flash("Only CSV, XLSX and XLS files are supported.", "error")
        return redirect(url_for("upload.upload"))

    # Remove old uploads
    for old_file in UPLOAD_DIR.iterdir():
        if old_file.is_file():
            old_file.unlink()

    filename = Path(uploaded_file.filename).name
    save_path = UPLOAD_DIR / filename

    uploaded_file.save(save_path)

    try:

        df = load_dataframe(save_path)

        rows = len(df)
        columns = len(df.columns)

        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())

        report = {
            "file_name": filename,
            "rows": rows,
            "columns": columns,
            "missing_cells": missing,
            "duplicate_rows": duplicates,
            "column_names": list(df.columns),
            "status": "uploaded"
        }

        with open(
            REPORT_DIR / "latest_upload.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False
            )

        flash(
            f"File uploaded successfully: {filename}",
            "success"
        )

        return render_template(
            "upload.html",
            report=report
        )

    except Exception as e:

        flash(
            f"Unable to read uploaded file: {str(e)}",
            "error"
        )

        return redirect(url_for("upload.upload"))


@upload_bp.route("/download-template")
def download_template():

    template = (
        ROOT
        / "data"
        / "templates"
        / "company_data_template.xlsx"
    )

    if not template.exists():
        flash("Template file does not exist.", "error")
        return redirect(url_for("upload.upload"))

    return send_file(
        template,
        as_attachment=True,
        download_name="company_data_template.xlsx"
    )
