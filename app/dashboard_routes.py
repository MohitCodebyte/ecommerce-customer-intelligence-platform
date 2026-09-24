from pathlib import Path
import json

from flask import Blueprint, render_template


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "reports"

dashboard_bp = Blueprint(
    "dashboard",
    __name__,
    url_prefix="/dashboard"
)


@dashboard_bp.route("/")
def dashboard():

    dashboard_file = (
        REPORT_DIR /
        "dashboard_data.json"
    )

    if not dashboard_file.exists():

        return render_template(
            "error.html",
            title="Dashboard Data Unavailable",
            message=(
                "Dashboard data has not been generated yet. "
                "Run the dashboard data service first."
            )
        ), 503


    try:

        with open(
            dashboard_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except Exception as exc:

        return render_template(
            "error.html",
            title="Dashboard Error",
            message=str(exc)
        ), 500


    return render_template(
        "dashboard.html",
        dashboard=data
    )
