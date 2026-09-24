from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

PIPELINE_STEPS = [
    {
        "name": "Data Validation",
        "script": ROOT / "src" / "data" / "validator.py"
    },
    {
        "name": "Data Standardization",
        "script": ROOT / "src" / "data" / "processor.py"
    }
]


# ============================================================
# HELPERS
# ============================================================

def run_step(name, script):

    print()
    print("=" * 80)
    print(name.upper())
    print("=" * 80)
    print()

    if not script.exists():

        return {
            "name": name,
            "status": "SKIPPED",
            "message": f"Script not found: {script}"
        }

    started = datetime.now()

    try:

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True
        )

        elapsed = (
            datetime.now() - started
        ).total_seconds()

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:

            return {
                "name": name,
                "status": "FAILED",
                "return_code": result.returncode,
                "duration_seconds": round(elapsed, 2),
                "message": "Pipeline step failed."
            }

        return {
            "name": name,
            "status": "SUCCESS",
            "return_code": 0,
            "duration_seconds": round(elapsed, 2)
        }

    except Exception as exc:

        return {
            "name": name,
            "status": "FAILED",
            "message": str(exc)
        }


# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_pipeline():

    print()
    print("=" * 80)
    print("E-COMMERCE CUSTOMER INTELLIGENCE")
    print("AUTOMATED DATA PIPELINE")
    print("=" * 80)

    started_at = datetime.now()

    results = []

    for step in PIPELINE_STEPS:

        result = run_step(
            step["name"],
            step["script"]
        )

        results.append(result)

        if result["status"] == "FAILED":

            print()
            print("PIPELINE STOPPED")
            print(f"Failed step: {step['name']}")

            break

    completed_at = datetime.now()

    success_count = sum(
        1 for result in results
        if result["status"] == "SUCCESS"
    )

    failed_count = sum(
        1 for result in results
        if result["status"] == "FAILED"
    )

    skipped_count = sum(
        1 for result in results
        if result["status"] == "SKIPPED"
    )

    overall_status = (
        "SUCCESS"
        if failed_count == 0
        else "FAILED"
    )

    report = {
        "pipeline": "E-Commerce Customer Intelligence Pipeline",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "status": overall_status,
        "successful_steps": success_count,
        "failed_steps": failed_count,
        "skipped_steps": skipped_count,
        "steps": results
    }

    report_path = REPORT_DIR / "pipeline_run.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    print(f"Status           : {overall_status}")
    print(f"Successful steps : {success_count}")
    print(f"Failed steps     : {failed_count}")
    print(f"Skipped steps    : {skipped_count}")
    print()
    print(f"Report           : {report_path}")
    print()


if __name__ == "__main__":
    run_pipeline()
