from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent_loop import run_agent_loop, run_agent_loop_mock


# =========================================================
# Optional imports
# =========================================================

try:
    from src.agent_memory import record_run
except Exception:
    record_run = None

try:
    from src.config import OUTPUTS_DIR
except Exception:
    OUTPUTS_DIR = Path("outputs")

try:
    from src.config import TABLES_DIR
except Exception:
    TABLES_DIR = Path("tables")

try:
    from src.bundle_results import run_bundle_results_pipeline
except Exception:
    run_bundle_results_pipeline = None

try:
    from src.cloud_s3 import upload_bundle_and_get_link
except Exception:
    upload_bundle_and_get_link = None

try:
    from src.email_results import send_email_with_s3_link
except Exception:
    send_email_with_s3_link = None


# =========================================================
# Helpers
# =========================================================

def ensure_output_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_backtest_metrics() -> dict[str, Any]:
    """
    Load backtest metrics from the summary CSV if available.
    Expected format:
        metric,value
        Strategy Sharpe Ratio,1.23
        Total Strategy Return,0.18
        ...
    """
    candidate_paths = [
        TABLES_DIR / "backtest_summary.csv",
        OUTPUTS_DIR / "backtest_summary.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if {"metric", "value"}.issubset(df.columns):
                    return dict(zip(df["metric"].astype(str), df["value"]))
            except Exception as exc:
                print(f"Warning: failed to read backtest summary from {path}: {exc}")

    return {}


def save_agent_result(result: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"agent_run_result_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, default=str)

    return output_path


def try_record_memory(recommendation: str, metrics: dict[str, Any]) -> None:
    if record_run is None:
        print("Memory module not available. Skipping memory recording.")
        return

    try:
        record_run(recommendation, metrics)
        print("Memory recorded successfully.")
    except Exception as exc:
        print(f"Warning: failed to record memory: {exc}")


def try_bundle_and_deliver() -> None:
    """
    Optional delivery stage.
    Runs only if the relevant modules exist.
    """
    if run_bundle_results_pipeline is None:
        print("Bundle module not available. Skipping result bundling.")
        return

    try:
        run_bundle_results_pipeline()
        print("Results bundled successfully.")
    except Exception as exc:
        print(f"Warning: bundle stage failed: {exc}")
        return

    if upload_bundle_and_get_link is None:
        print("S3 upload module not available. Skipping upload.")
        return

    try:
        s3_uri, download_url = upload_bundle_and_get_link()
        print(f"Bundle uploaded successfully: {s3_uri}")
    except Exception as exc:
        print(f"Warning: upload stage failed: {exc}")
        return

    if send_email_with_s3_link is None:
        print("Email module not available. Skipping email delivery.")
        return

    try:
        send_email_with_s3_link(download_url=download_url, s3_uri=s3_uri)
        print("Email sent successfully.")
    except Exception as exc:
        print(f"Warning: email stage failed: {exc}")


# =========================================================
# Main runner
# =========================================================

def main() -> dict[str, Any]:
    ensure_output_dirs()

    print("Agentic AI Crypto Trading System")
    print("=" * 50)

    # 1. Run the agent loop
    try:
        result = run_agent_loop(verbose=True)
        # result = run_agent_loop_mock(verbose=True)
    except Exception as exc:
        error_result = {
            "status": "error",
            "recommendation": "Agent execution failed.",
            "confidence": "Low",
            "rationale": str(exc),
            "executed_tools": [],
            "steps": [],
        }

        error_path = save_agent_result(error_result)
        print(f"\nAgent failed. Error result saved to: {error_path}")
        return error_result

    # 2. Save final agent output
    result_path = save_agent_result(result)
    print(f"\nAgent result saved to: {result_path}")

    # 3. Load backtest metrics if available
    metrics = load_backtest_metrics()
    if metrics:
        print("Backtest metrics loaded.")
    else:
        print("No backtest metrics found.")

    # 4. Record memory if module exists
    recommendation = result.get("recommendation", "")
    try_record_memory(recommendation, metrics)

    # 5. Optional bundle/upload/email delivery
    try_bundle_and_deliver()

    print("\nRun complete.")
    return result


if __name__ == "__main__":
    final_result = main()
    print("\nFinal recommendation:")
    print(final_result.get("recommendation", "No recommendation generated."))