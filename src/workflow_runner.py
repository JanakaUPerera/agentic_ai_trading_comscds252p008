from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from main import run_full_pipeline

OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_workflow(email_recipient: str) -> tuple[dict, str]:
    log_buffer = io.StringIO()

    with redirect_stdout(log_buffer):
        result = run_full_pipeline(email_recipient=email_recipient)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = OUTPUT_DIR / f"agent_run_result_{timestamp}.json"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    result.setdefault("output_files", [])
    result["output_files"].append(str(result_path))

    return result, log_buffer.getvalue()


def get_latest_result_files() -> list[str]:
    files = sorted(OUTPUT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:10]]