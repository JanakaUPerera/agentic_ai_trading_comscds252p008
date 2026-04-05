from __future__ import annotations

import json
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from main import run_full_pipeline

OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class StreamlitLogWriter:
    def __init__(self, placeholder, max_chars: int = 30000, min_refresh_seconds: float = 0.2):
        self.placeholder = placeholder
        self.max_chars = max_chars
        self.min_refresh_seconds = min_refresh_seconds
        self.buffer = ""
        self._last_refresh = 0.0

    def write(self, text: str) -> int:
        if not text:
            return 0

        self.buffer += text

        if len(self.buffer) > self.max_chars:
            self.buffer = self.buffer[-self.max_chars:]

        now = time.time()
        if now - self._last_refresh >= self.min_refresh_seconds or text.endswith("\n"):
            self.placeholder.code(self.buffer, language="text")
            self._last_refresh = now

        return len(text)

    def flush(self) -> None:
        self.placeholder.code(self.buffer, language="text")


def run_workflow(email_recipient: str, log_placeholder=None) -> tuple[dict, str]:
    if log_placeholder is not None:
        log_writer = StreamlitLogWriter(log_placeholder)
        with redirect_stdout(log_writer):
            result = run_full_pipeline(email_recipient=email_recipient)
        logs = log_writer.buffer
    else:
        import io

        log_buffer = io.StringIO()
        with redirect_stdout(log_buffer):
            result = run_full_pipeline(email_recipient=email_recipient)
        logs = log_buffer.getvalue()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = OUTPUT_DIR / f"agent_run_result_{timestamp}.json"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    result.setdefault("output_files", [])
    result["output_files"].append(str(result_path))

    return result, logs


def get_latest_result_files() -> list[str]:
    files = sorted(OUTPUT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:10]]