from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


# =========================================================
# Config
# =========================================================

try:
    from src.config import OUTPUTS_DIR
except Exception:
    OUTPUTS_DIR = Path("outputs")

MEMORY_FILE = OUTPUTS_DIR / "agent_memory.json"
MAX_RUN_HISTORY = 30


# =========================================================
# Helpers
# =========================================================

def _ensure_output_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "").replace(",", "")
            if cleaned == "":
                return None
            return float(cleaned)
        return float(value)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": []}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return {"runs": []}
        if "runs" not in data or not isinstance(data["runs"], list):
            data["runs"] = []
        return data
    except Exception:
        return {"runs": []}


def _save_json(path: Path, data: dict[str, Any]) -> None:
    _ensure_output_dir()
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def _extract_metric(
    metrics: dict[str, Any],
    possible_keys: list[str],
) -> Any:
    for key in possible_keys:
        if key in metrics:
            return metrics[key]
    return None


# =========================================================
# Public API
# =========================================================

def load_memory() -> dict[str, Any]:
    return _load_json(MEMORY_FILE)


def save_memory(memory: dict[str, Any]) -> None:
    _save_json(MEMORY_FILE, memory)


def record_run(recommendation: str, metrics: dict[str, Any]) -> None:
    """
    Store one completed agent run.

    Expected metrics examples:
    - Strategy Sharpe Ratio
    - Total Strategy Return
    - Strategy Max Drawdown
    - Win Rate
    - Trade Count
    """
    memory = load_memory()

    sharpe = _extract_metric(
        metrics,
        ["Strategy Sharpe Ratio", "Sharpe Ratio", "Sharpe"],
    )
    total_return = _extract_metric(
        metrics,
        ["Total Strategy Return", "Strategy Total Return", "Total Return"],
    )
    drawdown = _extract_metric(
        metrics,
        ["Strategy Max Drawdown", "Max Drawdown"],
    )
    win_rate = _extract_metric(
        metrics,
        ["Win Rate"],
    )
    trade_count = _extract_metric(
        metrics,
        ["Trade Count", "Number of Trades", "Completed Trades"],
    )

    run_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "date": str(date.today()),
        "recommendation": recommendation,
        "metrics": {
            "sharpe": sharpe,
            "total_return": total_return,
            "drawdown": drawdown,
            "win_rate": win_rate,
            "trade_count": trade_count,
        },
    }

    memory["runs"].append(run_record)
    memory["runs"] = memory["runs"][-MAX_RUN_HISTORY:]

    save_memory(memory)


def get_recent_runs(limit: int = 5) -> list[dict[str, Any]]:
    memory = load_memory()
    runs = memory.get("runs", [])
    return runs[-limit:] if runs else []


def get_memory_context() -> str:
    """
    Return a short text summary that can be injected into the agent prompt.
    """
    recent_runs = get_recent_runs(limit=5)

    if not recent_runs:
        return "No previous runs recorded."

    last_run = recent_runs[-1]

    sharpe_values = []
    return_values = []
    drawdown_values = []

    for run in recent_runs:
        metrics = run.get("metrics", {})
        sharpe = _safe_float(metrics.get("sharpe"))
        total_return = _safe_float(metrics.get("total_return"))
        drawdown = _safe_float(metrics.get("drawdown"))

        if sharpe is not None:
            sharpe_values.append(sharpe)
        if total_return is not None:
            return_values.append(total_return)
        if drawdown is not None:
            drawdown_values.append(drawdown)

    avg_sharpe = sum(sharpe_values) / len(sharpe_values) if sharpe_values else None
    avg_return = sum(return_values) / len(return_values) if return_values else None
    avg_drawdown = sum(drawdown_values) / len(drawdown_values) if drawdown_values else None

    last_metrics = last_run.get("metrics", {})
    last_recommendation = str(last_run.get("recommendation", "")).strip()

    if len(last_recommendation) > 220:
        last_recommendation = last_recommendation[:220].rstrip() + "..."

    parts = [
        f"Previous run count available: {len(recent_runs)}",
        f"Last run date: {last_run.get('date', 'Unknown')}",
        f"Last recommendation: {last_recommendation or 'No recommendation stored.'}",
        (
            f"Last run metrics: Sharpe={last_metrics.get('sharpe')}, "
            f"Return={last_metrics.get('total_return')}, "
            f"Drawdown={last_metrics.get('drawdown')}, "
            f"Win Rate={last_metrics.get('win_rate')}, "
            f"Trades={last_metrics.get('trade_count')}"
        ),
    ]

    if avg_sharpe is not None:
        parts.append(f"Average Sharpe over recent runs: {avg_sharpe:.4f}")
    if avg_return is not None:
        parts.append(f"Average total return over recent runs: {avg_return:.4f}")
    if avg_drawdown is not None:
        parts.append(f"Average drawdown over recent runs: {avg_drawdown:.4f}")

    if avg_sharpe is not None:
        if avg_sharpe < 0:
            parts.append(
                "Recent performance has been weak on a risk-adjusted basis. Be conservative and investigate before giving strong buy recommendations."
            )
        elif avg_sharpe < 0.5:
            parts.append(
                "Recent risk-adjusted performance has been modest. Use medium confidence unless the latest analysis is clearly strong."
            )
        else:
            parts.append(
                "Recent risk-adjusted performance has been acceptable. Use current analysis to confirm whether confidence should remain high."
            )

    return "\n".join(parts)


# =========================================================
# Optional CLI test
# =========================================================

if __name__ == "__main__":
    print(get_memory_context())