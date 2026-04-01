from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


# =========================================================
# Registry
# =========================================================

@dataclass
class AgentTool:
    name: str
    description: str
    fn: Callable[[], dict[str, Any]]
    returns: str


TOOL_REGISTRY: dict[str, AgentTool] = {}


def register(name: str, description: str, returns: str):
    def decorator(fn: Callable[[], dict[str, Any]]):
        TOOL_REGISTRY[name] = AgentTool(
            name=name,
            description=description,
            fn=fn,
            returns=returns,
        )
        return fn
    return decorator


# =========================================================
# Imports from your existing project
# Adjust these import names only if your actual functions differ
# =========================================================

from src.fetch_data import fetch_crypto_data
from src.preprocess_data import preprocess_crypto_data
from src.eda import run_eda_pipeline
from src.features import run_feature_engineering_pipeline
from src.clustering import run_clustering_pipeline
from src.retrieve_news_info import run_news_info_retrieval_pipeline
from src.decision_engine import run_decision_engine_pipeline
from src.manage_risk import run_risk_management_pipeline
from src.backtest import run_backtesting_pipeline

try:
    from src.config import CRYPTO_ASSETS
except Exception:
    CRYPTO_ASSETS = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "TRX-USD", "AVAX-USD", "LINK-USD"
    ]

try:
    from src.config import OUTPUTS_DIR
except Exception:
    OUTPUTS_DIR = Path("outputs")

try:
    from src.config import TABLES_DIR
except Exception:
    TABLES_DIR = Path("tables")

try:
    from src.config import DATA_DIR
except Exception:
    DATA_DIR = Path("data")

try:
    from src.config import PROCESSED_DATA_DIR
except Exception:
    PROCESSED_DATA_DIR = DATA_DIR / "processed"


# =========================================================
# Helpers
# =========================================================

def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _date_range_from_df(df: pd.DataFrame) -> tuple[Any, Any]:
    if df is None or df.empty:
        return None, None

    candidate_cols = ["date", "Date", "datetime", "timestamp"]
    for col in candidate_cols:
        if col in df.columns:
            series = pd.to_datetime(df[col], errors="coerce").dropna()
            if not series.empty:
                return series.min().date(), series.max().date()

    return None, None


def _ticker_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0

    for col in ["ticker", "asset", "symbol"]:
        if col in df.columns:
            return int(df[col].nunique())

    return 0


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df is None or df.empty or column not in df.columns:
        return {}
    counts = df[column].astype(str).value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _numeric_summary(df: pd.DataFrame, column: str) -> dict[str, float]:
    if df is None or df.empty or column not in df.columns:
        return {}
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {}
    return {
        "min": float(series.min()),
        "mean": float(series.mean()),
        "max": float(series.max()),
    }


def _tool_result(
    tool: str,
    status: str,
    summary: str,
    details: dict[str, Any] | None = None,
    output_files: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tool": tool,
        "status": status,
        "summary": summary,
        "details": details or {},
        "output_files": output_files or [],
    }


# =========================================================
# Tool wrappers
# =========================================================

@register(
    name="fetch_market_data",
    description=(
        "Download fresh OHLCV market data for the tracked crypto assets. "
        "Use this first when data is missing or stale."
    ),
    returns="Row count, asset count, and date range of fetched market data.",
)
def tool_fetch_market_data() -> dict[str, Any]:
    df = fetch_crypto_data(CRYPTO_ASSETS)

    if not isinstance(df, pd.DataFrame):
        return _tool_result(
            tool="fetch_market_data",
            status="success",
            summary="Market data fetch executed, but no DataFrame was returned.",
        )

    start_date, end_date = _date_range_from_df(df)
    asset_count = _ticker_count(df)

    return _tool_result(
        tool="fetch_market_data",
        status="success",
        summary=(
            f"Fetched {len(df)} rows for {asset_count or len(CRYPTO_ASSETS)} assets "
            f"from {start_date} to {end_date}."
        ),
        details={
            "rows": int(len(df)),
            "asset_count": int(asset_count or len(CRYPTO_ASSETS)),
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
            "columns": list(df.columns),
        },
    )


@register(
    name="preprocess_market_data",
    description=(
        "Clean and standardize market data after fetching. "
        "Use this before feature engineering."
    ),
    returns="Processed row count, cleaned date range, and key preprocessing outputs.",
)
def tool_preprocess_market_data() -> dict[str, Any]:
    result = preprocess_crypto_data()

    if isinstance(result, pd.DataFrame):
        df = result
    else:
        candidate = _find_first_existing([
            PROCESSED_DATA_DIR / "cleaned_crypto_data.csv"
        ])
        df = _safe_read_csv(candidate) if candidate else None

    if df is None:
        return _tool_result(
            tool="preprocess_market_data",
            status="success",
            summary="Preprocessing pipeline executed, but no processed DataFrame was found.",
        )

    start_date, end_date = _date_range_from_df(df)
    missing_counts = {col: int(val) for col, val in df.isna().sum().to_dict().items()}

    return _tool_result(
        tool="preprocess_market_data",
        status="success",
        summary=(
            f"Preprocessed {len(df)} rows across {_ticker_count(df)} assets "
            f"from {start_date} to {end_date}."
        ),
        details={
            "rows": int(len(df)),
            "asset_count": int(_ticker_count(df)),
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
            "missing_values": missing_counts,
            "columns": list(df.columns),
        },
    )


@register(
    name="run_eda",
    description=(
        "Run exploratory data analysis, descriptive statistics, and visual diagnostics. "
        "Use when the agent wants context before decision-making."
    ),
    returns="EDA completion status and discovered output files.",
)
def tool_run_eda() -> dict[str, Any]:
    run_eda_pipeline()

    discovered = []
    for folder in [OUTPUTS_DIR, TABLES_DIR]:
        if Path(folder).exists():
            for file in Path(folder).glob("*"):
                if file.suffix.lower() in {".csv", ".png", ".jpg", ".jpeg", ".html", ".txt"}:
                    discovered.append(str(file))

    return _tool_result(
        tool="run_eda",
        status="success",
        summary="EDA pipeline executed successfully.",
        details={
            "output_count": len(discovered),
        },
        output_files=sorted(discovered)[:25],
    )


@register(
    name="run_feature_engineering",
    description=(
        "Compute technical indicators and engineered features such as moving averages, "
        "RSI, MACD, ATR, rolling volatility, and trading signals."
    ),
    returns="Feature row count, date range, and signal distribution summary.",
)
def tool_run_feature_engineering() -> dict[str, Any]:
    result = run_feature_engineering_pipeline()

    if isinstance(result, pd.DataFrame):
        df = result
    else:
        candidate = _find_first_existing([
            PROCESSED_DATA_DIR / "featured_crypto_data.csv",
        ])
        df = _safe_read_csv(candidate) if candidate else None

    if df is None:
        return _tool_result(
            tool="run_feature_engineering",
            status="success",
            summary="Feature engineering executed, but no output DataFrame was found.",
        )

    start_date, end_date = _date_range_from_df(df)

    signal_columns = [
        "combined_signal",
        "trend_signal",
        "signal",
        "trade_signal",
        "ma_signal",
        "rsi_signal",
        "macd_signal",
    ]
    signal_summaries = {}
    for col in signal_columns:
        if col in df.columns:
            signal_summaries[col] = _value_counts_dict(df, col)

    return _tool_result(
        tool="run_feature_engineering",
        status="success",
        summary=(
            f"Feature engineering completed for {len(df)} rows "
            f"from {start_date} to {end_date}."
        ),
        details={
            "rows": int(len(df)),
            "asset_count": int(_ticker_count(df)),
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
            "columns": list(df.columns),
            "signal_summaries": signal_summaries,
        },
    )


@register(
    name="run_clustering",
    description=(
        "Run clustering analysis to identify asset behavior patterns or market-state groupings. "
        "Use when deeper analytical segmentation is needed."
    ),
    returns="Cluster distribution and clustering output availability.",
)
def tool_run_clustering() -> dict[str, Any]:
    result = run_clustering_pipeline()

    if isinstance(result, pd.DataFrame):
        df = result
    else:
        candidate = _find_first_existing([
            PROCESSED_DATA_DIR / "clustered_crypto_data.csv",
        ])
        df = _safe_read_csv(candidate) if candidate else None

    if df is None:
        return _tool_result(
            tool="run_clustering",
            status="success",
            summary="Clustering executed, but no clustering result file was found.",
        )

    cluster_col = None
    for col in ["cluster", "Cluster", "kmeans_cluster"]:
        if col in df.columns:
            cluster_col = col
            break

    cluster_counts = _value_counts_dict(df, cluster_col) if cluster_col else {}

    return _tool_result(
        tool="run_clustering",
        status="success",
        summary="Clustering analysis completed successfully.",
        details={
            "rows": int(len(df)),
            "cluster_column": cluster_col,
            "cluster_counts": cluster_counts,
            "columns": list(df.columns),
        },
    )


@register(
    name="check_news_sentiment",
    description=(
        "Retrieve crypto news and compute sentiment or market impact signals. "
        "Use this to provide external information retrieval before decisions."
    ),
    returns="News article count, per-asset sentiment summary, and severe-risk flags.",
)
def tool_check_news_sentiment() -> dict[str, Any]:
    run_news_info_retrieval_pipeline()

    detailed_path = _find_first_existing([
        OUTPUTS_DIR / "crypto_news_headlines.csv",
    ])

    summary_path = _find_first_existing([
        OUTPUTS_DIR / "crypto_news_sentiment_summary.csv",
    ])

    news_df = _safe_read_csv(detailed_path) if detailed_path else None
    summary_df = _safe_read_csv(summary_path) if summary_path else None

    article_count = int(len(news_df)) if news_df is not None else 0

    summary_preview: list[dict[str, Any]] = []
    severe_assets: list[str] = []

    if summary_df is not None and not summary_df.empty:
        columns_needed = [
            col for col in [
                "asset", "ticker", "news_signal", "asset_news_bias",
                "market_impact", "dominant_sentiment"
            ]
            if col in summary_df.columns
        ]
        if columns_needed:
            summary_preview = summary_df[columns_needed].head(20).to_dict(orient="records")

        for risk_col in ["news_signal", "market_impact"]:
            if risk_col in summary_df.columns:
                flagged = summary_df[
                    summary_df[risk_col].astype(str).str.contains(
                        "severe|security|regulatory|negative",
                        case=False,
                        na=False,
                    )
                ]
                if not flagged.empty:
                    name_col = "asset" if "asset" in flagged.columns else "ticker"
                    severe_assets.extend(flagged[name_col].astype(str).tolist())

    return _tool_result(
        tool="check_news_sentiment",
        status="success",
        summary=(
            f"News retrieval completed with {article_count} articles. "
            f"Severe or cautionary news flags found for {len(set(severe_assets))} assets."
        ),
        details={
            "article_count": article_count,
            "summary_preview": summary_preview,
            "severe_assets": sorted(set(severe_assets)),
        },
        output_files=[
            str(p) for p in [detailed_path, summary_path] if p is not None
        ],
    )


@register(
    name="run_decisions",
    description=(
        "Apply the decision engine to produce Buy, Sell, or Hold outputs "
        "using technical and optional news-based signals."
    ),
    returns="Decision distribution, score summary, and decision output file availability.",
)
def tool_run_decisions() -> dict[str, Any]:
    result = run_decision_engine_pipeline()

    if isinstance(result, pd.DataFrame):
        df = result
    else:
        candidate = _find_first_existing([
            OUTPUTS_DIR / "market_data_with_decisions.csv",
        ])
        df = _safe_read_csv(candidate) if candidate else None

    summary_df = _safe_read_csv(_find_first_existing([
        OUTPUTS_DIR / "decision_summary.csv",
    ]) or Path("__missing__"))

    if df is None and summary_df is None:
        return _tool_result(
            tool="run_decisions",
            status="success",
            summary="Decision engine executed, but no decision outputs were found.",
        )

    decision_counts = {}
    score_stats = {}

    if df is not None:
        decision_col = None
        for col in ["final_decision", "risk_adjusted_decision", "decision", "raw_decision"]:
            if col in df.columns:
                decision_col = col
                break

        if decision_col:
            decision_counts = _value_counts_dict(df, decision_col)

        for score_col in ["decision_score", "score"]:
            if score_col in df.columns:
                score_stats = _numeric_summary(df, score_col)
                break

    if not decision_counts and summary_df is not None and not summary_df.empty:
        if {"final_decision", "count"}.issubset(summary_df.columns):
            decision_counts = (
                summary_df.groupby("final_decision")["count"].sum().astype(int).to_dict()
            )

    return _tool_result(
        tool="run_decisions",
        status="success",
        summary=f"Decision engine completed. Decision counts: {decision_counts}.",
        details={
            "decision_counts": decision_counts,
            "score_stats": score_stats,
            "rows": int(len(df)) if df is not None else None,
        },
    )


@register(
    name="apply_risk_controls",
    description=(
        "Apply volatility, news, and position risk rules to adjust or override raw decisions. "
        "Use this after running the decision engine."
    ),
    returns="Risk-adjusted decision distribution, override count, and high-risk summary.",
)
def tool_apply_risk_controls() -> dict[str, Any]:
    result = run_risk_management_pipeline()

    if isinstance(result, pd.DataFrame):
        df = result
    else:
        candidate = _find_first_existing([
            OUTPUTS_DIR / "market_data_with_risk_controls.csv",
        ])
        df = _safe_read_csv(candidate) if candidate else None

    summary_df = _safe_read_csv(_find_first_existing([
        OUTPUTS_DIR / "risk_summary.csv",
    ]) or Path("__missing__"))

    if df is None and summary_df is None:
        return _tool_result(
            tool="apply_risk_controls",
            status="success",
            summary="Risk management executed, but no risk output files were found.",
        )

    adjusted_counts = {}
    override_count = None

    if df is not None:
        for col in ["risk_adjusted_decision", "final_decision_after_risk", "final_decision"]:
            if col in df.columns:
                adjusted_counts = _value_counts_dict(df, col)
                break

        if {"raw_decision", "risk_adjusted_decision"}.issubset(df.columns):
            override_count = int((df["raw_decision"] != df["risk_adjusted_decision"]).sum())

    if summary_df is not None and not summary_df.empty:
        for col in ["risk_overrides", "override_count", "overrides"]:
            if col in summary_df.columns:
                override_count = int(pd.to_numeric(summary_df[col], errors="coerce").fillna(0).sum())
                break

    return _tool_result(
        tool="apply_risk_controls",
        status="success",
        summary=(
            f"Risk controls applied. Adjusted decision counts: {adjusted_counts}. "
            f"Overrides: {override_count}."
        ),
        details={
            "adjusted_counts": adjusted_counts,
            "override_count": override_count,
            "rows": int(len(df)) if df is not None else None,
        },
    )


@register(
    name="run_backtest",
    description=(
        "Run the backtesting engine on final risk-adjusted decisions "
        "to evaluate strategy performance."
    ),
    returns="Sharpe ratio, total return, max drawdown, win rate, and trade count.",
)
def tool_run_backtest() -> dict[str, Any]:
    run_backtesting_pipeline()

    summary_path = _find_first_existing([
        TABLES_DIR / "backtest_summary.csv",
    ])

    summary_df = _safe_read_csv(summary_path) if summary_path else None

    if summary_df is None or summary_df.empty:
        return _tool_result(
            tool="run_backtest",
            status="success",
            summary="Backtest executed, but no summary file was found.",
        )

    metrics = {}

    if {"metric", "value"}.issubset(summary_df.columns):
        metrics = dict(zip(summary_df["metric"].astype(str), summary_df["value"]))

    # Common keys from your earlier workflow notes / likely outputs
    sharpe = (
        metrics.get("Strategy Sharpe Ratio")
        or metrics.get("Sharpe Ratio")
        or metrics.get("Sharpe")
    )
    total_return = (
        metrics.get("Total Strategy Return")
        or metrics.get("Strategy Total Return")
        or metrics.get("Total Return")
    )
    max_drawdown = (
        metrics.get("Strategy Max Drawdown")
        or metrics.get("Max Drawdown")
    )
    win_rate = metrics.get("Win Rate")
    trade_count = (
        metrics.get("Trade Count")
        or metrics.get("Number of Trades")
        or metrics.get("Completed Trades")
    )

    return _tool_result(
        tool="run_backtest",
        status="success",
        summary=(
            f"Backtest completed. Sharpe={sharpe}, Return={total_return}, "
            f"Max Drawdown={max_drawdown}, Win Rate={win_rate}, Trades={trade_count}."
        ),
        details={
            "metrics": metrics,
        },
        output_files=[str(summary_path)] if summary_path else [],
    )


# =========================================================
# Convenience functions for the agent loop
# =========================================================

def get_tool_descriptions() -> str:
    lines = []
    for tool in TOOL_REGISTRY.values():
        lines.append(
            f"- {tool.name}: {tool.description} Returns: {tool.returns}"
        )
    return "\n".join(lines)


def list_tool_names() -> list[str]:
    return list(TOOL_REGISTRY.keys())


def run_tool(tool_name: str) -> dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        return _tool_result(
            tool=tool_name,
            status="error",
            summary=f"Unknown tool: {tool_name}",
        )

    try:
        return TOOL_REGISTRY[tool_name].fn()
    except Exception as exc:
        return _tool_result(
            tool=tool_name,
            status="error",
            summary=f"Tool '{tool_name}' failed: {exc}",
        )