from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
import os

from src.config import OUTPUTS_DIR, TABLES_DIR

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_API_URL,
    GROQ_TEMPERATURE,
    LLM_MAX_CONTEXT_CHARS,
    LLM_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT_SECONDS,
)

BACKTEST_SUMMARY_FILE = TABLES_DIR / "backtest_summary.csv"
CLUSTER_SUMMARY_FILE = TABLES_DIR / "cluster_summary.csv"
DECISION_SUMMARY_FILE = OUTPUTS_DIR / "decision_summary.csv"
RISK_SUMMARY_FILE = OUTPUTS_DIR / "risk_summary.csv"
NEWS_SUMMARY_FILE = OUTPUTS_DIR / "crypto_news_sentiment_summary.csv"
PORTFOLIO_RETURNS_FILE = OUTPUTS_DIR / "portfolio_daily_returns.csv"

INTERPRETATION_MD_FILE = OUTPUTS_DIR / "final_interpretation.md"
INTERPRETATION_DEBUG_FILE = OUTPUTS_DIR / "final_interpretation_debug_prompt.txt"

REQUIRED_SECTIONS = [
    "Executive Summary",
    "Performance Takeaways",
    "Trading Behaviour",
    "Risk View",
    "News and Market Context",
    "Asset Behaviour and Clustering",
    "Business Meaning",
    "Main Weaknesses",
    "Bottom Line",
]


def load_csv_if_exists(file_path: Path) -> Optional[pd.DataFrame]:
    if file_path.exists():
        return pd.read_csv(file_path)
    return None


def get_metric_value(summary_df: Optional[pd.DataFrame], metric_name: str) -> Optional[float]:
    if summary_df is None or summary_df.empty:
        return None

    row = summary_df.loc[summary_df["metric"] == metric_name]
    if row.empty:
        return None

    try:
        return float(row["value"].iloc[0])
    except Exception:
        return None


def safe_float(value: object) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def compact_number(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def compact_percent(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}%}"


def extract_key_metrics() -> str:
    """
    Extract a compact metrics section so the prompt stays focused.
    """
    backtest_df = load_csv_if_exists(BACKTEST_SUMMARY_FILE)
    risk_df = load_csv_if_exists(RISK_SUMMARY_FILE)
    news_df = load_csv_if_exists(NEWS_SUMMARY_FILE)
    decision_df = load_csv_if_exists(DECISION_SUMMARY_FILE)
    cluster_df = load_csv_if_exists(CLUSTER_SUMMARY_FILE)

    final_strategy_growth = get_metric_value(backtest_df, "Final Strategy Growth")
    final_benchmark_growth = get_metric_value(backtest_df, "Final Benchmark Growth")
    total_strategy_return = get_metric_value(backtest_df, "Total Strategy Return")
    total_benchmark_return = get_metric_value(backtest_df, "Total Benchmark Return")
    strategy_drawdown = get_metric_value(backtest_df, "Strategy Max Drawdown")
    strategy_sharpe = get_metric_value(backtest_df, "Strategy Sharpe Ratio")
    total_trades = get_metric_value(backtest_df, "Total Trades")
    win_rate = get_metric_value(backtest_df, "Win Rate")

    metrics_lines = [
        "Key Metrics:",
        f"- Final Strategy Growth: {compact_number(final_strategy_growth, 6)}",
        f"- Final Benchmark Growth: {compact_number(final_benchmark_growth, 6)}",
        f"- Total Strategy Return: {compact_percent(total_strategy_return)}",
        f"- Total Benchmark Return: {compact_percent(total_benchmark_return)}",
        f"- Strategy Max Drawdown: {compact_percent(strategy_drawdown)}",
        f"- Strategy Sharpe Ratio: {compact_number(strategy_sharpe, 4)}",
        f"- Total Trades: {int(total_trades) if total_trades is not None else 'N/A'}",
        f"- Win Rate: {compact_percent(win_rate)}",
    ]

    if risk_df is not None and not risk_df.empty:
        total_overrides = int(risk_df["risk_overrides"].sum()) if "risk_overrides" in risk_df.columns else None
        avg_volatility = safe_float(risk_df["avg_rolling_volatility"].mean()) if "avg_rolling_volatility" in risk_df.columns else None
        top_risk_assets = (
            ", ".join(risk_df.sort_values(by="risk_overrides", ascending=False).head(3)["ticker"].astype(str).tolist())
            if "risk_overrides" in risk_df.columns and "ticker" in risk_df.columns
            else "N/A"
        )
        metrics_lines.extend([
            f"- Total Risk Overrides: {total_overrides if total_overrides is not None else 'N/A'}",
            f"- Average Rolling Volatility: {compact_number(avg_volatility, 6)}",
            f"- Top Risk-Override Assets: {top_risk_assets}",
        ])

    if news_df is not None and not news_df.empty:
        if "headline_count" in news_df.columns and "asset" in news_df.columns:
            top_news_assets_df = news_df.sort_values(by="headline_count", ascending=False).head(3)
            top_news_assets = ", ".join(
                f"{row['asset']} ({int(row['headline_count'])})" for _, row in top_news_assets_df.iterrows()
            )
        else:
            top_news_assets = "N/A"

        strong_positive_assets = (
            ", ".join(news_df.loc[news_df["asset_news_bias"] == "Strong Positive", "asset"].astype(str).tolist())
            if "asset_news_bias" in news_df.columns and "asset" in news_df.columns
            else ""
        )
        strong_negative_assets = (
            ", ".join(news_df.loc[news_df["asset_news_bias"] == "Strong Negative", "asset"].astype(str).tolist())
            if "asset_news_bias" in news_df.columns and "asset" in news_df.columns
            else ""
        )

        metrics_lines.extend([
            f"- Top News Assets: {top_news_assets}",
            f"- Strong Positive News Bias Assets: {strong_positive_assets or 'None'}",
            f"- Strong Negative News Bias Assets: {strong_negative_assets or 'None'}",
        ])

    if decision_df is not None and not decision_df.empty and {"final_decision", "count"}.issubset(decision_df.columns):
        buy_count = int(decision_df.loc[decision_df["final_decision"] == "Buy", "count"].sum())
        sell_count = int(decision_df.loc[decision_df["final_decision"] == "Sell", "count"].sum())
        hold_count = int(decision_df.loc[decision_df["final_decision"] == "Hold", "count"].sum())
        metrics_lines.extend([
            f"- Total Buy Decisions: {buy_count}",
            f"- Total Sell Decisions: {sell_count}",
            f"- Total Hold Decisions: {hold_count}",
        ])

    if cluster_df is not None and not cluster_df.empty and {"cluster", "assets"}.issubset(cluster_df.columns):
        for _, row in cluster_df.iterrows():
            metrics_lines.append(f"- Cluster {row['cluster']}: {row['assets']}")

    return "\n".join(metrics_lines)


def dataframe_to_compact_text(
    dataframe: Optional[pd.DataFrame],
    title: str,
    max_rows: int = 12,
    max_cols: int = 12,
) -> str:
    """
    Convert dataframe into compact prompt text with controlled width.
    """
    if dataframe is None or dataframe.empty:
        return f"{title}:\nNot available.\n"

    working_df = dataframe.copy()

    if len(working_df.columns) > max_cols:
        working_df = working_df.iloc[:, :max_cols]

    preview = working_df.head(max_rows).to_string(index=False, max_cols=max_cols)
    return f"{title}:\n{preview}\n"


def trim_context_to_limit(text: str, max_chars: int = LLM_MAX_CONTEXT_CHARS) -> str:
    """
    Trim context safely if it grows too large.
    """
    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars]
    last_newline = trimmed.rfind("\n")
    if last_newline > 0:
        trimmed = trimmed[:last_newline]

    return trimmed + "\n\n[Context trimmed to fit prompt size.]"


def build_interpretation_context() -> str:
    """
    Build compact context from output files.
    """
    backtest_df = load_csv_if_exists(BACKTEST_SUMMARY_FILE)
    cluster_df = load_csv_if_exists(CLUSTER_SUMMARY_FILE)
    decision_df = load_csv_if_exists(DECISION_SUMMARY_FILE)
    risk_df = load_csv_if_exists(RISK_SUMMARY_FILE)
    news_df = load_csv_if_exists(NEWS_SUMMARY_FILE)
    portfolio_df = load_csv_if_exists(PORTFOLIO_RETURNS_FILE)

    sections = [
        extract_key_metrics(),
        "",
        dataframe_to_compact_text(backtest_df, "Backtest Summary", max_rows=20),
        dataframe_to_compact_text(cluster_df, "Cluster Summary", max_rows=10),
        dataframe_to_compact_text(decision_df, "Decision Summary", max_rows=20),
        dataframe_to_compact_text(risk_df, "Risk Summary", max_rows=12),
        dataframe_to_compact_text(news_df, "News Sentiment Summary", max_rows=12),
        dataframe_to_compact_text(portfolio_df, "Portfolio Daily Returns", max_rows=15, max_cols=6),
    ]

    context = "\n".join(sections)
    return trim_context_to_limit(context)


def build_prompt(context: str) -> str:
    """
    Build a practical, non-academic interpretation prompt.
    """
    return f"""
You are a sharp crypto market analyst who writes practical, clear, executive-style result interpretations.

Your job is to interpret the results of an automated crypto trading workflow in a practical, business-style way.

Write clearly, directly, and naturally.
Do not sound academic, theoretical, or overly formal.
Do not mention coursework, university, academic framing, or research framing.
Do not use vague filler language.
Do not invent numbers or claims.
If something is missing, say it was not available.

Write the interpretation in markdown.

Use this exact structure:

# Final Interpretation

## Executive Summary
Give a short overall view of what happened.

## Performance Takeaways
Explain:
- whether the strategy did better or worse than the benchmark
- how strong or weak the returns look
- whether the risk-adjusted performance looks convincing

## Trading Behaviour
Explain:
- whether the system was aggressive or cautious
- whether it produced many Buy, Sell, or Hold decisions
- what that says about market conditions

## Risk View
Explain:
- whether risk controls were used often
- whether volatility or news risk seems to have had strong influence
- whether the risk layer helped reduce exposure

## News and Market Context
Explain:
- which assets got the most news attention
- whether sentiment looked positive, negative, security-related, or regulatory
- how that may have influenced decisions

## Asset Behaviour and Clustering
Explain:
- what the clusters suggest
- whether some assets behaved similarly
- whether some assets looked more volatile or more defensive

## Business Meaning
Explain what these results mean from a practical decision-making or trading support perspective.

## Main Weaknesses
Call out the biggest weaknesses or limits in a realistic way.

## Bottom Line
End with a short final verdict on whether the workflow looks useful.

Style rules:
- Sound like an analyst explaining results to a founder, manager, or trading lead.
- Keep sentences clean and natural.
- Be specific when numbers are available.
- Avoid academic phrases.
- Prefer short paragraphs.

Project Output Context:
{context}
""".strip()


def validate_groq_config() -> None:
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in .env")


def extract_response_text(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as error:
        raise RuntimeError(
            f"Unable to extract Groq response content. Raw response: {json.dumps(data)[:2000]}"
        ) from error


def clean_markdown_response(content: str) -> str:
    """
    Clean common markdown issues.
    """
    content = content.strip()

    # Remove surrounding code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    # Normalize excessive blank lines
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    # Ensure title exists
    if not content.startswith("# Final Interpretation"):
        content = "# Final Interpretation\n\n" + content

    return content


def add_missing_sections(content: str) -> str:
    """
    Ensure all required sections exist.
    """
    for section in REQUIRED_SECTIONS:
        heading = f"## {section}"
        if heading not in content:
            content += f"\n\n{heading}\nNot available from the generated interpretation.\n"
    return content.strip() + "\n"


def call_groq_llm(prompt: str) -> str:
    """
    Call Groq with retries.
    """
    validate_groq_config()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "temperature": GROQ_TEMPERATURE,
        "include_reasoning": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a sharp crypto market analyst who writes practical, clear, executive-style result interpretations.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            return extract_response_text(data)
        except Exception as error:
            last_error = error
            print(f"Groq request failed on attempt {attempt}/{LLM_MAX_RETRIES}: {error}")
            if attempt < LLM_MAX_RETRIES:
                time.sleep(2 * attempt)

    raise RuntimeError(f"Groq request failed after {LLM_MAX_RETRIES} attempts.") from last_error


def save_debug_prompt(prompt: str) -> None:
    INTERPRETATION_DEBUG_FILE.write_text(prompt, encoding="utf-8")
    print(f"Saved debug prompt to {INTERPRETATION_DEBUG_FILE}")


def save_interpretation_files(content: str) -> None:
    INTERPRETATION_MD_FILE.write_text(content, encoding="utf-8")
    print(f"Saved markdown interpretation to {INTERPRETATION_MD_FILE}")


def run_llm_interpreter_agent() -> str:
    """
    Run the Groq-based LLM interpreter.
    """
    print("Building interpretation context from output files...")
    context = build_interpretation_context()

    print("Building prompt for Groq...")
    prompt = build_prompt(context)
    save_debug_prompt(prompt)

    print(f"Calling Groq model: {GROQ_MODEL}")
    interpretation = call_groq_llm(prompt)

    print("Cleaning interpretation output...")
    interpretation = clean_markdown_response(interpretation)
    interpretation = add_missing_sections(interpretation)

    print("Saving interpretation files...")
    save_interpretation_files(interpretation)

    print("Groq LLM interpreter completed successfully.")
    return interpretation


if __name__ == "__main__":
    try:
        run_llm_interpreter_agent()
    except Exception as e:
        print(f"Error running interpret-agent pipeline: {e}")