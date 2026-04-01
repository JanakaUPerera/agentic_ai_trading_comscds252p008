from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import OUTPUTS_DIR


MARKET_NEWS_INPUT_FILE = OUTPUTS_DIR / "market_data_with_news_signal.csv"
DECISION_OUTPUT_FILE = OUTPUTS_DIR / "market_data_with_decisions.csv"
DECISION_SUMMARY_FILE = OUTPUTS_DIR / "decision_summary.csv"


def load_market_news_data(file_path: Path = MARKET_NEWS_INPUT_FILE) -> pd.DataFrame:
    """
    Load the market dataset enriched with news signals.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Market-news dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    return dataframe


def map_technical_signal_to_score(signal: str) -> int:
    """
    Convert technical combined signal to numeric score.
    """
    signal_map = {
        "Buy": 2,
        "Hold": 0,
        "Sell": -2,
    }
    return signal_map.get(str(signal).strip(), 0)


def map_market_impact_to_score(signal: str) -> int:
    """
    Convert market impact signal from news into numeric score.
    """
    signal_map = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1,
    }
    return signal_map.get(str(signal).strip(), 0)


def map_asset_news_bias_to_score(bias: str) -> int:
    """
    Convert asset-level news bias into numeric score.
    """
    bias_map = {
        "Strong Positive": 2,
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1,
        "Strong Negative": -2,
    }
    return bias_map.get(str(bias).strip(), 0)

def map_trend_to_score(trend: int) -> int:
    if trend == 1:
        return 2   # strong bullish bias
    if trend == -1:
        return -2  # strong bearish bias
    return 0

def apply_news_category_penalty(news_signal: str) -> int:
    """
    Apply category-aware penalty or adjustment based on the dominant news type.
    Security and strongly negative regulatory news should reduce trading confidence.
    """
    signal = str(news_signal).strip()

    if signal == "Security":
        return -2
    if signal == "Regulatory":
        return -1
    if signal == "Bearish":
        return -1
    if signal == "Bullish":
        return 1

    return 0


def derive_directional_news_bonus(score: float) -> int:
    """
    Convert directional weighted news score into a compact bonus.
    """
    if score >= 5:
        return 2
    if score >= 2:
        return 1
    if score <= -5:
        return -2
    if score <= -2:
        return -1
    return 0

def momentum_bonus(return_7d: float, return_14d: float) -> int:
    """
    Calculate a momentum bonus based on recent price returns.
    """
    if pd.notna(return_7d) and pd.notna(return_14d):
        if return_7d > 0.05 and return_14d > 0.08:
            return 1
        if return_7d < -0.05 and return_14d < -0.08:
            return -1
    return 0

def calculate_decision_components(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create component scores used by the decision engine.
    """
    dataframe = dataframe.copy()

    dataframe["technical_score"] = dataframe["combined_signal"].apply(map_technical_signal_to_score)
    dataframe["market_impact_score"] = dataframe["market_impact_signal"].apply(map_market_impact_to_score)
    dataframe["news_bias_score"] = dataframe["asset_news_bias"].apply(map_asset_news_bias_to_score)
    dataframe["news_category_adjustment"] = dataframe["news_signal"].apply(apply_news_category_penalty)
    dataframe["directional_news_bonus"] = dataframe["directional_news_score_sum"].apply(derive_directional_news_bonus)
    dataframe["trend_score"] = dataframe["trend_signal"].apply(map_trend_to_score)
    dataframe["momentum_bonus"] = dataframe.apply(lambda row: momentum_bonus(row.get("return_7d", 0), row.get("return_14d", 0)), axis=1)

    dataframe["decision_score"] = (
        dataframe["technical_score"]
        + dataframe["market_impact_score"]
        + dataframe["news_bias_score"]
        + dataframe["news_category_adjustment"]
        + dataframe["directional_news_bonus"]
        + dataframe["trend_score"]
        + dataframe["momentum_bonus"]
    )

    return dataframe

def apply_decision_rules(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Apply final Buy/Sell/Hold decision rules.
    """
    dataframe = dataframe.copy()

    final_decisions = []
    reasons = []

    for _, row in dataframe.iterrows():
        score = float(row["decision_score"])
        combined_signal = str(row["combined_signal"]).strip()
        news_signal = str(row["news_signal"]).strip()
        trend_signal = int(row.get("trend_signal", 0))
        asset_news_bias = str(row["asset_news_bias"]).strip()

        if news_signal == "Security":
            final_decision = "Sell"
            reason = "Security risk dominates."
        elif news_signal == "Regulatory":
            final_decision = "Sell"
            reason = "Negative regulatory context and weak score."
        elif trend_signal == 1 and score >= 3:
            final_decision = "Buy"
            reason = "Bull trend with supportive multi-signal score."
        elif trend_signal == -1:
            final_decision = "Sell"
            reason = "Bear trend with supportive multi-signal score."
        elif combined_signal == "Buy" and asset_news_bias in {"Negative", "Strong Negative"}:
            final_decision = "Hold"
            reason = "Technical buy weakened by negative asset news bias."
        elif combined_signal == "Sell" and asset_news_bias in {"Positive", "Strong Positive"}:
            final_decision = "Hold"
            reason = "Technical sell softened by positive asset news bias."
        elif score >= 5:
            final_decision = "Buy"
            reason = "Strong positive total decision score."
        elif score <= -5:
            final_decision = "Sell"
            reason = "Strong negative total decision score."
        else:
            final_decision = "Hold"
            reason = "Signal strength below trading threshold."

        final_decisions.append(final_decision)
        reasons.append(reason)

    dataframe["final_decision"] = final_decisions
    dataframe["decision_reason"] = reasons
    return dataframe


def summarize_decisions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize final decisions by ticker.
    """
    summary = (
        dataframe.groupby(["ticker", "final_decision"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["ticker", "count"], ascending=[True, False])
    )
    return summary


def save_decision_outputs(dataframe: pd.DataFrame) -> Tuple[Path, Path]:
    """
    Save row-level decisions and summary.
    """
    dataframe.to_csv(DECISION_OUTPUT_FILE, index=False)
    print(f"Saved decision dataset to {DECISION_OUTPUT_FILE}")

    summary = summarize_decisions(dataframe)
    summary.to_csv(DECISION_SUMMARY_FILE, index=False)
    print(f"Saved decision summary to {DECISION_SUMMARY_FILE}")

    return DECISION_OUTPUT_FILE, DECISION_SUMMARY_FILE


def run_decision_engine_pipeline() -> pd.DataFrame:
    """
    Run the full decision engine pipeline.
    """
    print("Loading market dataset with news signals...")
    dataframe = load_market_news_data()

    print("Calculating decision components...")
    dataframe = calculate_decision_components(dataframe)

    print("Applying final decision rules...")
    dataframe = apply_decision_rules(dataframe)

    print("Saving decision outputs...")
    save_decision_outputs(dataframe)

    print("Decision engine pipeline completed successfully.")
    return dataframe


if __name__ == "__main__":
    try:
        run_decision_engine_pipeline()
    except Exception as e:
        print(f"Error running decision engine pipeline: {e}")