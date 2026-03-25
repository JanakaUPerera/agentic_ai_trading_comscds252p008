from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import OUTPUTS_DIR


DECISION_INPUT_FILE = OUTPUTS_DIR / "market_data_with_decisions.csv"
RISK_OUTPUT_FILE = OUTPUTS_DIR / "market_data_with_risk_controls.csv"
RISK_SUMMARY_FILE = OUTPUTS_DIR / "risk_summary.csv"


def load_decision_data(file_path: Path = DECISION_INPUT_FILE) -> pd.DataFrame:
    """
    Load the decision-engine output dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Decision dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    return dataframe

def classify_volatility_risk(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Classify volatility risk level within each asset using rolling volatility quantiles.
    (Includes Severe level while preserving original dataframe structure)
    """
    dataframe = dataframe.copy()

    def classify_group(series: pd.Series) -> pd.Series:
        low = series.quantile(0.33)
        high = series.quantile(0.66)
        severe = series.quantile(0.90)

        def classify_value(value):
            if pd.isna(value):
                return "Unknown"
            if value >= severe:
                return "Severe"
            elif value >= high:
                return "High"
            elif value >= low:
                return "Medium"
            else:
                return "Low"

        return series.apply(classify_value)

    dataframe["volatility_risk_level"] = (
        dataframe.groupby("ticker")["rolling_volatility"]
        .transform(classify_group)
    )

    return dataframe


def classify_news_risk(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Derive news risk level from news category and asset news bias.
    """
    dataframe = dataframe.copy()

    def derive_news_risk(row: pd.Series) -> str:
        news_signal = str(row.get("news_signal", "Neutral")).strip()
        asset_news_bias = str(row.get("asset_news_bias", "Neutral")).strip()
        market_impact_signal = str(row.get("market_impact_signal", "Neutral")).strip()

        if news_signal == "Security":
            return "Severe"
        if news_signal == "Regulatory" and market_impact_signal == "Negative":
            return "High"
        if asset_news_bias == "Strong Negative":
            return "High"
        if asset_news_bias == "Negative":
            return "Medium"
        return "Low"

    dataframe["news_risk_level"] = dataframe.apply(derive_news_risk, axis=1)
    return dataframe


def classify_position_risk(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Classify position risk based on volatility, momentum extremes, and recent returns.
    """
    dataframe = dataframe.copy()

    def derive_position_risk(row: pd.Series) -> str:
        atr_pct = row.get("atr_pct", None)
        rsi_value = row.get("rsi_14", None)
        return_7d = row.get("return_7d", 0)
        trend_signal = row.get("trend_signal", 0)

        if pd.notna(atr_pct) and atr_pct >= 0.10:
            return "High"
        if pd.notna(rsi_value) and (rsi_value >= 80 or rsi_value <= 20) and trend_signal != 1:
            return "High"
        if pd.notna(return_7d) and abs(return_7d) >= 0.25:
            return "High"
        if pd.notna(atr_pct) and atr_pct >= 0.06:
            return "Medium"
        return "Low"

    dataframe["position_risk_level"] = dataframe.apply(derive_position_risk, axis=1)
    return dataframe


def assign_overall_risk_level(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Combine volatility risk, news risk, and position risk into one overall risk level.
    """
    dataframe = dataframe.copy()

    risk_rank = {
        "Unknown": 1,
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Severe": 4,
    }

    def combine_risks(row: pd.Series) -> str:
        levels = [
            str(row.get("volatility_risk_level", "Unknown")).strip(),
            str(row.get("news_risk_level", "Unknown")).strip(),
            str(row.get("position_risk_level", "Unknown")).strip(),
        ]

        max_level = max(levels, key=lambda value: risk_rank.get(value, 0))
        return max_level

    dataframe["overall_risk_level"] = dataframe.apply(combine_risks, axis=1)
    return dataframe


def apply_risk_overrides(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Apply risk controls to convert raw final decisions into risk-adjusted decisions.
    Combines broad risk coverage with stricter confirmation for high-risk buy cases.
    """
    dataframe = dataframe.copy()

    adjusted_decisions = []
    adjusted_reasons = []

    for _, row in dataframe.iterrows():
        raw_decision = str(row.get("final_decision", "Hold")).strip()
        overall_risk = str(row.get("overall_risk_level", "Unknown")).strip()
        news_risk = str(row.get("news_risk_level", "Unknown")).strip()
        volatility_risk = str(row.get("volatility_risk_level", "Unknown")).strip()
        position_risk = str(row.get("position_risk_level", "Unknown")).strip()
        trend_signal = int(row.get("trend_signal", 0))
        score = float(row.get("decision_score", 0))

        if overall_risk == "Severe":
            if raw_decision == "Buy":
                adjusted_decision = "Hold"
                adjusted_reason = "Severe risk blocks new long exposure."
            else:
                adjusted_decision = "Sell"
                adjusted_reason = "Severe risk supports defensive sell action."

        elif news_risk == "High" and raw_decision == "Buy":
            adjusted_decision = "Hold"
            adjusted_reason = "High news risk blocks new buy exposure."

        elif overall_risk == "High" and raw_decision == "Buy":
            if trend_signal == 1 and score >= 3:
                adjusted_decision = "Buy"
                adjusted_reason = "High risk but strong bullish trend and score allow controlled buy."
            else:
                adjusted_decision = "Hold"
                adjusted_reason = "High combined risk downgrades buy decision to hold."

        elif volatility_risk == "High" and raw_decision == "Buy":
            adjusted_decision = "Hold"
            adjusted_reason = "High volatility risk reduces aggressive long positioning."

        elif position_risk == "High" and raw_decision == "Buy" and trend_signal != 1:
            adjusted_decision = "Hold"
            adjusted_reason = "High position risk defers additional buy exposure."

        elif overall_risk == "High" and raw_decision == "Sell":
            adjusted_decision = "Sell"
            adjusted_reason = "High risk supports defensive sell decision."

        else:
            adjusted_decision = raw_decision
            adjusted_reason = "No additional risk override required."

        adjusted_decisions.append(adjusted_decision)
        adjusted_reasons.append(adjusted_reason)

    dataframe["risk_adjusted_decision"] = adjusted_decisions
    dataframe["risk_override_reason"] = adjusted_reasons

    return dataframe


def add_risk_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple binary flags for easier analysis and reporting.
    """
    dataframe = dataframe.copy()

    dataframe["high_volatility_flag"] = (dataframe["volatility_risk_level"] == "High").astype(int)
    dataframe["negative_news_flag"] = dataframe["news_risk_level"].isin(["High", "Severe"]).astype(int)
    dataframe["high_position_risk_flag"] = (dataframe["position_risk_level"] == "High").astype(int)
    dataframe["risk_override_flag"] = (
        dataframe["risk_adjusted_decision"] != dataframe["final_decision"]
    ).astype(int)

    return dataframe


def summarize_risk(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize risk profile and risk-adjusted decisions by ticker.
    """
    summary = (
        dataframe.groupby("ticker")
        .agg(
            rows=("ticker", "size"),
            avg_rolling_volatility=("rolling_volatility", "mean"),
            high_volatility_days=("high_volatility_flag", "sum"),
            negative_news_days=("negative_news_flag", "sum"),
            high_position_risk_days=("high_position_risk_flag", "sum"),
            risk_overrides=("risk_override_flag", "sum"),
            buy_count=("risk_adjusted_decision", lambda values: (values == "Buy").sum()),
            sell_count=("risk_adjusted_decision", lambda values: (values == "Sell").sum()),
            hold_count=("risk_adjusted_decision", lambda values: (values == "Hold").sum()),
        )
        .reset_index()
    )

    summary["avg_rolling_volatility"] = summary["avg_rolling_volatility"].round(6)
    return summary


def save_risk_outputs(dataframe: pd.DataFrame) -> Tuple[Path, Path]:
    """
    Save row-level risk-managed dataset and summary.
    """
    dataframe.to_csv(RISK_OUTPUT_FILE, index=False)
    print(f"Saved risk-managed dataset to {RISK_OUTPUT_FILE}")

    summary = summarize_risk(dataframe)
    summary.to_csv(RISK_SUMMARY_FILE, index=False)
    print(f"Saved risk summary to {RISK_SUMMARY_FILE}")

    return RISK_OUTPUT_FILE, RISK_SUMMARY_FILE


def run_risk_management_pipeline() -> pd.DataFrame:
    """
    Run the full risk management pipeline.
    """
    print("Loading decision dataset...")
    dataframe = load_decision_data()

    print("Classifying volatility risk...")
    dataframe = classify_volatility_risk(dataframe)
    
    print("Classifying news risk...")
    dataframe = classify_news_risk(dataframe)

    print("Classifying position risk...")
    dataframe = classify_position_risk(dataframe)

    print("Assigning overall risk level...")
    dataframe = assign_overall_risk_level(dataframe)

    print("Applying risk overrides...")
    dataframe = apply_risk_overrides(dataframe)

    print("Adding risk flags...")
    dataframe = add_risk_flags(dataframe)

    print("Saving risk management outputs...")
    save_risk_outputs(dataframe)

    print("Risk management pipeline completed successfully.")
    return dataframe


if __name__ == "__main__":
    try:
        run_risk_management_pipeline()
    except Exception as e:
        print(f"Error running risk management pipeline: {e}")