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
    """
    dataframe = dataframe.copy()

    def classify_group(group):
        low = group.quantile(0.33)
        high = group.quantile(0.66)

        return pd.cut(
            group,
            bins=[-float("inf"), low, high, float("inf")],
            labels=["Low", "Medium", "High"]
        )

    dataframe["volatility_risk_level"] = (
        dataframe.groupby("ticker")["rolling_volatility"]
        .transform(classify_group)
    )
    
    dataframe["volatility_risk_level"] = dataframe["volatility_risk_level"].astype(object)

    dataframe.loc[
        dataframe["rolling_volatility"].isna(),
        "volatility_risk_level"
    ] = "Unknown"

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
    Derive position risk using volatility, recent returns, RSI, and decision score.
    """
    dataframe = dataframe.copy()

    def derive_position_risk(row: pd.Series) -> str:
        volatility_risk = str(row.get("volatility_risk_level", "Unknown")).strip()
        rsi_value = row.get("rsi_14", None)
        decision_score = row.get("decision_score", 0)
        return_7d = row.get("return_7d", 0)

        if volatility_risk == "High" and abs(decision_score) >= 3:
            return "High"

        if pd.notna(rsi_value):
            if rsi_value >= 75 or rsi_value <= 25:
                return "High"

        if pd.notna(return_7d):
            if abs(return_7d) >= 0.20:
                return "High"

        if volatility_risk == "Medium":
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

        if overall_risk == "Severe":
            adjusted_decision = "Hold" if raw_decision == "Buy" else "Sell"
            adjusted_reason = "Severe risk conditions triggered protective override."
        elif news_risk == "High" and raw_decision == "Buy":
            adjusted_decision = "Hold"
            adjusted_reason = "Negative news risk blocks new buy exposure."
        elif overall_risk == "High" and raw_decision == "Buy":
            adjusted_decision = "Hold"
            adjusted_reason = "High combined risk downgrades buy decision to hold."
        elif volatility_risk == "High" and raw_decision == "Buy":
            adjusted_decision = "Hold"
            adjusted_reason = "High volatility risk reduces aggressive long positioning."
        elif position_risk == "High" and raw_decision == "Buy":
            adjusted_decision = "Hold"
            adjusted_reason = "Position risk is high, so buy action is deferred."
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
    # try:
        run_risk_management_pipeline()
    # except Exception as e:
    #     print(f"Error running risk management pipeline: {e}")