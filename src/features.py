from __future__ import annotations

from pathlib import Path

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

from src.config import PROCESSED_DATA_DIR


EDA_ENRICHED_FILE = PROCESSED_DATA_DIR / "eda_enriched_crypto_data.csv"
FEATURED_OUTPUT_FILE = PROCESSED_DATA_DIR / "featured_crypto_data.csv"


def load_eda_data(file_path: Path = EDA_ENRICHED_FILE) -> pd.DataFrame:
    """
    Load the EDA-enriched crypto dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"EDA-enriched dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    return dataframe


def add_moving_averages(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add short-term and long-term simple moving averages for each ticker.
    """
    dataframe = dataframe.copy()

    dataframe["ma_7"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.rolling(window=7).mean())
    )
    dataframe["ma_21"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.rolling(window=21).mean())
    )
    dataframe["ma_50"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.rolling(window=50).mean())
    )

    return dataframe


def add_exponential_moving_averages(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add exponential moving averages for faster trend response.
    """
    dataframe = dataframe.copy()

    dataframe["ema_12"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.ewm(span=12, adjust=False).mean())
    )
    dataframe["ema_26"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.ewm(span=26, adjust=False).mean())
    )

    return dataframe


def add_trend_regime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend regime features based on moving average crossovers.
    """
    dataframe = dataframe.copy()

    dataframe["ma_100"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.rolling(window=100).mean())
    )
    dataframe["ma_200"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.rolling(window=200).mean())
    )

    dataframe["trend_signal"] = 0
    dataframe.loc[
        (dataframe["ma_50"] > dataframe["ma_200"]) & (dataframe["close"] > dataframe["ma_50"]),
        "trend_signal"
    ] = 1

    dataframe.loc[
        (dataframe["ma_50"] < dataframe["ma_200"]) & (dataframe["close"] < dataframe["ma_50"]),
        "trend_signal"
    ] = -1

    return dataframe


def add_atr_feature(dataframe: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) as a volatility feature for each ticker.
    """
    atr_frames = []

    for _, group in dataframe.groupby("ticker", sort=False):
        group = group.copy()
        atr_indicator = AverageTrueRange(
            high=group["high"],
            low=group["low"],
            close=group["close"],
            window=window,
        )
        group["atr_14"] = atr_indicator.average_true_range()
        group["atr_pct"] = group["atr_14"] / group["close"]
        atr_frames.append(group)

    return pd.concat(atr_frames, ignore_index=True)


def add_rsi(dataframe: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) for each ticker.
    """
    dataframe = dataframe.copy()
    rsi_values = []

    for _, group in dataframe.groupby("ticker", sort=False):
        group = group.copy()
        indicator = RSIIndicator(close=group["close"], window=window)
        group["rsi_14"] = indicator.rsi()
        rsi_values.append(group)

    dataframe = pd.concat(rsi_values, ignore_index=True)
    return dataframe


def add_macd(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACD, MACD signal, and MACD histogram for each ticker.
    """
    macd_frames = []

    for _, group in dataframe.groupby("ticker", sort=False):
        group = group.copy()

        macd_indicator = MACD(close=group["close"])
        group["macd"] = macd_indicator.macd()
        group["macd_signal"] = macd_indicator.macd_signal()
        group["macd_diff"] = macd_indicator.macd_diff()

        macd_frames.append(group)

    dataframe = pd.concat(macd_frames, ignore_index=True)
    return dataframe


def add_momentum_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum and price-change based features.
    """
    dataframe = dataframe.copy()

    dataframe["return_3d"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.pct_change(periods=3))
    )
    dataframe["return_7d"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.pct_change(periods=7))
    )
    dataframe["return_14d"] = (
        dataframe.groupby("ticker")["close"]
        .transform(lambda group: group.pct_change(periods=14))
    )

    dataframe["volume_change"] = (
        dataframe.groupby("ticker")["volume"]
        .transform(lambda group: group.pct_change())
    )

    return dataframe


def add_signal_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple rule-based Buy/Sell/Hold signals using moving averages,
    RSI, and MACD conditions.
    """
    dataframe = dataframe.copy()

    dataframe["ma_signal"] = "Hold"
    dataframe.loc[dataframe["ma_7"] > dataframe["ma_21"], "ma_signal"] = "Buy"
    dataframe.loc[dataframe["ma_7"] < dataframe["ma_21"], "ma_signal"] = "Sell"

    dataframe["rsi_signal"] = "Hold"
    dataframe.loc[dataframe["rsi_14"] < 30, "rsi_signal"] = "Buy"
    dataframe.loc[dataframe["rsi_14"] > 70, "rsi_signal"] = "Sell"

    dataframe["macd_signal_label"] = "Hold"
    dataframe.loc[dataframe["macd"] > dataframe["macd_signal"], "macd_signal_label"] = "Buy"
    dataframe.loc[dataframe["macd"] < dataframe["macd_signal"], "macd_signal_label"] = "Sell"

    return dataframe


def add_combined_signal(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined signal by combining MA, RSI, and MACD signals.
    Majority rule:
    - Buy if at least 2 Buy
    - Sell if at least 2 Sell
    - Otherwise Hold
    """
    dataframe = dataframe.copy()

    def combine_row_signals(row: pd.Series) -> str:
        signals = [row["ma_signal"], row["rsi_signal"], row["macd_signal_label"]]
        buy_count = signals.count("Buy")
        sell_count = signals.count("Sell")

        if buy_count >= 2:
            return "Buy"
        if sell_count >= 2:
            return "Sell"
        return "Hold"

    dataframe["combined_signal"] = dataframe.apply(combine_row_signals, axis=1)
    return dataframe


def fill_feature_gaps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Fill or preserve NaNs created naturally by rolling indicators.
    These early NaNs are expected, but for downstream analysis we can
    forward-fill within ticker where appropriate.
    """
    dataframe = dataframe.copy()

    feature_columns = [
        "ma_7",
        "ma_21",
        "ma_50",
        "ma_100",
        "ma_200",
        "trend_signal",
        "atr_14",
        "atr_pct",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_diff",
        "return_3d",
        "return_7d",
        "return_14d",
        "volume_change",
    ]

    existing_columns = [column for column in feature_columns if column in dataframe.columns]

    dataframe[existing_columns] = (
        dataframe.groupby("ticker")[existing_columns]
        .transform(lambda group: group.ffill())
    )

    return dataframe


def save_featured_data(
    dataframe: pd.DataFrame,
    output_path: Path = FEATURED_OUTPUT_FILE,
) -> Path:
    """
    Save the featured dataset.
    """
    dataframe.to_csv(output_path, index=False)
    print(f"Featured dataset saved to {output_path}")
    return output_path


def summarize_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize combined trading signals by ticker.
    """
    summary = (
        dataframe.groupby(["ticker", "combined_signal"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["ticker", "count"], ascending=[True, False])
    )
    return summary


def save_signal_summary(dataframe: pd.DataFrame) -> Path:
    """
    Save signal summary table.
    """
    output_path = PROCESSED_DATA_DIR / "signal_summary.csv"
    dataframe.to_csv(output_path, index=False)
    print(f"Signal summary saved to {output_path}")
    return output_path


def run_feature_engineering_pipeline() -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.
    """
    print("Loading EDA-enriched dataset...")
    dataframe = load_eda_data()

    print("Sorting dataset...")
    dataframe = dataframe.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    print("Adding moving averages...")
    dataframe = add_moving_averages(dataframe)

    print("Adding exponential moving averages...")
    dataframe = add_exponential_moving_averages(dataframe)

    print("Adding trend regime features...")
    dataframe = add_trend_regime_features(dataframe)

    print("Adding ATR volatility feature...")
    dataframe = add_atr_feature(dataframe)

    print("Adding RSI...")
    dataframe = add_rsi(dataframe)

    print("Adding MACD...")
    dataframe = add_macd(dataframe)

    print("Adding momentum features...")
    dataframe = add_momentum_features(dataframe)

    print("Adding signal columns...")
    dataframe = add_signal_columns(dataframe)

    print("Adding combined decision signal...")
    dataframe = add_combined_signal(dataframe)

    print("Filling feature gaps...")
    dataframe = fill_feature_gaps(dataframe)

    print("Saving featured dataset...")
    save_featured_data(dataframe)

    print("Saving signal summary...")
    signal_summary = summarize_signals(dataframe)
    save_signal_summary(signal_summary)

    print("Feature engineering pipeline completed successfully.")
    return dataframe


if __name__ == "__main__":
    try:
        run_feature_engineering_pipeline()
    except Exception as e:
        print(f"Error during feature engineering pipeline: {e}")