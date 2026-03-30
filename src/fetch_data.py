from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

from src.config import CRYPTO_ASSETS, END_DATE, RAW_DATA_DIR, START_DATE


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return dataframe


def validate_downloaded_data(dataframe: pd.DataFrame, ticker: str) -> None:
    """
    Validate that the downloaded dataframe is not empty and contains key columns.
    """
    if dataframe.empty:
        raise ValueError(f"No data downloaded for {ticker}.")

    required_columns = {"date", "open", "high", "low", "close", "volume"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"{ticker} is missing required columns: {missing_columns}")


def download_single_asset(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical OHLCV data for a single crypto asset from Yahoo Finance.
    """
    print(f"Downloading data for {ticker}...")

    dataframe = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    dataframe = dataframe.reset_index()
    dataframe = standardize_columns(dataframe)
    dataframe["ticker"] = ticker

    validate_downloaded_data(dataframe, ticker)
    print(f"Downloaded {len(dataframe)} rows for {ticker}.")

    return dataframe


def save_asset_csv(dataframe: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    """
    Save a single asset dataframe to CSV.
    """
    safe_ticker = ticker.replace("-", "_").lower()
    output_path = output_dir / f"{safe_ticker}.csv"
    dataframe.to_csv(output_path, index=False)
    print(f"Saved raw data for {ticker} to {output_path}")
    return output_path


def combine_asset_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all individual asset dataframes into one dataset.
    """
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    combined_dataframe = combined_dataframe.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    return combined_dataframe


def save_combined_csv(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    """
    Save the combined dataset to CSV.
    """
    output_path = output_dir / "combined_crypto_data.csv"
    dataframe.to_csv(output_path, index=False)
    print(f"Saved combined raw dataset to {output_path}")
    return output_path


def fetch_crypto_data(
    tickers: List[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    output_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """
    Download, validate, and save crypto data for all selected assets.
    Returns the combined dataframe.
    """
    all_dataframes: List[pd.DataFrame] = []

    print("Starting crypto data fetching...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Number of assets: {len(tickers)}")

    for ticker in tickers:
        try:
            asset_dataframe = download_single_asset(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            save_asset_csv(asset_dataframe, ticker, output_dir)
            all_dataframes.append(asset_dataframe)
        except Exception as error:
            print(f"Failed to process {ticker}: {error}")

    if not all_dataframes:
        raise RuntimeError("Data fetching failed for all assets.")

    combined_dataframe = combine_asset_dataframes(all_dataframes)
    save_combined_csv(combined_dataframe, output_dir)

    print("Crypto data fetching completed successfully.")
    print(f"Combined dataset shape: {combined_dataframe.shape}")

    return combined_dataframe


if __name__ == "__main__":
    try:
        fetch_crypto_data(CRYPTO_ASSETS)
    except Exception as e:
        print(f"An error occurred during data fetching: {e}")