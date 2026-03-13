from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


RAW_COMBINED_FILE = RAW_DATA_DIR / "combined_crypto_data.csv"
CLEANED_OUTPUT_FILE = PROCESSED_DATA_DIR / "cleaned_crypto_data.csv"


def load_raw_data(file_path: Path = RAW_COMBINED_FILE) -> pd.DataFrame:
    """
    Load the combined raw crypto dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    return dataframe


def convert_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.
    """
    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    if "ticker" in dataframe.columns:
        dataframe["ticker"] = dataframe["ticker"].astype(str).str.strip()

    return dataframe


def remove_duplicates(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows.
    """
    before_count = len(dataframe)
    dataframe = dataframe.drop_duplicates()
    after_count = len(dataframe)
    removed_count = before_count - after_count
    return dataframe, removed_count


def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    Strategy:
    - Drop rows where date or ticker is missing
    - Forward-fill price fields within each ticker
    - Fill missing volume with 0
    """
    dataframe = dataframe.copy()

    dataframe = dataframe.dropna(subset=["date", "ticker"])

    price_columns = ["open", "high", "low", "close", "adj_close"]
    existing_price_columns = [column for column in price_columns if column in dataframe.columns]

    if existing_price_columns:
        dataframe[existing_price_columns] = (
            dataframe.groupby("ticker")[existing_price_columns]
            .transform(lambda group: group.ffill().bfill())
        )

    if "volume" in dataframe.columns:
        dataframe["volume"] = dataframe["volume"].fillna(0)

    return dataframe


def remove_invalid_rows(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove clearly invalid rows such as:
    - non-positive prices
    - negative volume
    """
    dataframe = dataframe.copy()
    before_count = len(dataframe)

    price_columns = ["open", "high", "low", "close", "adj_close"]
    existing_price_columns = [column for column in price_columns if column in dataframe.columns]

    for column in existing_price_columns:
        dataframe = dataframe[dataframe[column] > 0]

    if "volume" in dataframe.columns:
        dataframe = dataframe[dataframe["volume"] >= 0]

    after_count = len(dataframe)
    removed_count = before_count - after_count
    return dataframe, removed_count


def sort_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Sort data by ticker and date.
    """
    dataframe = dataframe.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    return dataframe


def generate_cleaning_summary(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    """
    Print a summary of cleaning results.
    """
    print("\nPreprocessing Summary")
    print("-" * 40)
    print(f"Initial shape: {before_df.shape}")
    print(f"Final shape:   {after_df.shape}")
    print(f"Rows removed:  {len(before_df) - len(after_df)}")
    print("\nMissing values after cleaning:")
    print(after_df.isnull().sum())
    print("-" * 40)


def save_cleaned_data(dataframe: pd.DataFrame, output_path: Path = CLEANED_OUTPUT_FILE) -> Path:
    """
    Save cleaned dataset to processed directory.
    """
    dataframe.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return output_path


def preprocess_crypto_data(
    input_path: Path = RAW_COMBINED_FILE,
    output_path: Path = CLEANED_OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for crypto data.
    """
    print("Loading raw dataset...")
    raw_dataframe = load_raw_data(input_path)
    working_dataframe = raw_dataframe.copy()

    print(f"Raw dataset shape: {working_dataframe.shape}")

    print("Converting data types...")
    working_dataframe = convert_data_types(working_dataframe)

    print("Removing duplicate rows...")
    working_dataframe, duplicate_count = remove_duplicates(working_dataframe)
    print(f"Removed duplicates: {duplicate_count}")

    print("Handling missing values...")
    working_dataframe = handle_missing_values(working_dataframe)

    print("Removing invalid rows...")
    working_dataframe, invalid_count = remove_invalid_rows(working_dataframe)
    print(f"Removed invalid rows: {invalid_count}")

    print("Sorting data...")
    cleaned_dataframe = sort_data(working_dataframe)

    generate_cleaning_summary(raw_dataframe, cleaned_dataframe)
    save_cleaned_data(cleaned_dataframe, output_path)

    return cleaned_dataframe


if __name__ == "__main__":
    try:
        preprocess_crypto_data()
    except Exception as e:
        print(f"Error during preprocessing: {e}")