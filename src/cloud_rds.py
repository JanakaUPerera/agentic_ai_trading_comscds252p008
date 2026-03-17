from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from src.config import OUTPUTS_DIR, RDS_DBNAME, RDS_HOST, RDS_PASSWORD, RDS_PORT, RDS_USER, TABLES_DIR


NEWS_SUMMARY_FILE = OUTPUTS_DIR / "crypto_news_sentiment_summary.csv"
DECISION_SUMMARY_FILE = OUTPUTS_DIR / "decision_summary.csv"
RISK_SUMMARY_FILE = OUTPUTS_DIR / "risk_summary.csv"
BACKTEST_SUMMARY_FILE = TABLES_DIR / "backtest_summary.csv"


def create_rds_engine():
    """
    Create SQLAlchemy engine for PostgreSQL RDS.
    """
    required_values = [RDS_HOST, RDS_DBNAME, RDS_USER, RDS_PASSWORD]
    if not all(required_values):
        raise ValueError("Missing RDS configuration in environment variables.")

    connection_url = (
        f"postgresql+psycopg2://{RDS_USER}:{RDS_PASSWORD}"
        f"@{RDS_HOST}:{RDS_PORT}/{RDS_DBNAME}"
    )
    return create_engine(connection_url)


def load_csv_if_exists(file_path: Path) -> pd.DataFrame | None:
    """
    Load CSV file if it exists.
    """
    if not file_path.exists():
        print(f"File not found, skipping: {file_path}")
        return None

    return pd.read_csv(file_path)


def upload_dataframe_to_rds(
    dataframe: pd.DataFrame,
    table_name: str,
    engine,
) -> None:
    """
    Upload dataframe to PostgreSQL RDS.
    """
    dataframe.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
    )
    print(f"Uploaded table to RDS: {table_name}")


def upload_project_tables_to_rds() -> None:
    """
    Upload selected structured outputs to RDS.
    """
    engine = create_rds_engine()

    table_file_map = {
        "news_sentiment_summary": NEWS_SUMMARY_FILE,
        "decision_summary": DECISION_SUMMARY_FILE,
        "risk_summary": RISK_SUMMARY_FILE,
        "backtest_summary": BACKTEST_SUMMARY_FILE,
    }

    for table_name, file_path in table_file_map.items():
        dataframe = load_csv_if_exists(file_path)
        if dataframe is not None and not dataframe.empty:
            upload_dataframe_to_rds(dataframe, table_name, engine)

    print("RDS upload completed.")


if __name__ == "__main__":
    try:
        upload_project_tables_to_rds()
    except Exception as e:
        print(f"Error during RDS upload: {e}")