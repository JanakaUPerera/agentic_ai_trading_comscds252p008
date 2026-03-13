from src.config import CRYPTO_ASSETS, END_DATE, START_DATE
from src.fetch_data import fetch_crypto_data


def main() -> None:
    print("Agentic AI Trading Workflow Project")
    print(f"Assets selected: {len(CRYPTO_ASSETS)}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    combined_dataframe = fetch_crypto_data(
        tickers=CRYPTO_ASSETS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print("Data collection summary:")
    print(combined_dataframe.head())
    print(combined_dataframe.info())


if __name__ == "__main__":
    main()