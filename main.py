from src.config import CRYPTO_ASSETS, END_DATE, START_DATE
from src.fetch_data import fetch_crypto_data
from src.preprocess_data import preprocess_crypto_data
from src.eda import run_eda_pipeline
from src.features import run_feature_engineering_pipeline


def main() -> None:
    print("Agentic AI Trading Workflow Project")
    print(f"Assets selected: {len(CRYPTO_ASSETS)}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    combined_dataframe = fetch_crypto_data(
        tickers=CRYPTO_ASSETS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print("Fetched dataset info:")
    print(combined_dataframe.info())
    
    cleaned_dataframe = preprocess_crypto_data()
    
    print("\nCleaned dataset info:")
    print(cleaned_dataframe.info())
    
    eda_dataframe = run_eda_pipeline()
    
    print("\nEDA dataset info:")
    print(eda_dataframe.info())
    
    featured_dataframe = run_feature_engineering_pipeline()
    
    print("\nFeatured dataset info:")
    print(featured_dataframe.info())

if __name__ == "__main__":
    main()