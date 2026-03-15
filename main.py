from src.config import CRYPTO_ASSETS, END_DATE, START_DATE
from src.fetch_data import fetch_crypto_data
from src.preprocess_data import preprocess_crypto_data
from src.eda import run_eda_pipeline
from src.features import run_feature_engineering_pipeline
from src.clustering import run_clustering_pipeline
from src.retrieve_news_info import run_news_info_retrieval_pipeline
from src.decision_engine import run_decision_engine_pipeline
from src.manage_risk import run_risk_management_pipeline
from src.backtest import run_backtesting_pipeline


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
    
    clustered_dataframe = run_clustering_pipeline()
    print("\nClustered dataset info:")
    print(clustered_dataframe.info())
    
    news_dataframe = run_news_info_retrieval_pipeline()
    print("\nNews information dataset info:")
    print(news_dataframe.info())
    
    decision_dataframe = run_decision_engine_pipeline()
    print("\nDecision dataset info:")
    print(decision_dataframe.info())
    
    risk_dataframe = run_risk_management_pipeline()
    print("\Risk dataset info:")
    print(risk_dataframe.info())
    
    asset_backtest_dataframe, portfolio_backtest_dataframe = run_backtesting_pipeline()
    print("\Asset backtest dataset info:")
    print(asset_backtest_dataframe.info())
    print("\Portfolio backtest dataset info:")
    print(portfolio_backtest_dataframe.info())

if __name__ == "__main__":
    main()