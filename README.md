# Agentic AI Workflow for Automated Cryptocurrency Trading

## Project Overview
This project is developed for Portfolio CW 2 under the finance domain option: **Agentic AI Workflow for Automated Trading**.

The aim of this project is to design and analyze an Agentic AI-based cryptocurrency trading workflow system that can autonomously:
- collect financial market data
- perform exploratory data analysis
- retrieve external financial information
- generate buy/sell/hold decisions
- apply risk controls
- evaluate trading performance through backtesting
- integrate cloud services using AWS or Azure

## Problem Statement
Cryptocurrency traders face difficulty making consistent trading decisions due to high market volatility, rapidly changing trends, and the need to combine technical indicators with external market information. This project addresses that problem by developing an Agentic AI workflow that supports automated decision-making using market analysis, information retrieval, decision rules, and risk management.

## Selected Assets
The project uses the following 10 crypto assets:
- BTC-USD
- ETH-USD
- BNB-USD
- SOL-USD
- XRP-USD
- ADA-USD
- DOGE-USD
- TRX-USD
- AVAX-USD
- LINK-USD

## Agentic Workflow
1. Collect historical cryptocurrency market data
2. Clean and preprocess datasets
3. Perform exploratory data analysis
4. Generate trading indicators
5. Cluster assets using K-Means
6. Retrieve cryptocurrency news from RSS feeds
7. Classify news sentiment
8. Generate Buy/Sell/Hold decisions
9. Apply risk management rules
10. Backtest the trading strategy
11. Perform LLM-based analytical interpretation using Groq API
12. Generate interpretation report
13. Package outputs and report into ZIP archive
14. Upload ZIP bundle to AWS S3
15. Generate download link
16. Send link via email

## Project Alignment
This project is structured to fully address:

1. **Data Collection**
   - Collect cryptocurrency market data using APIs (Yahoo Finance).
   - Retrieve real-time crypto news headlines using RSS feeds from multiple industry sources.
   - Store raw datasets locally during processing and upload them to cloud storage for persistence.

2. **Exploratory Data Analysis**
   - Data cleaning and preprocessing.
   - Handling missing values and anomalies.
   - Descriptive statistics and time-series visualization.
   - Correlation analysis and volatility estimation.
   - Asset clustering using K-Means.
   - Feature engineering including technical indicators such as Moving Averages, RSI, and MACD.

3. **Problem Solving**
   - Define a trading-related problem involving automated analysis of cryptocurrency markets.
   - Design an Agentic AI workflow architecture including:
     - Market Analysis Module
     - Information Retrieval Module (RSS-based crypto news retrieval)
     - Decision Engine (Buy / Sell / Hold signals)
     - Risk Management Module
   - Evaluate the trading workflow using historical backtesting metrics.

4. **Cloud Integration**
   - AWS S3 used for storing raw datasets, analytical outputs, figures, and generated report bundles.
   - AWS EC2 used as the computation environment where the full trading workflow pipeline is executed.
   - AWS RDS used for storing structured analytical summaries such as decision summaries, risk summaries, backtesting metrics, and news sentiment summaries.

5. **Automated Analytical Interpretation**
   - Analytical outputs from the workflow are processed by a Large Language Model agent.
   - The system uses the **Groq API with the model `openai/gpt-oss-120b`** to generate an analytical interpretation of the results.
   - The LLM analyzes trading signals, market patterns, clustering behaviour, news sentiment impact, and strategy performance.
   - A structured interpretation report is generated automatically.

6. **Automated Result Packaging and Distribution**
   - Essential analytical outputs, figures, tables, and the LLM interpretation report are bundled into a ZIP archive.
   - The ZIP bundle is uploaded to AWS S3.
   - A downloadable link for the uploaded bundle is generated.
   - The link is automatically sent to a specified email address for convenient access to the analysis results.

7. **Git Collaboration**
   - All development and experimentation are managed through a Git repository with meaningful commits documenting each stage of the workflow implementation.

8. **Documentation**
   - Architecture diagrams illustrating the Agentic AI workflow.
   ![Agentic AI Trading Workflow Architecture](docs/Agentic%20AI%20Trading%20Workflow%20Architecture.png)
   - Cloud infrastructure setup documentation.
   [aws_setup.md](docs/aws_setup.md)
   - Results and analytical insights derived from the workflow.
   [methodlogy.md](docs/methodlogy.md)
   
9. **Final Report**
   - Overall report with discussion of limitations and potential business/financial impact.
   [agentic_ai_trading_report_comscds252p008.pdf](docs/agentic_ai_trading_report_comscds252p008.pdf)

## Project Structure
```txt
📦agentic_ai_trading_comscds252p008
 ┣ 📂data
 ┃ ┣ 📂outputs
 ┃ ┃ ┣ 📂final_analysis_bundle
 ┃ ┃ ┃ ┣ 📂figures
 ┃ ┃ ┃ ┃ ┣ 📜crypto_asset_clusters.png
 ┃ ┃ ┃ ┃ ┣ 📜portfolio_growth.png
 ┃ ┃ ┃ ┃ ┗ 📜strategy_vs_benchmark.png
 ┃ ┃ ┃ ┣ 📂outputs
 ┃ ┃ ┃ ┃ ┣ 📜crypto_news_sentiment_summary.csv
 ┃ ┃ ┃ ┃ ┣ 📜decision_summary.csv
 ┃ ┃ ┃ ┃ ┣ 📜final_interpretation.md
 ┃ ┃ ┃ ┃ ┣ 📜final_interpretation_debug_prompt.txt
 ┃ ┃ ┃ ┃ ┣ 📜portfolio_daily_returns.csv
 ┃ ┃ ┃ ┃ ┗ 📜risk_summary.csv
 ┃ ┃ ┃ ┣ 📂tables
 ┃ ┃ ┃ ┃ ┣ 📜backtest_summary.csv
 ┃ ┃ ┃ ┃ ┣ 📜cluster_summary.csv
 ┃ ┃ ┃ ┃ ┣ 📜correlation_matrix.csv
 ┃ ┃ ┃ ┃ ┣ 📜descriptive_statistics.csv
 ┃ ┃ ┃ ┃ ┣ 📜missing_values_summary.csv
 ┃ ┃ ┃ ┃ ┗ 📜volatility_summary.csv
 ┃ ┃ ┃ ┗ 📜README_results.txt
 ┃ ┃ ┣ 📜backtest_results.csv
 ┃ ┃ ┣ 📜crypto_news_headlines.csv
 ┃ ┃ ┣ 📜crypto_news_sentiment_summary.csv
 ┃ ┃ ┣ 📜decision_summary.csv
 ┃ ┃ ┣ 📜final_analysis_bundle.zip
 ┃ ┃ ┣ 📜final_interpretation.md
 ┃ ┃ ┣ 📜final_interpretation_debug_prompt.txt
 ┃ ┃ ┣ 📜final_interpretation.md
 ┃ ┃ ┣ 📜market_data_with_decisions.csv
 ┃ ┃ ┣ 📜market_data_with_news_signal.csv
 ┃ ┃ ┣ 📜market_data_with_risk_controls.csv
 ┃ ┃ ┣ 📜portfolio_daily_returns.csv
 ┃ ┃ ┗ 📜risk_summary.csv
 ┃ ┣ 📂processed
 ┃ ┃ ┣ 📜cleaned_crypto_data.csv
 ┃ ┃ ┣ 📜clustered_crypto_data.csv
 ┃ ┃ ┣ 📜eda_enriched_crypto_data.csv
 ┃ ┃ ┣ 📜featured_crypto_data.csv
 ┃ ┃ ┗ 📜signal_summary.csv
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📜ada_usd.csv
 ┃ ┃ ┣ 📜avax_usd.csv
 ┃ ┃ ┣ 📜bnb_usd.csv
 ┃ ┃ ┣ 📜btc_usd.csv
 ┃ ┃ ┣ 📜combined_crypto_data.csv
 ┃ ┃ ┣ 📜doge_usd.csv
 ┃ ┃ ┣ 📜eth_usd.csv
 ┃ ┃ ┣ 📜link_usd.csv
 ┃ ┃ ┣ 📜sol_usd.csv
 ┃ ┃ ┣ 📜trx_usd.csv
 ┃ ┃ ┗ 📜xrp_usd.csv
 ┣ 📂docs
 ┃ ┣ 📜Agentic AI Trading Workflow Architecture.png
 ┃ ┣ 📜aws_setup.md
 ┃ ┗ 📜methodlogy.md
 ┣ 📂notebooks
 ┣ 📂reports
 ┃ ┣ 📂figures
 ┃ ┃ ┣ 📜correlation_heatmap.png
 ┃ ┃ ┣ 📜crypto_asset_clusters.png
 ┃ ┃ ┣ 📜daily_returns_subplots.png
 ┃ ┃ ┣ 📜interactive_correlation_heatmap.html
 ┃ ┃ ┣ 📜interactive_daily_returns.html
 ┃ ┃ ┣ 📜interactive_normalized_price_trends.html
 ┃ ┃ ┣ 📜interactive_rolling_volatility.html
 ┃ ┃ ┣ 📜normalized_price_trends.png
 ┃ ┃ ┣ 📜portfolio_growth.png
 ┃ ┃ ┣ 📜price_trends_subplots.png
 ┃ ┃ ┣ 📜rolling_volatility_subplots.png
 ┃ ┃ ┗ 📜strategy_vs_benchmark.png
 ┃ ┗ 📂tables
 ┃ ┃ ┣ 📜asset_clusters.csv
 ┃ ┃ ┣ 📜backtest_summary.csv
 ┃ ┃ ┣ 📜cluster_summary.csv
 ┃ ┃ ┣ 📜correlation_matrix.csv
 ┃ ┃ ┣ 📜descriptive_statistics.csv
 ┃ ┃ ┣ 📜missing_values_summary.csv
 ┃ ┃ ┗ 📜volatility_summary.csv
 ┣ 📜.env
 ┣ 📜main.py
 ┣ 📜README.md
 ┗ 📜requirements.txt
 ```