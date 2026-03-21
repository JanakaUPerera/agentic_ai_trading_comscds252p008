# Methodology

## 1. Data Collection
Historical daily cryptocurrency market data was collected using Yahoo Finance through the `yfinance` library. Ten cryptocurrency assets were selected to provide variation in volatility, market capitalization, and market behavior.

Real cryptocurrency news headlines were collected from multiple RSS feeds including CoinDesk, Cointelegraph, Decrypt, CryptoNews, NewsBTC, Bitcoin Magazine, The Block, and AMBCrypto.

## 2. Data Preprocessing
The market dataset was cleaned by:
- converting dates and numeric columns to correct types
- removing duplicate rows
- handling missing values
- removing invalid rows such as non-positive prices or negative volume

## 3. Exploratory Data Analysis
EDA included:
- descriptive statistics
- missing value assessment
- price trend visualization
- daily return calculation
- rolling volatility analysis
- return correlation analysis

## 4. Feature Engineering
Trading-oriented features were created, including:
- moving averages
- exponential moving averages
- RSI
- MACD
- short-horizon returns
- volume change

Rule-based technical signals were generated as Buy, Sell, or Hold.

## 5. Clustering
K-Means clustering was applied to group crypto assets by behavioral similarity using aggregated asset-level indicators such as returns, volatility, RSI, MACD, and momentum.

## 6. Information Retrieval
A news information retrieval module collected real crypto news headlines and classified them into:
- Bullish
- Bearish
- Regulatory
- Security
- Neutral

The module also applied source weighting, recency weighting, and confidence weighting to derive asset-level news signals.

## 7. Decision Engine
The decision engine combined:
- technical signals
- market impact from news
- asset-level news bias
- weighted directional news score

This produced final Buy, Sell, or Hold decisions.

## 8. Risk Management
Risk controls were added using:
- volatility risk
- news-driven risk
- position-level risk

High-risk situations triggered overrides that downgraded aggressive trades.

## 9. Backtesting
The risk-adjusted strategy was backtested using lagged positions to avoid look-ahead bias. Performance was evaluated using:
- cumulative growth
- benchmark comparison
- drawdown
- Sharpe ratio
- trade count
- win rate

## 10. Cloud Integration
The workflow was executed on AWS EC2. Data and outputs were uploaded to S3, while structured summary tables were uploaded to RDS PostgreSQL.

## 11. LLM-Based Analytical Interpretation
After generating the analytical outputs, a Large Language Model agent performs automated interpretation.
The system uses the **Groq API** with the model:

### Input to the LLM
The model receives:
- descriptive statistics
- clustering results
- decision summaries
- news sentiment summaries
- backtesting metrics
- portfolio growth curves

### LLM Tasks
The LLM produces a structured analytical report containing:
- market behaviour analysis
- asset clustering interpretation
- news sentiment impact on trading signals
- evaluation of buy/sell signal distribution
- explanation of backtesting performance
- risk insights and strategy limitations

### Output
The interpretation is saved as: 
- final_interpretation_debug_prompt.txt
- final_interpretation.md

## 12. Result Packaging
A bundle is created containing:
- analytical figures
- summary tables
- decision outputs
- risk summaries
- LLM interpretation report
These files are packaged into a ZIP archive.

## 13. Cloud Distribution
The ZIP archive (final_analysis_bundle.zip) is uploaded to **AWS S3** and a download link is generated. The link is then sent to the specified email recipient.
This allows automated delivery of analytical trading insights.