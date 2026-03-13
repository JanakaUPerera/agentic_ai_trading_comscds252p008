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

## Project Alignment
This project is structured to fully address:

1. **Data Collection**
   - Collect crypto market data using APIs
   - Store raw data locally and in cloud storage

2. **Exploratory Data Analysis**
   - Data cleaning and preprocessing
   - Missing value and anomaly handling
   - Statistical analysis and visualizations
   - Correlation, volatility, clustering, and feature engineering

3. **Problem Solving**
   - Define a trading-related problem
   - Design an Agentic AI workflow architecture including:
     - Market Analysis Module
     - Information Retrieval Module
     - Decision Engine
     - Risk Management Module
   - Backtesting and performance evaluation

4. **Cloud Integration**
   - S3 or Blob Storage for raw data
   - EC2 or VM for computation
   - RDS or SQL Database for structured storage

5. **Git Collaboration**
   - All development managed through Git with meaningful commits

6. **Final Reporting**
   - Architecture diagrams
   - Cloud setup
   - Results and insights
   - Limitations and business impact

## Project Structure
```bash
agentic-ai-trading-workflow/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── eda.py
│   ├── clustering.py
│   ├── market_analysis.py
│   ├── info_retrieval.py
│   ├── decision_engine.py
│   ├── risk_management.py
│   ├── backtesting.py
│   ├── cloud_s3.py
│   ├── cloud_rds.py
│   └── utils.py
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── docs/
├── tests/
├── requirements.txt
├── README.md
└── main.py