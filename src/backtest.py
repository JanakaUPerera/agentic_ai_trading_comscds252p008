from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, OUTPUTS_DIR, TABLES_DIR


RISK_INPUT_FILE = OUTPUTS_DIR / "market_data_with_risk_controls.csv"
BACKTEST_OUTPUT_FILE = OUTPUTS_DIR / "backtest_results.csv"
BACKTEST_SUMMARY_FILE = TABLES_DIR / "backtest_summary.csv"
PORTFOLIO_CURVE_FILE = FIGURES_DIR / "portfolio_growth.png"
BENCHMARK_COMPARISON_FILE = FIGURES_DIR / "strategy_vs_benchmark.png"


def load_risk_managed_data(file_path: Path = RISK_INPUT_FILE) -> pd.DataFrame:
    """
    Load the risk-managed dataset for backtesting.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Risk-managed dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    return dataframe


def map_decision_to_position(decision: str) -> int:
    """
    Map decision to trading position.
    Buy = 1, Hold = 0, Sell = -1
    """
    decision_map = {
        "Buy": 1,
        "Hold": 0,
        "Sell": -1,
    }
    return decision_map.get(str(decision).strip(), 0)


def prepare_backtest_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare required columns for backtesting.
    """
    dataframe = dataframe.copy()
    dataframe = dataframe.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    dataframe["position"] = dataframe["risk_adjusted_decision"].apply(map_decision_to_position)

    price_column = "adj_close" if "adj_close" in dataframe.columns else "close"
    dataframe["asset_return"] = dataframe.groupby("ticker")[price_column].pct_change()

    # Shift the signal forward by one day to avoid look-ahead bias
    dataframe["lagged_position"] = dataframe.groupby("ticker")["position"].shift(1).fillna(0)

    dataframe["strategy_return"] = dataframe["lagged_position"] * dataframe["asset_return"]
    dataframe["benchmark_return"] = dataframe["asset_return"]

    return dataframe


def add_transaction_costs(
    dataframe: pd.DataFrame,
    cost_per_trade: float = 0.001,
) -> pd.DataFrame:
    """
    Apply simple transaction costs when position changes.
    """
    dataframe = dataframe.copy()

    dataframe["position_change"] = (
        dataframe.groupby("ticker")["lagged_position"]
        .diff()
        .abs()
        .fillna(0)
    )

    dataframe["transaction_cost"] = dataframe["position_change"] * cost_per_trade
    dataframe["net_strategy_return"] = dataframe["strategy_return"] - dataframe["transaction_cost"]

    return dataframe


def create_portfolio_returns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate asset-level returns into equal-weight portfolio-level daily returns.
    """
    portfolio = (
        dataframe.groupby("date")
        .agg(
            strategy_return=("net_strategy_return", "mean"),
            benchmark_return=("benchmark_return", "mean"),
        )
        .reset_index()
        .sort_values(by="date")
    )

    portfolio["strategy_growth"] = (1 + portfolio["strategy_return"].fillna(0)).cumprod()
    portfolio["benchmark_growth"] = (1 + portfolio["benchmark_return"].fillna(0)).cumprod()

    return portfolio


def calculate_max_drawdown(growth_series: pd.Series) -> float:
    """
    Calculate maximum drawdown from a cumulative growth curve.
    """
    running_max = growth_series.cummax()
    drawdown = (growth_series - running_max) / running_max
    return float(drawdown.min())


def calculate_sharpe_ratio(return_series: pd.Series) -> float:
    """
    Calculate a simple daily Sharpe ratio annualized to 252 trading days.
    """
    cleaned = return_series.dropna()
    if cleaned.empty or cleaned.std() == 0:
        return 0.0

    sharpe = (cleaned.mean() / cleaned.std()) * np.sqrt(252)
    return float(sharpe)


def summarize_backtest(
    asset_dataframe: pd.DataFrame,
    portfolio_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create summary performance metrics for the strategy and benchmark.
    """
    strategy_returns = portfolio_dataframe["strategy_return"].dropna()
    benchmark_returns = portfolio_dataframe["benchmark_return"].dropna()

    final_strategy_growth = portfolio_dataframe["strategy_growth"].iloc[-1]
    final_benchmark_growth = portfolio_dataframe["benchmark_growth"].iloc[-1]

    total_trades = int((asset_dataframe["position_change"] > 0).sum())
    active_days = int((asset_dataframe["lagged_position"] != 0).sum())

    winning_days = int((portfolio_dataframe["strategy_return"] > 0).sum())
    losing_days = int((portfolio_dataframe["strategy_return"] < 0).sum())
    total_signal_days = winning_days + losing_days
    win_rate = (winning_days / total_signal_days) if total_signal_days > 0 else 0.0

    summary = pd.DataFrame([
        {
            "metric": "Final Strategy Growth",
            "value": round(float(final_strategy_growth), 6),
        },
        {
            "metric": "Final Benchmark Growth",
            "value": round(float(final_benchmark_growth), 6),
        },
        {
            "metric": "Total Strategy Return",
            "value": round(float(final_strategy_growth - 1), 6),
        },
        {
            "metric": "Total Benchmark Return",
            "value": round(float(final_benchmark_growth - 1), 6),
        },
        {
            "metric": "Strategy Max Drawdown",
            "value": round(calculate_max_drawdown(portfolio_dataframe["strategy_growth"]), 6),
        },
        {
            "metric": "Benchmark Max Drawdown",
            "value": round(calculate_max_drawdown(portfolio_dataframe["benchmark_growth"]), 6),
        },
        {
            "metric": "Strategy Sharpe Ratio",
            "value": round(calculate_sharpe_ratio(strategy_returns), 6),
        },
        {
            "metric": "Benchmark Sharpe Ratio",
            "value": round(calculate_sharpe_ratio(benchmark_returns), 6),
        },
        {
            "metric": "Average Daily Strategy Return",
            "value": round(float(strategy_returns.mean()), 6) if not strategy_returns.empty else 0.0,
        },
        {
            "metric": "Average Daily Benchmark Return",
            "value": round(float(benchmark_returns.mean()), 6) if not benchmark_returns.empty else 0.0,
        },
        {
            "metric": "Total Trades",
            "value": total_trades,
        },
        {
            "metric": "Active Position Days",
            "value": active_days,
        },
        {
            "metric": "Winning Days",
            "value": winning_days,
        },
        {
            "metric": "Losing Days",
            "value": losing_days,
        },
        {
            "metric": "Win Rate",
            "value": round(float(win_rate), 6),
        },
    ])

    return summary


def save_backtest_summary(summary_dataframe: pd.DataFrame) -> Path:
    """
    Save backtest summary table.
    """
    summary_dataframe.to_csv(BACKTEST_SUMMARY_FILE, index=False)
    print(f"Saved backtest summary to {BACKTEST_SUMMARY_FILE}")
    return BACKTEST_SUMMARY_FILE


def save_backtest_results(dataframe: pd.DataFrame) -> Path:
    """
    Save detailed backtest data.
    """
    dataframe.to_csv(BACKTEST_OUTPUT_FILE, index=False)
    print(f"Saved detailed backtest results to {BACKTEST_OUTPUT_FILE}")
    return BACKTEST_OUTPUT_FILE


def plot_portfolio_growth(portfolio_dataframe: pd.DataFrame) -> Path:
    """
    Plot cumulative strategy growth.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_dataframe["date"], portfolio_dataframe["strategy_growth"])
    plt.title("Portfolio Growth from Risk-Adjusted Strategy")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(PORTFOLIO_CURVE_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved portfolio growth plot to {PORTFOLIO_CURVE_FILE}")
    return PORTFOLIO_CURVE_FILE


def plot_strategy_vs_benchmark(portfolio_dataframe: pd.DataFrame) -> Path:
    """
    Plot strategy growth versus benchmark growth.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_dataframe["date"], portfolio_dataframe["strategy_growth"], label="Strategy")
    plt.plot(portfolio_dataframe["date"], portfolio_dataframe["benchmark_growth"], label="Benchmark")
    plt.title("Strategy vs Benchmark Growth")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(BENCHMARK_COMPARISON_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved strategy vs benchmark plot to {BENCHMARK_COMPARISON_FILE}")
    return BENCHMARK_COMPARISON_FILE


def run_backtesting_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete backtesting pipeline.
    """
    print("Loading risk-managed dataset...")
    asset_dataframe = load_risk_managed_data()

    print("Preparing backtest data...")
    asset_dataframe = prepare_backtest_data(asset_dataframe)

    print("Applying transaction costs...")
    asset_dataframe = add_transaction_costs(asset_dataframe, cost_per_trade=0.001)

    print("Creating equal-weight portfolio returns...")
    portfolio_dataframe = create_portfolio_returns(asset_dataframe)

    print("Saving detailed backtest results...")
    save_backtest_results(asset_dataframe)

    print("Generating performance summary...")
    summary_dataframe = summarize_backtest(asset_dataframe, portfolio_dataframe)
    save_backtest_summary(summary_dataframe)

    print("Creating backtesting plots...")
    plot_portfolio_growth(portfolio_dataframe)
    plot_strategy_vs_benchmark(portfolio_dataframe)

    portfolio_output_file = OUTPUTS_DIR / "portfolio_daily_returns.csv"
    portfolio_dataframe.to_csv(portfolio_output_file, index=False)
    print(f"Saved portfolio daily returns to {portfolio_output_file}")

    print("Backtesting pipeline completed successfully.")
    return asset_dataframe, portfolio_dataframe


if __name__ == "__main__":
    try:
        run_backtesting_pipeline()
    except Exception as e:
        print(f"Error running backtest pipeline: {e}")