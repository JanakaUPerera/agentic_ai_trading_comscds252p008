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
DRAWDOWN_CURVE_FILE = FIGURES_DIR / "drawdown_curve.png"
RETURN_DISTRIBUTION_FILE = FIGURES_DIR / "return_distribution.png"
WIN_LOSS_BY_PAIR_FILE = FIGURES_DIR / "win_loss_by_pair.png"

def load_risk_managed_data(file_path: Path = RISK_INPUT_FILE) -> pd.DataFrame:
    """
    Load the risk-managed dataset for backtesting.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Risk-managed dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    return dataframe


def map_execution_position(row: pd.Series) -> float:
    decision = str(row.get("risk_adjusted_decision", "Hold")).strip()
    score = float(row.get("decision_score", 0))
    overall_risk = str(row.get("overall_risk_level", "Low")).strip()
    trend_signal = int(row.get("trend_signal", 0))

    if decision == "Hold":
        return 0.0

    if abs(score) < 2:
        return 0.0
    
    if decision == "Buy":
        if score >= 6:
            position = 1.78
        elif score >= 5:
            position = 1.30
        elif score >= 4:
            position = 0.80
        elif score >= 2:
            position = 0.60

        atr_pct = float(row.get("atr_pct", 0))
        if atr_pct > 0:
            position *= 1 / (1 + atr_pct)
        
        if overall_risk == "High" and trend_signal != 1:
            position *= 0.50
        elif overall_risk == "Medium":
            position *= 0.75

        if trend_signal == 1 and score >= 4:
            position *= 2.5

        return min(position, 2.0)

    if decision == "Sell":
        if score <= -6:
            position = -1.0
        elif score <= -4:
            position = -0.75
        elif score <= -2:
            position = -0.50
        else:
            position = -0.25

        if overall_risk == "High":
            position *= 1.00
        elif overall_risk == "Medium":
            position *= 0.85

        return max(position, -1.0)

    return 0.0


def apply_trend_holding(position_series: pd.Series, df_slice: pd.DataFrame) -> pd.Series:
    positions = position_series.to_numpy(copy=True)

    for i in range(1, len(positions)):
        prev_pos = positions[i - 1]
        trend_signal = df_slice.iloc[i]["trend_signal"]
        # Exit if momentum weakens
        return_7d = df_slice.iloc[i]["return_7d"]

        # If already in a long position and trend is still bullish → HOLD
        if prev_pos > 0:
            if trend_signal == 1 and return_7d > -0.03:
                positions[i] = prev_pos
            else:
                positions[i] = 0.0  # Early exit if momentum weakens or trend reverses

        # If already in a short position and trend is bearish → HOLD
        elif prev_pos < 0 and trend_signal == -1:
            positions[i] = prev_pos

    return pd.Series(positions, index=position_series.index)


def prepare_backtest_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare required columns for backtesting.
    """
    dataframe = dataframe.copy()
    dataframe = dataframe.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    dataframe["position"] = dataframe.apply(map_execution_position, axis=1)
    # --- Trend Holding Logic ---
    dataframe["position"] = dataframe.groupby("ticker", group_keys=False)["position"].transform(
        lambda group: apply_trend_holding(group, dataframe.loc[group.index])
    )

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
    
    df = portfolio_dataframe.copy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["strategy_growth"], label="Strategy")
    plt.plot(df["date"], df["benchmark_growth"], label="Benchmark")
    plt.title("Strategy vs Benchmark Growth")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.fill_between(
        df["date"],
        df["strategy_growth"],
        df["benchmark_growth"],
        where=(df["strategy_growth"] < df["benchmark_growth"]),
        alpha=0.2,
    )

    plt.savefig(BENCHMARK_COMPARISON_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved strategy vs benchmark plot to {BENCHMARK_COMPARISON_FILE}")
    return BENCHMARK_COMPARISON_FILE


def calculate_drawdown_series(growth_series: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from cumulative growth curve.
    """
    running_max = growth_series.cummax()
    drawdown = (growth_series - running_max) / running_max
    return drawdown


def plot_drawdown_curve(portfolio_dataframe: pd.DataFrame) -> Path:
    """
    Plot drawdown curves for strategy and benchmark.
    """
    df = portfolio_dataframe.copy()
    df["strategy_drawdown"] = calculate_drawdown_series(df["strategy_growth"])
    df["benchmark_drawdown"] = calculate_drawdown_series(df["benchmark_growth"])

    plt.figure(figsize=(12, 6))
    plt.plot(
        df["date"], 
        df["strategy_drawdown"], 
        label="Strategy Drawdown", 
        linewidth=2
    )
    plt.plot(
        df["date"],
        df["benchmark_drawdown"],
        label="Benchmark Drawdown",
        linewidth=2,
    )

    plt.title("Drawdown Curve")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(DRAWDOWN_CURVE_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved drawdown curve to {DRAWDOWN_CURVE_FILE}")
    return DRAWDOWN_CURVE_FILE


def plot_return_distribution(portfolio_dataframe: pd.DataFrame) -> Path:
    """
    Plot return distribution of strategy and benchmark daily returns.
    """
    plt.figure(figsize=(12, 6))

    strategy_returns = portfolio_dataframe["strategy_return"].dropna()
    benchmark_returns = portfolio_dataframe["benchmark_return"].dropna()

    plt.hist(strategy_returns, bins=50, alpha=0.6, label="Strategy Returns", density=True)
    plt.hist(benchmark_returns, bins=50, alpha=0.6, label="Benchmark Returns", density=True)

    plt.title("Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(RETURN_DISTRIBUTION_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved return distribution plot to {RETURN_DISTRIBUTION_FILE}")
    return RETURN_DISTRIBUTION_FILE


def create_win_loss_by_pair(asset_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create win/loss summary by crypto pair.
    A win is counted when net strategy return > 0 and a loss when < 0.
    """
    summary = (
        asset_dataframe.groupby("ticker")
        .agg(
            win_days=("net_strategy_return", lambda values: (values > 0).sum()),
            loss_days=("net_strategy_return", lambda values: (values < 0).sum()),
        )
        .reset_index()
    )

    return summary


def plot_win_loss_by_pair(asset_dataframe: pd.DataFrame) -> Path:
    """
    Plot win/loss days by crypto pair.
    """
    summary = create_win_loss_by_pair(asset_dataframe)

    x = np.arange(len(summary))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, summary["win_days"], width=width, label="Win Days")
    plt.bar(x + width / 2, summary["loss_days"], width=width, label="Loss Days")

    plt.xticks(x, summary["ticker"], rotation=45)
    plt.title("Win/Loss by Pair")
    plt.xlabel("Crypto Pair")
    plt.ylabel("Number of Days")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()

    plt.savefig(WIN_LOSS_BY_PAIR_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved win/loss by pair plot to {WIN_LOSS_BY_PAIR_FILE}")
    return WIN_LOSS_BY_PAIR_FILE


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
    plot_drawdown_curve(portfolio_dataframe)
    plot_return_distribution(portfolio_dataframe)
    plot_win_loss_by_pair(asset_dataframe)

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