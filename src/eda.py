from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR


CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_crypto_data.csv"


def load_cleaned_data(file_path: Path = CLEANED_DATA_FILE) -> pd.DataFrame:
    """
    Load the cleaned crypto dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    return dataframe


def generate_descriptive_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate descriptive statistics for key numeric columns grouped by ticker.
    """
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    existing_columns = [column for column in numeric_columns if column in dataframe.columns]

    stats_dataframe = (
        dataframe.groupby("ticker")[existing_columns]
        .agg(["mean", "median", "std", "min", "max"])
        .round(4)
    )

    return stats_dataframe


def save_descriptive_statistics(stats_dataframe: pd.DataFrame) -> Path:
    """
    Save descriptive statistics to CSV.
    """
    output_path = TABLES_DIR / "descriptive_statistics.csv"
    stats_dataframe.to_csv(output_path)
    print(f"Saved descriptive statistics to {output_path}")
    return output_path


def create_missing_values_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a missing values summary table.
    """
    summary_dataframe = pd.DataFrame({
        "column": dataframe.columns,
        "missing_count": dataframe.isnull().sum().values,
        "missing_percentage": (dataframe.isnull().mean().values * 100).round(4),
    })
    return summary_dataframe


def save_missing_values_summary(summary_dataframe: pd.DataFrame) -> Path:
    """
    Save missing values summary to CSV.
    """
    output_path = TABLES_DIR / "missing_values_summary.csv"
    summary_dataframe.to_csv(output_path, index=False)
    print(f"Saved missing values summary to {output_path}")
    return output_path


def calculate_daily_returns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns for each ticker using adjusted close if available,
    otherwise close.
    """
    dataframe = dataframe.copy()
    price_column = "adj_close" if "adj_close" in dataframe.columns else "close"

    dataframe = dataframe.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    dataframe["daily_return"] = dataframe.groupby("ticker")[price_column].pct_change()

    return dataframe


def calculate_rolling_volatility(
    dataframe: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    Calculate rolling volatility based on daily returns.
    """
    dataframe = dataframe.copy()
    dataframe["rolling_volatility"] = (
        dataframe.groupby("ticker")["daily_return"]
        .transform(lambda group: group.rolling(window=window).std())
    )
    return dataframe


def create_price_pivot(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create pivot table of adjusted close prices for time-series analysis.
    """
    price_column = "adj_close" if "adj_close" in dataframe.columns else "close"

    pivot_dataframe = dataframe.pivot_table(
        index="date",
        columns="ticker",
        values=price_column,
        aggfunc="mean",
    )

    return pivot_dataframe


def create_returns_pivot(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create pivot table of daily returns for correlation analysis.
    """
    pivot_dataframe = dataframe.pivot_table(
        index="date",
        columns="ticker",
        values="daily_return",
        aggfunc="mean",
    )
    return pivot_dataframe




def save_correlation_matrix(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, Path]:
    """
    Compute and save the correlation matrix of daily returns.
    """
    returns_pivot = create_returns_pivot(dataframe)
    correlation_matrix = returns_pivot.corr().round(4)

    output_path = TABLES_DIR / "correlation_matrix.csv"
    correlation_matrix.to_csv(output_path)

    print(f"Saved correlation matrix to {output_path}")
    return correlation_matrix, output_path


def summarize_volatility(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize mean, max, and std of rolling volatility by ticker.
    """
    summary_dataframe = (
        dataframe.groupby("ticker")["rolling_volatility"]
        .agg(["mean", "max", "std"])
        .round(6)
        .reset_index()
    )
    return summary_dataframe


def save_volatility_summary(summary_dataframe: pd.DataFrame) -> Path:
    """
    Save volatility summary to CSV.
    """
    output_path = TABLES_DIR / "volatility_summary.csv"
    summary_dataframe.to_csv(output_path, index=False)
    print(f"Saved volatility summary to {output_path}")
    return output_path


def plot_normalized_price_trends_static(dataframe: pd.DataFrame) -> Path:
    """
    Static normalized price trends for report use.
    """
    pivot_dataframe = create_price_pivot(dataframe)
    normalized_dataframe = pivot_dataframe.div(pivot_dataframe.iloc[0]).mul(100)

    plt.figure(figsize=(14, 7))
    for column in normalized_dataframe.columns:
        plt.plot(normalized_dataframe.index, normalized_dataframe[column], label=column)

    plt.title("Normalized Crypto Asset Price Trends (Base = 100)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price Index")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "normalized_price_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved normalized price trends plot to {output_path}")
    return output_path


def plot_price_trends_subplots_static(dataframe: pd.DataFrame) -> Path:
    """
    Static price trends as subplots for report use.
    """
    tickers = sorted(dataframe["ticker"].unique())
    figure, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 18), sharex=True)
    axes = axes.flatten()

    price_column = "adj_close" if "adj_close" in dataframe.columns else "close"

    for index, ticker in enumerate(tickers):
        asset_data = dataframe[dataframe["ticker"] == ticker]
        axes[index].plot(asset_data["date"], asset_data[price_column])
        axes[index].set_title(ticker)
        axes[index].set_ylabel("Price")
        axes[index].grid(True)

    for index in range(len(tickers), len(axes)):
        figure.delaxes(axes[index])

    figure.suptitle("Crypto Asset Price Trends by Asset", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = FIGURES_DIR / "price_trends_subplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved price trend subplots to {output_path}")
    return output_path


def plot_daily_returns_subplots_static(dataframe: pd.DataFrame) -> Path:
    """
    Static daily return subplots for report use.
    """
    tickers = sorted(dataframe["ticker"].unique())
    figure, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 18), sharex=True)
    axes = axes.flatten()

    for index, ticker in enumerate(tickers):
        asset_data = dataframe[dataframe["ticker"] == ticker]
        axes[index].plot(asset_data["date"], asset_data["daily_return"])
        axes[index].set_title(ticker)
        axes[index].set_ylabel("Return")
        axes[index].grid(True)

    for index in range(len(tickers), len(axes)):
        figure.delaxes(axes[index])

    figure.suptitle("Daily Returns by Asset", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = FIGURES_DIR / "daily_returns_subplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved daily return subplots to {output_path}")
    return output_path


def plot_rolling_volatility_subplots_static(dataframe: pd.DataFrame) -> Path:
    """
    Static rolling volatility subplots for report use.
    """
    tickers = sorted(dataframe["ticker"].unique())
    figure, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 18), sharex=True)
    axes = axes.flatten()

    for index, ticker in enumerate(tickers):
        asset_data = dataframe[dataframe["ticker"] == ticker]
        axes[index].plot(asset_data["date"], asset_data["rolling_volatility"])
        axes[index].set_title(ticker)
        axes[index].set_ylabel("Volatility")
        axes[index].grid(True)

    for index in range(len(tickers), len(axes)):
        figure.delaxes(axes[index])

    figure.suptitle("Rolling Volatility by Asset", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = FIGURES_DIR / "rolling_volatility_subplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved rolling volatility subplots to {output_path}")
    return output_path


def plot_correlation_heatmap_static(dataframe: pd.DataFrame) -> Path:
    """
    Static correlation heatmap for report use.
    """
    returns_pivot = create_returns_pivot(dataframe)
    correlation_matrix = returns_pivot.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.title("Correlation Heatmap of Daily Returns")
    plt.tight_layout()

    output_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved correlation heatmap to {output_path}")
    return output_path


def plot_interactive_normalized_price_trends(dataframe: pd.DataFrame) -> Path:
    """
    Interactive normalized price trends saved as HTML.
    """
    price_column = "adj_close" if "adj_close" in dataframe.columns else "close"

    working_df = dataframe[["date", "ticker", price_column]].copy()
    working_df = working_df.sort_values(["ticker", "date"])

    working_df["normalized_price"] = (
        working_df.groupby("ticker")[price_column]
        .transform(lambda series: (series / series.iloc[0]) * 100)
    )

    figure = px.line(
        working_df,
        x="date",
        y="normalized_price",
        color="ticker",
        title="Interactive Normalized Crypto Price Trends (Base = 100)",
        labels={
            "date": "Date",
            "normalized_price": "Normalized Price Index",
            "ticker": "Asset",
        },
    )

    max_date = working_df["date"].max()
    min_date = max_date - pd.DateOffset(months=3)
    
    figure.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            range=[min_date, max_date],
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            ),
            rangeslider=dict(visible=True),
            type="date",
            title="Date Range",
            showspikes=True
        )
    )
    
    output_path = FIGURES_DIR / "interactive_normalized_price_trends.html"
    figure.write_html(str(output_path))
    print(f"Saved interactive normalized price trends to {output_path}")
    return output_path


def plot_interactive_rolling_volatility(dataframe: pd.DataFrame) -> Path:
    """
    Interactive rolling volatility chart saved as HTML.
    """
    figure = px.line(
        dataframe,
        x="date",
        y="rolling_volatility",
        color="ticker",
        title="Interactive Rolling Volatility of Crypto Assets",
        labels={
            "date": "Date",
            "rolling_volatility": "Rolling Volatility",
            "ticker": "Asset",
        },
    )

    max_date = dataframe["date"].max()
    min_date = max_date - pd.DateOffset(months=3)

    figure.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            range=[min_date, max_date],
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            ),
            rangeslider=dict(visible=True),
            type="date",
            title="Date Range",
            showspikes=True,
        ),
    )

    output_path = FIGURES_DIR / "interactive_rolling_volatility.html"
    figure.write_html(str(output_path))
    print(f"Saved interactive rolling volatility chart to {output_path}")
    return output_path


def plot_interactive_daily_returns(dataframe: pd.DataFrame) -> Path:
    """
    Interactive daily returns chart saved as HTML.
    """
    figure = px.line(
        dataframe,
        x="date",
        y="daily_return",
        color="ticker",
        title="Interactive Daily Returns of Crypto Assets",
        labels={
            "date": "Date",
            "daily_return": "Daily Return",
            "ticker": "Asset",
        },
    )

    max_date = dataframe["date"].max()
    min_date = max_date - pd.DateOffset(months=3)

    figure.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            range=[min_date, max_date],
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            ),
            rangeslider=dict(visible=True),
            type="date",
            title="Date Range",
            showspikes=True,
        ),
    )

    output_path = FIGURES_DIR / "interactive_daily_returns.html"
    figure.write_html(str(output_path))
    print(f"Saved interactive daily returns chart to {output_path}")
    return output_path


def plot_interactive_correlation_heatmap(dataframe: pd.DataFrame) -> Path:
    """
    Interactive correlation heatmap saved as HTML.
    """
    returns_pivot = create_returns_pivot(dataframe)
    correlation_matrix = returns_pivot.corr().round(4)

    figure = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            text=correlation_matrix.values,
            texttemplate="%{text}",
            hoverongaps=False,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )

    figure.update_layout(
        title="Interactive Correlation Heatmap of Daily Returns",
        template="plotly_white",
        hovermode="closest",
        xaxis=dict(title="Asset"),
        yaxis=dict(title="Asset"),
    )

    output_path = FIGURES_DIR / "interactive_correlation_heatmap.html"
    figure.write_html(str(output_path))
    print(f"Saved interactive correlation heatmap to {output_path}")
    return output_path


def run_eda_pipeline() -> pd.DataFrame:
    """
    Run the EDA pipeline and save required static and interactive outputs.
    """
    print("Loading cleaned dataset for EDA...")
    dataframe = load_cleaned_data()

    print("Generating descriptive statistics...")
    stats_dataframe = generate_descriptive_statistics(dataframe)
    save_descriptive_statistics(stats_dataframe)

    print("Generating missing values summary...")
    missing_values_summary = create_missing_values_summary(dataframe)
    save_missing_values_summary(missing_values_summary)

    print("Calculating daily returns...")
    dataframe = calculate_daily_returns(dataframe)

    print("Calculating rolling volatility...")
    dataframe = calculate_rolling_volatility(dataframe, window=14)

    print("Saving correlation matrix...")
    save_correlation_matrix(dataframe)

    print("Saving volatility summary...")
    volatility_summary = summarize_volatility(dataframe)
    save_volatility_summary(volatility_summary)

    print("Creating static report-friendly visualizations...")
    plot_normalized_price_trends_static(dataframe)
    plot_price_trends_subplots_static(dataframe)
    plot_daily_returns_subplots_static(dataframe)
    plot_rolling_volatility_subplots_static(dataframe)
    plot_correlation_heatmap_static(dataframe)

    print("Creating interactive Plotly visualizations...")
    plot_interactive_normalized_price_trends(dataframe)
    plot_interactive_rolling_volatility(dataframe)
    plot_interactive_daily_returns(dataframe)
    plot_interactive_correlation_heatmap(dataframe)

    output_file = PROCESSED_DATA_DIR / "eda_enriched_crypto_data.csv"
    dataframe.to_csv(output_file, index=False)
    print(f"Saved EDA-enriched dataset to {output_file}")

    print("EDA pipeline completed successfully.")
    return dataframe


if __name__ == "__main__":
    run_eda_pipeline()