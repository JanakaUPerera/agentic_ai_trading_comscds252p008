from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR


FEATURED_DATA_FILE = PROCESSED_DATA_DIR / "featured_crypto_data.csv"
CLUSTERED_OUTPUT_FILE = PROCESSED_DATA_DIR / "clustered_crypto_data.csv"


def load_featured_data(file_path: Path = FEATURED_DATA_FILE) -> pd.DataFrame:
    """
    Load the feature-engineered crypto dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Featured dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    return dataframe


def build_asset_level_feature_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily features into asset-level features for clustering.
    Each ticker gets one summary row.
    """
    required_columns = [
        "daily_return",
        "rolling_volatility",
        "rsi_14",
        "macd",
        "macd_diff",
        "return_7d",
        "return_14d",
        "volume_change",
    ]

    existing_columns = [column for column in required_columns if column in dataframe.columns]
    if not existing_columns:
        raise ValueError("No required feature columns found for clustering.")

    asset_features = (
        dataframe.groupby("ticker")[existing_columns]
        .agg(["mean", "std"])
    )

    asset_features.columns = [
        f"{column}_{statistic}" for column, statistic in asset_features.columns
    ]
    asset_features = asset_features.reset_index()

    return asset_features


def prepare_clustering_matrix(
    asset_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the numeric feature matrix and keep ticker labels separately.
    """
    feature_matrix = asset_features.drop(columns=["ticker"]).copy()
    feature_matrix = feature_matrix.fillna(feature_matrix.mean(numeric_only=True))

    return asset_features[["ticker"]].copy(), feature_matrix


def scale_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numeric features for clustering.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(feature_matrix)

    scaled_dataframe = pd.DataFrame(
        scaled_array,
        columns=feature_matrix.columns,
        index=feature_matrix.index,
    )
    return scaled_dataframe


def run_kmeans_clustering(
    scaled_features: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> KMeans:
    """
    Fit K-Means clustering model.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    model.fit(scaled_features)
    return model


def attach_cluster_labels(
    asset_labels: pd.DataFrame,
    asset_features: pd.DataFrame,
    model: KMeans,
) -> pd.DataFrame:
    """
    Attach cluster labels to asset-level feature table.
    """
    clustered_assets = asset_labels.copy()
    clustered_assets = pd.concat(
        [clustered_assets, asset_features.drop(columns=["ticker"], errors="ignore")],
        axis=1,
    )
    clustered_assets["cluster"] = model.labels_
    return clustered_assets


def create_cluster_summary(clustered_assets: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table for each cluster.
    """
    summary = (
        clustered_assets.groupby("cluster")
        .agg({
            "ticker": lambda values: ", ".join(values),
            "daily_return_mean": "mean",
            "rolling_volatility_mean": "mean",
            "rsi_14_mean": "mean",
            "macd_mean": "mean",
        })
        .reset_index()
        .rename(columns={"ticker": "assets"})
        .round(6)
    )
    return summary


def save_cluster_summary(summary_dataframe: pd.DataFrame) -> Path:
    """
    Save cluster summary to CSV.
    """
    output_path = TABLES_DIR / "cluster_summary.csv"
    summary_dataframe.to_csv(output_path, index=False)
    print(f"Saved cluster summary to {output_path}")
    return output_path


def save_clustered_assets(clustered_assets: pd.DataFrame) -> Path:
    """
    Save asset-level clustering output to CSV.
    """
    output_path = TABLES_DIR / "asset_clusters.csv"
    clustered_assets.to_csv(output_path, index=False)
    print(f"Saved asset cluster assignments to {output_path}")
    return output_path


def map_clusters_to_daily_data(
    daily_dataframe: pd.DataFrame,
    clustered_assets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map cluster labels back to the daily featured dataset.
    """
    cluster_map = clustered_assets[["ticker", "cluster"]].copy()
    merged_dataframe = daily_dataframe.merge(cluster_map, on="ticker", how="left")
    return merged_dataframe


def save_daily_clustered_data(dataframe: pd.DataFrame) -> Path:
    """
    Save the daily dataset with cluster labels included.
    """
    dataframe.to_csv(CLUSTERED_OUTPUT_FILE, index=False)
    print(f"Saved clustered daily dataset to {CLUSTERED_OUTPUT_FILE}")
    return CLUSTERED_OUTPUT_FILE


def plot_cluster_scatter(
    scaled_features: pd.DataFrame,
    clustered_assets: pd.DataFrame,
) -> Path:
    """
    Reduce features to 2D using PCA and plot clusters.
    """
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(scaled_features)

    plot_dataframe = pd.DataFrame({
        "pca_1": reduced[:, 0],
        "pca_2": reduced[:, 1],
        "ticker": clustered_assets["ticker"],
        "cluster": clustered_assets["cluster"],
    })

    plt.figure(figsize=(10, 7))
    for cluster_id in sorted(plot_dataframe["cluster"].unique()):
        cluster_group = plot_dataframe[plot_dataframe["cluster"] == cluster_id]
        plt.scatter(cluster_group["pca_1"], cluster_group["pca_2"], label=f"Cluster {cluster_id}")

        for _, row in cluster_group.iterrows():
            plt.text(row["pca_1"], row["pca_2"], row["ticker"], fontsize=8)

    plt.title("K-Means Clustering of Crypto Assets")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "crypto_asset_clusters.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved cluster visualization to {output_path}")
    return output_path


def run_clustering_pipeline(n_clusters: int = 3) -> pd.DataFrame:
    """
    Run the full clustering pipeline.
    """
    print("Loading featured dataset...")
    daily_dataframe = load_featured_data()

    print("Building asset-level feature table...")
    asset_features = build_asset_level_feature_table(daily_dataframe)

    print("Preparing clustering matrix...")
    asset_labels, feature_matrix = prepare_clustering_matrix(asset_features)

    print("Scaling features...")
    scaled_features = scale_features(feature_matrix)

    print(f"Running K-Means clustering with {n_clusters} clusters...")
    model = run_kmeans_clustering(scaled_features, n_clusters=n_clusters)

    print("Attaching cluster labels...")
    clustered_assets = attach_cluster_labels(asset_labels, asset_features, model)

    print("Creating cluster summary...")
    cluster_summary = create_cluster_summary(clustered_assets)
    save_cluster_summary(cluster_summary)

    print("Saving asset cluster assignments...")
    save_clustered_assets(clustered_assets)

    print("Mapping clusters to daily dataset...")
    clustered_daily_dataframe = map_clusters_to_daily_data(daily_dataframe, clustered_assets)
    save_daily_clustered_data(clustered_daily_dataframe)

    print("Creating cluster visualization...")
    plot_cluster_scatter(scaled_features, clustered_assets)

    print("Clustering pipeline completed successfully.")
    return clustered_daily_dataframe


if __name__ == "__main__":
    try:
        run_clustering_pipeline(n_clusters=3)
    except Exception as e:
        print(f"Error during clustering pipeline: {e}")