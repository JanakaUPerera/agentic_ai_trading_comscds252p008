from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DATA_DIR


FEATURED_DATA_FILE = PROCESSED_DATA_DIR / "featured_crypto_data.csv"
NEWS_HEADLINES_FILE = OUTPUTS_DIR / "crypto_news_headlines.csv"
NEWS_SENTIMENT_FILE = OUTPUTS_DIR / "crypto_news_sentiment_summary.csv"
MARKET_NEWS_OUTPUT_FILE = OUTPUTS_DIR / "market_data_with_news_signal.csv"


RSS_FEEDS = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "decrypt": "https://decrypt.co/feed",
    "cryptonews": "https://cryptonews.com/news/feed/",
    "newsbtc": "https://www.newsbtc.com/feed/",
    "bitcoin_magazine": "https://bitcoinmagazine.com/.rss/full/",
    "theblock": "https://www.theblock.co/rss.xml",
    "ambcrypto": "https://ambcrypto.com/feed/",
}


SOURCE_WEIGHTS = {
    "cointelegraph": 1.00,
    "coindesk": 1.10,
    "decrypt": 1.00,
    "cryptonews": 0.95,
    "newsbtc": 0.90,
    "bitcoin_magazine": 0.95,
    "theblock": 1.10,
    "ambcrypto": 0.90,
}


CONFIDENCE_WEIGHTS = {
    "High": 1.00,
    "Medium": 0.75,
    "Low": 0.50,
}


ASSET_KEYWORDS = {
    "BTC-USD": ["bitcoin", "btc"],
    "ETH-USD": ["ethereum", "ether", "eth"],
    "BNB-USD": ["bnb", "binance coin", "binance"],
    "SOL-USD": ["solana", "sol"],
    "XRP-USD": ["xrp", "ripple"],
    "ADA-USD": ["cardano", "ada"],
    "DOGE-USD": ["dogecoin", "doge"],
    "TRX-USD": ["tron", "trx"],
    "AVAX-USD": ["avalanche", "avax"],
    "LINK-USD": ["chainlink", "link"],
}


BULLISH_KEYWORDS = {
    "surge": 3,
    "gain": 2,
    "gains": 2,
    "rise": 2,
    "rises": 2,
    "rally": 3,
    "soar": 3,
    "soars": 3,
    "jump": 2,
    "jumps": 2,
    "climb": 2,
    "climbs": 2,
    "spike": 2,
    "breakout": 3,
    "bullish": 4,
    "optimistic": 2,
    "confidence": 2,
    "strong": 1,
    "recovery": 2,
    "rebound": 2,
    "record": 2,
    "all-time high": 4,
    "ath": 3,
    "milestone": 2,
    "adoption": 3,
    "investment": 2,
    "institutional": 2,
    "inflow": 3,
    "funding": 2,
    "approval": 3,
    "approved": 3,
    "greenlight": 3,
    "legalized": 3,
    "launch": 2,
    "released": 2,
    "listing": 2,
    "listed": 2,
    "upgrade": 2,
    "update": 1,
    "partnership": 3,
    "collaboration": 2,
    "integration": 2,
    "scaling": 2,
    "improvement": 2,
    "innovation": 2,
    "expansion": 2,
}

BEARISH_KEYWORDS = {
    "fall": 2,
    "falls": 2,
    "drop": 2,
    "drops": 2,
    "plunge": 4,
    "plunges": 4,
    "crash": 5,
    "collapse": 5,
    "slump": 3,
    "dip": 1,
    "selloff": 4,
    "bearish": 4,
    "fear": 2,
    "panic": 3,
    "uncertainty": 2,
    "weakness": 2,
    "loss": 2,
    "losses": 2,
    "bankruptcy": 5,
    "insolvency": 5,
    "liquidation": 4,
    "volatile": 1,
    "warning": 2,
    "downgrade": 3,
    "concern": 2,
}

REGULATORY_KEYWORDS = {
    "ban": 4,
    "banned": 4,
    "lawsuit": 5,
    "investigation": 4,
    "probe": 4,
    "crackdown": 5,
    "regulation": 3,
    "penalty": 4,
    "approved": 2,
    "approval": 2,
    "legalized": 3,
    "greenlight": 2,
    "etf": 2,
    "sec": 3,
    "cftc": 3,
    "court": 3,
    "compliance": 2,
    "policy": 2,
    "sanction": 4,
    "license": 2,
}

SECURITY_KEYWORDS = {
    "hack": 5,
    "hacked": 5,
    "breach": 5,
    "exploit": 5,
    "attack": 4,
    "scam": 5,
    "fraud": 5,
    "rug pull": 5,
    "phishing": 4,
    "stolen": 4,
    "theft": 5,
    "drain": 3,
    "vulnerability": 4,
    "leak": 4,
    "compromised": 4,
    "malware": 4,
    "shutdown": 3,
    "suspend": 3,
    "halt": 3,
    "delisting": 3,
}


CATEGORY_KEYWORDS: Dict[str, Dict[str, int]] = {
    "Bullish": BULLISH_KEYWORDS,
    "Bearish": BEARISH_KEYWORDS,
    "Regulatory": REGULATORY_KEYWORDS,
    "Security": SECURITY_KEYWORDS,
}


@dataclass
class NewsItem:
    source: str
    title: str
    link: str
    published_at: Optional[datetime]
    asset: str
    sentiment_category: str
    category_score: int
    bullish_score: int
    bearish_score: int
    regulatory_score: int
    security_score: int
    confidence: str
    market_impact: str
    bullish_matches: str
    bearish_matches: str
    regulatory_matches: str
    security_matches: str
    classification_reason: str
    source_weight: float
    recency_weight: float
    confidence_weight: float
    weighted_news_score: float
    directional_news_score: float


def load_featured_data(file_path: Path = FEATURED_DATA_FILE) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Featured dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    return dataframe


def parse_datetime_safe(date_value: Optional[str]) -> Optional[datetime]:
    if not date_value:
        return None

    try:
        parsed = parsedate_to_datetime(date_value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title_for_dedup(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\b(update|updated|report|reports|analysis|analyst|today)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_weighted_score_with_matches(
    text: str,
    weighted_keywords: Dict[str, int],
) -> tuple[int, List[str]]:
    normalized_text = normalize_text(text)
    score = 0
    matches: List[str] = []

    sorted_keywords = sorted(weighted_keywords.items(), key=lambda item: len(item[0]), reverse=True)

    for keyword, weight in sorted_keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        occurrences = len(re.findall(pattern, normalized_text))
        if occurrences > 0:
            score += weight * occurrences
            matches.extend([keyword] * occurrences)

    return score, matches


def derive_market_impact(
    sentiment_category: str,
    bullish_score: int,
    bearish_score: int,
) -> str:
    if sentiment_category == "Bullish":
        return "Positive"

    if sentiment_category in {"Bearish", "Security"}:
        return "Negative"

    if sentiment_category == "Regulatory":
        if bullish_score > bearish_score:
            return "Positive"
        if bearish_score > bullish_score:
            return "Negative"
        return "Neutral"

    if bullish_score > bearish_score:
        return "Positive"
    if bearish_score > bullish_score:
        return "Negative"

    return "Neutral"


def classify_crypto_sentiment(headline: str) -> Dict[str, Any]:
    bullish_score, bullish_matches = compute_weighted_score_with_matches(headline, BULLISH_KEYWORDS)
    bearish_score, bearish_matches = compute_weighted_score_with_matches(headline, BEARISH_KEYWORDS)
    regulatory_score, regulatory_matches = compute_weighted_score_with_matches(headline, REGULATORY_KEYWORDS)
    security_score, security_matches = compute_weighted_score_with_matches(headline, SECURITY_KEYWORDS)

    category_scores = {
        "Bullish": bullish_score,
        "Bearish": bearish_score,
        "Regulatory": regulatory_score,
        "Security": security_score,
    }

    ranked_scores = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
    top_category, top_score = ranked_scores[0]
    second_category, second_score = ranked_scores[1]

    if top_score == 0:
        sentiment_category = "Neutral"
        category_score = 0
        confidence = "Low"
        reason = "No category keywords matched the headline."
    elif security_score >= 5 and security_score >= max(bullish_score, bearish_score, regulatory_score):
        sentiment_category = "Security"
        category_score = security_score
        confidence = "High"
        reason = "Security-related keywords strongly dominate the headline."
    elif regulatory_score >= 5 and regulatory_score > bullish_score and regulatory_score > bearish_score:
        sentiment_category = "Regulatory"
        category_score = regulatory_score
        confidence = "High"
        reason = "Regulatory/legal keywords strongly dominate the headline."
    elif top_score - second_score <= 1:
        sentiment_category = "Neutral"
        category_score = top_score
        confidence = "Low"
        reason = f"Top scores are too close: {top_category}={top_score}, {second_category}={second_score}."
    else:
        sentiment_category = top_category
        category_score = top_score
        confidence = "High" if top_score >= 5 and (top_score - second_score) >= 2 else "Medium"
        reason = f"{top_category} has the highest score with a clear margin."

    market_impact = derive_market_impact(
        sentiment_category=sentiment_category,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
    )

    return {
        "sentiment_category": sentiment_category,
        "category_score": category_score,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "regulatory_score": regulatory_score,
        "security_score": security_score,
        "confidence": confidence,
        "market_impact": market_impact,
        "bullish_matches": bullish_matches,
        "bearish_matches": bearish_matches,
        "regulatory_matches": regulatory_matches,
        "security_matches": security_matches,
        "classification_reason": reason,
    }


def detect_related_asset(headline: str) -> str:
    text = normalize_text(headline)

    for asset, keywords in ASSET_KEYWORDS.items():
        for keyword in sorted(keywords, key=len, reverse=True):
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            if re.search(pattern, text):
                return asset

    return "MARKET"


def compute_recency_weight(
    published_at: Optional[datetime],
    reference_time: Optional[datetime] = None,
) -> float:
    if published_at is None:
        return 0.60

    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    age_hours = max((reference_time - published_at).total_seconds() / 3600, 0)

    if age_hours <= 6:
        return 1.20
    if age_hours <= 24:
        return 1.00
    if age_hours <= 48:
        return 0.85
    if age_hours <= 72:
        return 0.70
    return 0.50


def compute_directional_news_score(
    market_impact: str,
    weighted_news_score: float,
) -> float:
    if market_impact == "Positive":
        return weighted_news_score
    if market_impact == "Negative":
        return -weighted_news_score
    return 0.0


def fetch_feed_entries(feed_name: str, feed_url: str, max_items: int = 50) -> List[NewsItem]:
    print(f"Fetching RSS feed from {feed_name}: {feed_url}")
    parsed_feed = feedparser.parse(feed_url)

    if getattr(parsed_feed, "bozo", False):
        print(f"Warning: feedparser reported a parsing issue for {feed_name}")

    news_items: List[NewsItem] = []
    now_utc = datetime.now(timezone.utc)

    for entry in parsed_feed.entries[:max_items]:
        title = str(getattr(entry, "title", "")).strip()
        link = str(getattr(entry, "link", "")).strip()
        published_raw = getattr(entry, "published", None) or getattr(entry, "updated", None)

        if not title:
            continue

        published_at = parse_datetime_safe(published_raw)
        asset = detect_related_asset(title)
        classification = classify_crypto_sentiment(title)

        source_weight = SOURCE_WEIGHTS.get(feed_name, 1.00)
        confidence_weight = CONFIDENCE_WEIGHTS.get(classification["confidence"], 0.50)
        recency_weight = compute_recency_weight(published_at, reference_time=now_utc)

        weighted_news_score = (
            classification["category_score"]
            * source_weight
            * confidence_weight
            * recency_weight
        )

        directional_news_score = compute_directional_news_score(
            market_impact=classification["market_impact"],
            weighted_news_score=weighted_news_score,
        )

        news_items.append(
            NewsItem(
                source=feed_name,
                title=title,
                link=link,
                published_at=published_at,
                asset=asset,
                sentiment_category=classification["sentiment_category"],
                category_score=classification["category_score"],
                bullish_score=classification["bullish_score"],
                bearish_score=classification["bearish_score"],
                regulatory_score=classification["regulatory_score"],
                security_score=classification["security_score"],
                confidence=classification["confidence"],
                market_impact=classification["market_impact"],
                bullish_matches=", ".join(classification["bullish_matches"]),
                bearish_matches=", ".join(classification["bearish_matches"]),
                regulatory_matches=", ".join(classification["regulatory_matches"]),
                security_matches=", ".join(classification["security_matches"]),
                classification_reason=classification["classification_reason"],
                source_weight=round(source_weight, 4),
                recency_weight=round(recency_weight, 4),
                confidence_weight=round(confidence_weight, 4),
                weighted_news_score=round(weighted_news_score, 4),
                directional_news_score=round(directional_news_score, 4),
            )
        )

    print(f"Collected {len(news_items)} headlines from {feed_name}")
    return news_items


def fetch_all_crypto_news(max_items_per_feed: int = 50) -> pd.DataFrame:
    all_items: List[NewsItem] = []

    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            all_items.extend(fetch_feed_entries(feed_name, feed_url, max_items=max_items_per_feed))
        except Exception as error:
            print(f"Failed to fetch {feed_name}: {error}")

    if not all_items:
        raise RuntimeError("No news headlines were collected from RSS feeds.")

    dataframe = pd.DataFrame([
        {
            "source": item.source,
            "title": item.title,
            "link": item.link,
            "published_at": item.published_at,
            "asset": item.asset,
            "sentiment_category": item.sentiment_category,
            "category_score": item.category_score,
            "bullish_score": item.bullish_score,
            "bearish_score": item.bearish_score,
            "regulatory_score": item.regulatory_score,
            "security_score": item.security_score,
            "confidence": item.confidence,
            "market_impact": item.market_impact,
            "bullish_matches": item.bullish_matches,
            "bearish_matches": item.bearish_matches,
            "regulatory_matches": item.regulatory_matches,
            "security_matches": item.security_matches,
            "classification_reason": item.classification_reason,
            "source_weight": item.source_weight,
            "recency_weight": item.recency_weight,
            "confidence_weight": item.confidence_weight,
            "weighted_news_score": item.weighted_news_score,
            "directional_news_score": item.directional_news_score,
        }
        for item in all_items
    ])

    dataframe["published_at"] = pd.to_datetime(dataframe["published_at"], errors="coerce", utc=True)

    dataframe["normalized_title"] = dataframe["title"].apply(normalize_title_for_dedup)
    dataframe = dataframe.sort_values(by="published_at", ascending=False, na_position="last").reset_index(drop=True)
    dataframe = dataframe.drop_duplicates(subset=["source", "normalized_title"], keep="first")
    dataframe = dataframe.drop_duplicates(subset=["title", "link"], keep="first")
    dataframe = dataframe.drop(columns=["normalized_title"]).reset_index(drop=True)

    return dataframe


def save_news_headlines(
    dataframe: pd.DataFrame,
    output_path: Path = NEWS_HEADLINES_FILE,
) -> Path:
    dataframe.to_csv(output_path, index=False)
    print(f"Saved news headlines to {output_path}")
    return output_path


def summarize_news_sentiment(news_dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        news_dataframe.groupby("asset")
        .agg(
            headline_count=("title", "count"),
            avg_category_score=("category_score", "mean"),
            bullish_score_mean=("bullish_score", "mean"),
            bearish_score_mean=("bearish_score", "mean"),
            regulatory_score_mean=("regulatory_score", "mean"),
            security_score_mean=("security_score", "mean"),
            bullish_count=("sentiment_category", lambda values: (values == "Bullish").sum()),
            bearish_count=("sentiment_category", lambda values: (values == "Bearish").sum()),
            regulatory_count=("sentiment_category", lambda values: (values == "Regulatory").sum()),
            security_count=("sentiment_category", lambda values: (values == "Security").sum()),
            neutral_count=("sentiment_category", lambda values: (values == "Neutral").sum()),
            positive_impact_count=("market_impact", lambda values: (values == "Positive").sum()),
            negative_impact_count=("market_impact", lambda values: (values == "Negative").sum()),
            neutral_impact_count=("market_impact", lambda values: (values == "Neutral").sum()),
            high_confidence_count=("confidence", lambda values: (values == "High").sum()),
            medium_confidence_count=("confidence", lambda values: (values == "Medium").sum()),
            low_confidence_count=("confidence", lambda values: (values == "Low").sum()),
            weighted_news_score_sum=("weighted_news_score", "sum"),
            directional_news_score_sum=("directional_news_score", "sum"),
            weighted_news_score_mean=("weighted_news_score", "mean"),
            directional_news_score_mean=("directional_news_score", "mean"),
        )
        .reset_index()
    )

    numeric_columns = [
        "avg_category_score",
        "bullish_score_mean",
        "bearish_score_mean",
        "regulatory_score_mean",
        "security_score_mean",
        "weighted_news_score_sum",
        "directional_news_score_sum",
        "weighted_news_score_mean",
        "directional_news_score_mean",
    ]
    summary[numeric_columns] = summary[numeric_columns].round(4)

    def derive_dominant_signal(row: pd.Series) -> str:
        category_strength = {
            "Bullish": row["bullish_score_mean"],
            "Bearish": row["bearish_score_mean"],
            "Regulatory": row["regulatory_score_mean"],
            "Security": row["security_score_mean"],
        }

        ranked = sorted(category_strength.items(), key=lambda item: item[1], reverse=True)
        top_category, top_value = ranked[0]
        second_value = ranked[1][1]

        if top_value == 0:
            return "Neutral"
        if top_category == "Security" and top_value >= 2:
            return "Security"
        if top_category == "Regulatory" and top_value >= 2 and (top_value - second_value) > 0.25:
            return "Regulatory"
        if (top_value - second_value) <= 0.25:
            return "Neutral"
        return top_category

    def derive_market_signal(row: pd.Series) -> str:
        score = row["directional_news_score_sum"]
        if score >= 2:
            return "Positive"
        if score <= -2:
            return "Negative"
        return "Neutral"

    def derive_asset_news_bias(row: pd.Series) -> str:
        score = row["directional_news_score_sum"]
        if score >= 5:
            return "Strong Positive"
        if score >= 2:
            return "Positive"
        if score <= -5:
            return "Strong Negative"
        if score <= -2:
            return "Negative"
        return "Neutral"

    summary["news_signal"] = summary.apply(derive_dominant_signal, axis=1)
    summary["market_impact_signal"] = summary.apply(derive_market_signal, axis=1)
    summary["asset_news_bias"] = summary.apply(derive_asset_news_bias, axis=1)

    return summary.sort_values(
        by=["headline_count", "weighted_news_score_sum"],
        ascending=[False, False],
    ).reset_index(drop=True)


def save_news_sentiment_summary(
    dataframe: pd.DataFrame,
    output_path: Path = NEWS_SENTIMENT_FILE,
) -> Path:
    dataframe.to_csv(output_path, index=False)
    print(f"Saved news sentiment summary to {output_path}")
    return output_path


def merge_news_signal_with_market_data(
    market_dataframe: pd.DataFrame,
    news_summary: pd.DataFrame,
) -> pd.DataFrame:
    merge_columns = [
        "asset",
        "headline_count",
        "avg_category_score",
        "bullish_score_mean",
        "bearish_score_mean",
        "regulatory_score_mean",
        "security_score_mean",
        "positive_impact_count",
        "negative_impact_count",
        "neutral_impact_count",
        "high_confidence_count",
        "medium_confidence_count",
        "low_confidence_count",
        "weighted_news_score_sum",
        "directional_news_score_sum",
        "weighted_news_score_mean",
        "directional_news_score_mean",
        "news_signal",
        "market_impact_signal",
        "asset_news_bias",
    ]

    merged = market_dataframe.merge(
        news_summary[merge_columns],
        left_on="ticker",
        right_on="asset",
        how="left",
    )

    fill_zero_columns = [
        "headline_count",
        "avg_category_score",
        "bullish_score_mean",
        "bearish_score_mean",
        "regulatory_score_mean",
        "security_score_mean",
        "positive_impact_count",
        "negative_impact_count",
        "neutral_impact_count",
        "high_confidence_count",
        "medium_confidence_count",
        "low_confidence_count",
        "weighted_news_score_sum",
        "directional_news_score_sum",
        "weighted_news_score_mean",
        "directional_news_score_mean",
    ]

    for column in fill_zero_columns:
        merged[column] = merged[column].fillna(0)

    merged["news_signal"] = merged["news_signal"].fillna("Neutral")
    merged["market_impact_signal"] = merged["market_impact_signal"].fillna("Neutral")
    merged["asset_news_bias"] = merged["asset_news_bias"].fillna("Neutral")
    merged = merged.drop(columns=["asset"], errors="ignore")

    return merged


def run_news_info_retrieval_pipeline(max_items_per_feed: int = 50) -> pd.DataFrame:
    print("Loading featured market dataset...")
    market_dataframe = load_featured_data()

    print("Fetching real crypto news headlines from RSS feeds...")
    news_dataframe = fetch_all_crypto_news(max_items_per_feed=max_items_per_feed)
    save_news_headlines(news_dataframe)

    print("Summarizing crypto news categories by asset...")
    news_summary = summarize_news_sentiment(news_dataframe)
    save_news_sentiment_summary(news_summary)

    print("Merging news classifications into market dataset...")
    enriched_market_dataframe = merge_news_signal_with_market_data(market_dataframe, news_summary)

    enriched_market_dataframe.to_csv(MARKET_NEWS_OUTPUT_FILE, index=False)
    print(f"Saved market dataset with news signals to {MARKET_NEWS_OUTPUT_FILE}")

    print("Information retrieval pipeline completed successfully.")
    return enriched_market_dataframe


if __name__ == "__main__":
    try:
        run_news_info_retrieval_pipeline(max_items_per_feed=50)
    except Exception as error:
        print(f"Error during news information retrieval pipeline: {error}")