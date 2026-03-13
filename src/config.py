from datetime import date, timedelta
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Report directories
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Documentation and notebooks
DOCS_DIR = BASE_DIR / "docs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Selected crypto assets
CRYPTO_ASSETS = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "SOL-USD",
    "XRP-USD",
    "ADA-USD",
    "DOGE-USD",
    "TRX-USD",
    "AVAX-USD",
    "LINK-USD",
]

# Data collection date range
START_DATE = "2015-01-01"
yesterday = date.today() - timedelta(days=1)
END_DATE = yesterday.strftime("%Y-%m-%d")

# Create required directories automatically
REQUIRED_DIRS = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    DOCS_DIR,
    NOTEBOOKS_DIR,
]

for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)