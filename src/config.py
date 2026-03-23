from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

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
startDate = date.today() - timedelta(days=(365 * 3))  # Last 3 years
START_DATE = startDate.strftime("%Y-%m-%d")
yesterday = date.today() - timedelta(days=1)
END_DATE = yesterday.strftime("%Y-%m-%d")

# AWS config
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# RDS config
RDS_HOST = os.getenv("RDS_HOST")
RDS_PORT = os.getenv("RDS_PORT", "5432")
RDS_DBNAME = os.getenv("RDS_DBNAME")
RDS_USER = os.getenv("RDS_USER")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")

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

# Groq API config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
LLM_MAX_CONTEXT_CHARS = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "18000"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "180"))

# Email config
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")

for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)