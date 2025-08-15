"""
Configuration settings for CustomerIQ platform
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/customeriq.db")

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# RFM Analysis settings
RFM_QUANTILES = 5  # 1-5 scale for RFM scoring

# Churn prediction settings
CHURN_THRESHOLD_DAYS = 90  # Days without purchase to consider churn
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# CLV settings
CLV_PREDICTION_PERIOD = 365  # Days to predict CLV for

# Streamlit settings
PAGE_TITLE = "CustomerIQ - Intelligent Customer Analytics"
PAGE_ICON = "🎯"
LAYOUT = "wide"

# Visualization settings
PLOTLY_THEME = "plotly_white"
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Customer segments
CUSTOMER_SEGMENTS = {
    "Champions": {"description": "Best customers - high value, frequent buyers"},
    "Loyal Customers": {"description": "Regular customers with good value"},
    "Potential Loyalists": {"description": "Recent customers with potential"},
    "At Risk": {"description": "Customers showing declining engagement"},
    "Cannot Lose Them": {"description": "High-value customers at risk"},
    "Hibernating": {"description": "Previously good customers, now inactive"},
    "New Customers": {"description": "Recent first-time buyers"},
    "Lost": {"description": "Customers who haven't returned"}
}

# Risk categories
RISK_CATEGORIES = {
    "Low": {"threshold": 0.3, "color": "#2ca02c"},
    "Medium": {"threshold": 0.7, "color": "#ff7f0e"},
    "High": {"threshold": 1.0, "color": "#d62728"}
}

# File upload settings
MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls']

# Export settings
REPORT_FORMATS = ['PDF', 'CSV', 'Excel']
