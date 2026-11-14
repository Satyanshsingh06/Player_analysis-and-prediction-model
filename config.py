"""
Configuration file for CricketIQ project.
"""

from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
DATA_PATH = Path(
    r"C:\Users\aatif\OneDrive\Desktop\CricketIQ_master_v1.csv"
)

# Output directories
REPORT_DIR = BASE_DIR / "reports"
CSV_DIR = REPORT_DIR / "csv"
GRAPH_DIR = REPORT_DIR / "graphs"

# Minimum balls faced for bowler ranking
MIN_BALLS = 30

# Batter name
BATTER_NAME = "Virat Kohli"

# Create output directories
CSV_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
