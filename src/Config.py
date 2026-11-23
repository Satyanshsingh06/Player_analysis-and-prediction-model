from pathlib import Path
# Root directory of the CricketIQ project
BASE_DIR = Path(__file__).resolve().parent.parent

# Main folders
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DIR = PROCESSED_DIR / "cleaned"
MASTER_DIR = PROCESSED_DIR / "master"
ML_DIR = DATA_DIR / "ml"
FEATURES_DIR = ML_DIR / "features"
TARGETS_DIR = ML_DIR / "targets"

MODELS_DIR = BASE_DIR / "Models"
REPORT_DIR = BASE_DIR / "Report"

ORIGINAL_FILE = RAW_DIR / "KOhli_delivery_Combined_2009-2026.csv"
MAPPED_FILE = CLEANED_DIR / "kohli_ipl_with_match_metadata.csv"

# Final master dataset
MASTER_FILE = MASTER_DIR / "CricketIQ_master_v1.csv"

# Shared documentation and data-quality outputs.
DATA_DICTIONARY_FILE = BASE_DIR / "DATA_DICTIONARY.md"
DATA_QUALITY_FILE = REPORT_DIR / "data_quality_summary.csv"
# Create folders if they do not exist
for folder in [
    RAW_DIR,
    CLEANED_DIR,
    MASTER_DIR,
    FEATURES_DIR,
    TARGETS_DIR,
    MODELS_DIR,
    REPORT_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Project directory:", BASE_DIR)
    print("Raw data directory:", RAW_DIR)
    print("Processed data directory:", PROCESSED_DIR)
    print("Original file exists:", ORIGINAL_FILE.exists())
    print("Mapped file exists:", MAPPED_FILE.exists())
