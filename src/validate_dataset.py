import pandas as pd
from Config import MASTER_FILE


def validate_dataset():
    print("CRICKETIQ - MASTER DATASET VALIDATION")

    if not MASTER_FILE.is_file():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_FILE}")

    if MASTER_FILE.suffix.lower() == ".csv":
        df = pd.read_csv(MASTER_FILE)
    elif MASTER_FILE.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(MASTER_FILE)
    else:
        raise ValueError(f"Unsupported master dataset format: {MASTER_FILE.suffix}")

    print("\nDataset shape:", df.shape)

    # Inspect columns
    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Duplicate delivery IDs
    duplicates = df["delivery_id"].duplicated().sum()
    print("\nDuplicate delivery IDs:", duplicates)

    # Season coverage
    print("\nSeason-wise delivery counts:")
    print(df.groupby("season").size())

    # Batter identity
    print("\nUnique batters:")
    print(df["batter"].unique())

    # Check runs distribution
    print("\nRuns distribution:")
    print(df["batter_runs"].value_counts().sort_index())

    # Check scoring consistency
    if {"batter_runs", "extras", "total_runs"}.issubset(df.columns):
        invalid_runs = (
            df["batter_runs"] + df["extras"] != df["total_runs"]
        ).sum()
        print("\nRun calculation errors:", invalid_runs)

    metadata_pairs = [
        ("season", "season_metadata"),
        ("match_date", "match_date_metadata"),
        ("opponent", "opponent_metadata"),
        ("venue", "venue_metadata"),
    ]

    for original, metadata in metadata_pairs:
        if original not in df.columns or metadata not in df.columns:
            print(f"{original}: metadata unavailable")
            continue

        differences = (
            df[original].astype(str).str.strip()
            != df[metadata].astype(str).str.strip()
        ).sum()

        print(f"{original}: {differences} differences")

    print("\n--- Additional Validation ---")

    # Invalid dates
    dates = pd.to_datetime(df["match_date"], errors="coerce")
    print("Invalid dates:", dates.isna().sum())

    # Validate total runs
    invalid_runs = (
        df["batter_runs"] + df["extras"] != df["total_runs"]
    ).sum()
    print("Run errors:", invalid_runs)

    # Validate extras components
    extra_columns = ["wides", "noballs", "byes", "legbyes", "penalty"]

    invalid_extras = (
        df[extra_columns].sum(axis=1) != df["extras"]
    ).sum()
    print("Extras calculation errors:", invalid_extras)

    # Inspect dismissals
    print("\nDismissal flags:")
    print(df["is_kohli_out"].value_counts(dropna=False))

    print("\nBowler-credited dismissals:")
    print(df["bowler_credited_dismissal"].value_counts(dropna=False))

    print("\nValidation inspection completed.")

if __name__ == "__main__":
    validate_dataset()
