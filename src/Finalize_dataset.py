from pathlib import Path
import pandas as pd
from Config import MASTER_FILE

OUTPUT_FILE = MASTER_FILE
METADATA_PAIRS = {
    "season": "season_metadata",
    "match_date": "match_date_metadata",
    "opponent": "opponent_metadata",
    "venue": "venue_metadata",
}   # i pair the duplicate columns having same intity to cross check

def season_matches(season: pd.Series, metadata: pd.Series) -> pd.Series:
    season = pd.to_numeric(season, errors="coerce")
    metadata = metadata.astype("string")

    start_year = pd.to_numeric(
        metadata.str.extract(r"^(\d{4})", expand=False),
        errors="coerce",
    )
    end_year = pd.to_numeric(
        metadata.str.replace(
            r"^(\d{2})\d{2}/(\d{2})$",
            r"\1\2",
            regex=True,
        ).str.extract(r"^(\d{4})", expand=False),
        errors="coerce",
    )

    return season.eq(start_year) | season.eq(end_year)


def finalize_dataset():
    if MASTER_FILE.suffix.lower() == ".csv":
        df = pd.read_csv(MASTER_FILE)
    else:
        df = pd.read_excel(MASTER_FILE)

    print("\nOriginal shape:", df.shape)

    # Check required columns
    required = [
        "delivery_id",
        "match_id",
        "season",
        "match_date",
        "batter",
        "bowler",
        "batter_runs",
        "extras",
        "total_runs",
        "wides",
        "noballs",
        "is_legal_delivery",
        "counts_as_ball_faced",
        "is_kohli_out",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate metadata before removing duplicates
    print("\nMetadata comparison:")

    for original, metadata in METADATA_PAIRS.items():
        if metadata not in df.columns:
            continue

        if original == "match_date":
            left = pd.to_datetime(df[original], errors="coerce")
            right = pd.to_datetime(df[metadata], errors="coerce")
        elif original == "season":
            matches = season_matches(df[original], df[metadata])
        else:
            left = df[original].astype("string").str.strip()
            right = df[metadata].astype("string").str.strip()

        if original != "season":
            matches = left.eq(right)

        differences = (~matches.fillna(False)).sum()

        print(f"{original}: {differences} differences")

        if differences:
            examples = df.loc[
                ~matches.fillna(False),
                ["match_id", original, metadata]
            ].drop_duplicates().head(10)

            print(examples.to_string(index=False))

            raise ValueError(
                f"Metadata conflict in '{original}'. "
                "Resolve differences before finalizing."
            )

    # Remove verified redundant columns
    columns_to_remove = [
        col for col in METADATA_PAIRS.values()
        if col in df.columns
    ]

    df = df.drop(columns=columns_to_remove)

    # duplicate delivery identifiers
    if df["delivery_id"].duplicated().any():
        raise ValueError("Duplicate delivery IDs detected.")

    # Validate runs
    numeric_cols = [
        "batter_runs", "extras", "total_runs",
        "wides", "noballs"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not (
        df["batter_runs"] + df["extras"]
        == df["total_runs"]
    ).all():
        raise ValueError("Run calculation inconsistency detected.")

    # Validate batter identity
    if not df["batter"].eq("Virat Kohli").all():
        raise ValueError("Unexpected batter found.")

    # Validate match dates
    dates = pd.to_datetime(df["match_date"], errors="coerce")

    if dates.isna().any():
        raise ValueError("Invalid match dates detected.")

    df["match_date"] = dates.dt.strftime("%Y-%m-%d")

    # Preserve chronological delivery order
    df = df.sort_values(
        [
            "match_date",
            "match_id",
            "innings",
            "over",
            "delivery_in_over",
        ]
    ).reset_index(drop=True)

    # Save the new version
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    #  Verify the saved dataset
    verified = pd.read_csv(OUTPUT_FILE)

    if verified.shape != df.shape:
        raise ValueError("Saved dataset shape mismatch.")

    print("\nFinal shape:", verified.shape)
    print("Unique matches:", verified["match_id"].nunique())
    print("Duplicate IDs:", verified["delivery_id"].duplicated().sum())
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    finalize_dataset()
