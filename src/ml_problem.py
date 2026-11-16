"""This module prepares supervised learning targets, checks their validity,
and creates chronological train, validation, and test datasets.

It does not train or evaluate a machine learning model.
"""

from pathlib import Path
import pandas as pd
from Config import MASTER_FILE, PROCESSED_DIR, REPORT_DIR


# Keep prediction targets centralized so later modules use the same names.
TARGET_COLUMNS = [
    "target_runs",
    "target_boundary",
    "target_dismissal",
]

# Separate historical learning from future model evaluation.
TRAIN_END_YEAR = 2024
VALIDATION_YEAR = 2025
TEST_YEAR = 2026


def load_master_dataset(file_path: Path) -> pd.DataFrame:  # Load the canonical dataset and verify essential columns.

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Master dataset not found: {file_path}"
        )

    deliveries = pd.read_csv(file_path)

    required_columns = {
        "delivery_id",
        "match_id",
        "season",
        "match_date",
        "batter",
        "bowler",
        "batter_runs",
        "is_four",
        "is_six",
        "is_kohli_out",
    }

    missing_columns = required_columns - set(deliveries.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    if deliveries.empty:
        raise ValueError("The master dataset is empty.")

    if deliveries["delivery_id"].isna().any():
        raise ValueError("Some delivery identifiers are missing.")

    if deliveries["delivery_id"].duplicated().any():
        raise ValueError("Duplicate delivery identifiers detected.")

    if not deliveries["batter"].eq("Virat Kohli").all():
        raise ValueError("The dataset contains another batter.")

    return deliveries


def validate_binary_column(
    deliveries: pd.DataFrame,
    column_name: str,
) -> None:  # Ensure a binary outcome contains only valid zero/one values.

    values = deliveries[column_name]

    if values.isna().any():
        raise ValueError(f"{column_name} contains missing values.")

    # Accept genuine booleans and numeric 0/1, but reject text labels.
    valid_values = values.map(
        lambda value: isinstance(value, (bool, int, float))
        and value in (0, 1)
    )

    if not valid_values.all():
        raise ValueError(
            f"{column_name} must contain only 0, 1, True, or False."
        )


def create_prediction_targets(
    deliveries: pd.DataFrame,
) -> pd.DataFrame:  # Create three targets from observed delivery outcomes.

    prepared = deliveries.copy()

    runs = pd.to_numeric(
        prepared["batter_runs"],
        errors="coerce",
    )

    if runs.isna().any():
        raise ValueError("Batter runs contain invalid or missing values.")

    if not runs.between(0, 6).all():
        raise ValueError(
            "Unexpected batter-run outcome outside 0-6. "
            "Review the records before continuing."
        )

    if not (runs == runs.astype(int)).all():
        raise ValueError("Batter runs must be whole numbers.")

    prepared["target_runs"] = runs.astype(int)

    for column in ["is_four", "is_six", "is_kohli_out"]:
        validate_binary_column(prepared, column)

    # A boundary is either a four or a six scored off the bat.
    prepared["target_boundary"] = (
        prepared["is_four"].astype(int)
        | prepared["is_six"].astype(int)
    )

    prepared["target_dismissal"] = (
        prepared["is_kohli_out"].astype(int)
    )

    # Verify that boundary labels agree with the batter's runs.
    invalid_fours = (
        (prepared["is_four"] == 1)
        & (prepared["target_runs"] != 4)
    )

    invalid_sixes = (
        (prepared["is_six"] == 1)
        & (prepared["target_runs"] != 6)
    )

    if invalid_fours.any() or invalid_sixes.any():
        raise ValueError(
            "Boundary flags conflict with recorded batter runs."
        )

    return prepared


def create_chronological_splits(
    deliveries: pd.DataFrame,
) -> dict[str, pd.DataFrame]: # Split deliveries by season without mixing future seasons.

    prepared = deliveries.copy()

    prepared["season"] = pd.to_numeric(
        prepared["season"],
        errors="coerce",
    )

    if prepared["season"].isna().any():
        raise ValueError("Some IPL seasons are invalid or missing.")

    prepared["match_date"] = pd.to_datetime(
        prepared["match_date"],
        errors="coerce",
    )

    if prepared["match_date"].isna().any():
        raise ValueError("Some match dates are invalid.")

    # Sort records chronologically for later sequential features.
    prepared = prepared.sort_values(
        ["match_date", "match_id", "innings", "over",
         "delivery_in_over"]
    ).reset_index(drop=True)

    splits = {
        "train": prepared[
            prepared["season"] <= TRAIN_END_YEAR
        ].copy(),
        "validation": prepared[
            prepared["season"] == VALIDATION_YEAR
        ].copy(),
        "test": prepared[
            prepared["season"] == TEST_YEAR
        ].copy(),
    }

    # Every evaluation period must contain real observations.
    for split_name, split_data in splits.items():
        if split_data.empty:
            raise ValueError(
                f"The {split_name} split contains no deliveries."
            )

    total_split_rows = sum(len(data) for data in splits.values())

    if total_split_rows != len(prepared):
        raise ValueError(
            "Some deliveries were excluded from the chronological split."
        )

    return splits


def summarize_targets(
    deliveries: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame: # Summarize target frequencies and class imbalance.
    records = []
    for target in TARGET_COLUMNS:
        counts = (
            deliveries[target]
            .value_counts()
            .sort_index()
        )

        for outcome, count in counts.items():
            records.append({
                "split": split_name,
                "target": target,
                "outcome": int(outcome),
                "count": int(count),
                "percentage": round(
                    count / len(deliveries) * 100, 2
                ),
            })

    return pd.DataFrame(records)

def save_prepared_datasets(
    splits: dict[str, pd.DataFrame],
) -> None: # Export reproducible datasets for future feature engineering.

    output_directory = PROCESSED_DIR / "ml"

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name, split_data in splits.items():
        output_path = (
            output_directory / f"kohli_{split_name}_targets.csv"
        )

        split_data.to_csv(output_path, index=False)

        print(f"Saved {split_name}: {output_path}")


def main() -> None: # Run the target preparation and chronological splitting pipeline.
    deliveries = load_master_dataset(MASTER_FILE)
    prepared = create_prediction_targets(deliveries)
    splits = create_chronological_splits(prepared)
    summaries = []

    for split_name, split_data in splits.items():
        print(
            f"{split_name.title()}: "
            f"{len(split_data):,} deliveries, "
            f"{split_data['match_id'].nunique()} matches"
        )

        summaries.append(
            summarize_targets(split_data, split_name)
        )

    summary = pd.concat(summaries, ignore_index=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORT_DIR / "ml_target_distribution.csv"

    summary.to_csv(summary_path, index=False)

    save_prepared_datasets(splits)

    print("\nTarget distribution:")
    print(summary.to_string(index=False))    # it convert target distribution tothe string.


if __name__ == "__main__":
    main()
