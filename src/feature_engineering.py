
"""Build point-in-time batting features for CricketIQ.

Every historical feature excludes the delivery being predicted.
The complete chronological dataset is processed before splitting
so that later seasons can use genuinely earlier match history.
"""

import numpy as np
import pandas as pd

try:
    from .Config import MASTER_FILE, PROCESSED_DIR
except ImportError:
    from Config import MASTER_FILE, PROCESSED_DIR


OUTPUT_DIR = PROCESSED_DIR / "ml"

TRAIN_END_YEAR = 2024
VALIDATION_YEAR = 2025
TEST_YEAR = 2026

# A small smoothing constant avoids undefined strike rates.
# We also retain sample counts so models can learn reliability.
SMOOTHING_BALLS = 10

FEATURE_COLUMNS = [
    "season",
    "bowler",
    "opponent",
    "venue",
    "innings",
    "over",
    "phase",
    "innings_runs_before",
    "innings_balls_before",
    "innings_strike_rate_before",
    "career_balls_before",
    "career_strike_rate_before",
    "matchup_balls_before",
    "matchup_strike_rate_before",
    "matchup_dismissals_before",
    "recent_10_runs",
    "recent_10_boundary_rate",
]


def load_deliveries() -> pd.DataFrame:
    """Load and chronologically order Kohli's delivery records."""

    deliveries = pd.read_csv(MASTER_FILE)

    required = {
        "delivery_id", "match_id", "match_date", "season",
        "innings", "over", "delivery_in_over",
        "batter_runs", "bowler", "counts_as_ball_faced",
        "is_four", "is_six", "bowler_credited_dismissal",
        "is_kohli_out", "opponent", "venue", "phase",
    }

    missing = required - set(deliveries.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if deliveries["delivery_id"].duplicated().any():
        raise ValueError("Duplicate delivery IDs found.")

    if not deliveries["batter"].eq("Virat Kohli").all():
        raise ValueError("Unexpected batter found.")

    deliveries["match_date"] = pd.to_datetime(
        deliveries["match_date"], errors="raise"
    )

    deliveries["season"] = pd.to_numeric(
        deliveries["season"], errors="raise"
    ).astype(int)

    # Use delivery_in_over, not over.ball notation, because
    # wides and no-balls can produce repeated legal-ball labels.
    return deliveries.sort_values(
        [
            "match_date", "match_id", "innings",
            "over", "delivery_in_over",
        ],
        kind="stable",
    ).reset_index(drop=True)


def prepare_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the observed outcomes used in calculations."""

    data = df.copy()

    numeric_columns = [
        "batter_runs",
        "counts_as_ball_faced",
        "is_four",
        "is_six",
        "is_kohli_out",
        "bowler_credited_dismissal",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="raise"
        )

    for column in numeric_columns[1:]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"Invalid binary values: {column}")

    data["boundary"] = (
        (data["is_four"] == 1) | (data["is_six"] == 1)
    ).astype(int)

    data["target_runs"] = data["batter_runs"].astype(int)
    data["target_boundary"] = data["boundary"]
    data["target_dismissal"] = data["is_kohli_out"].astype(int)

    return data


def add_innings_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Kohli's innings statistics before each delivery."""

    data = df.copy()

    innings = data.groupby(
        ["match_id", "innings"], sort=False
    )

    # Subtracting the current observation excludes its outcome.
    data["innings_runs_before"] = (
        innings["batter_runs"].cumsum()
        - data["batter_runs"]
    )

    data["innings_balls_before"] = (
        innings["counts_as_ball_faced"].cumsum()
        - data["counts_as_ball_faced"]
    )

    data["innings_strike_rate_before"] = np.where(
        data["innings_balls_before"] > 0,
        100 * data["innings_runs_before"]
        / data["innings_balls_before"].replace(0, np.nan),
        0.0,
    )

    return data


def add_career_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate historical batting form across all IPL matches."""

    data = df.copy()

    # Rows are already ordered chronologically.
    data["career_balls_before"] = (
        data["counts_as_ball_faced"].cumsum()
        - data["counts_as_ball_faced"]
    )

    career_runs_before = (
        data["batter_runs"].cumsum()
        - data["batter_runs"]
    )

    data["career_strike_rate_before"] = np.where(
        data["career_balls_before"] > 0,
        100 * career_runs_before
        / data["career_balls_before"].replace(0, np.nan),
        0.0,
    )

    return data


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate historical Kohli-versus-bowler statistics."""

    data = df.copy()

    matchup = data.groupby("bowler", sort=False)

    data["matchup_balls_before"] = (
        matchup["counts_as_ball_faced"].cumsum()
        - data["counts_as_ball_faced"]
    )

    matchup_runs_before = (
        matchup["batter_runs"].cumsum()
        - data["batter_runs"]
    )

    data["matchup_dismissals_before"] = (
        matchup["bowler_credited_dismissal"].cumsum()
        - data["bowler_credited_dismissal"]
    )

    # Shrink small-sample matchup rates toward the batter's
    # previous career scoring rate to reduce extreme estimates.
    prior_runs_per_ball = (
        data["career_strike_rate_before"] / 100
    )

    smoothed_runs = (
        matchup_runs_before
        + SMOOTHING_BALLS * prior_runs_per_ball
    )

    smoothed_balls = (
        data["matchup_balls_before"] + SMOOTHING_BALLS
    )

    data["matchup_strike_rate_before"] = (
        100 * smoothed_runs / smoothed_balls
    )

    return data


def add_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate form over the previous 10 recorded deliveries."""

    data = df.copy()

    innings_keys = ["match_id", "innings"]

    # These rolling windows use recorded striker deliveries,
    # including wides. They are not official balls-faced windows.
    data["recent_10_runs"] = (
        data.groupby(innings_keys)["batter_runs"]
        .transform(
            lambda values: values.shift(1)
            .rolling(10, min_periods=1)
            .sum()
        )
        .fillna(0)
    )

    data["recent_10_boundary_rate"] = (
        data.groupby(innings_keys)["boundary"]
        .transform(
            lambda values: values.shift(1)
            .rolling(10, min_periods=1)
            .mean()
        )
        .fillna(0)
    )

    return data


def validate_features(df: pd.DataFrame) -> None:
    """Check feature completeness and basic point-in-time rules."""

    if df[FEATURE_COLUMNS].isna().any().any():
        missing = df[FEATURE_COLUMNS].isna().sum()
        raise ValueError(
            f"Missing feature values:\n{missing[missing > 0]}"
        )

    if (df["innings_balls_before"] < 0).any():
        raise ValueError("Negative innings ball count found.")

    if (df["matchup_balls_before"] < 0).any():
        raise ValueError("Negative matchup ball count found.")

    if not df["recent_10_boundary_rate"].between(0, 1).all():
        raise ValueError("Invalid recent boundary rate.")

    # The first delivery in each innings has no prior innings data.
    first_deliveries = df.groupby(
        ["match_id", "innings"], sort=False
    ).head(1)

    if not (first_deliveries["innings_runs_before"] == 0).all():
        raise ValueError("First-ball innings leakage detected.")

    if not (first_deliveries["innings_balls_before"] == 0).all():
        raise ValueError("First-ball ball-count leakage detected.")


def save_feature_splits(df: pd.DataFrame) -> None:
    """Save feature and target records by chronological season."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": df[df["season"] <= TRAIN_END_YEAR],
        "validation": df[df["season"] == VALIDATION_YEAR],
        "test": df[df["season"] == TEST_YEAR],
    }

    if sum(map(len, splits.values())) != len(df):
        raise ValueError("Some seasons are outside the split plan.")

    export_columns = (
        ["delivery_id", "match_id", "match_date"]
        + FEATURE_COLUMNS
        + ["target_runs", "target_boundary", "target_dismissal"]
    )

    for name, subset in splits.items():
        if subset.empty:
            raise ValueError(f"Empty {name} split.")

        path = OUTPUT_DIR / f"kohli_{name}_features.csv"

        subset[export_columns].to_csv(path, index=False)

        print(f"{name.title()}: {len(subset):,} rows")
        print(f"Saved: {path}")


def main() -> None:
    """Execute the CricketIQ feature-engineering pipeline."""


    data = load_deliveries()
    data = prepare_outcomes(data)

    data = add_innings_features(data)
    data = add_career_features(data)
    data = add_matchup_features(data)
    data = add_recent_form_features(data)

    validate_features(data)
    save_feature_splits(data)

if __name__ == "__main__":
    main()
