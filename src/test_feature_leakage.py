
"""Audit point-in-time correctness of CricketIQ features.

Compares exported historical features with independently
recalculated values from the canonical delivery dataset.
"""

import numpy as np
import pandas as pd

from Config import MASTER_FILE, PROCESSED_DIR


FEATURE_DIR = PROCESSED_DIR / "ml"


def load_audit_data() -> pd.DataFrame:
    """Join raw delivery outcomes with exported feature records."""

    raw = pd.read_csv(MASTER_FILE)

    feature_files = [
        FEATURE_DIR / "kohli_train_features.csv",
        FEATURE_DIR / "kohli_validation_features.csv",
        FEATURE_DIR / "kohli_test_features.csv",
    ]

    features = pd.concat(
        [pd.read_csv(path) for path in feature_files],
        ignore_index=True,
    )

    if raw["delivery_id"].duplicated().any():
        raise ValueError("Duplicate IDs in master dataset.")

    if features["delivery_id"].duplicated().any():
        raise ValueError("Duplicate IDs in feature datasets.")

    if set(raw["delivery_id"]) != set(features["delivery_id"]):
        raise ValueError("Master and feature delivery IDs differ.")

    # Source outcomes are used for auditing, never as model inputs.
    merged = features.merge(
        raw[
            [
                "delivery_id",
                "batter_runs",
                "counts_as_ball_faced",
                "bowler_credited_dismissal",
                "delivery_in_over",
            ]
        ],
        on="delivery_id",
        how="inner",
        validate="one_to_one",
    )

    merged["match_date"] = pd.to_datetime(
        merged["match_date"], errors="raise"
    )

    return merged.sort_values(
        [
            "match_date",
            "match_id",
            "innings",
            "over",
            "delivery_in_over",
        ],
        kind="stable",
    ).reset_index(drop=True)


def check_previous_counts(data: pd.DataFrame) -> None:
    """Recalculate innings and career counts excluding each ball."""

    innings = data.groupby(
        ["match_id", "innings"], sort=False
    )

    expected_innings_runs = (
        innings["batter_runs"].cumsum()
        - data["batter_runs"]
    )

    expected_innings_balls = (
        innings["counts_as_ball_faced"].cumsum()
        - data["counts_as_ball_faced"]
    )

    expected_career_balls = (
        data["counts_as_ball_faced"].cumsum()
        - data["counts_as_ball_faced"]
    )

    checks = {
        "innings_runs_before": expected_innings_runs,
        "innings_balls_before": expected_innings_balls,
        "career_balls_before": expected_career_balls,
    }

    for feature, expected in checks.items():
        if not np.allclose(
            data[feature].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
        ):
            raise AssertionError(
                f"Point-in-time mismatch: {feature}"
            )

        print(f"{feature}: PASSED")


def check_matchup_history(data: pd.DataFrame) -> None:
    """Verify historical bowler counts and dismissal totals."""

    matchup = data.groupby("bowler", sort=False)

    expected_balls = (
        matchup["counts_as_ball_faced"].cumsum()
        - data["counts_as_ball_faced"]
    )

    expected_dismissals = (
        matchup["bowler_credited_dismissal"].cumsum()
        - data["bowler_credited_dismissal"]
    )

    if not np.allclose(
        data["matchup_balls_before"],
        expected_balls,
    ):
        raise AssertionError("Bowler matchup counts are incorrect.")

    if not np.allclose(
        data["matchup_dismissals_before"],
        expected_dismissals,
    ):
        raise AssertionError("Bowler dismissal history is incorrect.")

    print("Bowler matchup history: PASSED")


def main() -> None:
    """Run historical consistency checks."""

    data = load_audit_data()

    check_previous_counts(data)
    check_matchup_history(data)

if __name__ == "__main__":
    main()
