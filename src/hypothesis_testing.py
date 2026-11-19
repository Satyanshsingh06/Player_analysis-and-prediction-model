
"""Hypothesis testing for CricketIQ batting-phase boundary rates.

Research question:
Does Kohli's boundary rate differ between powerplay and death overs?

Uses:
- Match-cluster bootstrap confidence interval
- Match-level paired sign-flip permutation test
- Descriptive boundary rates

The dataset is not modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .Config import MASTER_FILE, REPORT_DIR
except ImportError:
    from src.Config import MASTER_FILE, REPORT_DIR


OUTPUT_DIR = REPORT_DIR / "hypothesis_testing"

PHASES = ("powerplay", "death")
ALPHA = 0.05
N_RESAMPLES = 5000
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load valid powerplay and death-over observations."""

    data = pd.read_csv(MASTER_FILE)

    required = {
        "match_id",
        "phase",
        "is_four",
        "is_six",
        "counts_as_ball_faced",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if data[list(required)].isna().any().any():
        raise ValueError("Missing values in required columns.")

    for column in (
        "is_four",
        "is_six",
        "counts_as_ball_faced",
    ):
        if not data[column].isin([0, 1, True, False]).all():
            raise ValueError(f"Invalid binary values: {column}")

    data = data.loc[
        data["phase"].isin(PHASES)
        & data["counts_as_ball_faced"].astype(bool)
    ].copy()

    data["boundary"] = (
        data["is_four"].astype(int)
        | data["is_six"].astype(int)
    ).astype(int)

    if data.empty:
        raise ValueError("No eligible deliveries found.")

    return data


def describe_phases(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate boundary frequency for each batting phase."""

    summary = (
        data.groupby("phase")
        .agg(
            deliveries=("boundary", "size"),
            boundaries=("boundary", "sum"),
            matches=("match_id", "nunique"),
        )
        .reindex(PHASES)
    )

    if summary.isna().any().any():
        raise ValueError("One batting phase is missing.")

    summary["boundary_rate"] = (
        summary["boundaries"] / summary["deliveries"]
    )

    return summary.reset_index()


def bootstrap_difference(
    data: pd.DataFrame,
) -> tuple[float, float, float]:
    """Bootstrap complete matches to estimate the rate difference."""

    grouped = (
        data.groupby(["match_id", "phase"])["boundary"]
        .agg(["sum", "count"])
        .unstack("phase", fill_value=0)
    )

    # Each row contains both phase-specific statistics for one match.
    matrix = np.column_stack([
        grouped[("sum", "powerplay")].to_numpy(),
        grouped[("count", "powerplay")].to_numpy(),
        grouped[("sum", "death")].to_numpy(),
        grouped[("count", "death")].to_numpy(),
    ]).astype(float)

    observed = (
        matrix[:, 2].sum() / matrix[:, 3].sum()
        - matrix[:, 0].sum() / matrix[:, 1].sum()
    )

    rng = np.random.default_rng(RANDOM_STATE)
    differences = []

    for _ in range(N_RESAMPLES):
        indices = rng.integers(
            0, len(matrix), size=len(matrix)
        )

        sample = matrix[indices]

        powerplay_balls = sample[:, 1].sum()
        death_balls = sample[:, 3].sum()

        if powerplay_balls == 0 or death_balls == 0:
            continue

        difference = (
            sample[:, 2].sum() / death_balls
            - sample[:, 0].sum() / powerplay_balls
        )

        differences.append(difference)

    if len(differences) < N_RESAMPLES * 0.95:
        raise ValueError("Too many invalid bootstrap samples.")

    lower, upper = np.percentile(
        differences, [2.5, 97.5]
    )

    return float(observed), float(lower), float(upper)


def paired_permutation_test(data: pd.DataFrame) -> tuple[float, int]:
    """Test paired within-match differences using sign flips.

    Only matches containing observations from both phases are used.
    Each match contributes equally to this paired test.
    """

    match_rates = (
        data.groupby(["match_id", "phase"])["boundary"]
        .mean()
        .unstack()
        .dropna(subset=list(PHASES))
    )

    differences = (
        match_rates["death"] - match_rates["powerplay"]
    ).to_numpy(dtype=float)

    if len(differences) < 2:
        raise ValueError("Not enough matches with both phases.")

    observed_statistic = abs(differences.mean())

    rng = np.random.default_rng(RANDOM_STATE)
    extreme = 0

    for _ in range(N_RESAMPLES):
        signs = rng.choice([-1, 1], size=len(differences))

        statistic = abs(
            np.mean(differences * signs)
        )

        if statistic >= observed_statistic:
            extreme += 1

    p_value = (extreme + 1) / (N_RESAMPLES + 1)

    return float(p_value), len(differences)


def main() -> None:
    """Run the complete batting-phase hypothesis test."""

    print("=" * 60)
    print("CRICKETIQ - HYPOTHESIS TESTING")
    print("=" * 60)

    data = load_data()

    summary = describe_phases(data)

    difference, lower, upper = bootstrap_difference(data)

    p_value, paired_matches = paired_permutation_test(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / "phase_boundary_rates.csv",
        index=False,
    )

    results = {
        "powerplay_boundary_rate": float(
            summary.loc[
                summary["phase"] == "powerplay",
                "boundary_rate",
            ].iloc[0]
        ),
        "death_boundary_rate": float(
            summary.loc[
                summary["phase"] == "death",
                "boundary_rate",
            ].iloc[0]
        ),
        "difference_death_minus_powerplay": difference,
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "paired_matches": paired_matches,
        "p_value": p_value,
        "alpha": ALPHA,
        "significant": p_value < ALPHA,
    }

    pd.DataFrame([results]).to_csv(
        OUTPUT_DIR / "hypothesis_results.csv",
        index=False,
    )

    print("\nPHASE BOUNDARY RATES")
    print(summary.to_string(index=False))

    print(f"\nDifference: {difference:.4f}")
    print(f"95% bootstrap CI: [{lower:.4f}, {upper:.4f}]")
    print(f"Paired matches: {paired_matches}")
    print(f"Permutation p-value: {p_value:.5f}")

    if p_value < ALPHA:
        print("Result: Reject H0 for the paired-match test.")
    else:
        print("Result: Insufficient evidence to reject H0.")



if __name__ == "__main__":
    main()
