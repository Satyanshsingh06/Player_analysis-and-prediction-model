
"""Approved input features for CricketIQ prediction models.

Only information available before the delivery may be used.
Outcome columns, delivery identifiers, and future information
must never enter the model's feature matrix.
"""

CATEGORICAL_FEATURES = [
    "bowler",
    "opponent",
    "venue",
    "phase",
]

NUMERICAL_FEATURES = [
    "innings",
    "over",
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

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

TARGETS = {
    "runs": "target_runs",
    "boundary": "target_boundary",
    "dismissal": "target_dismissal",
}
