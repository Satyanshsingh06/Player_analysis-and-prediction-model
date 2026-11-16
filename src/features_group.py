# Fields used to identify, order, and audit delivery records.

IDENTIFIER_COLUMNS = [
    "delivery_id",
    "match_id",
    "match_date",
]

# Information available before the delivery takes place.
# These are candidate features, not a finalized model input set.
PRE_DELIVERY_FEATURES = [
    "bowler",
    "opponent",
    "venue",
    "innings",
    "over",
    "phase",
]

# Actual delivery outcomes that our models may predict.
TARGET_COLUMNS = [
    "batter_runs",
    "is_four",
    "is_six",
    "is_kohli_out",
]

# Fields that describe what happened during or after the ball.
# Never use these directly to predict that same delivery.
POST_DELIVERY_COLUMNS = [
    "extras",
    "total_runs",
    "wides",
    "noballs",
    "byes",
    "legbyes",
    "penalty",
    "is_legal_delivery",
    "counts_as_ball_faced",
    "dismissal_type",
    "bowler_credited_dismissal",
]

# Additional columns retained for analysis and feature engineering.
CONTEXT_COLUMNS = [
    "season",
    "delivery_in_over",
    "batting_team",
    "batter",
    "non_striker",
]
