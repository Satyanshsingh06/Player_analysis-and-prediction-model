"""Command-line predictor for Virat Kohli's IPL delivery outcomes."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
MASTER_FILE = ROOT / "Data" / "processed" / "master" / "CricketIQ_master_v1.csv"
CATEGORICAL = ["bowler", "venue", "opponent"]
NUMERIC = ["innings", "over"]
FEATURES = CATEGORICAL + NUMERIC


def load_deliveries() -> pd.DataFrame:
    if not MASTER_FILE.is_file():
        raise FileNotFoundError(f"Master delivery data not found: {MASTER_FILE}")
    data = pd.read_csv(MASTER_FILE)
    required = set(FEATURES + ["batter", "batter_runs", "is_six", "counts_as_ball_faced"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Master dataset is missing columns: {sorted(missing)}")
    data = data.loc[data["batter"].eq("Virat Kohli")].copy()
    if data.empty:
        raise ValueError("No Virat Kohli deliveries were found in the master dataset.")
    data["six"] = pd.to_numeric(data["is_six"], errors="coerce").fillna(0).astype(int)
    runs = pd.to_numeric(data["batter_runs"], errors="coerce").fillna(0)
    faced = pd.to_numeric(data["counts_as_ball_faced"], errors="coerce").fillna(0).astype(bool)
    data["dot"] = (faced & runs.eq(0)).astype(int)
    data["innings"] = pd.to_numeric(data["innings"], errors="coerce")
    data["over"] = pd.to_numeric(data["over"], errors="coerce")
    return data.dropna(subset=FEATURES + ["six", "dot"])


def make_model() -> Pipeline:
    preprocessing = ColumnTransformer([
        ("category", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
        ("number", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def predict_from_context(data: pd.DataFrame, bowler: str, innings: int,
                         over: int, venue: str, opponent: str) -> dict[str, float]:
    """Fit outcome models on the available dataset and predict this context."""
    context = pd.DataFrame([{
        "bowler": bowler, "innings": innings, "over": over,
        "venue": venue, "opponent": opponent,
    }], columns=FEATURES)
    results = {}
    for label in ("six", "dot"):
        model = make_model().fit(data[FEATURES], data[label])
        classes = list(model.named_steps["classifier"].classes_)
        results[label] = (
            float(model.predict_proba(context)[0, classes.index(1)])
            if 1 in classes else 0.0
        )
    return results


def main() -> None:
    data = load_deliveries()
    print("Virat Kohli next-delivery predictor")
    print(f"Historical deliveries used: {len(data):,}")
    bowler = input("Bowler: ").strip()
    innings = int(input("Innings (1 or 2): "))
    over = int(input("Over (0-19): "))
    venue = input("Venue: ").strip()
    opponent = input("Opponent: ").strip()
    if innings not in (1, 2) or not 0 <= over <= 19:
        raise ValueError("Enter innings 1 or 2 and an over from 0 to 19.")
    probabilities = predict_from_context(data, bowler, innings, over, venue, opponent)
    print("\nEstimated probabilities (other outcomes are possible):")
    print(f"Six:      {probabilities['six']:.1%}")
    print(f"Dot ball: {probabilities['dot']:.1%}")


if __name__ == "__main__":
    main()
