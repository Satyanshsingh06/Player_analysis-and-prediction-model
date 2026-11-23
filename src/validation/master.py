"""Combined inspection and cricket-rule checks for the master dataset."""

import pandas as pd

from src.Config import MASTER_FILE


def load_master() -> pd.DataFrame:
    if not MASTER_FILE.is_file():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_FILE}")
    if MASTER_FILE.suffix.lower() == ".csv":
        return pd.read_csv(MASTER_FILE)
    if MASTER_FILE.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(MASTER_FILE)
    raise ValueError(f"Unsupported master dataset format: {MASTER_FILE.suffix}")


def validate_master(data: pd.DataFrame) -> None:
    """Print structural, scoring, metadata, and dismissal checks."""
    print("CRICKETIQ MASTER DATASET VALIDATION")
    print("Shape:", data.shape)
    print("Missing values:\n", data.isna().sum())

    if "delivery_id" in data:
        print("Duplicate delivery IDs:", data["delivery_id"].duplicated().sum())
    if "season" in data:
        print("Deliveries by season:\n", data.groupby("season").size())
    if "batter_runs" in data:
        print("Batter run distribution:\n", data["batter_runs"].value_counts().sort_index())

    pairs = [
        ("season", "season_metadata"),
        ("match_date", "match_date_metadata"),
        ("opponent", "opponent_metadata"),
        ("venue", "venue_metadata"),
    ]
    for source, metadata in pairs:
        if source in data and metadata in data:
            left = data[source].astype("string").str.strip()
            right = data[metadata].astype("string").str.strip()
            print(f"{source} metadata differences:", (left != right).fillna(True).sum())

    if "match_date" in data:
        print("Invalid dates:", pd.to_datetime(data["match_date"], errors="coerce").isna().sum())

    if {"batter_runs", "extras", "total_runs"}.issubset(data):
        print("Run total errors:", (data["batter_runs"] + data["extras"] != data["total_runs"]).sum())

    extras = ["wides", "noballs", "byes", "legbyes", "penalty"]
    if set(extras + ["extras"]).issubset(data):
        print("Extras component errors:", (data[extras].sum(axis=1) != data["extras"]).sum())

    if {"wides", "noballs", "is_legal_delivery"}.issubset(data):
        expected_legal = data["wides"].eq(0) & data["noballs"].eq(0)
        print("Legal delivery errors:", (data["is_legal_delivery"].astype(bool) != expected_legal).sum())
    if {"wides", "counts_as_ball_faced"}.issubset(data):
        expected_faced = data["wides"].eq(0)
        print("Balls faced errors:", (data["counts_as_ball_faced"].astype(bool) != expected_faced).sum())

    dismissal_cols = {"is_kohli_out", "bowler_credited_dismissal", "dismissal_type"}
    if dismissal_cols.issubset(data):
        dismissed = data["is_kohli_out"].astype(bool)
        credited = data["bowler_credited_dismissal"].astype(bool)
        print("Dismissals missing type:", (dismissed & data["dismissal_type"].isna()).sum())
        print("Bowler credit without dismissal:", (credited & ~dismissed).sum())
        non_bowler = {"run out", "retired hurt", "retired out", "obstructing the field"}
        wrong_credit = credited & data["dismissal_type"].astype("string").str.lower().isin(non_bowler)
        print("Incorrect bowler credits:", wrong_credit.sum())

    print("Rows:", len(data))
    if "match_id" in data:
        print("Unique matches:", data["match_id"].nunique())


if __name__ == "__main__":
    validate_master(load_master())
