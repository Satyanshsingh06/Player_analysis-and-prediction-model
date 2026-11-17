
import pandas as pd
from Config import MASTER_FILE


def validate_cricket_rules():
    if MASTER_FILE.suffix.lower() == ".csv":
        df = pd.read_csv(MASTER_FILE)
    elif MASTER_FILE.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(MASTER_FILE)
    else:
        raise ValueError(f"Unsupported master dataset format: {MASTER_FILE.suffix}")

    pairs = [
        ("season", "season_metadata"),
        ("match_date", "match_date_metadata"),
        ("opponent", "opponent_metadata"),
        ("venue", "venue_metadata")
    ]

    for original, metadata in pairs:
        if metadata in df.columns:
            left = df[original].astype("string").str.strip()
            right = df[metadata].astype("string").str.strip()

            differences = (left != right).fillna(True).sum()

            print(f"{original}: {differences} differences")

    print("DATE VALIDATION")
    dates = pd.to_datetime(df["match_date"], errors="coerce")
    print("Invalid dates:", dates.isna().sum())
    
    print("RUNS AND EXTRAS")
    run_errors = (
        df["batter_runs"] + df["extras"]
        != df["total_runs"]
    ).sum()

    print("Run calculation errors:", run_errors)

    extra_columns = [
        "wides", "noballs", "byes", "legbyes", "penalty"
    ]
    extra_errors = (
        df[extra_columns].sum(axis=1) != df["extras"]
    ).sum()

    print("Extras calculation errors:", extra_errors)
    print("LEGAL DELIVERY RULES")

    expected_legal = (
        (df["wides"] == 0) &
        (df["noballs"] == 0)
    )

    legal_errors = (
        df["is_legal_delivery"].astype(bool)
        != expected_legal
    ).sum()

    print("Legal delivery errors:", legal_errors)
    print("BALLS FACED RULES")

    expected_balls_faced = df["wides"] == 0

    ball_errors = (
        df["counts_as_ball_faced"].astype(bool)
        != expected_balls_faced
    ).sum()
    print("Balls faced errors:", ball_errors)

    print("DISMISSAL VALIDATION")
    kohli_out = df["is_kohli_out"].astype(bool)
    credited = df["bowler_credited_dismissal"].astype(bool)

    missing_dismissal_type = (
        kohli_out & df["dismissal_type"].isna()
    ).sum()

    print(
        "Kohli dismissals missing type:",
        missing_dismissal_type
    )

    # A bowler cannot receive a wicket credit
    # unless Kohli was dismissed.
    invalid_credited = (
        credited & ~kohli_out
    ).sum()

    print("Invalid bowler-credited wickets:", invalid_credited)

    non_bowler_dismissals = [
        "run out",
        "retired hurt",
        "retired out",
        "obstructing the field"
    ]

    wrong_credit = (
        credited &
        df["dismissal_type"].isin(non_bowler_dismissals)
    ).sum()
    print("Wrong bowler wicket credits:", wrong_credit)

    print("DATASET SUMMARY")

    print("Total records:", len(df))
    print("Unique matches:", df["match_id"].nunique())
    print("Total batter runs:", df["batter_runs"].sum())
    print("Official balls faced:", expected_balls_faced.sum())
    print("Kohli dismissals:", kohli_out.sum())
    print("Bowler-credited dismissals:", credited.sum())

if __name__ == "__main__":
    validate_cricket_rules()
