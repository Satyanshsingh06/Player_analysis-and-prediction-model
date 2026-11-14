import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import logging

try:
    from .Config import MASTER_FILE, REPORT_DIR
except ImportError:
    from Config import MASTER_FILE, REPORT_DIR

DATA_PATH = MASTER_FILE
CSV_DIR = REPORT_DIR / "eda"
GRAPH_DIR = CSV_DIR / "charts"

MIN_BALLS = 30
BATTER_NAME = "Virat Kohli"

# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def _ensure_output_dirs() -> None:
    """Create the EDA report directories before writing outputs."""

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

def load_data():

    """Load and validate the cricket dataset."""

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "delivery_id",
        "match_id",
        "season",
        "innings",
        "phase",
        "batter",
        "bowler",
        "batter_runs",
        "total_runs",
        "wides",
        "noballs",
        "byes",
        "legbyes",
        "penalty",
        "is_legal_delivery",
        "counts_as_ball_faced",
        "is_four",
        "is_six",
        "is_kohli_out",
        "bowler_credited_dismissal"
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Convert numeric columns
    numeric_columns = [
        "season",
        "innings",
        "batter_runs",
        "total_runs",
        "wides",
        "noballs",
        "byes",
        "legbyes",
        "penalty"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

        if df[col].isna().any():
            raise ValueError(
                f"Missing numeric values in column: {col}"
            )

    # Convert boolean columns safely
    boolean_columns = [
        "is_legal_delivery",
        "counts_as_ball_faced",
        "is_four",
        "is_six",
        "is_kohli_out",
        "bowler_credited_dismissal"
    ]
    for col in boolean_columns:
        converted = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

        if converted.isna().any():
            raise ValueError(
                f"Invalid boolean values in column: {col}"
            )

    df[col] = converted.astype(bool)

    # Keep only the required batter
    df = df[
        df["batter"] == BATTER_NAME
    ].copy()

    # --------------------------------------------------
    # CRICKET RULES
    # --------------------------------------------------

    # Wide balls are not balls faced
    df["balls_faced"] = df[
        "counts_as_ball_faced"
    ]

    # Legal deliveries
    df["legal_balls"] = df[
        "is_legal_delivery"
    ]

    # Batter runs
    # Byes and leg-byes are already excluded
    df["runs"] = df["batter_runs"]

    # Bowler conceded runs
    # Byes, leg-byes and penalty runs are excluded
    df["bowler_runs"] = (
        df["total_runs"]
        - df["byes"]
        - df["legbyes"]
        - df["penalty"]
    ).clip(lower=0)

    # Boundaries
    df["boundaries"] = (
        df["is_four"].astype(int)
        + df["is_six"].astype(int)
    )
    # Batter dot balls: official balls faced on which Kohli
    # scored no runs off the bat.
    df["dot_ball"] = (
        df["counts_as_ball_faced"].astype(bool)
        & df["batter_runs"].eq(0)
    )

    # Dismissals
    df["dismissals"] = df["is_kohli_out"]

    logger.info(
        "Dataset loaded successfully: %s deliveries",
        len(df)
    )

    return df


# --------------------------------------------------
# 2. REUSABLE BATTING METRICS
# --------------------------------------------------

def calculate_metrics(data):

    """Calculate batting statistics."""

    runs = data["runs"].sum()

    balls = data["balls_faced"].sum()

    dismissals = data["dismissals"].sum()

    fours = data["is_four"].sum()

    sixes = data["is_six"].sum()

    dots = data["dot_ball"].sum()

    boundaries = fours + sixes

    strike_rate = (
        runs / balls * 100
        if balls > 0 else 0
    )

    average = (
        runs / dismissals
        if dismissals > 0 else 0
    )

    dot_percentage = (
        dots / balls * 100
        if balls > 0 else 0
    )

    boundary_percentage = (
        boundaries / balls * 100
        if balls > 0 else 0
    )

    return {
        "Runs": int(runs),
        "Balls": int(balls),
        "Strike Rate": round(strike_rate, 2),
        "Batting Average": round(average, 2),
        "Fours": int(fours),
        "Sixes": int(sixes),
        "Dismissals": int(dismissals),
        "Dot Balls": int(dots),
        "Dot Ball %": round(dot_percentage, 2),
        "Boundary %": round(boundary_percentage, 2)
    }


# --------------------------------------------------
# 3. OVERALL ANALYSIS
# --------------------------------------------------

def overall_analysis(df):

    """Generate overall batting performance."""

    logger.info("Calculating overall performance...")
    _ensure_output_dirs()

    metrics = calculate_metrics(df)

    report = pd.DataFrame(
        [metrics]
    )

    report.to_csv(
        CSV_DIR / "overall_analysis.csv",
        index=False
    )

    print("\nOVERALL PERFORMANCE")

    print(report.to_string(index=False))

    return report


# --------------------------------------------------
# 4. SEASON ANALYSIS
# --------------------------------------------------

def season_analysis(df):

    """Analyze performance season-wise."""

    logger.info("Calculating season analysis...")
    _ensure_output_dirs()

    results = []

    for season, group in df.groupby("season"):

        metrics = calculate_metrics(group)

        metrics["Season"] = int(season)

        metrics["Matches"] = (
            group["match_id"].nunique()
        )

        results.append(metrics)

    report = pd.DataFrame(results)

    report = report.sort_values("Season")

    report.to_csv(
        CSV_DIR / "season_analysis.csv",
        index=False
    )

    print("\nSEASON ANALYSIS")

    print(report.to_string(index=False))

    return report


# --------------------------------------------------
# 5. BOWLER ANALYSIS
# --------------------------------------------------

def bowler_analysis(df):

    """Analyze Kohli's performance against each bowler."""

    logger.info("Calculating bowler analysis...")
    _ensure_output_dirs()

    results = []

    for bowler, group in df.groupby("bowler"):

        metrics = calculate_metrics(group)

        balls_faced = group[
            "balls_faced"
        ].sum()

        wickets = group[
            "bowler_credited_dismissal"
        ].sum()

        legal_balls = group[
            "legal_balls"
        ].sum()

        runs_conceded = group[
            "bowler_runs"
        ].sum()

        if balls_faced < MIN_BALLS:
            continue

        bowler_strike_rate = (
            legal_balls / wickets
            if wickets > 0 else np.nan
        )

        economy = (
            runs_conceded / legal_balls * 6
            if legal_balls > 0 else np.nan
        )

        dismissal_rate = (
            wickets / balls_faced * 100
        )

        results.append({

            "Bowler": bowler,

            "Runs Against": metrics["Runs"],

            "Balls Faced": int(balls_faced),

            "Kohli Strike Rate":
                metrics["Strike Rate"],

            "Bowler Runs Conceded":
                int(runs_conceded),

            "Legal Balls": int(legal_balls),

            "Economy vs Kohli (striker deliveries)": round(economy, 2)
                if pd.notna(economy) else np.nan,

            "Bowler Strike Rate":
                round(bowler_strike_rate, 2)
                if pd.notna(bowler_strike_rate)
                else np.nan,

            "Fours": metrics["Fours"],

            "Sixes": metrics["Sixes"],

            "Dot Balls": metrics["Dot Balls"],

            "Dot Ball %": metrics["Dot Ball %"],

            "Boundary %": metrics["Boundary %"],

            "Dismissals": int(wickets),

            "Dismissal Rate": round(
                dismissal_rate, 2
            ),

            "Matches": group[
                "match_id"
            ].nunique()

        })

    report = pd.DataFrame(results)

    report = report.sort_values(
        by="Dismissals",
        ascending=False
    )

    report.to_csv(
        CSV_DIR / "bowler_analysis.csv",
        index=False
    )

    print("\nBOWLER ANALYSIS")

    print(report.head(15).to_string(index=False))

    return report


# --------------------------------------------------
# 6. PHASE ANALYSIS
# --------------------------------------------------

def phase_analysis(df):

    """Compare powerplay, middle and death overs."""

    logger.info("Calculating phase analysis...")
    _ensure_output_dirs()

    results = []

    for phase, group in df.groupby("phase"):

        metrics = calculate_metrics(group)

        metrics["Phase"] = phase

        metrics["Matches"] = (
            group["match_id"].nunique()
        )

        results.append(metrics)

    report = pd.DataFrame(results)

    report.to_csv(
        CSV_DIR / "phase_analysis.csv",
        index=False
    )

    print("\nPHASE ANALYSIS")

    print(report.to_string(index=False))

    return report


# --------------------------------------------------
# 7. BATTING FIRST VS CHASING
# --------------------------------------------------

def innings_analysis(df):

    """Compare batting first and chasing."""

    logger.info("Calculating innings analysis...")
    _ensure_output_dirs()
    
    innings_mapping = {
        1: "First Innings",
        2: "Second Innings",
        3: "Super Over - First Innings",
        4: "Super Over - Second Innings",
    }
    df = df.copy()

    df["Innings Type"] = (
        df["innings"]
        .map(innings_mapping)
        .fillna("Other")
    )

    results = []

    for innings_type, group in df.groupby(
        "Innings Type"
    ):

        metrics = calculate_metrics(group)

        metrics["Innings Type"] = innings_type

        results.append(metrics)

    report = pd.DataFrame(results)

    report.to_csv(
        CSV_DIR / "innings_analysis.csv",
        index=False
    )

    print("\nBATTING FIRST VS CHASING")

    print(report.to_string(index=False))

    return report


# --------------------------------------------------
# 8. BOWLER DIFFICULTY INDEX
# --------------------------------------------------

def bowler_difficulty_index(bowlers):

    """
    Calculate custom Bowler Difficulty Index.

    Higher score = More difficult bowler.

    Factors:
    1. Lower Kohli strike rate.
    2. Higher dot-ball percentage.
    3. Lower boundary percentage.
    4. Higher dismissal rate.
    """

    logger.info("Calculating Bowler Difficulty Index...")
    _ensure_output_dirs()

    result = bowlers.copy()

    if result.empty:

        result["Bowler Difficulty Index"] = []

        return result

    # Convert all factors into percentile scores
    # so that they have comparable scales.

    result["SR Score"] = (
        -result["Kohli Strike Rate"]
    ).rank(pct=True) * 100

    result["Dot Score"] = (
        result["Dot Ball %"]
    ).rank(pct=True) * 100

    result["Boundary Score"] = (
        -result["Boundary %"]
    ).rank(pct=True) * 100

    result["Dismissal Score"] = (
        result["Dismissal Rate"]
    ).rank(pct=True) * 100

    # Equal weighting of all four components
    result["Bowler Difficulty Index"] = (

        result["SR Score"] * 0.25

        + result["Dot Score"] * 0.25

        + result["Boundary Score"] * 0.25

        + result["Dismissal Score"] * 0.25

    ).round(2)

    result = result.sort_values(
        "Bowler Difficulty Index",
        ascending=False
    )

    result["Difficulty Rank"] = range(
        1, len(result) + 1
    )

    result.to_csv(
        CSV_DIR / "bowler_difficulty_index.csv",
        index=False
    )

    print("\nBOWLER DIFFICULTY INDEX")

    print(
        result[
            [
                "Difficulty Rank",
                "Bowler",
                "Bowler Difficulty Index"
            ]
        ].head(15).to_string(index=False)
    )

    return result


# --------------------------------------------------
# 9. DATA VISUALIZATIONS
# --------------------------------------------------

def create_visualizations(
    season,
    bowlers,
    phases,
    difficulty
):

    """Create and save PNG graphs."""

    logger.info("Generating visualizations...")
    _ensure_output_dirs()

    sns.set_theme(style="whitegrid")

    # ----------------------------------------------
    # GRAPH 1: Season-wise Runs
    # ----------------------------------------------

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=season,
        x="Season",
        y="Runs"
    )

    plt.title(
        "Virat Kohli Season-wise Runs"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "season_runs.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------
    # GRAPH 2: Season-wise Strike Rate
    # ----------------------------------------------

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=season,
        x="Season",
        y="Strike Rate",
        marker="o"
    )

    plt.title(
        "Season-wise Strike Rate"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "season_strike_rate.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------
    # GRAPH 3: Bowler Matchups
    # ----------------------------------------------

    if not bowlers.empty:

        top_bowlers = bowlers.nlargest(
            15,
            "Balls Faced"
        )

        plt.figure(figsize=(12, 7))

        sns.barplot(
            data=top_bowlers,
            x="Kohli Strike Rate",
            y="Bowler"
        )

        plt.title(
            "Kohli vs Top 15 Bowlers"
        )

        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / "bowler_matchups.png",
            dpi=300
        )

        plt.close()

    # ----------------------------------------------
    # GRAPH 4: Phase-wise Strike Rate
    # ----------------------------------------------

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=phases,
        x="Phase",
        y="Strike Rate",
        order=["powerplay", "middle", "death"]
    )

    plt.title(
        "Phase-wise Strike Rate"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "phase_strike_rate.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------
    # GRAPH 5: Phase-wise Dot Ball %
    # ----------------------------------------------

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=phases,
        x="Phase",
        y="Dot Ball %",
        order=["powerplay", "middle", "death"]
    )

    plt.title(
        "Phase-wise Dot Ball Percentage"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "phase_dot_percentage.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------
    # GRAPH 6: Difficulty Index
    # ----------------------------------------------

    if not difficulty.empty:

        top_difficult = difficulty.head(15)

        plt.figure(figsize=(12, 7))

        sns.barplot(
            data=top_difficult,
            x="Bowler Difficulty Index",
            y="Bowler"
        )

        plt.title(
            "Top 15 Difficult Bowlers for Kohli"
        )

        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / "bowler_difficulty.png",
            dpi=300
        )

        plt.close()

    logger.info(
        "All graphs saved successfully."
    )


def main():

    """Execute complete CricketIQ analysis."""

    # Load dataset
    df = load_data()

    # Overall analysis
    overall_analysis(df)

    # Season analysis
    season = season_analysis(df)

    # Bowler analysis
    bowlers = bowler_analysis(df)

    # Phase analysis
    phases = phase_analysis(df)

    # Batting first vs chasing
    innings_analysis(df)

    # Difficulty index
    difficulty = bowler_difficulty_index(
        bowlers
    )

    # Generate graphs
    create_visualizations(
        season,
        bowlers,
        phases,
        difficulty
    )

    
    print("ANALYSIS COMPLETED")

    print("\nCSV reports saved at:", CSV_DIR)

    print("\nGraphs saved at:", GRAPH_DIR)

if __name__ == "__main__":

    main()
