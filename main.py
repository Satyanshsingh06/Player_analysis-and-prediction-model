import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "Data/processed/CricketIQ_master_v1.csv"
REPORT_DIR = ROOT / "Report"

def heading(title):
    print(title.center(65))

def read_csv(path):
    if not path.is_file():
        print(f"\nFile not found: {path}")
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def show_table(rows, columns, limit=12):
    if not rows:
        print("No records available.")
        return

    columns = [col for col in columns if col in rows[0]]

    if not columns:
        print("Expected report columns were not found.")
        print("Available columns:", ", ".join(rows[0]))
        return

    widths = {
        col: min(
            28,
            max(
                len(col),
                *(len(str(row.get(col, ""))) for row in rows[:limit]),
            ),
        )
        for col in columns
    }

    print(" | ".join(col[:widths[col]].ljust(widths[col])
                     for col in columns))
    print("-" * (sum(widths.values()) + 3 * (len(columns) - 1)))

    for row in rows[:limit]:
        print(" | ".join(
            str(row.get(col, ""))[:widths[col]].ljust(widths[col])
            for col in columns
        ))

def dataset_overview():
    heading("1. DATASET OVERVIEW")

    if not DATA_FILE.is_file():
        print("Master dataset is missing.")
        print(DATA_FILE)
        return

    with DATA_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []

        count = 0
        matches = set()
        seasons = set()
        deliveries = set()
        duplicate_ids = 0

        for row in reader:
            count += 1
            matches.add(row.get("match_id", ""))
            seasons.add(row.get("season", ""))

            delivery_id = row.get("delivery_id", "")

            if delivery_id in deliveries:
                duplicate_ids += 1

            deliveries.add(delivery_id)

    print(f"Dataset       : {DATA_FILE.name}")
    print(f"Total records : {count:,}")
    print(f"Attributes    : {len(columns)}")
    print(f"Matches       : {len(matches)}")
    print(f"Seasons       : {len(seasons)}")
    print(f"Duplicate IDs : {duplicate_ids}")

    print("\nDATASET COLUMNS")
    for index, column in enumerate(columns, 1):
        print(f"{index:2}. {column}")


def performance_analysis():
    heading("2. VIRAT KOHLI PERFORMANCE ANALYSIS")

    folder = REPORT_DIR / "eda"

    overall = read_csv(folder / "overall_analysis.csv")
    seasons = read_csv(folder / "season_analysis.csv")
    phases = read_csv(folder / "phase_analysis.csv")
    bowlers = read_csv(folder / "bowler_analysis.csv")

    if overall:
        print("\nOVERALL PERFORMANCE")
        for key, value in overall[0].items():
            print(f"{key:<24}: {value}")

    if seasons:
        print("\nSEASON-WISE PERFORMANCE")
        show_table(
            seasons,
            ["Season", "Runs", "Balls", "Strike Rate", "Matches"],
            limit=25,
        )

    if phases:
        print("\nBATTING PHASE ANALYSIS")
        show_table(
            phases,
            ["Phase", "Runs", "Balls", "Strike Rate", "Boundary %"],
        )

    if bowlers:
        print("\nBOWLER MATCHUPS")
        show_table(
            bowlers,
            [
                "Bowler",
                "Runs Against",
                "Balls Faced",
                "Kohli Strike Rate",
                "Dismissals",
            ],
            limit=10,
        )


def model_comparison():
    heading("3. MACHINE LEARNING MODEL COMPARISON")

    possible_files = [
        REPORT_DIR / "model_comparison/model_summary.csv",
        REPORT_DIR / "syllabus_model_comparison/model_summary.csv",
    ]

    report = next(
        (path for path in possible_files if path.is_file()),
        None,
    )

    if report is None:
        print("Model comparison summary not found.")
        return

    rows = read_csv(report)

    for task in ("runs", "boundary", "dismissal"):
        selected = [
            row for row in rows
            if row.get("task") == task
        ]

        if not selected:
            continue

        print(f"\nTASK: {task.upper()}")

        selected.sort(
            key=lambda row: float(row["mean_score"])
        )

        show_table(
            selected,
            [
                "algorithm",
                "primary_metric",
                "mean_score",
                "std_score",
            ],
            limit=20,
        )


def load_predictor():

    from src.inference.predict import CricketIQPredictor
    return CricketIQPredictor()


def next_ball_prediction():
    heading("4. NEXT BALL PREDICTION")

    print("\nMATCH INFORMATION")

    bowler = input("Bowler: ").strip()
    opponent = input("Opponent: ").strip()
    venue = input("Venue: ").strip()

    if not all([bowler, opponent, venue]):
        print("Bowler, opponent and venue are required.")
        return

    try:
        innings = int(input("Innings (1/2): "))
        over = int(input("Zero-based over (0-19): "))
    except ValueError:
        print("Innings and over must be integers.")
        return

    if innings not in (1, 2) or not 0 <= over <= 19:
        print("Invalid innings or over.")
        return

    print("\nENTER COMPLETED KOHLI DELIVERIES")

    try:
        number = int(input("\nNumber of previous deliveries: "))

        if number < 0:
            raise ValueError

    except ValueError:
        print("Enter a non-negative integer.")
        return

    previous = []

    for index in range(number):
        print(f"\nCompleted delivery {index + 1}")

        try:
            runs = int(input("Batter runs: "))
            faced = int(input("Counts as ball faced : "))

            if runs not in range(7) or faced not in (0, 1):
                raise ValueError

            if faced == 0 and runs > 0:
                print(
                    "This simplified demo cannot represent "
                    "runs off the bat on an uncounted ball."
                )
                return

        except ValueError:
            print("Invalid delivery values.")
            return

        previous.append({
            "batter_runs": runs,
            "counts_as_ball_faced": faced,
            "is_four": int(runs == 4),
            "is_six": int(runs == 6),
        })

    print("\nLoading frozen models...")

    try:
        predictor = load_predictor()

        result = predictor.predict_from_match(
            bowler=bowler,
            opponent=opponent,
            venue=venue,
            innings=innings,
            over=over,
            previous_deliveries=previous,
        )

    except Exception as error:
        print("\nPrediction could not be completed.")
        print(f"{type(error).__name__}: {error}")
        return

    predictions = result["predictions"]

    runs_so_far = sum(d["batter_runs"] for d in previous)
    balls_so_far = sum(
        d["counts_as_ball_faced"] for d in previous
    )

    strike_rate = (
        runs_so_far / balls_so_far * 100
        if balls_so_far else 0
    )

    heading("CRICKETIQ - PREDICTION RESULT")

    print(f"Batter          : Virat Kohli")
    print(f"Bowler          : {bowler}")
    print(f"Opponent        : {opponent}")
    print(f"Venue           : {venue}")
    print(f"Innings         : {innings}")
    print(f"Current over    : {over + 1}")

    print("\nCURRENT BATTING STATISTICS")
    print(f"Runs scored     : {runs_so_far}")
    print(f"Balls faced     : {balls_so_far}")
    print(f"Strike rate     : {strike_rate:.2f}")

    print("\nNEXT DELIVERY PREDICTION")

    print(
        f"Expected runs   : "
        f"{predictions['expected_runs']:.3f}"
    )

    print(
        f"Boundary chance : "
        f"{predictions['boundary_probability']:.2%}"
    )

    print(
        f"Dismissal risk  : "
        f"{predictions['dismissal_probability']:.2%}"
    )

    run_probs = predictions.get("run_probabilities")

    if run_probs is not None:
        print("\nRUN OUTCOME PROBABILITIES")

        for run, probability in sorted(
            run_probs.items(),
            key=lambda item: int(item[0]),
        ):
            probability = float(probability)

            bar = "*" * round(probability * 40)

            print(
                f"{run} runs | {probability:7.2%} | {bar}"
            )
    print("\nPrediction generated using frozen ML models.")

def main():
    while True:
        heading("CRICKETIQ - DATA SCIENCE DEMONSTRATION")

        print("1. Dataset Overview")
        print("2. Virat Kohli Performance Analysis")
        print("3. Machine Learning Model Comparison")
        print("4. Predict Next Ball")
        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()

        actions = {
            "1": dataset_overview,
            "2": performance_analysis,
            "3": model_comparison,
            "4": next_ball_prediction,
        }

        if choice == "0":
            print("\nThank you for using CricketIQ.")
            break

        action = actions.get(choice)

        if action is None:
            print("\nInvalid option.")
            continue

        try:
            action()
        except Exception as error:
            print(
                f"\nError: {type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    main()