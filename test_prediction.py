
"""Smoke test for CricketIQ's next-ball prediction engine."""

from src.predict import CricketIQPredictor


def main():
    """Generate a sample prediction using completed deliveries."""

    predictor = CricketIQPredictor()

    previous_deliveries = [
        {
            "batter_runs": 1,
            "counts_as_ball_faced": 1,
            "is_four": 0,
            "is_six": 0,
        },
        {
            "batter_runs": 4,
            "counts_as_ball_faced": 1,
            "is_four": 1,
            "is_six": 0,
        },
        {
            "batter_runs": 0,
            "counts_as_ball_faced": 0,
            "is_four": 0,
            "is_six": 0,
        },
    ]

    result = predictor.predict_from_match(
        bowler="Jasprit Bumrah",
        opponent="Mumbai Indians",
        venue="Wankhede Stadium",
        innings=1,
        over=4,
        previous_deliveries=previous_deliveries,
    )

    print("\nCRICKETIQ - NEXT BALL PREDICTION")
    print("-" * 45)
    print(f"Bowler: {result['bowler']}")
    print(f"Expected runs: {result['predictions']['expected_runs']}")
    print(
        "Boundary probability: "
        f"{result['predictions']['boundary_probability']:.2%}"
    )
    print(
        "Dismissal probability: "
        f"{result['predictions']['dismissal_probability']:.2%}"
    )


if __name__ == "__main__":
    main()
