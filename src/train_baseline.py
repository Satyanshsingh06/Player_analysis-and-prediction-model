
"""Train and evaluate CricketIQ's statistical baseline models.

The baselines estimate outcome probabilities using training data only.
Validation data is used for evaluation, while the 2026 test set remains
untouched until final model selection.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from Config import PROCESSED_DIR, REPORT_DIR
DATA_DIRECTORY = PROCESSED_DIR / "ml"
TRAIN_FILE = DATA_DIRECTORY / "kohli_train_features.csv"
VALIDATION_FILE = DATA_DIRECTORY / "kohli_validation_features.csv"

# The complete set of possible batter-run outcomes.
RUN_CLASSES = np.arange(7)
TARGET_COLUMNS = [
    "target_runs",
    "target_boundary",
    "target_dismissal",
]

# A small pseudocount prevents zero-probability predictions.
# This is important because five runs does not occur in training.
SMOOTHING_ALPHA = 1.0

def load_dataset(file_path: Path) -> pd.DataFrame: # Load a feature dataset and verify the prediction targets.
    dataset = pd.read_csv(file_path)
    return dataset


def calculate_run_probabilities(
    training_data: pd.DataFrame,
) -> np.ndarray: # Estimate run-class probabilities from historical frequencies.

    counts = (
        training_data["target_runs"]
        .value_counts()
        .reindex(RUN_CLASSES, fill_value=0)
        .to_numpy(dtype=float)
    )

    # Laplace smoothing assigns a small nonzero probability
    # to classes that were absent from the training data.
    smoothed_counts = counts + SMOOTHING_ALPHA
    probabilities = smoothed_counts / smoothed_counts.sum()
    return probabilities


def evaluate_run_baseline(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> dict:  # Evaluate constant multiclass run probabilities.
    class_probabilities = calculate_run_probabilities(training_data)
    # Every validation delivery receives the same class distribution.
    predictions = np.tile(
        class_probabilities,
        (len(validation_data), 1),
    )

    actual_runs = validation_data["target_runs"].to_numpy()
    predicted_classes = RUN_CLASSES[
        np.argmax(predictions, axis=1)
    ]
    metrics = {
        "model": "Run frequency baseline",
        "target": "target_runs",
        "log_loss": log_loss(
            actual_runs,
            predictions,
            labels=RUN_CLASSES,
        ),
        "accuracy": accuracy_score(
            actual_runs,
            predicted_classes,
        ),
    }

# from here run prediction baseline strts.
# where we find class probability.
    for run_value, probability in zip(
        RUN_CLASSES, class_probabilities
    ):
        print(f"  {run_value} runs: {probability:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def evaluate_binary_baseline(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    target_column: str,
    model_name: str,
) -> dict:
    """Evaluate a constant-probability binary classifier."""

    historical_probability = training_data[target_column].mean()

    # Constant predictions provide a reference for future models.
    predicted_probabilities = np.full(
        len(validation_data),
        historical_probability,
        dtype=float,
    )

    actual_outcomes = validation_data[target_column].to_numpy()

    predicted_labels = (
        predicted_probabilities >= 0.5
    ).astype(int)

    metrics = {
        "model": model_name,
        "target": target_column,
        "brier_score": brier_score_loss(
            actual_outcomes,
            predicted_probabilities,
        ),
        "average_precision": average_precision_score(
            actual_outcomes,
            predicted_probabilities,
        ),
        "accuracy": accuracy_score(
            actual_outcomes,
            predicted_labels,
        ),
    }

    print(f"\n{model_name.upper()}")
    print(
        f"Historical probability: "
        f"{historical_probability:.4f}"
    )
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(
        f"Average precision: "
        f"{metrics['average_precision']:.4f}"
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def save_metrics(metrics: list[dict]) -> None: # Save validation results for comparison with future models.

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = REPORT_DIR / "baseline_metrics.csv"

    results = pd.DataFrame(metrics)

    results.to_csv(output_file, index=False)

    print(f"\nSaved baseline metrics: {output_file}")


def main() -> None:  #  Execute the baseline evaluation pipeline.


    train_data = load_dataset(TRAIN_FILE)
    validation_data = load_dataset(VALIDATION_FILE)

    # Verify that model development does not mix delivery records.
    overlapping_ids = set(train_data["delivery_id"]) & set(
        validation_data["delivery_id"]
    )

    if overlapping_ids:
        raise ValueError(
            "Training and validation deliveries overlap."
        )
    if not train_data["season"].le(2024).all():
        raise ValueError("Training data contains future seasons.")
    if not validation_data["season"].eq(2025).all():
        raise ValueError("Validation data contains incorrect seasons.")
    results = [
        evaluate_run_baseline(train_data, validation_data),
        evaluate_binary_baseline(
            train_data,
            validation_data,
            "target_boundary",
            "Boundary frequency baseline",
        ),
        evaluate_binary_baseline(
            train_data,
            validation_data,
            "target_dismissal",
            "Dismissal frequency baseline",
        ),
    ]
    save_metrics(results)

if __name__ == "__main__":
    main()
