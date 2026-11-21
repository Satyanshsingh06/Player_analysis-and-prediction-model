
"""Evaluate CricketIQ's frozen models on the held-out IPL 2026 season.


This script:
- Verifies frozen model artifacts and metadata.
- Checks chronological and match-level separation.
- Evaluates three prediction tasks without retraining.
- Compares models against training-only frequency baselines.
- Saves delivery-level predictions and consolidated metrics.

Do not use the test results for model tuning or selection.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
)

try:
    from src.Config import PROCESSED_DIR, REPORT_DIR, MODELS_DIR
    from src.models.probability_utils import align_run_probabilities
except ImportError:
    from ..Config import PROCESSED_DIR, REPORT_DIR, MODELS_DIR
    from ..models.probability_utils import align_run_probabilities


ML_DATA_DIR = PROCESSED_DIR / "ml"
DATA_DIR = ML_DATA_DIR if (ML_DATA_DIR / "kohli_train_features.csv").is_file() else PROCESSED_DIR

TRAIN_FILE = DATA_DIR / "kohli_train_features.csv"
VALIDATION_FILE = DATA_DIR / "kohli_validation_features.csv"
TEST_FILE = DATA_DIR / "kohli_test_features.csv"

MODEL_DIR = MODELS_DIR / "final"
if not (MODEL_DIR / "model_metadata.json").is_file():
    MODEL_DIR = MODELS_DIR
OUTPUT_DIR = REPORT_DIR / "final"

METADATA_FILE = MODEL_DIR / "model_metadata.json"

TASKS = {
    "runs": "target_runs",
    "boundary": "target_boundary",
    "dismissal": "target_dismissal",
}

RUN_CLASSES = np.arange(7)
LAPLACE_ALPHA = 1.0
PROBABILITY_FLOOR = 1e-6

# A completion marker prevents accidental repeated final evaluation.
COMPLETION_FILE = OUTPUT_DIR / "test_evaluation_complete.json"


def calculate_sha256(file_path: Path) -> str:
    """Calculate the checksum of a frozen model artifact."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_metadata() -> dict:
    """Load and validate the frozen model manifest."""

    if not METADATA_FILE.is_file():
        raise FileNotFoundError(
            f"Model metadata missing: {METADATA_FILE}"
        )

    metadata = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )

    if metadata.get("status") != "frozen_development_candidates":
        raise ValueError("Models are not marked as frozen.")

    if metadata.get("final_test_evaluated") is not False:
        raise ValueError("Metadata does not confirm an unused test.")

    if metadata.get("training_seasons") != "2008-2024":
        raise ValueError("Unexpected training period.")

    if metadata.get("final_test_season") != 2026:
        raise ValueError("Unexpected final test season.")

    if set(metadata.get("models", {})) != set(TASKS):
        raise ValueError("Frozen model manifest is incomplete.")

    if not np.isclose(
        float(metadata.get("probability_floor", -1)),
        PROBABILITY_FLOOR,
    ):
        raise ValueError("Probability floor differs from frozen settings.")

    return metadata


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load datasets and verify chronological separation."""

    paths = {
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
        "test": TEST_FILE,
    }

    datasets = {}

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

        data = pd.read_csv(path)

        if data.empty:
            raise ValueError(f"{name} dataset is empty.")

        required = {
            "delivery_id", "match_id", "season",
            *TASKS.values(),
        }

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

        if data[list(required)].isna().any().any():
            raise ValueError(f"Missing required values in {name}.")

        if data["delivery_id"].duplicated().any():
            raise ValueError(f"Duplicate delivery IDs in {name}.")

        datasets[name] = data

    train = datasets["train"]
    validation = datasets["validation"]
    test = datasets["test"]

    if not train["season"].between(2008, 2024).all():
        raise ValueError("Unexpected training seasons.")

    if not validation["season"].eq(2025).all():
        raise ValueError("Unexpected validation seasons.")

    if not test["season"].eq(2026).all():
        raise ValueError("Test data must contain only IPL 2026.")

    for identifier in ("delivery_id", "match_id"):
        train_ids = set(train[identifier])
        validation_ids = set(validation[identifier])
        test_ids = set(test[identifier])

        if (
            train_ids & validation_ids
            or train_ids & test_ids
            or validation_ids & test_ids
        ):
            raise ValueError(
                f"Overlapping {identifier} across datasets."
            )

    for task in ("boundary", "dismissal"):
        for name, data in datasets.items():
            if not data[TASKS[task]].isin([0, 1]).all():
                raise ValueError(
                    f"Invalid {task} targets in {name}."
                )

    for name, data in datasets.items():
        if not data["target_runs"].isin(RUN_CLASSES).all():
            raise ValueError(f"Invalid run targets in {name}.")

    return train, validation, test


def load_frozen_model(task: str, metadata: dict):
    """Verify checksum and feature schema before loading a model."""

    info = metadata["models"][task]

    expected_name = f"{task}.joblib"

    if info.get("artifact") != expected_name:
        raise ValueError(f"Unexpected artifact for {task}.")

    path = MODEL_DIR / expected_name

    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen model: {path}")

    if calculate_sha256(path) != info["sha256"]:
        raise ValueError(f"Checksum mismatch for {task}.")

    features = info["features"]

    if not features or len(features) != len(set(features)):
        raise ValueError(f"Invalid feature schema for {task}.")

    # Load only trusted artifacts generated by this project.
    model = joblib.load(path)

    if list(model.feature_names_in_) != features:
        raise ValueError(
            f"Saved model feature schema mismatch: {task}."
        )

    return model, features


def get_baseline(training: pd.DataFrame, task: str):
    """Estimate historical outcome probabilities from training only."""

    target = TASKS[task]

    if task == "runs":
        counts = (
            training[target]
            .value_counts()
            .reindex(RUN_CLASSES, fill_value=0)
            .to_numpy(dtype=float)
        )

        counts += LAPLACE_ALPHA
        return counts / counts.sum()

    return float(training[target].mean())


def evaluate_runs(
    model,
    features: pd.DataFrame,
    actual: pd.Series,
    baseline: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate multiclass run predictions and expected runs."""

    learned_classes = model.named_steps[
        "classifier"
    ].classes_

    probabilities = align_run_probabilities(
        model.predict_proba(features),
        learned_classes,
    )

    if probabilities.shape != (len(features), 7):
        raise ValueError("Incorrect run probability shape.")

    predicted_runs = RUN_CLASSES[
        probabilities.argmax(axis=1)
    ]

    expected_runs = probabilities @ RUN_CLASSES

    actual_values = actual.to_numpy(dtype=int)

    baseline_predictions = np.tile(
        baseline, (len(actual_values), 1)
    )

    metrics = {
        "model_error": float(
            log_loss(
                actual_values,
                probabilities,
                labels=RUN_CLASSES,
            )
        ),
        "baseline_error": float(
            log_loss(
                actual_values,
                baseline_predictions,
                labels=RUN_CLASSES,
            )
        ),
        "accuracy": float(
            accuracy_score(actual_values, predicted_runs)
        ),
        "expected_runs_mae": float(
            mean_absolute_error(actual_values, expected_runs)
        ),
        "mean_expected_runs": float(expected_runs.mean()),
        "mean_actual_runs": float(actual_values.mean()),
    }

    predictions = pd.DataFrame({
        "actual": actual_values,
        "predicted_class": predicted_runs,
        "expected_runs": expected_runs,
    })

    for run_class in RUN_CLASSES:
        predictions[f"prob_runs_{run_class}"] = (
            probabilities[:, run_class]
        )

    return metrics, predictions


def evaluate_binary(
    model,
    features: pd.DataFrame,
    actual: pd.Series,
    baseline: float,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate binary probability quality and event ranking."""

    classes = model.named_steps["classifier"].classes_

    positive_index = list(classes).index(1)

    probabilities = model.predict_proba(
        features
    )[:, positive_index]

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Invalid binary model probabilities.")

    actual_values = actual.to_numpy(dtype=int)

    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    baseline_predictions = np.full(
        len(actual_values),
        baseline,
        dtype=float,
    )

    metrics = {
        "model_error": float(
            brier_score_loss(actual_values, probabilities)
        ),
        "baseline_error": float(
            brier_score_loss(
                actual_values, baseline_predictions
            )
        ),
        "log_loss": float(
            log_loss(
                actual_values, probabilities, labels=[0, 1]
            )
        ),
        "average_precision": float(
            average_precision_score(
                actual_values, probabilities
            )
        ) if actual_values.sum() > 0 else None,
        "accuracy": float(
            accuracy_score(actual_values, predicted_labels)
        ),
        "actual_event_rate": float(actual_values.mean()),
        "mean_predicted_probability": float(
            probabilities.mean()
        ),
    }

    predictions = pd.DataFrame({
        "actual": actual_values,
        "predicted_class": predicted_labels,
        "predicted_probability": probabilities,
    })

    return metrics, predictions


def evaluate_all_models(
    training: pd.DataFrame,
    test: pd.DataFrame,
    metadata: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all three frozen models without fitting anything."""

    metric_records = []
    prediction_records = []

    for task, target in TASKS.items():
        model, selected_features = load_frozen_model(
            task, metadata
        )

        missing = set(selected_features) - set(test.columns)

        if missing:
            raise ValueError(
                f"Missing test features: {sorted(missing)}"
            )

        if test[selected_features].isna().any().any():
            raise ValueError(f"Missing test feature values: {task}")

        baseline = get_baseline(training, task)

        if task == "runs":
            metrics, predictions = evaluate_runs(
                model,
                test[selected_features],
                test[target],
                baseline,
            )
            metric_name = "log_loss"
        else:
            metrics, predictions = evaluate_binary(
                model,
                test[selected_features],
                test[target],
                baseline,
            )
            metric_name = "brier_score"

        improvement = (
            metrics["baseline_error"] - metrics["model_error"]
        )

        metric_records.append({
            "task": task,
            "metric": metric_name,
            "test_deliveries": len(test),
            **metrics,
            "absolute_improvement": improvement,
            "relative_improvement_pct": (
                100 * improvement / metrics["baseline_error"]
            ),
            "beats_baseline": bool(improvement > 0),
        })

        predictions.insert(
            0, "delivery_id", test["delivery_id"].to_numpy()
        )
        predictions.insert(
            1, "match_id", test["match_id"].to_numpy()
        )
        predictions.insert(
            2, "season", test["season"].to_numpy()
        )
        predictions.insert(3, "task", task)

        prediction_records.append(predictions)

        print(
            f"{task:<12} | {metric_name:<12} "
            f"| model={metrics['model_error']:.5f} "
            f"| baseline={metrics['baseline_error']:.5f}"
        )

    return (
        pd.DataFrame(metric_records),
        pd.concat(prediction_records, ignore_index=True),
    )


def save_results(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    metadata: dict,
) -> None:
    """Save test results and record that evaluation was completed."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = metrics[
        [
            "task", "metric", "baseline_error", "model_error",
            "absolute_improvement", "relative_improvement_pct",
            "beats_baseline",
        ]
    ].copy()

    metrics.to_csv(
        OUTPUT_DIR / "test_metrics_2026.csv", index=False
    )

    comparison.to_csv(
        OUTPUT_DIR / "test_comparison_2026.csv", index=False
    )

    predictions.to_csv(
        OUTPUT_DIR / "test_predictions_2026.csv", index=False
    )

    report = {
        "project": "CricketIQ",
        "evaluation_season": 2026,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_status": metadata["status"],
        "model_checksums": {
            task: metadata["models"][task]["sha256"]
            for task in TASKS
        },
        "results": json.loads(
            metrics.to_json(orient="records")
        ),
        "note": (
            "Held-out final evaluation. Models were not retrained "
            "or tuned on 2026 outcomes."
        ),
    }

    (OUTPUT_DIR / "test_report_2026.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Write completion marker only after all reports are saved.
    COMPLETION_FILE.write_text(
        json.dumps({
            "completed_at_utc": report["evaluated_at_utc"],
            "evaluation_season": 2026,
            "model_checksums": report["model_checksums"],
        }, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Execute the final held-out evaluation exactly once."""


    if COMPLETION_FILE.exists():
        print(
            "Final evaluation was already completed. "
            f"Review the saved report in {OUTPUT_DIR}."
        )
        return

    metadata = load_metadata()

    training, validation, test = load_datasets()

    print(f"Training deliveries: {len(training):,}")
    print(f"Validation deliveries: {len(validation):,}")
    print(f"Test deliveries: {len(test):,}")

    metrics, predictions = evaluate_all_models(
        training, test, metadata
    )

    save_results(metrics, predictions, metadata)

    print("\nFINAL TEST RESULTS")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
