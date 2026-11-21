
"""Finalize CricketIQ's development-stage prediction models.

Block 8C.3D

- Verify selected model artifacts and 2025 predictions.
- Estimate uncertainty using match-level bootstrap sampling.
- Compare context-only and full-feature dismissal models.
- Freeze selected candidates with reproducibility metadata.
- Keep IPL 2026 completely untouched.
"""

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from src.Config import MODELS_DIR, PROCESSED_DIR, REPORT_DIR
from src.features.model_features import MODEL_FEATURES, TARGETS
from src.models.model_selection import (
    get_model_configuration,
    get_selected_features,
    validate_model_configurations,
)
from src.models.probability_utils import align_run_probabilities
from src.models.train_logistic import create_pipeline


DATA_DIR = PROCESSED_DIR / "ml"
PREDICTION_DIR = REPORT_DIR / "selected_validation"
CANDIDATE_DIR = MODELS_DIR / "candidates"

FINAL_MODEL_DIR = MODELS_DIR / "final"
FINAL_REPORT_DIR = REPORT_DIR / "final"

TRAIN_FILE = DATA_DIR / "kohli_train_features.csv"
VALIDATION_FILE = DATA_DIR / "kohli_validation_features.csv"

TASKS = ("runs", "boundary", "dismissal")
RUN_CLASSES = np.arange(7)

N_BOOTSTRAPS = 2000
RANDOM_SEED = 42
LAPLACE_ALPHA = 1.0

# Fixed before final evaluation; not tuned on 2026.
PROBABILITY_FLOOR = 1e-6


def file_sha256(path: Path) -> str:
    """Calculate a SHA-256 checksum for artifact verification."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical training and 2025 validation data."""

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)

    required = (
        set(MODEL_FEATURES)
        | set(TARGETS.values())
        | {"delivery_id", "match_id", "season"}
    )

    for name, data in [("train", train), ("validation", validation)]:
        if data.empty:
            raise ValueError(f"{name} dataset is empty.")

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing {name} columns: {sorted(missing)}"
            )

        if data[list(required)].isna().any().any():
            raise ValueError(f"Missing values in {name}.")

        if data["delivery_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {name}.")

    if not train["season"].between(2008, 2024).all():
        raise ValueError("Unexpected training seasons.")

    if not validation["season"].eq(2025).all():
        raise ValueError("Validation must contain only 2025.")

    for identifier in ("delivery_id", "match_id"):
        overlap = set(train[identifier]) & set(validation[identifier])

        if overlap:
            raise ValueError(f"Overlapping {identifier} values.")

    return train, validation


def get_baseline(training: pd.DataFrame, task: str):
    """Calculate historical probabilities without validation outcomes."""

    target = TARGETS[task]

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


def load_candidate_predictions(
    task: str,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Verify saved predictions against the validation dataset."""

    path = PREDICTION_DIR / f"{task}_predictions_2025.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")

    predictions = pd.read_csv(path)

    if predictions["delivery_id"].duplicated().any():
        raise ValueError(f"Duplicate predictions: {task}")

    if set(predictions["delivery_id"]) != set(validation["delivery_id"]):
        raise ValueError(f"Prediction IDs differ for {task}.")

    predictions = validation[
        ["delivery_id", "match_id", "season", TARGETS[task]]
    ].merge(
        predictions,
        on="delivery_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_saved"),
    )

    if task == "runs":
        actual = predictions[TARGETS[task]].to_numpy(dtype=int)

        if not np.array_equal(
            actual,
            predictions["actual_runs"].to_numpy(dtype=int),
        ):
            raise ValueError("Saved runs targets do not match.")

        columns = [f"prob_runs_{r}" for r in RUN_CLASSES]
        probabilities = predictions[columns].to_numpy(dtype=float)

    else:
        actual = predictions[TARGETS[task]].to_numpy(dtype=int)

        if not np.array_equal(
            actual,
            predictions["actual"].to_numpy(dtype=int),
        ):
            raise ValueError(f"Saved {task} targets do not match.")

        probabilities = predictions[
            "predicted_probability"
        ].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(f"Invalid probabilities for {task}.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"Probabilities outside [0,1]: {task}")

    if task == "runs":
        if not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError("Run probabilities do not sum to one.")

    return predictions


def materialize_candidate_inputs(
    validation: pd.DataFrame,
) -> None:
    """Create missing 2025 candidate inputs from selected local artifacts."""

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        candidate_path = CANDIDATE_DIR / f"selected_{task}_2025.joblib"
        prediction_path = PREDICTION_DIR / f"{task}_predictions_2025.csv"

        if candidate_path.is_file() and prediction_path.is_file():
            continue

        source_path = (
            candidate_path
            if candidate_path.is_file()
            else MODELS_DIR / f"{task}.joblib"
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Cannot create {task} candidate inputs; selected model "
                f"is missing: {source_path}"
            )

        model = joblib.load(source_path)
        features = get_selected_features(task)

        if list(model.feature_names_in_) != features:
            raise ValueError(
                f"Selected model feature schema mismatch: {task}"
            )

        if not candidate_path.is_file():
            shutil.copy2(source_path, candidate_path)

        if prediction_path.is_file():
            continue

        probabilities = model.predict_proba(validation[features])
        classes = model.named_steps["classifier"].classes_

        predictions = validation[["delivery_id"]].copy()

        if task == "runs":
            aligned = align_run_probabilities(probabilities, classes)
            predictions["actual_runs"] = validation[TARGETS[task]].to_numpy(
                dtype=int
            )

            for run_class in RUN_CLASSES:
                predictions[f"prob_runs_{run_class}"] = aligned[:, run_class]
        else:
            positive_index = list(classes).index(1)
            predictions["actual"] = validation[TARGETS[task]].to_numpy(
                dtype=int
            )
            predictions["predicted_probability"] = (
                probabilities[:, positive_index]
            )

        predictions.to_csv(prediction_path, index=False)


def verify_candidate_model(
    task: str,
    validation: pd.DataFrame,
    predictions: pd.DataFrame,
):
    """Verify that the saved pipeline reproduces 2025 predictions."""

    path = CANDIDATE_DIR / f"selected_{task}_2025.joblib"

    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate model: {path}")

    # Only load model artifacts generated locally by this project.
    model = joblib.load(path)

    features = get_selected_features(task)

    if list(model.feature_names_in_) != features:
        raise ValueError(f"Model feature schema mismatch: {task}")

    ordered = validation.set_index("delivery_id").loc[
        predictions["delivery_id"]
    ]

    raw = model.predict_proba(ordered[features])
    classes = model.named_steps["classifier"].classes_

    if task == "runs":
        reproduced = align_run_probabilities(raw, classes)

        columns = [f"prob_runs_{r}" for r in RUN_CLASSES]
        saved = predictions[columns].to_numpy(dtype=float)

    else:
        positive_index = list(classes).index(1)
        reproduced = raw[:, positive_index]

        saved = predictions[
            "predicted_probability"
        ].to_numpy(dtype=float)

    if not np.allclose(reproduced, saved, atol=1e-7, rtol=1e-6):
        raise ValueError(
            f"Saved model does not reproduce {task} predictions."
        )

    return path


def bootstrap_reliability(
    task: str,
    predictions: pd.DataFrame,
    baseline,
) -> dict:
    """Estimate paired error improvement using complete-match resampling."""

    if task == "runs":
        actual = predictions["actual_runs"].to_numpy(dtype=int)

        columns = [f"prob_runs_{r}" for r in RUN_CLASSES]
        probabilities = predictions[columns].to_numpy(dtype=float)

        model_probability = probabilities[
            np.arange(len(actual)), actual
        ]

        baseline_probability = np.asarray(baseline)[actual]

        model_errors = -np.log(
            np.clip(model_probability, PROBABILITY_FLOOR, 1)
        )

        baseline_errors = -np.log(
            np.clip(baseline_probability, PROBABILITY_FLOOR, 1)
        )

        metric = "log_loss"

    else:
        actual = predictions["actual"].to_numpy(dtype=int)

        probabilities = predictions[
            "predicted_probability"
        ].to_numpy(dtype=float)

        model_errors = (probabilities - actual) ** 2
        baseline_errors = (float(baseline) - actual) ** 2

        metric = "brier_score"

    errors = pd.DataFrame({
        "match_id": predictions["match_id"].to_numpy(),
        "baseline_error": baseline_errors,
        "model_error": model_errors,
    })

    grouped = errors.groupby("match_id", sort=True).agg(
        n=("model_error", "size"),
        baseline_sum=("baseline_error", "sum"),
        model_sum=("model_error", "sum"),
    )

    clusters = grouped.to_numpy(dtype=float)
    match_count = len(clusters)

    if match_count < 2:
        raise ValueError("At least two matches are required.")

    rng = np.random.default_rng(RANDOM_SEED)
    improvements = []

    for _ in range(N_BOOTSTRAPS):
        indices = rng.integers(0, match_count, size=match_count)
        sample = clusters[indices]

        n = sample[:, 0].sum()

        baseline_error = sample[:, 1].sum() / n
        model_error = sample[:, 2].sum() / n

        improvements.append(baseline_error - model_error)

    lower, upper = np.percentile(improvements, [2.5, 97.5])

    observed_baseline = float(baseline_errors.mean())
    observed_model = float(model_errors.mean())

    return {
        "task": task,
        "metric": metric,
        "matches": match_count,
        "deliveries": len(predictions),
        "baseline_error": observed_baseline,
        "model_error": observed_model,
        "improvement": observed_baseline - observed_model,
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "bootstrap_iterations": N_BOOTSTRAPS,
        "interval_excludes_zero": bool(lower > 0 or upper < 0),
    }


def compare_dismissal_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the frozen context candidate with a full-feature alternative."""

    target = TARGETS["dismissal"]
    records = []

    experiments = [
        ("context", get_selected_features("dismissal"), 0.01),
        ("full", MODEL_FEATURES, 1.0),
    ]

    for name, features, c_value in experiments:
        model = create_pipeline(
            selected_features=features,
            c_value=c_value,
        )

        model.fit(train[features], train[target])

        classes = model.named_steps["classifier"].classes_
        positive_index = list(classes).index(1)

        probabilities = model.predict_proba(
            validation[features]
        )[:, positive_index]

        actual = validation[target]

        records.append({
            "configuration": name,
            "feature_count": len(features),
            "C": c_value,
            "brier_score": float(
                brier_score_loss(actual, probabilities)
            ),
            "log_loss": float(
                log_loss(actual, probabilities, labels=[0, 1])
            ),
            "average_precision": float(
                average_precision_score(actual, probabilities)
            ),
            "mean_predicted_probability": float(
                probabilities.mean()
            ),
        })

    return pd.DataFrame(records)


def freeze_models(
    verified_paths: dict[str, Path],
) -> dict:
    """Copy verified candidate artifacts and record reproducibility metadata."""

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "project": "CricketIQ",
        "status": "frozen_development_candidates",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seasons": "2008-2024",
        "development_validation_season": 2025,
        "final_test_season": 2026,
        "final_test_evaluated": False,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "probability_floor": PROBABILITY_FLOOR,
        "models": {},
    }

    for task, source in verified_paths.items():
        destination = FINAL_MODEL_DIR / f"{task}.joblib"

        shutil.copy2(source, destination)

        configuration = get_model_configuration(task)

        metadata["models"][task] = {
            "algorithm": configuration.algorithm,
            "feature_set": configuration.feature_set,
            "features": get_selected_features(task),
            "regularization_c": configuration.regularization_c,
            "artifact": destination.name,
            "sha256": file_sha256(destination),
        }

    metadata_path = FINAL_MODEL_DIR / "model_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    """Verify, evaluate, and freeze CricketIQ's selected candidates."""

    print("=" * 60)
    print("CRICKETIQ - FINAL MODEL DEVELOPMENT CHECKPOINT")
    print("=" * 60)

    validate_model_configurations()

    train, validation = load_datasets()

    FINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    materialize_candidate_inputs(validation)

    verified_models = {}
    reliability_results = []

    for task in TASKS:
        print(f"\nChecking {task} model...")

        predictions = load_candidate_predictions(task, validation)

        verified_models[task] = verify_candidate_model(
            task, validation, predictions
        )

        baseline = get_baseline(train, task)

        reliability = bootstrap_reliability(
            task, predictions, baseline
        )

        reliability_results.append(reliability)

        print(
            f"Improvement: {reliability['improvement']:+.6f}"
        )
        print(
            f"95% CI: [{reliability['ci_lower_95']:.6f}, "
            f"{reliability['ci_upper_95']:.6f}]"
        )

    reliability_report = pd.DataFrame(reliability_results)

    reliability_report.to_csv(
        FINAL_REPORT_DIR / "bootstrap_summary.csv",
        index=False,
    )

    dismissal_comparison = compare_dismissal_models(
        train, validation
    )

    dismissal_comparison.to_csv(
        FINAL_REPORT_DIR / "dismissal_comparison.csv",
        index=False,
    )

    # Freeze the previously selected candidates, not whichever
    # alternative happens to win the additional 2025 comparison.
    metadata = freeze_models(verified_models)

    final_report = {
        "status": "development_models_frozen",
        "bootstrap_reliability": reliability_results,
        "dismissal_diagnostics": dismissal_comparison.to_dict(
            orient="records"
        ),
        "model_metadata": metadata,
        "notes": [
            "2025 was used during model development.",
            "Bootstrap intervals reflect match-sampling uncertainty.",
            "The 2026 test set remains unopened.",
            "Frozen models must not be tuned using 2026 outcomes.",
        ],
    }

    (FINAL_REPORT_DIR / "final_report.json").write_text(
        json.dumps(final_report, indent=2),
        encoding="utf-8",
    )

    print("\nRELIABILITY SUMMARY")
    print(reliability_report.to_string(index=False))

    print("\nDISMISSAL COMPARISON")
    print(dismissal_comparison.to_string(index=False))



if __name__ == "__main__":
    main()
