
"""Probability utilities shared by CricketIQ prediction models.

Aligns model outputs to a consistent run-class order and applies
a small fixed probability floor for numerical stability.
"""

import numpy as np


RUN_CLASSES = np.arange(7)
PROBABILITY_FLOOR = 1e-6


def align_run_probabilities(
    probabilities: np.ndarray,
    learned_classes: np.ndarray,
) -> np.ndarray:
    """Return valid probabilities ordered by run classes 0 through 6."""

    probabilities = np.asarray(probabilities, dtype=float)
    learned_classes = np.asarray(learned_classes)

    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array.")

    if probabilities.shape[1] != len(learned_classes):
        raise ValueError("Class count does not match probability columns.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be between zero and one.")

    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Model probabilities must sum to one.")

    aligned = np.zeros(
        (len(probabilities), len(RUN_CLASSES)),
        dtype=float,
    )

    seen_classes = set()

    for index, run_class in enumerate(learned_classes):
        run_class = int(run_class)

        if run_class not in RUN_CLASSES:
            raise ValueError(f"Unexpected run class: {run_class}")

        if run_class in seen_classes:
            raise ValueError(f"Duplicate run class: {run_class}")

        seen_classes.add(run_class)
        aligned[:, run_class] = probabilities[:, index]

    # Apply the same numerical safeguard to every model.
    aligned = np.maximum(aligned, PROBABILITY_FLOOR)

    # Renormalize to obtain a valid probability distribution.
    aligned /= aligned.sum(axis=1, keepdims=True)

    return aligned
