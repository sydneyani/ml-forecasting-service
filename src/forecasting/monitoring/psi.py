"""Population stability index.

PSI summarizes how a current sample differs from a reference sample.
Values under 0.1 are a small shift, 0.1 to 0.2 are worth a look, and 0.2
or more is the usual threshold for opening an investigation. The threshold
is not a deploy gate by itself: a feature can move because the world moved
(solar capacity) while the model is still the right one to ship.
"""

import numpy as np


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """PSI of `current` against `reference`. Non-finite values are ignored."""
    reference = _finite(reference)
    current = _finite(current)
    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI needs finite values in both samples")
    unique = np.unique(reference)
    if unique.size <= 10:
        return _categorical_psi(reference, current)
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return _categorical_psi(reference, current)
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    return _psi_from_counts(reference_counts, current_counts)


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _categorical_psi(reference: np.ndarray, current: np.ndarray) -> float:
    categories = np.unique(np.concatenate([reference, current]))
    reference_counts = np.array([(reference == category).sum() for category in categories])
    current_counts = np.array([(current == category).sum() for category in categories])
    return _psi_from_counts(reference_counts, current_counts)


def _psi_from_counts(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
    reference_share = _share(reference_counts)
    current_share = _share(current_counts)
    terms = (current_share - reference_share) * np.log(current_share / reference_share)
    return float(np.sum(terms))


def _share(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    share = counts.astype(float) / total
    share = np.clip(share, 1e-6, None)
    return share / share.sum()
