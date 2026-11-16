"""Expanding-window splits for time-ordered rows.

Each validation block is a contiguous slice at the end of the development
period. Training rows for a fold are every row strictly before that block, so
a later fold trains on the earlier folds' validation rows. Random k-fold is
not used: it would train on days that occur after the day being scored.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    valid_idx: np.ndarray


def expanding_window_splits(
    n_rows: int,
    n_splits: int,
    valid_size: int,
    min_train_size: int,
) -> list[Fold]:
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if valid_size < 1:
        raise ValueError("valid_size must be at least 1")
    if min_train_size < 1:
        raise ValueError("min_train_size must be at least 1")
    needed = min_train_size + n_splits * valid_size
    if n_rows < needed:
        raise ValueError(
            f"need at least {needed} rows for {n_splits} folds of {valid_size} "
            f"with min_train_size {min_train_size}, found {n_rows}"
        )

    folds: list[Fold] = []
    for index in range(n_splits):
        valid_end = n_rows - (n_splits - 1 - index) * valid_size
        valid_start = valid_end - valid_size
        folds.append(
            Fold(
                train_idx=np.arange(0, valid_start),
                valid_idx=np.arange(valid_start, valid_end),
            )
        )
    return folds
