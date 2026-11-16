import numpy as np
import pytest

from forecasting.evaluation.splits import expanding_window_splits


def test_expanding_windows_are_contiguous_and_causal():
    folds = expanding_window_splits(n_rows=1000, n_splits=4, valid_size=90, min_train_size=200)
    assert len(folds) == 4
    covered = np.concatenate([fold.valid_idx for fold in folds])
    assert len(covered) == len(set(covered.tolist()))
    assert folds[-1].valid_idx[-1] == 999
    assert folds[0].valid_idx[0] == 1000 - 360
    previous_train = -1
    previous_valid_end = None
    for fold in folds:
        assert fold.train_idx.max() < fold.valid_idx.min()
        assert len(fold.valid_idx) == 90
        assert len(fold.train_idx) > previous_train
        if previous_valid_end is not None:
            assert fold.valid_idx[0] == previous_valid_end
        previous_train = len(fold.train_idx)
        previous_valid_end = fold.valid_idx[-1] + 1


def test_splits_reject_a_series_that_is_too_short():
    with pytest.raises(ValueError, match="need at least"):
        expanding_window_splits(n_rows=100, n_splits=4, valid_size=90, min_train_size=200)
