import numpy as np
import pytest

from facerec.verify import accuracy_at, best_threshold, tar_at_far, ten_fold


def test_best_threshold_separates_cleanly_separable_pairs():
    d = np.array([0.2, 0.4, 0.5, 1.1, 1.3])
    same = np.array([True, True, True, False, False])
    t = best_threshold(d, same)
    assert 0.5 < t < 1.1
    assert accuracy_at(d, same, t) == 1.0


def test_ten_fold_never_chooses_a_threshold_using_the_fold_it_tests():
    # Fold 1 is labelled the opposite way to every other fold, so a threshold chosen on the
    # other folds must get all of fold 1 wrong.
    d = np.tile([0.3, 1.2], 10)
    folds = np.repeat(np.arange(10), 2)
    same = np.tile([True, False], 10)
    same[folds == 1] = ~same[folds == 1]
    result = ten_fold(d, same, folds)
    assert result["fold_accuracy"][1] == 0.0
    assert result["accuracy"] == pytest.approx(0.9)


def test_tar_at_far_respects_the_false_accept_limit():
    d = np.array([0.1, 0.2, 0.3, 0.9, 0.25, 1.0, 1.1, 1.2])
    same = np.array([True, True, True, True, False, False, False, False])
    tar, t = tar_at_far(d, same, far=0.0)
    assert ((d <= t) & ~same).sum() == 0
    assert tar == 0.5  # 0.1 and 0.2 only: accepting 0.3 would also accept the stranger at 0.25
