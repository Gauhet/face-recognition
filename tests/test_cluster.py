import numpy as np
import pytest

from facerec.cluster import bcubed, cluster, make_album, pairwise_scores


def test_well_separated_people_form_one_group_each():
    rng = np.random.default_rng(0)
    points = np.concatenate([c + 0.02 * rng.standard_normal((4, 3)) for c in np.eye(3)])
    labels = cluster(points, threshold=0.5)
    assert len(set(labels)) == 3
    assert all(len(set(labels[i * 4 : i * 4 + 4])) == 1 for i in range(3))


def test_a_single_photo_is_its_own_group():
    assert cluster(np.ones((1, 4)), threshold=1.0).tolist() == [0]


def test_perfect_grouping_scores_one():
    true = np.array([0, 0, 1, 2, 2, 2])
    assert pairwise_scores(true, true)["f1"] == 1.0
    assert bcubed(true, true)["f1"] == 1.0


def test_everything_in_one_group_has_full_recall_but_low_precision():
    true = np.array([0, 0, 1, 1])
    pred = np.zeros(4, dtype=int)
    pair = pairwise_scores(true, pred)
    assert pair["recall"] == 1.0 and pair["precision"] == pytest.approx(2 / 6)
    b = bcubed(true, pred)
    assert b["recall"] == 1.0 and b["precision"] == 0.5


def test_album_caps_photos_per_person():
    photos = {"a": list(range(20)), "b": [100, 101]}
    idx, labels = make_album(photos, ["a", "b"], max_photos=5, rng=np.random.default_rng(0))
    assert (labels == 0).sum() == 5 and (labels == 1).sum() == 2
    assert set(idx[labels == 1]) == {100, 101}
