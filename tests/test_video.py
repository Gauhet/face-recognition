import numpy as np
import pytest

from facerec.video import FaceTracker, box_iou, label_changes


def test_box_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]])
    assert box_iou(a, b)[0] == pytest.approx([50 / 150, 0])


def test_a_moving_face_keeps_its_track_and_a_new_face_gets_a_new_one():
    tracker = FaceTracker()
    first = tracker.update(0, np.array([[0, 0, 10, 10]]))
    second = tracker.update(1, np.array([[1, 0, 11, 10], [50, 50, 60, 60]]))
    assert second[0] is first[0]
    assert second[1].id != first[0].id


def test_two_faces_are_not_given_the_same_track():
    tracker = FaceTracker()
    tracker.update(0, np.array([[0, 0, 10, 10]]))
    tracks = tracker.update(1, np.array([[0, 0, 10, 10], [1, 1, 11, 11]]))
    assert tracks[0] is not tracks[1]


def test_a_face_that_jumps_is_matched_by_its_embedding():
    tracker = FaceTracker()
    e = np.array([[1.0, 0.0]])
    first = tracker.update(0, np.array([[0, 0, 10, 10]]), e)
    later = tracker.update(1, np.array([[100, 100, 110, 110]]), e)
    assert later[0] is first[0]


def test_lost_faces_are_held_for_a_few_frames_then_dropped():
    tracker = FaceTracker(max_missing=5)
    tracker.update(0, np.array([[0, 0, 10, 10]]))
    assert len(tracker.missing(3, hold=5)) == 1
    assert tracker.missing(9, hold=5) == []
    assert tracker.update(9, np.array([[0, 0, 10, 10]]))[0].id == 1  # too late: a new track


def test_label_changes():
    assert label_changes(["a", "a", None, "a", "b"]) == 3
