import numpy as np

from facerec.identify import Gallery, open_set_rates, split_people


def unit(*v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def make_gallery():
    g = Gallery()
    g.add("ann", np.stack([unit(1, 0, 0), unit(1, 0.1, 0)]))
    g.add("bob", unit(0, 1, 0))
    return g


def test_closest_person_is_named_and_strangers_are_not():
    g = make_gallery()
    faces = np.stack([unit(1, 0.05, 0), unit(0.1, 1, 0), unit(0, 0, 1)])
    assert g.identify(faces, threshold=0.5) == ["ann", "bob", None]


def test_mean_and_nearest_can_disagree():
    g = Gallery()
    g.add("ann", np.stack([unit(1, 0, 0), unit(0, 0, 1)]))  # two very different photos
    g.add("bob", unit(0.9, 0.5, 0.5))
    face = unit(1, 0.02, 0)
    assert g.match(face, "nearest")[0] == ["ann"]  # one photo of Ann is almost identical
    assert g.match(face, "mean")[0] == ["bob"]  # the average of Ann is pulled away by the other photo


def test_adding_photos_of_the_same_person_extends_them():
    g = make_gallery()
    g.add("bob", unit(0, 1, 0.1))
    assert len(g) == 2 and len(g.people["bob"]) == 2


def test_empty_gallery_names_nobody():
    assert Gallery().identify(np.stack([unit(1, 0, 0)]), threshold=2.0) == [None]


def test_open_set_rates_count_only_correct_names_below_threshold():
    correct = np.array([True, True, False])
    known_d = np.array([0.3, 0.9, 0.2])
    stranger_d = np.array([0.4, 1.2])
    dir_, far = open_set_rates(correct, known_d, stranger_d, np.array([0.5, 1.0]))
    assert dir_.tolist() == [1 / 3, 2 / 3]
    assert far.tolist() == [0.5, 0.5]


def test_split_tests_galleries_of_every_size_on_the_same_faces():
    photos = {f"p{i}": list(range(10 * i, 10 * i + n)) for i, n in enumerate([8, 8, 7, 9, 1, 2])}
    splits = [split_people(photos, enrol, probes=2, max_enrol=5, rng=np.random.default_rng(1)) for enrol in (1, 3, 5)]
    assert splits[0][1] == splits[1][1] == splits[2][1]  # same known tests
    assert splits[0][2] == splits[2][2]  # same strangers
    gallery, known, strangers = splits[2]
    assert len(gallery) == 2 and all(len(v) == 5 for v in gallery.values())
    for name, photo in known:
        assert photo not in gallery[name]
    assert len(strangers) == len(photos) - len(gallery)
