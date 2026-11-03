"""Identification: naming a face from a gallery of people added beforehand, or calling it unknown.

A closed-set matcher gives every face the name of the closest person in the gallery, so a stranger
always gets someone's name. Here a face is only named when it is closer than a threshold.
"""

import numpy as np

from facerec.faces import distance

METHODS = ("mean", "nearest")


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Gallery:
    def __init__(self):
        self.people: dict[str, np.ndarray] = {}

    def add(self, name: str, embeddings: np.ndarray) -> None:
        embeddings = np.atleast_2d(embeddings)
        if name in self.people:
            embeddings = np.concatenate([self.people[name], embeddings])
        self.people[name] = embeddings

    def __len__(self) -> int:
        return len(self.people)

    def distances(self, embeddings: np.ndarray, method: str = "mean") -> np.ndarray:
        """Distance from each face (rows) to each person (columns).

        "mean" compares with the average of a person's photos (scaled back to length 1);
        "nearest" with whichever of their photos is closest.
        """
        if method == "mean":
            centres = np.stack([normalise(e.mean(axis=0)) for e in self.people.values()])
            return distance(embeddings, centres)
        if method == "nearest":
            return np.stack([distance(embeddings, e).min(axis=1) for e in self.people.values()], axis=1)
        raise ValueError(f"method must be one of {METHODS}")

    def match(self, embeddings: np.ndarray, method: str = "mean") -> tuple[list[str], np.ndarray]:
        """The closest person to each face and how far away they are."""
        d = self.distances(np.atleast_2d(embeddings), method)
        names = list(self.people)
        best = d.argmin(axis=1)
        return [names[i] for i in best], d[np.arange(len(d)), best]

    def identify(self, embeddings: np.ndarray, threshold: float, method: str = "mean") -> list[str | None]:
        """A name for each face, or None when nobody in the gallery is closer than the threshold."""
        if not self.people:
            return [None] * len(np.atleast_2d(embeddings))
        names, d = self.match(embeddings, method)
        return [n if dist < threshold else None for n, dist in zip(names, d)]


def open_set_rates(
    known_correct: np.ndarray, known_distance: np.ndarray, unknown_distance: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each threshold: the share of added people's photos given the right name (detection and
    identification rate), and the share of strangers given anyone's name (false accept rate)."""
    dir_ = ((known_distance[None, :] < thresholds[:, None]) & known_correct[None, :]).mean(axis=1)
    far = (unknown_distance[None, :] < thresholds[:, None]).mean(axis=1)
    return dir_, far


def split_people(
    people_photos: dict[str, list[int]], enrol: int, probes: int, max_enrol: int, rng: np.random.Generator
) -> tuple[dict[str, list[int]], list[tuple[str, int]], list[int]]:
    """Choose who is added to the gallery and which photos test it.

    Half of the people with at least max_enrol + 1 photos are added, with `enrol` photos each; up to
    `probes` of their other photos are the known tests. The photos used for testing never depend on
    `enrol`, so galleries of 1, 3 or 5 photos are tested on the same faces. Everyone else is a
    stranger, tested with one photo each.

    people_photos maps a name to the indices of that person's photos.
    """
    names = sorted(people_photos)
    eligible = [n for n in names if len(people_photos[n]) > max_enrol]
    known = set(rng.permutation(eligible)[: len(eligible) // 2].tolist())
    gallery, known_tests, strangers = {}, [], []
    for name in names:
        photos = list(rng.permutation(people_photos[name]))
        if name in known:
            gallery[name] = photos[:enrol]
            known_tests += [(name, int(i)) for i in photos[max_enrol : max_enrol + probes]]
        else:
            strangers.append(int(photos[0]))
    return gallery, known_tests, strangers
