"""Faces in video: follow each face from frame to frame, name it once from all its frames, and
either label or blur it.

Labelling each frame on its own makes names flicker: a turned head or motion blur gives one bad
embedding, and the name changes for a frame. A track averages the embeddings it has collected, so
the name settles. For blurring, the tracker also keeps hiding a face for a few frames after the
detector loses it, since a single missed frame would show the face.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from facerec.draw import blur_faces, draw_faces
from facerec.faces import FaceModel, distance
from facerec.identify import Gallery, normalise


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Overlap (intersection over union) of every box in a with every box in b."""
    a, b = np.atleast_2d(a)[:, None, :], np.atleast_2d(b)[None, :, :]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = w * h
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


@dataclass(eq=False)  # tracks are compared by identity
class Track:
    id: int
    box: np.ndarray
    last_seen: int
    embeddings: list[np.ndarray] = field(default_factory=list)
    frames: int = 0

    def centre(self, recent: int = 30) -> np.ndarray | None:
        """Average embedding over the most recent frames."""
        return normalise(np.mean(self.embeddings[-recent:], axis=0)) if self.embeddings else None


class FaceTracker:
    """Links each frame's detections to existing tracks, by box overlap first and, for a face that
    moved too far for its boxes to overlap, by embedding distance."""

    def __init__(self, min_iou: float = 0.3, reid_distance: float = 0.9, max_missing: int = 15):
        self.min_iou = min_iou
        self.reid_distance = reid_distance
        self.max_missing = max_missing
        self.tracks: list[Track] = []
        self.next_id = 0

    def update(self, frame: int, boxes: np.ndarray, embeddings: np.ndarray | None = None) -> list[Track]:
        """Assign this frame's detections to tracks; returns the track of each detection."""
        live = [t for t in self.tracks if frame - t.last_seen <= self.max_missing]
        score = np.full((len(live), len(boxes)), -np.inf)
        if live and len(boxes):
            iou = box_iou(np.stack([t.box for t in live]), boxes)
            score = np.where(iou >= self.min_iou, iou, -np.inf)
            if embeddings is not None:
                for i, t in enumerate(live):
                    c = t.centre()
                    if c is not None:
                        close = distance(c, embeddings)[0] < self.reid_distance
                        # A small positive score: overlap wins whenever there is any.
                        score[i] = np.where(np.isinf(score[i]) & close, 1e-3, score[i])

        assigned: list[Track | None] = [None] * len(boxes)
        # Greedy: best-scoring pairs first, each track and detection used once.
        for flat in np.argsort(-score, axis=None):
            i, j = np.unravel_index(flat, score.shape)
            if np.isinf(score[i, j]):
                break
            if assigned[j] is None and live[i] not in assigned:
                assigned[j] = live[i]
        for j, box in enumerate(boxes):
            track = assigned[j]
            if track is None:
                track = Track(self.next_id, box, frame)
                self.next_id += 1
                self.tracks.append(track)
                assigned[j] = track
            track.box, track.last_seen = box, frame
            track.frames += 1
            if embeddings is not None:
                track.embeddings.append(embeddings[j])
        return assigned

    def missing(self, frame: int, hold: int) -> list[Track]:
        """Tracks not seen this frame but seen within the last `hold` frames."""
        return [t for t in self.tracks if 0 < frame - t.last_seen <= hold]


def label_changes(labels: list[str | None]) -> int:
    return sum(a != b for a, b in zip(labels, labels[1:]))


def process_video(
    model: FaceModel,
    path: str,
    out_path: str,
    mode: str = "label",
    gallery: Gallery | None = None,
    threshold: float = 1.0,
    hold: int = 10,
    keep_known: bool = False,
    max_frames: int | None = None,
    min_frames: int = 3,
) -> dict:
    """Label ("label") or blur ("blur") every face in a video and write the result.

    In blur mode with keep_known, people in the gallery are left visible. In label mode a face is
    only drawn once its track has lasted min_frames, which hides one-frame false detections (blur
    mode hides everything, since blurring a non-face costs nothing). Returns counts that describe
    what the tracker did.
    """
    gallery = gallery or Gallery()
    need_embeddings = mode == "label" or keep_known
    tracker = FaceTracker()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*("VP80" if out_path.endswith(".webm") else "mp4v"))
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)

    per_frame: dict[int, list[str | None]] = {}  # each track's frame-by-frame name, for comparison
    tracked: dict[int, list[str | None]] = {}  # each track's name from its averaged embedding
    stats = {"frames": 0, "faces": 0, "held": 0}
    frame_index = 0
    while max_frames is None or frame_index < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes, _ = model.detect(image)
        embeddings = model.embed([model.crop(image, b) for b in boxes]) if need_embeddings else None
        tracks = tracker.update(frame_index, boxes, embeddings)
        stats["faces"] += len(boxes)

        names: list[str | None] = [None] * len(tracks)
        if need_embeddings and len(tracks):
            frame_names = gallery.identify(embeddings, threshold)
            names = gallery.identify(np.stack([t.centre() for t in tracks]), threshold)
            for t, f, n in zip(tracks, frame_names, names):
                per_frame.setdefault(t.id, []).append(f)
                tracked.setdefault(t.id, []).append(n)

        if mode == "blur":
            hide = [b for b, n in zip(boxes, names) if not (keep_known and n)]
            held = tracker.missing(frame_index, hold)
            stats["held"] += len(held)
            out = blur_faces(image, np.array(hide + [t.box for t in held]).reshape(-1, 4))
        else:
            shown = [k for k, t in enumerate(tracks) if t.frames >= min_frames]
            out = draw_faces(image, boxes[shown], [names[k] for k in shown])
        writer.write(cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR))
        frame_index += 1
    cap.release()
    writer.release()

    stats["frames"] = frame_index
    stats["tracks"] = sum(t.frames >= min_frames for t in tracker.tracks)
    if need_embeddings:
        stats["label_changes_per_frame"] = sum(label_changes(v) for v in per_frame.values())
        stats["label_changes_tracked"] = sum(label_changes(v) for v in tracked.values())
        # How often each name was given, over faces in tracks long enough to be drawn.
        real = {t.id for t in tracker.tracks if t.frames >= min_frames}
        for key, labels in (("names_per_frame", per_frame), ("names_tracked", tracked)):
            given = [n or "unknown" for i, v in labels.items() if i in real for n in v]
            stats[key] = {n: given.count(n) for n in sorted(set(given))}
    return stats
