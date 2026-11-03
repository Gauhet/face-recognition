"""Labeled Faces in the Wild (LFW): download, the official test pairs, and the list of people.

LFW has 13,233 news photos of 5,749 people, each roughly centred on one face. Its standard test
is 6,000 pairs of photos (half the same person, half different people) split into 10 folds.
"""

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# The scikit-learn mirror of the original LFW files.
LFW_URL = "https://ndownloader.figshare.com/files/5976018"
PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"


@dataclass(frozen=True)
class Pair:
    first: str  # image paths relative to the lfw/ folder, e.g. "Abel_Pacheco/Abel_Pacheco_0001.jpg"
    second: str
    same: bool
    fold: int


def image_name(person: str, number: int) -> str:
    return f"{person}/{person}_{number:04d}.jpg"


def parse_pairs(text: str) -> list[Pair]:
    """Read LFW's pairs.txt. The first line gives the number of folds and pairs per class; each
    fold then lists that many same-person lines ("name i j") and different-person lines
    ("name1 i name2 j")."""
    lines = [line.split() for line in text.strip().splitlines()]
    folds, per_class = int(lines[0][0]), int(lines[0][1])
    pairs = []
    for k, fields in enumerate(lines[1:]):
        fold = k // (2 * per_class)
        if len(fields) == 3:
            name, i, j = fields
            pairs.append(Pair(image_name(name, int(i)), image_name(name, int(j)), True, fold))
        else:
            a, i, b, j = fields
            pairs.append(Pair(image_name(a, int(i)), image_name(b, int(j)), False, fold))
    if len(pairs) != folds * 2 * per_class:
        raise ValueError(f"expected {folds * 2 * per_class} pairs, found {len(pairs)}")
    return pairs


def download_lfw(data_dir: str | Path = "data") -> Path:
    """Download and unpack LFW on first use; return the folder holding one subfolder per person."""
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "pairs.txt").exists():
        urllib.request.urlretrieve(PAIRS_URL, raw / "pairs.txt")
    if not (raw / "lfw").exists():
        archive = raw / "lfw.tgz"
        if not archive.exists():
            print("downloading LFW (about 180 MB)...")
            urllib.request.urlretrieve(LFW_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(raw, filter="data")
        archive.unlink()
    return raw / "lfw"


def load_pairs(data_dir: str | Path = "data") -> list[Pair]:
    download_lfw(data_dir)
    return parse_pairs((Path(data_dir) / "raw" / "pairs.txt").read_text())


def list_images(lfw_dir: Path) -> list[str]:
    """Every photo, as a path relative to lfw_dir, sorted so the order never changes."""
    return sorted(p.relative_to(lfw_dir).as_posix() for p in lfw_dir.glob("*/*.jpg"))


def person_of(image: str) -> str:
    return image.split("/")[0]
