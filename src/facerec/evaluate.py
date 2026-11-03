"""Evaluate face verification, identification with strangers, robustness and album clustering on
LFW, and draw the README figures.

    python -m facerec.evaluate
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import roc_curve
from tqdm import tqdm

from facerec.cluster import bcubed, cluster, make_album, pairwise_scores
from facerec.data import download_lfw, load_pairs, list_images, person_of
from facerec.faces import CROP_SIZE, FaceModel
from facerec.identify import METHODS, Gallery, open_set_rates, split_people
from facerec.verify import tar_at_far, ten_fold

BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--results", default="results/metrics.json")
    return parser.parse_args()


# ---------------------------------------------------------------- embeddings


def central_face(boxes: np.ndarray, size: tuple[int, int]) -> int:
    """LFW photos are centred on the person they're labelled with; other faces can appear too."""
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    return int(np.argmin(np.linalg.norm(centres - np.array(size) / 2, axis=1)))


def face_boxes(model: FaceModel, lfw_dir: Path, images: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The labelled person's face box in every photo. Where the detector finds no face, the
    middle of the photo is used, which is where LFW's own face detector put the face."""
    boxes, found = np.zeros((len(images), 4), dtype=np.float32), np.zeros(len(images), dtype=bool)
    for i, name in enumerate(tqdm(images, desc="detecting faces")):
        image = Image.open(lfw_dir / name).convert("RGB")
        b, _ = model.detect(image)
        if len(b):
            boxes[i], found[i] = b[central_face(b, image.size)], True
        else:
            w, h = image.size
            boxes[i] = (0.3 * w, 0.28 * h, 0.7 * w, 0.72 * h)
    return boxes, found


def crops_for(lfw_dir: Path, images: list[str], boxes: np.ndarray) -> torch.Tensor:
    crops = [FaceModel.crop(Image.open(lfw_dir / n).convert("RGB"), b) for n, b in zip(images, boxes)]
    return torch.stack(crops).round().clamp(0, 255).byte()


class ImageNetBaseline:
    """A ResNet-50 trained to classify ImageNet objects, not faces: its last hidden layer used as a
    face embedding, on the same face crops, to show what face-specific training adds."""

    def __init__(self, device: str):
        self.device = device
        self.net = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval().to(device)
        self.net.fc = torch.nn.Identity()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(self, crops: torch.Tensor, batch: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(crops), batch):
            x = crops[i : i + batch].float().to(self.device) / 255
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            out.append(F.normalize(self.net((x - self.mean) / self.std), dim=1).cpu().numpy())
        return np.concatenate(out)


def lfw_embeddings(model: FaceModel, lfw_dir: Path, cache: Path) -> dict:
    """Boxes and FaceNet and baseline embeddings for every LFW photo, computed once and cached."""
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return {k: saved[k] for k in saved.files}
    images = list_images(lfw_dir)
    boxes, found = face_boxes(model, lfw_dir, images)
    baseline = ImageNetBaseline(model.device)
    facenet, imagenet = [], []
    for i in tqdm(range(0, len(images), 512), desc="embedding"):
        crops = crops_for(lfw_dir, images[i : i + 512], boxes[i : i + 512])
        facenet.append(model.embed(crops))
        imagenet.append(baseline.embed(crops))
    data = {"images": np.array(images), "boxes": boxes, "found": found,
            "facenet": np.concatenate(facenet), "imagenet": np.concatenate(imagenet)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


# ---------------------------------------------------------------- verification


def pair_arrays(pairs, index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = np.array([index[p.first] for p in pairs])
    second = np.array([index[p.second] for p in pairs])
    return first, second, np.array([p.same for p in pairs]), np.array([p.fold for p in pairs])


def verification(emb: dict, pairs, index) -> dict:
    first, second, same, folds = pair_arrays(pairs, index)
    out = {}
    for name in ("facenet", "imagenet"):
        e = emb[name]
        d = np.linalg.norm(e[first] - e[second], axis=1)
        result = ten_fold(d, same, folds)
        result["tar_at_far_1pct"], _ = tar_at_far(d, same, 0.01)
        result["tar_at_far_0.1pct"], _ = tar_at_far(d, same, 0.001)
        fpr, tpr, _ = roc_curve(same, -d)
        result["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        result["distances"] = d.tolist()
        out[name] = result
    return out


DEGRADATIONS = {
    "resolution": [160, 80, 40, 24, 16, 12],  # face width in pixels
    "blur": [0, 1, 2, 3, 4, 6],  # Gaussian blur sigma, in pixels of the 160 px crop
    "covered": ["none", "eyes", "mouth and nose"],
}


def degrade(crops: torch.Tensor, kind: str, level) -> torch.Tensor:
    x = crops.float()
    if kind == "resolution" and level < CROP_SIZE:
        small = F.interpolate(x, size=level, mode="area")
        x = F.interpolate(small, size=CROP_SIZE, mode="bilinear", align_corners=False)
    elif kind == "blur" and level > 0:
        x = torchvision.transforms.functional.gaussian_blur(x, kernel_size=2 * int(3 * level) + 1, sigma=level)
    elif kind == "covered" and level != "none":
        x = x.clone()
        rows = slice(int(0.30 * CROP_SIZE), int(0.50 * CROP_SIZE)) if level == "eyes" else slice(int(0.55 * CROP_SIZE), None)
        x[:, :, rows, :] = 0.0  # a black band like sunglasses, or a dark face mask
    return x


def robustness(model: FaceModel, lfw_dir: Path, emb: dict, pairs, index) -> dict:
    """Accuracy when the second photo of every pair is degraded (the first stays clean), using the
    face boxes found in the clean photos so only the embedding is tested, not detection."""
    first, second, same, folds = pair_arrays(pairs, index)
    crops = crops_for(lfw_dir, [str(emb["images"][i]) for i in second], emb["boxes"][second])
    a = emb["facenet"][first]
    out = {}
    for kind, levels in DEGRADATIONS.items():
        out[kind] = {}
        for level in levels:
            b = np.concatenate([model.embed(degrade(crops[i : i + 1000], kind, level)) for i in range(0, len(crops), 1000)])
            out[kind][str(level)] = ten_fold(np.linalg.norm(a - b, axis=1), same, folds)["accuracy"]
    return out


# ---------------------------------------------------------------- identification and clustering


def people_photos(images: np.ndarray) -> dict[str, list[int]]:
    photos = defaultdict(list)
    for i, name in enumerate(images):
        photos[person_of(str(name))].append(i)
    return dict(photos)


def identification(emb: dict, threshold: float, repeats: int = 5, probes: int = 5, max_enrol: int = 5) -> dict:
    """Galleries of 1, 3 or 5 photos per person, tested with photos of people in the gallery and of
    strangers, over several random choices of who is in the gallery."""
    e, photos = emb["facenet"], people_photos(emb["images"])
    thresholds = np.linspace(0.3, 1.6, 261)
    out = {"thresholds": thresholds.tolist(), "threshold": threshold}
    for enrol in (1, 3, 5):
        for method in METHODS:
            curves, at_t, sizes = [], [], []
            for seed in range(repeats):
                gallery_ids, known, strangers = split_people(photos, enrol, probes, max_enrol, np.random.default_rng(seed))
                gallery = Gallery()
                for name, ids in gallery_ids.items():
                    gallery.add(name, e[ids])
                names, kd = gallery.match(e[[i for _, i in known]], method)
                correct = np.array([n == truth for n, (truth, _) in zip(names, known)])
                _, ud = gallery.match(e[strangers], method)
                curves.append(open_set_rates(correct, kd, ud, thresholds))
                at_t.append(open_set_rates(correct, kd, ud, np.array([threshold])))
                sizes.append((len(gallery), len(known), len(strangers), float(correct.mean())))
            dir_, far = np.mean(curves, axis=0)
            # The loosest threshold that names at most 1% of strangers.
            strict = int(np.nonzero(far <= 0.01)[0].max())
            out[f"{enrol}_{method}"] = {
                "dir": dir_.tolist(), "far": far.tolist(),
                "dir_at_threshold": float(np.mean([c[0][0] for c in at_t])),
                "far_at_threshold": float(np.mean([c[1][0] for c in at_t])),
                "strict_threshold": float(thresholds[strict]),
                "dir_at_strict": float(dir_[strict]), "far_at_strict": float(far[strict]),
                # Naming every face after its closest match, as a closed-set system does.
                "closed_set_correct": float(np.mean([s[3] for s in sizes])),
                "gallery_people": sizes[0][0], "known_tests": sizes[0][1], "strangers": sizes[0][2],
            }
    return out


def clustering(emb: dict, albums: int = 20, people: int = 40, singles: int = 20, max_photos: int = 15) -> dict:
    """Pretend albums of people with several photos each plus people seen only once. The threshold
    is chosen on albums made from one half of LFW's people and tested on albums from the other."""
    e, photos = emb["facenet"], people_photos(emb["images"])
    rng = np.random.default_rng(0)
    names = rng.permutation(sorted(photos))
    halves = {"val": names[: len(names) // 2], "test": names[len(names) // 2 :]}
    thresholds = np.round(np.arange(0.6, 1.45, 0.05), 2)
    scores = {}
    for split, pool in halves.items():
        several = [n for n in pool if len(photos[n]) >= 2]
        once = [n for n in pool if len(photos[n]) == 1]
        rows = []
        for _ in range(albums):
            chosen = list(rng.choice(several, people, replace=False)) + list(rng.choice(once, singles, replace=False))
            idx, labels = make_album(photos, chosen, max_photos, rng)
            row = []
            for t in thresholds:
                pred = cluster(e[idx], t)
                row.append({**{f"bcubed_{k}": v for k, v in bcubed(labels, pred).items()},
                            **{f"pair_{k}": v for k, v in pairwise_scores(labels, pred).items()},
                            "groups": int(pred.max() + 1), "people": len(chosen), "photos": len(idx)})
            rows.append(row)
        scores[split] = {str(t): {k: float(np.mean([r[j][k] for r in rows])) for k in rows[0][0]}
                         for j, t in enumerate(thresholds)}
    best = max(scores["val"], key=lambda t: scores["val"][t]["bcubed_f1"])
    return {"thresholds": thresholds.tolist(), "best_threshold": float(best), "val": scores["val"],
            "test": scores["test"], "test_at_best": scores["test"][best]}


# ---------------------------------------------------------------- figures


def plot_roc(ver: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, label, colour in (("facenet", "FaceNet (trained on faces)", BLUE),
                                ("imagenet", "ResNet-50 (trained on ImageNet objects)", GREY)):
        r = ver[name]
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], color=colour, label=f"{label}: {100 * r['accuracy']:.1f}% accuracy")
    ax.set(xscale="log", xlim=(1e-3, 1), ylim=(0, 1.02), xlabel="different-person pairs wrongly accepted",
           ylabel="same-person pairs accepted", title="LFW verification, 6,000 pairs")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_robustness(rob: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    titles = {"resolution": "Smaller faces", "blur": "Blurrier faces", "covered": "Covered faces"}
    xlabels = {"resolution": "face width (px)", "blur": "blur sigma (px, 160 px face)", "covered": ""}
    for ax, (kind, values) in zip(axes, rob.items()):
        x, y = list(values), [100 * v for v in values.values()]
        if kind == "covered":
            bars = ax.bar(x, y, color=BLUE)
            ax.bar_label(bars, fmt="%.1f", fontsize=8)
        else:
            ax.plot(range(len(x)), y, "o-", color=BLUE)
            ax.set_xticks(range(len(x)), x)
            for i, v in enumerate(y):
                ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set(title=titles[kind], xlabel=xlabels[kind], ylim=(50, 102))
        ax.axhline(50, color="grey", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("verification accuracy (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_strangers(ident: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colours = {1: GREY, 3: ORANGE, 5: BLUE}
    for enrol in (1, 3, 5):
        # With one photo per person, the average and the closest photo are the same thing.
        for method, style in (("mean", "-"), ("nearest", "--")) if enrol > 1 else (("mean", "-"),):
            r = ident[f"{enrol}_{method}"]
            label = "1 photo each" if enrol == 1 else f"{enrol} photos each, {'average' if method == 'mean' else 'closest photo'}"
            ax.plot(np.array(r["far"]) * 100, np.array(r["dir"]) * 100, style, color=colours[enrol], label=label)
        r = ident[f"{enrol}_mean"]
        ax.plot(100 * r["far_at_threshold"], 100 * r["dir_at_threshold"], "o", color=colours[enrol])
    ax.axvline(1, color="black", lw=0.8, ls=":")
    r = ident["5_mean"]
    ax.annotate(f"verification threshold ({ident['threshold']:.2f}):\n{100 * r['far_at_threshold']:.0f}% of strangers named",
                (100 * r["far_at_threshold"], 100 * r["dir_at_threshold"]), xytext=(4, 72), fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.annotate("1% of strangers named", (1.08, 52), fontsize=8)
    ax.set(xscale="log", xlim=(0.1, 100), ylim=(50, 100), xlabel="strangers given someone's name (%)",
           ylabel="gallery faces named correctly (%)", title="Naming faces when strangers are in the photos")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_clustering(clu: dict, path: Path) -> None:
    t = [str(x) for x in clu["thresholds"]]
    test = clu["test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for key, label, colour in (("bcubed_precision", "precision", ORANGE), ("bcubed_recall", "recall", GREY),
                               ("bcubed_f1", "F1", BLUE)):
        axes[0].plot(clu["thresholds"], [100 * test[x][key] for x in t], color=colour, label=label)
    axes[0].axvline(clu["best_threshold"], color="black", lw=0.8, ls=":")
    axes[0].set(xlabel="distance threshold", ylabel="B-cubed score (%)", title="Grouping quality (test albums)", ylim=(0, 102))
    axes[0].legend(fontsize=8)
    axes[1].plot(clu["thresholds"], [test[x]["groups"] for x in t], color=BLUE, label="groups found")
    axes[1].axhline(test[t[0]]["people"], color="black", lw=0.8, label="people in album")
    axes[1].axvline(clu["best_threshold"], color="black", lw=0.8, ls=":")
    axes[1].set(xlabel="distance threshold", ylabel="count", title="Groups found against people present")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir, assets = Path(args.data_dir), Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    lfw_dir = download_lfw(data_dir)
    pairs = load_pairs(data_dir)
    model = FaceModel()
    emb = lfw_embeddings(model, lfw_dir, data_dir / "embeddings.npz")
    index = {str(n): i for i, n in enumerate(emb["images"])}
    print(f"{len(index):,} photos, face found in {emb['found'].mean():.1%}")

    ver = verification(emb, pairs, index)
    for name in ("facenet", "imagenet"):
        r = ver[name]
        print(f"{name:<9} accuracy {r['accuracy']:.4f} ± {r['accuracy_std']:.4f} | threshold {r['threshold']:.3f} | "
              f"TAR@FAR1% {r['tar_at_far_1pct']:.4f} | TAR@FAR0.1% {r['tar_at_far_0.1pct']:.4f}")
    threshold = ver["facenet"]["threshold"]

    rob = robustness(model, lfw_dir, emb, pairs, index)
    print("robustness", json.dumps(rob))
    ident = identification(emb, threshold)
    for key in (f"{k}_{m}" for k in (1, 3, 5) for m in METHODS):
        r = ident[key]
        print(f"gallery {key:<10} at threshold: named correctly {r['dir_at_threshold']:.3f}, strangers named "
              f"{r['far_at_threshold']:.3f} | at {r['strict_threshold']:.3f}: named correctly {r['dir_at_strict']:.3f}, "
              f"strangers named {r['far_at_strict']:.3f} | closed set correct {r['closed_set_correct']:.3f} | "
              f"{r['gallery_people']} people, {r['known_tests']} tests, {r['strangers']} strangers")
    clu = clustering(emb)
    print("clustering threshold", clu["best_threshold"], json.dumps(clu["test_at_best"]))

    plot_roc(ver, assets / "roc.png")
    plot_robustness(rob, assets / "robustness.png")
    plot_strangers(ident, assets / "strangers.png")
    plot_clustering(clu, assets / "clustering.png")

    for r in ver.values():
        del r["roc"], r["distances"]
    for key in list(ident):
        if isinstance(ident[key], dict):
            del ident[key]["dir"], ident[key]["far"]
    del ident["thresholds"]
    metrics = {"photos": len(index), "faces_found": float(emb["found"].mean()), "verification": ver,
               "robustness": rob, "identification": ident,
               "clustering": {k: clu[k] for k in ("best_threshold", "test_at_best")}}
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(metrics, indent=2))
    print(f"metrics written to {args.results}, figures to {assets}/")


if __name__ == "__main__":
    main()
