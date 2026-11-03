"""Finding faces and turning each one into an embedding.

Detection is MTCNN and the embedding network is Inception-ResNet V1 trained on VGGFace2 (FaceNet),
both pretrained, from facenet-pytorch. An embedding is a 512-number vector of length 1; photos of
the same person land close together, so "same person?" becomes "closer than a threshold?".
"""

from dataclasses import dataclass

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
from PIL import Image

CROP_SIZE = 160
# Extra pixels around the detected box (in the 160 px crop), the setting FaceNet was trained with.
MARGIN = 14


@dataclass
class Face:
    box: np.ndarray  # (x1, y1, x2, y2) in pixels
    prob: float  # detector confidence
    embedding: np.ndarray  # 512 numbers, length 1


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class FaceModel:
    def __init__(self, device: str | None = None, min_prob: float = 0.9):
        self.device = device or default_device()
        self.min_prob = min_prob
        self.detector = MTCNN(keep_all=True, device=self.device)
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def detect(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Boxes and confidences of every face the detector is at least min_prob sure of."""
        boxes, probs = self.detector.detect(image)
        if boxes is None:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        # With recent numpy, MTCNN returns these as arrays of Python objects.
        boxes, probs = np.asarray(boxes, dtype=np.float32), np.asarray(probs, dtype=np.float32)
        keep = probs >= self.min_prob
        return boxes[keep], probs[keep]

    @staticmethod
    def crop(image: Image.Image, box: np.ndarray) -> torch.Tensor:
        """The face as a 3x160x160 tensor of 0-255 pixel values."""
        return extract_face(image, box, CROP_SIZE, MARGIN)

    @torch.no_grad()
    def embed(self, crops: torch.Tensor | list[torch.Tensor], batch: int = 128) -> np.ndarray:
        """Embeddings for a stack of 0-255 crops, one row per crop."""
        if len(crops) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        crops = torch.stack(list(crops)) if isinstance(crops, list) else crops
        out = []
        for i in range(0, len(crops), batch):
            x = (crops[i : i + batch].float().to(self.device) - 127.5) / 128.0  # the scaling FaceNet expects
            out.append(self.net(x).cpu().numpy())  # the network already scales each vector to length 1
        return np.concatenate(out)

    def faces(self, image: Image.Image) -> list[Face]:
        image = image.convert("RGB")
        boxes, probs = self.detect(image)
        embeddings = self.embed([self.crop(image, b) for b in boxes])
        return [Face(b, float(p), e) for b, p, e in zip(boxes, probs, embeddings)]


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between embeddings (rows of a against rows of b), from 0 to 2."""
    return np.linalg.norm(np.atleast_2d(a)[:, None, :] - np.atleast_2d(b)[None, :, :], axis=-1)
