import numpy as np
from PIL import Image

from facerec.draw import blur_faces, draw_faces


def noisy_image():
    return Image.fromarray(np.random.default_rng(0).integers(0, 256, (100, 200, 3), dtype=np.uint8))


def test_blur_changes_only_the_face_and_its_surroundings():
    image = noisy_image()
    out = np.asarray(blur_faces(image, np.array([[40, 40, 60, 60]]), grow=0.25)).astype(int)
    before = np.asarray(image).astype(int)
    assert np.abs(out[45:55, 45:55] - before[45:55, 45:55]).mean() > 20  # face hidden
    assert (out[:, 100:] == before[:, 100:]).all()  # far away untouched
    assert (out[:30] == before[:30]).all()


def test_blur_ignores_boxes_outside_the_image():
    image = noisy_image()
    out = blur_faces(image, np.array([[500, 500, 520, 520]]))
    assert (np.asarray(out) == np.asarray(image)).all()


def test_drawing_keeps_the_size_and_leaves_the_original_alone():
    image = noisy_image()
    original = np.asarray(image).copy()
    out = draw_faces(image, np.array([[10, 30, 50, 70], [120, 30, 160, 70]]), ["Ann", None])
    assert out.size == image.size
    assert (np.asarray(image) == original).all()
    assert (np.asarray(out) != original).any()
