"""Browser demo: compare two faces, add people and name them in photos, group an album by person,
and blur or label faces in photos and video.

    python app.py
"""

import json
import random
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from facerec.cluster import cluster
from facerec.data import list_images
from facerec.draw import blur_faces, draw_faces
from facerec.faces import FaceModel, distance
from facerec.identify import Gallery
from facerec.video import process_video

MAX_VIDEO_SECONDS = 20
MAX_ALBUM_PHOTOS = 60
ROOT = Path(__file__).parent
LFW_DIR = ROOT / "data" / "raw" / "lfw"

# Thresholds measured on LFW by evaluate.py, with fallbacks if it hasn't been run.
metrics_path = ROOT / "results" / "metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
MATCH_THRESHOLD = round(metrics.get("verification", {}).get("facenet", {}).get("threshold", 1.06), 2)
# Naming a face checks it against everyone in the gallery, so a stranger gets many chances to match
# someone: this stricter threshold names at most 1% of strangers on LFW.
NAME_THRESHOLD = round(metrics.get("identification", {}).get("5_mean", {}).get("strict_threshold", 0.78), 2)
GROUP_THRESHOLD = round(metrics.get("clustering", {}).get("best_threshold", 0.95), 2)

model = FaceModel()
lfw_images = list_images(LFW_DIR) if LFW_DIR.exists() else []


def to_gallery(people: dict | None) -> Gallery:
    g = Gallery()
    for name, embeddings in (people or {}).items():
        g.add(name, np.array(embeddings))
    return g


def largest_face(image: Image.Image):
    faces = model.faces(image)
    return max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1])) if faces else None


def crop_for_display(image: Image.Image, box) -> Image.Image:
    return Image.fromarray(model.crop(image.convert("RGB"), box).permute(1, 2, 0).byte().numpy())


# ---------------------------------------------------------------- compare


def compare(a: Image.Image, b: Image.Image, threshold: float):
    if a is None or b is None:
        return None, None, "Add two photos."
    fa, fb = largest_face(a), largest_face(b)
    if fa is None or fb is None:
        return None, None, f"No face found in the {'first' if fa is None else 'second'} photo."
    d = float(distance(fa.embedding, fb.embedding)[0, 0])
    verdict = "**Same person**" if d < threshold else "**Different people**"
    note = (f"{verdict}: distance {d:.2f}, threshold {threshold:.2f}. Distances run from 0 (identical) to 2; "
            f"on LFW, same-person pairs are usually under 1.")
    return crop_for_display(a, fa.box), crop_for_display(b, fb.box), note


def random_lfw_pair(same: bool):
    people = {}
    for name in lfw_images:
        people.setdefault(name.split("/")[0], []).append(name)
    if same:
        photos = random.choice([v for v in people.values() if len(v) >= 2])
        a, b = random.sample(photos, 2)
    else:
        p, q = random.sample(list(people), 2)
        a, b = random.choice(people[p]), random.choice(people[q])
    return Image.open(LFW_DIR / a), Image.open(LFW_DIR / b)


# ---------------------------------------------------------------- people


def describe(people: dict | None) -> str:
    if not people:
        return "Nobody added yet."
    return "\n".join(f"- **{name}**: {len(e)} photo{'s' if len(e) > 1 else ''}" for name, e in people.items())


def add_person(name: str, files, people: dict | None):
    people = dict(people or {})
    name = (name or "").strip()
    if not name or not files:
        return people, describe(people), "Give a name and at least one photo."
    found, skipped = [], 0
    for f in files:
        face = largest_face(Image.open(f if isinstance(f, str) else f.name).convert("RGB"))
        if face is None:
            skipped += 1
        else:
            found.append(face.embedding.tolist())
    if found:
        people[name] = people.get(name, []) + found
    message = f"Added {len(found)} photo{'s' if len(found) != 1 else ''} of {name}."
    if skipped:
        message += f" No face found in {skipped}."
    return people, describe(people), message


def clear_people():
    return {}, describe({}), "Everyone removed."


# ---------------------------------------------------------------- photos


def name_faces(image: Image.Image, people: dict | None, threshold: float):
    if image is None:
        return None, ""
    faces = model.faces(image)
    names = to_gallery(people).identify(np.array([f.embedding for f in faces]), threshold) if faces else []
    known = sum(n is not None for n in names)
    summary = f"{len(faces)} face{'s' if len(faces) != 1 else ''}: {known} named, {len(faces) - known} unknown."
    if not people:
        summary += " Add people in the first tab to have them named."
    return draw_faces(image, np.array([f.box for f in faces]).reshape(-1, 4), names), summary


def blur_photo(image: Image.Image, keep_known: bool, people: dict | None, threshold: float):
    if image is None:
        return None, ""
    image = image.convert("RGB")
    faces = model.faces(image)
    names = to_gallery(people).identify(np.array([f.embedding for f in faces]), threshold) if faces else []
    hide = [f.box for f, n in zip(faces, names) if not (keep_known and n)]
    return blur_faces(image, np.array(hide).reshape(-1, 4)), f"Blurred {len(hide)} of {len(faces)} faces."


def group_album(files, threshold: float):
    if not files:
        return [], "Add some photos."
    files = files[:MAX_ALBUM_PHOTOS]
    crops, embeddings = [], []
    for f in files:
        image = Image.open(f if isinstance(f, str) else f.name).convert("RGB")
        for face in model.faces(image):
            crops.append(crop_for_display(image, face.box))
            embeddings.append(face.embedding)
    if not crops:
        return [], "No faces found."
    groups = cluster(np.array(embeddings), threshold)
    # Biggest groups first, so the people who appear most come first.
    order = sorted(range(len(crops)), key=lambda i: (-np.sum(groups == groups[i]), groups[i]))
    rank = {g: k + 1 for k, g in enumerate(dict.fromkeys(groups[i] for i in order))}
    items = [(crops[i], f"Person {rank[groups[i]]}") for i in order]
    return items, f"{len(crops)} faces in {len(files)} photos, grouped into {len(rank)} people."


# ---------------------------------------------------------------- video


def video_faces(path: str, mode: str, keep_known: bool, people: dict | None, threshold: float):
    if not path:
        return None, ""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    out_path = tempfile.NamedTemporaryFile(suffix=".webm", delete=False).name
    stats = process_video(model, path, out_path, mode="blur" if mode.startswith("Blur") else "label",
                          gallery=to_gallery(people), threshold=threshold, keep_known=keep_known,
                          max_frames=int(MAX_VIDEO_SECONDS * fps))
    summary = f"{stats['frames']} frames, {stats['faces']} face detections, {stats['tracks']} tracks."
    if "label_changes_tracked" in stats:
        summary += (f" Name changes within tracks: {stats['label_changes_per_frame']} naming each frame on its own, "
                    f"{stats['label_changes_tracked']} with tracking.")
    if stats["held"]:
        summary += f" {stats['held']} face-frames stayed blurred after the detector lost them."
    return out_path, summary


INTRO = f"""# Face Recognition
Pretrained FaceNet (Inception-ResNet V1, trained on VGGFace2) with an MTCNN face detector. Two photos
count as the same person when their face embeddings are closer than **{MATCH_THRESHOLD}**. Naming a face
uses a stricter **{NAME_THRESHOLD}**, because it is checked against everyone you've added and a looser
threshold names most strangers as someone. Both thresholds were measured on LFW. Photos you add stay in
this browser session and are not saved.
"""

with gr.Blocks(title="Face Recognition") as demo:
    gr.Markdown(INTRO)
    people = gr.State({})
    threshold = gr.Slider(0.5, 1.5, value=NAME_THRESHOLD, step=0.01,
                          label="Naming threshold (lower is stricter: fewer strangers named, more people missed)")

    with gr.Tab("Add people"):
        gr.Markdown("Add a few photos of each person, ideally from different days and angles. "
                    "The largest face in each photo is used.")
        with gr.Row():
            with gr.Column():
                person_name = gr.Textbox(label="Name")
                person_files = gr.File(label="Photos", file_count="multiple", file_types=["image"])
                with gr.Row():
                    add_button = gr.Button("Add", variant="primary")
                    clear_button = gr.Button("Remove everyone")
                add_message = gr.Markdown()
            people_list = gr.Markdown(describe({}))
        add_button.click(add_person, [person_name, person_files, people], [people, people_list, add_message])
        clear_button.click(clear_people, None, [people, people_list, add_message])

    with gr.Tab("Who's in this photo"):
        with gr.Row():
            who_in = gr.Image(type="pil", label="Photo")
            who_out = gr.Image(type="pil", label="Faces")
        who_summary = gr.Markdown()
        gr.Button("Find faces", variant="primary").click(name_faces, [who_in, people, threshold], [who_out, who_summary])

    with gr.Tab("Compare two faces"):
        with gr.Row():
            cmp_a = gr.Image(type="pil", label="First photo")
            cmp_b = gr.Image(type="pil", label="Second photo")
        with gr.Row():
            face_a = gr.Image(label="Face used", height=160)
            face_b = gr.Image(label="Face used", height=160)
        match_threshold = gr.Slider(0.5, 1.5, value=MATCH_THRESHOLD, step=0.01, label="Same-person threshold")
        cmp_result = gr.Markdown()
        with gr.Row():
            gr.Button("Compare", variant="primary").click(compare, [cmp_a, cmp_b, match_threshold], [face_a, face_b, cmp_result])
            if lfw_images:
                gr.Button("Random LFW pair: same person").click(lambda: random_lfw_pair(True), None, [cmp_a, cmp_b])
                gr.Button("Random LFW pair: different people").click(lambda: random_lfw_pair(False), None, [cmp_a, cmp_b])

    with gr.Tab("Group an album"):
        gr.Markdown(f"Upload up to {MAX_ALBUM_PHOTOS} photos. Faces are grouped by person without any names.")
        album_files = gr.File(label="Photos", file_count="multiple", file_types=["image"])
        group_threshold = gr.Slider(0.5, 1.5, value=GROUP_THRESHOLD, step=0.05, label="Grouping threshold")
        album_out = gr.Gallery(label="Faces by person", columns=8, height="auto")
        album_summary = gr.Markdown()
        gr.Button("Group faces", variant="primary").click(group_album, [album_files, group_threshold], [album_out, album_summary])

    with gr.Tab("Blur faces"):
        keep_photo = gr.Checkbox(label="Leave people I've added visible")
        with gr.Row():
            blur_in = gr.Image(type="pil", label="Photo")
            blur_out = gr.Image(type="pil", label="Blurred")
        blur_summary = gr.Markdown()
        gr.Button("Blur", variant="primary").click(blur_photo, [blur_in, keep_photo, people, threshold], [blur_out, blur_summary])

    with gr.Tab("Video"):
        gr.Markdown(f"The first {MAX_VIDEO_SECONDS} seconds are processed. Faces are tracked across frames, so names "
                    "don't flicker and a face stays blurred for a moment after the detector loses it.")
        with gr.Row():
            video_mode = gr.Radio(["Name faces", "Blur faces"], value="Blur faces", label="Mode")
            keep_video = gr.Checkbox(label="When blurring, leave people I've added visible")
        with gr.Row():
            video_in = gr.Video(label="Video")
            video_out = gr.Video(label="Result")
        video_summary = gr.Markdown()
        gr.Button("Process video", variant="primary").click(
            video_faces, [video_in, video_mode, keep_video, people, threshold], [video_out, video_summary])

if __name__ == "__main__":
    demo.launch()
