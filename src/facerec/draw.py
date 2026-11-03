"""Drawing labelled boxes and blurring faces."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

KNOWN = (34, 197, 94)  # green
UNKNOWN = (239, 68, 68)  # red


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def draw_faces(image: Image.Image, boxes: np.ndarray, labels: list[str | None]) -> Image.Image:
    """A copy of the image with a box round each face: green with the name if known, red if not."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(2, round(max(out.size) / 300))
    text = font(max(12, round(max(out.size) / 45)))
    for box, label in zip(boxes, labels):
        colour = KNOWN if label else UNKNOWN
        x1, y1, x2, y2 = (float(v) for v in box)
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=width)
        caption = label or "unknown"
        left, top, right, bottom = draw.textbbox((x1, y1), caption, font=text)
        lift = bottom - top + 2 * width
        y = y1 - lift if y1 - lift >= 0 else y2
        draw.rectangle((x1, y, x1 + right - left + 2 * width, y + lift), fill=colour)
        draw.text((x1 + width, y + width - (top - y1)), caption, fill="white", font=text)
    return out


def blur_faces(image: Image.Image, boxes: np.ndarray, grow: float = 0.25) -> Image.Image:
    """A copy of the image with each face, plus a little around it, hidden under a strong blur.

    The blur is pasted through an oval mask so the edges look natural; the oval is grown by `grow`
    of the box size because detector boxes are tight round the eyes and mouth and miss hair and ears.
    """
    out = image.convert("RGB").copy()
    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = x2 - x1, y2 - y1
        x1, y1 = max(0, int(x1 - grow * w)), max(0, int(y1 - grow * h))
        x2, y2 = min(out.width, int(x2 + grow * w)), min(out.height, int(y2 + grow * h))
        if x2 <= x1 or y2 <= y1:
            continue
        region = out.crop((x1, y1, x2, y2))
        # Pixelate then blur: either alone can leave a face recognisable at small sizes.
        small = region.resize((max(1, region.width // 8), max(1, region.height // 8)), Image.Resampling.BILINEAR)
        hidden = small.resize(region.size, Image.Resampling.NEAREST).filter(
            ImageFilter.GaussianBlur(max(2, region.width / 10))
        )
        mask = Image.new("L", region.size, 0)
        ImageDraw.Draw(mask).ellipse((0, 0, region.width - 1, region.height - 1), fill=255)
        out.paste(hidden, (x1, y1), mask)
    return out
