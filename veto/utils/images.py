from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def add_label(img: Image.Image, text: str, position: str = "top-right") -> Image.Image:
    """
    Overlay a black pill/rectangle label with centered white text onto an image.

    Args:
        img:      Source PIL image (not modified in place).
        text:     Label string to render.
        position: One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.

    Returns:
        A new PIL image with the label composited on top.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    font_size = max(14, img.width // 40)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x, pad_y = int(font_size * 0.7), int(font_size * 0.45)
    box_w = text_w + 2 * pad_x
    box_h = text_h + 2 * pad_y

    margin = int(font_size * 0.5)

    if position == "top-left":
        box_x, box_y = margin, margin
    elif position == "top-right":
        box_x, box_y = img.width - box_w - margin, margin
    elif position == "bottom-left":
        box_x, box_y = margin, img.height - box_h - margin
    else:
        box_x, box_y = img.width - box_w - margin, img.height - box_h - margin

    radius = int(box_h * 0.3)
    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=radius,
        fill=(0, 0, 0, 220),
    )

    text_x = box_x + pad_x
    text_y = box_y + pad_y - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

    return img


def resize_image(image: Image.Image, side: int) -> Image.Image:
    image = image.convert("RGB")
    if image.size == (side, side):
        return image
    return image.resize((side, side), Image.Resampling.LANCZOS)


def tensor01_to_pil(t: torch.Tensor) -> Image.Image:
    v = t[0].clamp(0, 1)
    u8 = (v * 255).round().to(dtype=torch.uint8).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(u8)


def pil_to_tensor01(pil: Image.Image) -> torch.Tensor:
    pil = pil.convert("RGB")
    t = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0)


def load_x01(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    if image_size > 0:
        pil = resize_image(pil, image_size)
    return pil_to_tensor01(pil).to(device)


def save_tensor01_png(t: torch.Tensor, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor01_to_pil(t).save(path, format="PNG")
