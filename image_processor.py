from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PURE_GREEN = (0, 255, 0)
DEFAULT_TOLERANCE = 90
DEFAULT_LIGHTNESS_THRESHOLD = 200
DEFAULT_NEUTRAL_CHROMA_THRESHOLD = 55
MODE_COLOR = "color"
MODE_LIGHT_NEUTRAL = "light_neutral"
OUTPUT_TRANSPARENT = "transparent"
OUTPUT_SOLID = "solid"
DEFAULT_REPLACEMENT_COLOR = (255, 255, 255)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ProcessResult:
    image: Image.Image
    removed_pixels: int


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not is_supported_image(image_path):
        raise ValueError("Only PNG, JPG, and JPEG images are supported.")

    with Image.open(image_path) as image:
        return image.convert("RGBA")


def remove_background(
    image: Image.Image,
    background_color: tuple[int, int, int] = PURE_GREEN,
    mode: str = MODE_COLOR,
    tolerance: int = 0,
    lightness_threshold: int = DEFAULT_LIGHTNESS_THRESHOLD,
    neutral_chroma_threshold: int = DEFAULT_NEUTRAL_CHROMA_THRESHOLD,
    output_mode: str = OUTPUT_TRANSPARENT,
    replacement_color: tuple[int, int, int] = DEFAULT_REPLACEMENT_COLOR,
) -> ProcessResult:
    rgba_image = image.convert("RGBA")
    pixels = np.array(rgba_image, dtype=np.uint8)
    rgb_pixels = pixels[:, :, :3].astype(np.int16)

    if mode == MODE_COLOR:
        tolerance = max(0, int(tolerance))
        target = np.array(background_color, dtype=np.int16)
        color_distance = np.linalg.norm(rgb_pixels - target, axis=2)
        mask = color_distance <= tolerance
    elif mode == MODE_LIGHT_NEUTRAL:
        lightness_threshold = max(0, min(255, int(lightness_threshold)))
        neutral_chroma_threshold = max(0, min(255, int(neutral_chroma_threshold)))
        max_channel = rgb_pixels.max(axis=2)
        min_channel = rgb_pixels.min(axis=2)
        chroma = max_channel - min_channel
        mask = (max_channel >= lightness_threshold) & (
            chroma <= neutral_chroma_threshold
        )
    else:
        raise ValueError(f"Unsupported background removal mode: {mode}")

    if output_mode == OUTPUT_TRANSPARENT:
        pixels[mask, 3] = 0
    elif output_mode == OUTPUT_SOLID:
        replacement = np.array(replacement_color, dtype=np.uint8)
        pixels[mask, :3] = replacement
        pixels[mask, 3] = 255
    else:
        raise ValueError(f"Unsupported output mode: {output_mode}")

    cleaned_image = Image.fromarray(pixels, mode="RGBA")
    return ProcessResult(image=cleaned_image, removed_pixels=int(mask.sum()))


def remove_green_background(
    image: Image.Image,
    background_color: tuple[int, int, int] = PURE_GREEN,
    tolerance: int = 0,
) -> ProcessResult:
    return remove_background(
        image=image,
        background_color=background_color,
        mode=MODE_COLOR,
        tolerance=tolerance,
    )


def save_png(image: Image.Image, path: str | Path) -> None:
    output_path = Path(path)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
