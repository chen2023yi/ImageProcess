from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


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
REMBG_MODEL_NAME = "u2net_human_seg"
REMBG_MODEL_DIR = Path(__file__).resolve().parent / "models" / "rembg"
REMBG_MODEL_SIZE = 175_997_641
REMBG_MODEL_MD5 = "c09ddc2e0104f800e3e1bb4652583d1f"
REMBG_MODEL_DIR_ENV = "IMAGEPRO_REMBG_MODEL_DIR"
REMBG_DISABLE_ENV = "IMAGEPRO_DISABLE_REMBG"
_LOCAL_MODEL_SESSION: object | None = None
_LOCAL_MODEL_SESSION_DIR: Path | None = None
_REMBG_VALID_MODEL_DIR: Path | None = None


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


def crop_image(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    rgba_image = image.convert("RGBA")
    left, top, right, bottom = (int(value) for value in box)

    if left < 0 or top < 0 or right > rgba_image.width or bottom > rgba_image.height:
        raise ValueError("Crop area must stay inside the image bounds.")
    if right <= left or bottom <= top:
        raise ValueError("Crop area must have a positive width and height.")

    return rgba_image.crop((left, top, right, bottom))


def crop_and_resize_image(
    image: Image.Image,
    box: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> Image.Image:
    width, height = (int(value) for value in output_size)
    if width <= 0 or height <= 0:
        raise ValueError("Output size must have a positive width and height.")

    cropped_image = crop_image(image, box)
    return cropped_image.resize((width, height), Image.Resampling.LANCZOS)


def crop_and_resize_mask(
    mask: Image.Image,
    box: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> Image.Image:
    width, height = (int(value) for value in output_size)
    if width <= 0 or height <= 0:
        raise ValueError("Output size must have a positive width and height.")

    mask_image = mask.convert("L")
    left, top, right, bottom = (int(value) for value in box)
    if left < 0 or top < 0 or right > mask_image.width or bottom > mask_image.height:
        raise ValueError("Crop area must stay inside the mask bounds.")
    if right <= left or bottom <= top:
        raise ValueError("Crop area must have a positive width and height.")

    cropped_mask = mask_image.crop((left, top, right, bottom))
    return cropped_mask.resize((width, height), Image.Resampling.LANCZOS)


def extract_subject(
    image: Image.Image,
    polygon: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    output_mode: str = OUTPUT_TRANSPARENT,
    replacement_color: tuple[int, int, int] = DEFAULT_REPLACEMENT_COLOR,
) -> ProcessResult:
    rgba_image = image.convert("RGBA")
    subject_mask = _polygon_to_mask(rgba_image.size, polygon)
    return extract_subject_from_mask(
        rgba_image,
        subject_mask,
        output_mode=output_mode,
        replacement_color=replacement_color,
    )


def extract_subject_from_mask(
    image: Image.Image,
    subject_mask: Image.Image,
    output_mode: str = OUTPUT_TRANSPARENT,
    replacement_color: tuple[int, int, int] = DEFAULT_REPLACEMENT_COLOR,
) -> ProcessResult:
    rgba_image = image.convert("RGBA")
    subject_mask = _normalize_subject_mask(subject_mask, rgba_image.size)
    mask_pixels = np.array(subject_mask, dtype=np.uint8)

    if output_mode == OUTPUT_TRANSPARENT:
        pixels = np.array(rgba_image, dtype=np.uint8)
        alpha = pixels[:, :, 3].astype(np.uint16)
        mask_alpha = mask_pixels.astype(np.uint16)
        pixels[:, :, 3] = ((alpha * mask_alpha + 127) // 255).astype(np.uint8)
        result_image = Image.fromarray(pixels, mode="RGBA")
    elif output_mode == OUTPUT_SOLID:
        replacement_image = Image.new(
            "RGBA",
            rgba_image.size,
            (*replacement_color, 255),
        )
        result_image = Image.composite(rgba_image, replacement_image, subject_mask)
    else:
        raise ValueError(f"Unsupported output mode: {output_mode}")

    background_pixels = int((mask_pixels < 128).sum())
    return ProcessResult(image=result_image, removed_pixels=background_pixels)


def detect_subject_mask(image: Image.Image) -> Image.Image:
    rgba_image = image.convert("RGBA")
    model_mask = _detect_subject_mask_with_local_model(rgba_image)
    if model_mask is not None:
        return model_mask

    rgb_pixels = np.array(rgba_image, dtype=np.uint8)[:, :, :3]

    grabcut_mask = _detect_subject_mask_with_grabcut(rgb_pixels)
    if grabcut_mask is not None:
        return grabcut_mask
    return _detect_subject_mask_fallback(rgb_pixels)


def _detect_subject_mask_with_local_model(image: Image.Image) -> Image.Image | None:
    if os.environ.get(REMBG_DISABLE_ENV) == "1":
        return None
    if importlib.util.find_spec("onnxruntime") is None:
        return None

    model_dir = _find_local_rembg_model_dir()
    if model_dir is None:
        return None

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        global _LOCAL_MODEL_SESSION, _LOCAL_MODEL_SESSION_DIR
        if _LOCAL_MODEL_SESSION is None or _LOCAL_MODEL_SESSION_DIR != model_dir:
            model_path = str(model_dir / f"{REMBG_MODEL_NAME}.onnx")
            _LOCAL_MODEL_SESSION = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            _LOCAL_MODEL_SESSION_DIR = model_dir
        return _run_u2net_mask(_LOCAL_MODEL_SESSION, image)
    except BaseException:  # noqa: BLE001
        return None


def _find_local_rembg_model_dir() -> Path | None:
    global _REMBG_VALID_MODEL_DIR
    if _REMBG_VALID_MODEL_DIR is not None:
        cached_model = _REMBG_VALID_MODEL_DIR / f"{REMBG_MODEL_NAME}.onnx"
        if cached_model.is_file():
            return _REMBG_VALID_MODEL_DIR

    configured_dir = os.environ.get(REMBG_MODEL_DIR_ENV) or os.environ.get("U2NET_HOME")
    candidate_dirs = []
    if configured_dir:
        candidate_dirs.append(Path(configured_dir).expanduser())
    candidate_dirs.extend([REMBG_MODEL_DIR, Path.home() / ".u2net"])

    seen: set[Path] = set()
    for model_dir in candidate_dirs:
        resolved_dir = model_dir.resolve() if model_dir.exists() else model_dir
        if resolved_dir in seen:
            continue
        seen.add(resolved_dir)
        model_path = model_dir / f"{REMBG_MODEL_NAME}.onnx"
        if _is_valid_rembg_model(model_path):
            _REMBG_VALID_MODEL_DIR = model_dir
            return model_dir
    return None


def _is_valid_rembg_model(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == REMBG_MODEL_SIZE
        and _file_md5(path) == REMBG_MODEL_MD5
    )


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - used only to verify rembg's model hash.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_u2net_mask(session: object, image: Image.Image) -> Image.Image:
    input_size = (320, 320)
    source_size = image.size
    resized_image = image.convert("RGB").resize(input_size, Image.Resampling.BILINEAR)
    pixels = np.array(resized_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((pixels - mean) / std).transpose(2, 0, 1)[np.newaxis, :, :, :]

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor.astype(np.float32)})[0]
    prediction = np.squeeze(output).astype(np.float32)
    min_value = float(np.min(prediction))
    max_value = float(np.max(prediction))
    if max_value <= min_value:
        return Image.new("L", source_size, 0)

    normalized = (prediction - min_value) / (max_value - min_value)
    mask_pixels = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    mask = Image.fromarray(mask_pixels, mode="L").resize(
        source_size,
        Image.Resampling.LANCZOS,
    )
    return _refine_local_model_mask(mask, image)


def _refine_local_model_mask(mask: Image.Image, image: Image.Image) -> Image.Image:
    mask_pixels = np.array(mask, dtype=np.uint8)
    binary = np.where(mask_pixels >= 72, 255, 0).astype(np.uint8)

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        binary = _keep_main_subject_components(binary)
        binary = _fill_small_subject_holes(binary)
        return Image.fromarray(binary, mode="L").filter(ImageFilter.GaussianBlur(1.2))

    rgb_pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    if rgb_pixels.shape[:2] != mask_pixels.shape:
        return Image.fromarray(binary, mode="L")

    grabcut_mask = np.full(mask_pixels.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[mask_pixels >= 52] = cv2.GC_PR_FGD
    grabcut_mask[mask_pixels <= 8] = cv2.GC_BGD

    height, width = mask_pixels.shape
    kernel_size = max(3, int(round(min(width, height) * 0.006)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    sure_foreground = cv2.erode(
        np.where(mask_pixels >= 172, 255, 0).astype(np.uint8),
        kernel,
        iterations=1,
    )
    grabcut_mask[sure_foreground > 0] = cv2.GC_FGD

    border_width = max(2, int(round(min(width, height) * 0.012)))
    low_top = mask_pixels[:border_width, :] <= 40
    low_bottom = mask_pixels[-border_width:, :] <= 40
    low_left = mask_pixels[:, :border_width] <= 40
    low_right = mask_pixels[:, -border_width:] <= 40
    grabcut_mask[:border_width, :][low_top] = cv2.GC_BGD
    grabcut_mask[-border_width:, :][low_bottom] = cv2.GC_BGD
    grabcut_mask[:, :border_width][low_left] = cv2.GC_BGD
    grabcut_mask[:, -border_width:][low_right] = cv2.GC_BGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            rgb_pixels,
            grabcut_mask,
            None,
            bg_model,
            fg_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
    except Exception:  # noqa: BLE001
        pass

    grabcut_foreground = (
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)
    )
    binary = np.where(grabcut_foreground & (mask_pixels >= 36), 255, 0).astype(np.uint8)
    binary = _remove_local_model_background_fringe(binary, rgb_pixels, mask_pixels)
    binary = _keep_main_subject_components(binary)
    binary = _fill_small_subject_holes(binary)
    smoothed = _smooth_subject_mask(
        binary,
        fill_holes=False,
        close_ratio=0.006,
        close_iterations=1,
        open_ratio=0.003,
        blur_ratio=0.006,
    )
    smoothed = _tighten_subject_alpha(smoothed)
    return Image.fromarray(smoothed, mode="L")


def _remove_local_model_background_fringe(
    mask: np.ndarray,
    rgb_pixels: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return mask

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return binary

    height, width = binary.shape
    kernel_size = max(3, int(round(min(width, height) * 0.008)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    edge_band = cv2.dilate(binary, kernel, iterations=1) != cv2.erode(
        binary,
        kernel,
        iterations=1,
    )

    background_distance = _background_distance_score(rgb_pixels)
    skin_score = _skin_likelihood_score(rgb_pixels)
    background_like = (background_distance <= 0.16) & (skin_score <= 0.42)
    weak_confidence = confidence <= 168
    very_weak_confidence = confidence <= 112
    removable = (
        (binary > 0)
        & background_like
        & (weak_confidence & edge_band | very_weak_confidence)
    )

    cleaned = binary.copy()
    cleaned[removable] = 0
    return cleaned


def _tighten_subject_alpha(
    mask: np.ndarray,
    low: int = 46,
    high: int = 218,
) -> np.ndarray:
    alpha = mask.astype(np.float32)
    alpha = (alpha - float(low)) * (255.0 / max(1, high - low))
    return np.clip(alpha, 0, 255).astype(np.uint8)


def paint_subject_mask(
    subject_mask: Image.Image,
    points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    radius: int,
    value: int,
) -> Image.Image:
    mask = subject_mask.convert("L").copy()
    if not points:
        return mask

    width, height = mask.size
    radius = max(1, int(radius))
    value = 255 if int(value) >= 128 else 0
    clamped_points = [
        (
            max(0, min(width - 1, int(round(x)))),
            max(0, min(height - 1, int(round(y)))),
        )
        for x, y in points
    ]

    draw = ImageDraw.Draw(mask)
    if len(clamped_points) > 1:
        draw.line(clamped_points, fill=value, width=radius * 2)

    for x, y in clamped_points:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=value,
        )
    return mask


def _normalize_subject_mask(
    subject_mask: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    mask = subject_mask.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.BILINEAR)
    return mask


def _detect_subject_mask_with_grabcut(rgb_pixels: np.ndarray) -> Image.Image | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None

    height, width = rgb_pixels.shape[:2]
    if width < 3 or height < 3:
        return None

    max_dimension = 900
    scale = min(1.0, max_dimension / max(width, height))
    if scale < 1.0:
        small_size = (max(2, round(width * scale)), max(2, round(height * scale)))
        small_image = Image.fromarray(rgb_pixels, mode="RGB").resize(
            small_size,
            Image.Resampling.BILINEAR,
        )
        working_pixels = np.array(small_image, dtype=np.uint8)
    else:
        working_pixels = rgb_pixels

    work_height, work_width = working_pixels.shape[:2]
    image_bgr = cv2.cvtColor(working_pixels, cv2.COLOR_RGB2BGR)
    candidates: list[np.ndarray] = []

    for rect in _subject_candidate_rects(work_width, work_height):
        candidate = _grabcut_from_rect(cv2, image_bgr, rect)
        if candidate is not None:
            candidates.append(candidate)

    fallback_candidate = np.array(
        _detect_subject_mask_fallback(working_pixels),
        dtype=np.uint8,
    )
    candidates.append(fallback_candidate)

    vote = np.mean([(candidate > 0).astype(np.float32) for candidate in candidates], axis=0)
    center_prior = _center_subject_prior(work_width, work_height)
    border_distance = _background_distance_score(working_pixels)
    skin_score = _skin_likelihood_score(working_pixels)
    combined_score = (
        vote * 0.50
        + border_distance * 0.22
        + center_prior * 0.16
        + skin_score * 0.12
    )

    seed_mask = np.full((work_height, work_width), cv2.GC_PR_BGD, dtype=np.uint8)
    seed_mask[combined_score >= 0.34] = cv2.GC_PR_FGD
    seed_mask[combined_score >= 0.68] = cv2.GC_FGD

    border = max(2, int(min(work_width, work_height) * 0.025))
    seed_mask[:border, :] = cv2.GC_BGD
    seed_mask[-border:, :] = cv2.GC_BGD
    seed_mask[:, :border] = cv2.GC_BGD
    seed_mask[:, -border:] = cv2.GC_BGD

    core = ((vote >= 0.55) & (center_prior >= 0.18)).astype(np.uint8) * 255
    core_kernel_size = max(3, int(round(min(work_width, work_height) * 0.012)))
    if core_kernel_size % 2 == 0:
        core_kernel_size += 1
    core_kernel = np.ones((core_kernel_size, core_kernel_size), dtype=np.uint8)
    core = cv2.erode(core, core_kernel, iterations=1)
    seed_mask[core > 0] = cv2.GC_FGD
    skin_support = (
        (skin_score >= 0.58)
        & (center_prior >= 0.08)
        & (border_distance >= 0.12)
    )
    seed_mask[skin_support & (seed_mask != cv2.GC_BGD)] = cv2.GC_PR_FGD
    if not np.any(seed_mask == cv2.GC_FGD):
        center_rect = (
            int(work_width * 0.38),
            int(work_height * 0.32),
            max(1, int(work_width * 0.24)),
            max(1, int(work_height * 0.34)),
        )
        x, y, rect_width, rect_height = center_rect
        seed_mask[y : y + rect_height, x : x + rect_width] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image_bgr,
        seed_mask,
        None,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    foreground = np.where(
        (seed_mask == cv2.GC_FGD) | (seed_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    vote_foreground = (
        ((combined_score >= 0.54) & (vote >= 0.35))
        | (skin_support & (center_prior >= 0.16))
    ).astype(np.uint8) * 255
    foreground = np.maximum(foreground, vote_foreground)
    foreground = _keep_main_subject_components(foreground)
    foreground = _remove_border_leakage(
        foreground,
        combined_score,
        border_distance,
        skin_score,
        vote,
    )
    foreground = _bridge_subject_gaps(foreground, combined_score, center_prior)
    foreground = _fill_subject_holes(foreground)
    foreground = _smooth_subject_mask(foreground)

    mask_image = Image.fromarray(foreground, mode="L")
    if mask_image.size != (width, height):
        mask_image = mask_image.resize((width, height), Image.Resampling.BILINEAR)
    return mask_image


def _subject_candidate_rects(width: int, height: int) -> list[tuple[int, int, int, int]]:
    margin_sets = [
        (0.02, 0.01, 0.01),
        (0.05, 0.025, 0.015),
        (0.09, 0.045, 0.02),
        (0.14, 0.07, 0.025),
    ]
    rects: list[tuple[int, int, int, int]] = []
    for margin_x_ratio, margin_top_ratio, margin_bottom_ratio in margin_sets:
        margin_x = max(1, int(width * margin_x_ratio))
        margin_top = max(1, int(height * margin_top_ratio))
        margin_bottom = max(1, int(height * margin_bottom_ratio))
        rect = (
            margin_x,
            margin_top,
            max(1, width - margin_x * 2),
            max(1, height - margin_top - margin_bottom),
        )
        if rect[2] >= 2 and rect[3] >= 2:
            rects.append(rect)
    return rects


def _grabcut_from_rect(
    cv2_module,  # noqa: ANN001
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
) -> np.ndarray | None:
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2_module.grabCut(
            image_bgr,
            mask,
            rect,
            bgd_model,
            fgd_model,
            4,
            cv2_module.GC_INIT_WITH_RECT,
        )
    except Exception:  # noqa: BLE001
        return None

    return np.where(
        (mask == cv2_module.GC_FGD) | (mask == cv2_module.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)


def _center_subject_prior(width: int, height: int) -> np.ndarray:
    y_grid, x_grid = np.ogrid[:height, :width]
    normalized_x = (x_grid - width * 0.5) / max(width * 0.48, 1)
    normalized_y = (y_grid - height * 0.56) / max(height * 0.62, 1)
    ellipse = normalized_x * normalized_x + normalized_y * normalized_y
    return (1.0 - np.clip(ellipse, 0, 1)).astype(np.float32)


def _background_distance_score(rgb_pixels: np.ndarray) -> np.ndarray:
    height, width = rgb_pixels.shape[:2]
    border = max(1, int(min(width, height) * 0.08))
    border_pixels = np.concatenate(
        [
            rgb_pixels[:border, :, :].reshape(-1, 3),
            rgb_pixels[-border:, :, :].reshape(-1, 3),
            rgb_pixels[:, :border, :].reshape(-1, 3),
            rgb_pixels[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    centers = _sample_background_centers(border_pixels, 10)
    pixels = rgb_pixels.astype(np.float32)
    distances = np.linalg.norm(pixels[:, :, None, :] - centers[None, None, :, :], axis=3)
    return np.clip(distances.min(axis=2) / 180.0, 0, 1).astype(np.float32)


def _skin_likelihood_score(rgb_pixels: np.ndarray) -> np.ndarray:
    red = rgb_pixels[:, :, 0].astype(np.float32)
    green = rgb_pixels[:, :, 1].astype(np.float32)
    blue = rgb_pixels[:, :, 2].astype(np.float32)
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)

    rgb_rule = (
        (red > 80)
        & (green > 35)
        & (blue > 15)
        & ((max_channel - min_channel) > 12)
        & (red > blue)
        & (red >= green * 0.82)
    )

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        score = rgb_rule.astype(np.float32)
        return np.array(Image.fromarray((score * 255).astype(np.uint8)).filter(
            ImageFilter.BoxBlur(2)
        ), dtype=np.float32) / 255.0

    ycrcb = cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    cr_score = np.clip(1.0 - np.abs(cr - 150.0) / 38.0, 0, 1)
    cb_score = np.clip(1.0 - np.abs(cb - 112.0) / 34.0, 0, 1)
    y_score = np.clip((ycrcb[:, :, 0] - 35.0) / 90.0, 0, 1)
    score = np.maximum(rgb_rule.astype(np.float32) * 0.72, cr_score * cb_score * y_score)
    score = cv2.GaussianBlur(score.astype(np.float32), (5, 5), 0)
    return np.clip(score, 0, 1)


def _keep_main_subject_components(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return mask

    binary = (mask > 0).astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        8,
    )
    if component_count <= 1:
        return mask

    height, width = mask.shape
    center_x = width / 2
    center_y = height / 2
    component_scores: list[tuple[int, float, float]] = []
    for label in range(1, component_count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < width * height * 0.0015:
            continue
        x, y = centroids[label]
        distance = ((x - center_x) / width) ** 2 + ((y - center_y) / height) ** 2
        score = area * (1.0 - min(0.85, distance))
        component_scores.append((label, area, score))

    if not component_scores:
        return mask

    main_label, main_area, main_score = max(component_scores, key=lambda item: item[2])
    main_left = stats[main_label, cv2.CC_STAT_LEFT]
    main_top = stats[main_label, cv2.CC_STAT_TOP]
    main_right = main_left + stats[main_label, cv2.CC_STAT_WIDTH]
    main_bottom = main_top + stats[main_label, cv2.CC_STAT_HEIGHT]
    padding_x = int(width * 0.1)
    padding_y = int(height * 0.1)
    keep_labels = {main_label}

    for label, area, score in component_scores:
        if label == main_label:
            continue
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        right = left + stats[label, cv2.CC_STAT_WIDTH]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        near_main = (
            right >= main_left - padding_x
            and left <= main_right + padding_x
            and bottom >= main_top - padding_y
            and top <= main_bottom + padding_y
        )
        if area >= main_area * 0.05 or (near_main and score >= main_score * 0.025):
            keep_labels.add(label)

    return np.where(np.isin(labels, list(keep_labels)), 255, 0).astype(np.uint8)


def _remove_border_leakage(
    mask: np.ndarray,
    combined_score: np.ndarray,
    border_distance: np.ndarray,
    skin_score: np.ndarray,
    vote: np.ndarray,
) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return mask

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = binary.shape
    background_like = (
        (border_distance <= 0.30)
        & (skin_score <= 0.34)
        & (combined_score <= 0.66)
    )
    weak_foreground = (
        (combined_score <= 0.45)
        | ((vote <= 0.42) & (skin_score <= 0.42))
    )
    passable = ((binary > 0) & (background_like | weak_foreground)).astype(np.uint8)

    flood = np.zeros((height + 2, width + 2), dtype=np.uint8)
    leaked = passable.copy()
    seeds: list[tuple[int, int]] = []
    for x in range(width):
        if leaked[0, x]:
            seeds.append((x, 0))
    for y in range(height):
        if leaked[y, 0]:
            seeds.append((0, y))
        if leaked[y, width - 1]:
            seeds.append((width - 1, y))

    bottom_band = max(1, int(height * 0.03))
    for x in range(width):
        if leaked[height - 1, x] and (x < width * 0.08 or x > width * 0.92):
            seeds.append((x, height - 1))

    for seed in seeds:
        if leaked[seed[1], seed[0]]:
            cv2.floodFill(leaked, flood, seed, 2)

    remove = leaked == 2
    if bottom_band:
        remove[height - bottom_band :, int(width * 0.12) : int(width * 0.88)] = False

    cleaned = binary.copy()
    cleaned[remove] = 0
    return cleaned


def _bridge_subject_gaps(
    mask: np.ndarray,
    combined_score: np.ndarray,
    center_prior: np.ndarray,
) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return mask

    height, width = mask.shape
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    close_size = max(7, int(round(min(width, height) * 0.025)))
    if close_size % 2 == 0:
        close_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    inverse = np.where(binary > 0, 0, 255).astype(np.uint8)
    distance_to_subject = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
    max_bridge_distance = max(4, min(width, height) * 0.038)
    support = (combined_score >= 0.24) | (center_prior >= 0.34)
    bridge = (
        (closed > 0)
        & (binary == 0)
        & (distance_to_subject <= max_bridge_distance)
        & support
    )

    bridged = binary.copy()
    bridged[bridge] = 255
    return bridged


def _fill_subject_holes(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return mask

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    flood_filled = binary.copy()
    height, width = binary.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    for seed in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if flood_filled[seed[1], seed[0]] == 0:
            cv2.floodFill(flood_filled, flood_mask, seed, 255)

    holes = cv2.bitwise_not(flood_filled)
    return cv2.bitwise_or(binary, holes)


def _fill_small_subject_holes(
    mask: np.ndarray,
    max_area_ratio: float = 0.0009,
    max_dimension_ratio: float = 0.055,
) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return mask

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    flood_filled = binary.copy()
    height, width = binary.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    for seed in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if flood_filled[seed[1], seed[0]] == 0:
            cv2.floodFill(flood_filled, flood_mask, seed, 255)

    holes = np.where(flood_filled == 0, 255, 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    if component_count <= 1:
        return binary

    max_area = max(4, int(width * height * max_area_ratio))
    max_width = max(3, int(width * max_dimension_ratio))
    max_height = max(3, int(height * max_dimension_ratio))
    filled = binary.copy()
    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        component_width = stats[label, cv2.CC_STAT_WIDTH]
        component_height = stats[label, cv2.CC_STAT_HEIGHT]
        if area <= max_area and component_width <= max_width and component_height <= max_height:
            filled[labels == label] = 255
    return filled


def _smooth_subject_mask(
    mask: np.ndarray,
    fill_holes: bool = True,
    close_ratio: float = 0.018,
    close_iterations: int = 2,
    open_ratio: float = 0.004,
    blur_ratio: float = 0.01,
) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return mask

    height, width = mask.shape
    close_size = max(3, int(round(min(width, height) * close_ratio)))
    if close_size % 2 == 0:
        close_size += 1
    open_size = max(3, int(round(min(width, height) * open_ratio)))
    if open_size % 2 == 0:
        open_size += 1
    blur_size = max(5, int(round(min(width, height) * blur_ratio)))
    if blur_size % 2 == 0:
        blur_size += 1

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_size, open_size),
    )
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=max(1, int(close_iterations)),
    )
    if fill_holes:
        binary = _fill_subject_holes(binary)
    else:
        binary = _fill_small_subject_holes(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    binary = cv2.medianBlur(binary, open_size)
    return cv2.GaussianBlur(binary, (blur_size, blur_size), 0)


def _detect_subject_mask_fallback(rgb_pixels: np.ndarray) -> Image.Image:
    height, width = rgb_pixels.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Subject detection requires a non-empty image.")

    border = max(1, int(min(width, height) * 0.08))
    border_pixels = np.concatenate(
        [
            rgb_pixels[:border, :, :].reshape(-1, 3),
            rgb_pixels[-border:, :, :].reshape(-1, 3),
            rgb_pixels[:, :border, :].reshape(-1, 3),
            rgb_pixels[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    centers = _sample_background_centers(border_pixels, 8)

    pixels = rgb_pixels.astype(np.float32)
    distances = np.linalg.norm(pixels[:, :, None, :] - centers[None, None, :, :], axis=3)
    background_distance = distances.min(axis=2) / 441.7

    y_grid, x_grid = np.ogrid[:height, :width]
    normalized_x = (x_grid - width * 0.5) / max(width * 0.45, 1)
    normalized_y = (y_grid - height * 0.55) / max(height * 0.55, 1)
    center_prior = 1.0 - np.clip(normalized_x * normalized_x + normalized_y * normalized_y, 0, 1)
    score = background_distance * 0.7 + center_prior * 0.45
    mask = (score >= 0.48).astype(np.uint8) * 255

    mask_image = Image.fromarray(mask, mode="L")
    scale = max(2, int(min(width, height) * 0.006))
    mask_image = mask_image.filter(ImageFilter.BoxBlur(scale))
    mask_array = np.array(mask_image, dtype=np.uint8)
    mask_array = np.where(mask_array >= 80, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_array, mode="L")


def _sample_background_centers(border_pixels: np.ndarray, center_count: int) -> np.ndarray:
    if len(border_pixels) <= center_count:
        return border_pixels

    luminance = (
        border_pixels[:, 0] * 0.2126
        + border_pixels[:, 1] * 0.7152
        + border_pixels[:, 2] * 0.0722
    )
    quantiles = np.linspace(0.05, 0.95, center_count)
    centers = []
    for quantile in quantiles:
        target = np.quantile(luminance, quantile)
        index = int(np.argmin(np.abs(luminance - target)))
        centers.append(border_pixels[index])
    return np.array(centers, dtype=np.float32)


def _polygon_to_mask(
    size: tuple[int, int],
    polygon: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> Image.Image:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Subject mask size must be positive.")

    points = _normalize_polygon_points(polygon, width, height)
    if len(points) < 3 or abs(_polygon_area(points)) < 1:
        raise ValueError("Subject selection must contain at least three area points.")

    scale = 4
    scaled_size = (width * scale, height * scale)
    scaled_points = [(x * scale, y * scale) for x, y in points]
    mask = Image.new("L", scaled_size, 0)
    ImageDraw.Draw(mask).polygon(scaled_points, fill=255)
    return mask.resize(size, Image.Resampling.BOX)


def _normalize_polygon_points(
    polygon: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for raw_x, raw_y in polygon:
        x = max(0, min(width - 1, int(round(raw_x))))
        y = max(0, min(height - 1, int(round(raw_y))))
        point = (x, y)
        if not points or points[-1] != point:
            points.append(point)

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _polygon_area(points: list[tuple[int, int]]) -> float:
    area = 0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2


def save_png(image: Image.Image, path: str | Path) -> None:
    output_path = Path(path)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
