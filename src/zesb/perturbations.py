"""Deterministic image perturbation transforms for Stage 1 robustness testing."""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter


def jpeg_compress(rgb_array: NDArray[np.uint8], quality: int = 30) -> NDArray[np.uint8]:
    """Re-encode an RGB array as JPEG at the requested quality level."""
    pil_img = Image.fromarray(rgb_array, mode="RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()


def center_crop(rgb_array: NDArray[np.uint8], fraction: float = 0.90) -> NDArray[np.uint8]:
    """Crop the central fraction of width and height from an RGB array."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0.0, 1.0]")
    height, width = rgb_array.shape[:2]
    crop_w = max(1, round(width * fraction))
    crop_h = max(1, round(height * fraction))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    pil_img = Image.fromarray(rgb_array, mode="RGB")
    cropped = pil_img.crop((left, top, left + crop_w, top + crop_h))
    return np.asarray(cropped, dtype=np.uint8).copy()


def gaussian_blur(rgb_array: NDArray[np.uint8], radius: float = 1.5) -> NDArray[np.uint8]:
    """Apply a Gaussian blur with the specified pixel radius."""
    pil_img = Image.fromarray(rgb_array, mode="RGB")
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.uint8).copy()


def scale_resize(rgb_array: NDArray[np.uint8], factor: float = 0.5) -> NDArray[np.uint8]:
    """Resize an RGB array by factor using bicubic resampling."""
    if factor <= 0.0:
        raise ValueError("factor must be positive")
    height, width = rgb_array.shape[:2]
    new_w = max(1, round(width * factor))
    new_h = max(1, round(height * factor))
    pil_img = Image.fromarray(rgb_array, mode="RGB")
    resized = pil_img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8).copy()


def horizontal_flip(rgb_array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Mirror an RGB array left-to-right."""
    pil_img = Image.fromarray(rgb_array, mode="RGB")
    flipped = pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return np.asarray(flipped, dtype=np.uint8).copy()


PerturbationFn = Callable[[NDArray[np.uint8]], NDArray[np.uint8]]

DEFAULT_PERTURBATIONS: dict[str, PerturbationFn] = {
    "jpeg_q30": lambda arr: jpeg_compress(arr, quality=30),
    "center_crop_90": lambda arr: center_crop(arr, fraction=0.90),
    "gaussian_blur_1_5": lambda arr: gaussian_blur(arr, radius=1.5),
    "scale_resize_0_5": lambda arr: scale_resize(arr, factor=0.5),
    "horizontal_flip": horizontal_flip,
}
