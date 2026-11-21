"""Shared synthetic media generators and pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


def make_checkerboard_rgb(
    width: int = 160,
    height: int = 160,
    tile: int = 16,
    fg: tuple[int, int, int] = (220, 20, 20),
    bg: tuple[int, int, int] = (240, 240, 240),
) -> NDArray[np.uint8]:
    """Generate a high-contrast synthetic checkerboard RGB image with asymmetric bar."""
    ys, xs = np.indices((height, width))
    mask = ((xs // tile) + (ys // tile)) % 2 == 0
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[mask] = fg
    arr[~mask] = bg
    # Break periodicity in one quadrant so pHash produces distinct bits from solid black
    arr[: height // 2, : width // 3] = fg
    return arr


def make_geometric_scene_rgb(
    width: int = 256,
    height: int = 256,
) -> NDArray[np.uint8]:
    """Generate a non-periodic multi-shape RGB scene with high PDQ quality."""
    img = Image.new("RGB", (width, height), (30, 50, 80))
    draw = ImageDraw.Draw(img)
    sx, sy = width / 256.0, height / 256.0
    draw.rectangle(
        [int(20 * sx), int(20 * sy), int(120 * sx), int(140 * sy)],
        fill=(220, 60, 40),
    )
    draw.ellipse(
        [int(90 * sx), int(60 * sy), int(220 * sx), int(200 * sy)],
        fill=(240, 210, 50),
    )
    draw.polygon(
        [
            (int(40 * sx), int(220 * sy)),
            (int(140 * sx), int(130 * sy)),
            (int(230 * sx), int(230 * sy)),
        ],
        fill=(40, 190, 120),
    )
    draw.rectangle(
        [int(150 * sx), int(30 * sy), int(235 * sx), int(95 * sy)],
        fill=(230, 230, 240),
    )
    return np.asarray(img, dtype=np.uint8).copy()


def make_concentric_rgb(
    width: int = 256,
    height: int = 256,
    ring: int = 16,
    c1: tuple[int, int, int] = (30, 180, 90),
    c2: tuple[int, int, int] = (15, 30, 45),
) -> NDArray[np.uint8]:
    """Generate a synthetic concentric-ring RGB image distinct from geometric scenes."""
    ys, xs = np.indices((height, width))
    cy, cx = height / 2.0, width / 2.0
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask = (dist // ring).astype(np.int32) % 2 == 0
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[mask] = c1
    arr[~mask] = c2
    return arr


def write_synthetic_png(path: Path, rgb_array: NDArray[np.uint8]) -> Path:
    """Save an RGB array as a PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_array, mode="RGB").save(path, format="PNG")
    return path


def write_multi_scene_mp4(
    path: Path,
    fps: float = 30.0,
    width: int = 160,
    height: int = 160,
) -> Path:
    """Write a synthetic MP4 switching from black to a 3-frame red checkerboard and back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")

    black_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    checker_rgb = make_checkerboard_rgb(width=width, height=height)
    checker_bgr = cv2.cvtColor(checker_rgb, cv2.COLOR_RGB2BGR)

    try:
        for frame_idx in range(120):
            if 60 <= frame_idx <= 62:
                writer.write(checker_bgr)
            else:
                writer.write(black_bgr)
    finally:
        writer.release()
    return path


def write_static_mp4(
    path: Path,
    rgb_array: NDArray[np.uint8],
    num_frames: int = 60,
    fps: float = 30.0,
) -> Path:
    """Write a static synthetic MP4 repeating a single RGB pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = rgb_array.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    try:
        for _ in range(num_frames):
            writer.write(bgr)
    finally:
        writer.release()
    return path
