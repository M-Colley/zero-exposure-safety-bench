"""Media hashing, image decoding, hybrid video scene sampling, and pHash deduplication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imagehash
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps
from scenedetect import ContentDetector, detect

from zesb.schemas import DatasetSample, FrameCandidate, IngestConfig, MediaType

CHUNK_SIZE_BYTES = 65_536


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file using 64 KB streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> list[DatasetSample]:
    """Load and validate newline-delimited JSON manifest entries."""
    samples: list[DatasetSample] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                samples.append(DatasetSample.model_validate(payload))
            except Exception as exc:
                raise ValueError(f"Invalid manifest entry at line {line_no}: {exc}") from exc
    return samples


def decode_image(image_path: Path) -> FrameCandidate:
    """Decode an image to an EXIF-transposed, metadata-stripped RGB uint8 candidate."""
    with Image.open(image_path) as img:
        transposed = ImageOps.exif_transpose(img)
        rgb_image = transposed.convert("RGB")
        rgb_array = np.asarray(rgb_image, dtype=np.uint8).copy()
    return FrameCandidate(
        frame_index=0,
        timestamp_sec=0.0,
        pdq_bits=None,
        rgb_array=rgb_array,
    )


def deduplicate_frames(
    candidates: list[FrameCandidate],
    hamming_threshold: int = 4,
) -> list[FrameCandidate]:
    """Drop sequential frames whose 64-bit pHash distance is within threshold."""
    retained: list[FrameCandidate] = []
    last_phash: imagehash.ImageHash | None = None
    for candidate in candidates:
        pil_img = Image.fromarray(candidate.rgb_array, mode="RGB")
        current_phash = imagehash.phash(pil_img)
        if last_phash is not None and int(current_phash - last_phash) <= hamming_threshold:
            continue
        retained.append(candidate)
        last_phash = current_phash
    return retained


def _collect_candidate_indices(
    video_path: Path,
    total_frames: int,
    fps: float,
    scene_threshold: float,
    min_fps: float,
) -> list[int]:
    """Combine PySceneDetect start/mid/end frames with uniform floor-FPS indices."""
    if total_frames <= 0:
        return []

    indices: set[int] = {0}

    scenes = detect(
        str(video_path),
        ContentDetector(threshold=scene_threshold, min_scene_len=1),
    )
    for start_tc, end_tc in scenes:
        start_idx = max(0, min(total_frames - 1, int(start_tc.frame_num)))
        end_idx = max(start_idx, min(total_frames - 1, int(end_tc.frame_num) - 1))
        mid_idx = (start_idx + end_idx) // 2
        indices.update((start_idx, mid_idx, end_idx))

    step_sec = 1.0 / min_fps
    duration_sec = total_frames / fps
    current_sec = 0.0
    while current_sec < duration_sec:
        frame_idx = min(total_frames - 1, round(current_sec * fps))
        indices.add(frame_idx)
        current_sec += step_sec

    indices.add(total_frames - 1)
    return sorted(indices)


def extract_video_candidates(
    video_path: Path,
    config: IngestConfig | None = None,
) -> list[FrameCandidate]:
    """Extract deduplicated RGB keyframes using scene boundaries and floor FPS."""
    cfg = config or IngestConfig()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open video stream for decoding")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = raw_fps if raw_fps > 0.0 else 30.0
        target_indices = _collect_candidate_indices(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            scene_threshold=cfg.scene_threshold,
            min_fps=cfg.min_fps,
        )

        raw_candidates: list[FrameCandidate] = []
        if target_indices:
            target_set = set(target_indices)
            max_target = target_indices[-1]
            frame_idx = 0
            while frame_idx <= max_target:
                ok, bgr_frame = cap.read()
                if not ok or bgr_frame is None:
                    break
                if frame_idx in target_set:
                    rgb_frame: NDArray[np.uint8] = np.asarray(
                        cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
                        dtype=np.uint8,
                    )
                    raw_candidates.append(
                        FrameCandidate(
                            frame_index=frame_idx,
                            timestamp_sec=float(frame_idx / fps),
                            pdq_bits=None,
                            rgb_array=rgb_frame,
                        )
                    )
                frame_idx += 1
    finally:
        cap.release()

    return deduplicate_frames(
        raw_candidates,
        hamming_threshold=cfg.dedup_hamming_threshold,
    )


def ingest_sample(
    sample: DatasetSample,
    media_root: Path,
    config: IngestConfig | None = None,
) -> tuple[str, list[FrameCandidate]]:
    """Compute SHA-256 digest and decode frame candidates for one manifest sample."""
    media_path = sample.resolve_media_path(media_root)
    if not media_path.is_file():
        raise FileNotFoundError(f"Media file not found for sample {sample.sample_id}")
    sha256 = compute_sha256(media_path)
    if sample.media_type == MediaType.IMAGE:
        return sha256, [decode_image(media_path)]
    return sha256, extract_video_candidates(media_path, config=config)
