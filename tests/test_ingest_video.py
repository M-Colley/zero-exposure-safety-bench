"""Ingestion, scene detection, floor-FPS sampling, and pHash deduplication tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.conftest import (
    make_geometric_scene_rgb,
    write_multi_scene_mp4,
    write_static_mp4,
    write_synthetic_png,
)

from zesb.ingest import compute_sha256, decode_image, extract_video_candidates, ingest_sample
from zesb.schemas import DatasetSample, GroundTruthState, IngestConfig, MediaType


def test_ingest_image_computes_sha256_and_strips_exif(tmp_path: Path) -> None:
    """Verify streaming SHA-256 digest and RGB decoding for synthetic PNGs."""
    rgb = make_geometric_scene_rgb(width=160, height=160)
    img_path = write_synthetic_png(tmp_path / "safe" / "scene.png", rgb)
    sample = DatasetSample(
        sample_id="img_1",
        relative_path="safe/scene.png",
        media_type=MediaType.IMAGE,
        ground_truth=GroundTruthState.SAFE,
    )

    digest, frames = ingest_sample(sample, tmp_path)
    assert digest == compute_sha256(img_path)
    assert len(frames) == 1
    assert frames[0].frame_index == 0
    assert frames[0].timestamp_sec == 0.0
    assert np.array_equal(frames[0].rgb_array, rgb)

    decoded = decode_image(img_path)
    assert decoded.rgb_array.dtype == np.uint8
    assert decoded.rgb_array.shape == (160, 160, 3)


def test_extract_video_captures_brief_anomaly_and_deduplicates_static_segments(
    tmp_path: Path,
) -> None:
    """Verify PySceneDetect captures a 3-frame anomaly while pHash drops static black frames."""
    video_path = write_multi_scene_mp4(tmp_path / "multi_scene.mp4", fps=30.0)
    cfg = IngestConfig(scene_threshold=27.0, min_fps=1.0, dedup_hamming_threshold=4)
    candidates = extract_video_candidates(video_path, config=cfg)

    # Static black frames before frame 60 collapse to 1 frame.
    # The 3-frame red checkerboard around frame 60..62 is retained as a scene frame.
    # Static black frames after the checkerboard collapse to 1 frame.
    assert 2 <= len(candidates) <= 4
    has_red_anomaly = any(int(c.rgb_array[:, :, 0].mean()) > 100 for c in candidates)
    assert has_red_anomaly, (
        f"Expected red checkerboard frame in {[c.frame_index for c in candidates]}"
    )
    assert all(
        candidates[i].timestamp_sec < candidates[i + 1].timestamp_sec
        for i in range(len(candidates) - 1)
    )


def test_static_video_deduplicates_to_single_representative_frame(tmp_path: Path) -> None:
    """Ensure a completely static video collapses via pHash deduplication."""
    static_rgb = make_geometric_scene_rgb(width=160, height=160)
    video_path = write_static_mp4(tmp_path / "static.mp4", static_rgb, num_frames=60, fps=30.0)
    candidates = extract_video_candidates(video_path)
    assert len(candidates) == 1
    assert candidates[0].frame_index == 0


def test_corrupted_video_error_omits_sensitive_file_path(tmp_path: Path) -> None:
    """Ensure video decode failures never embed raw file paths in exception messages."""
    secret_file = tmp_path / "secret_vault" / "sensitive_broken_clip.mp4"
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    secret_file.write_bytes(b"not-a-valid-mp4-stream")
    with pytest.raises(ValueError) as exc_info:
        extract_video_candidates(secret_file)
    assert "sensitive_broken_clip.mp4" not in str(exc_info.value)
    assert "secret_vault" not in str(exc_info.value)
