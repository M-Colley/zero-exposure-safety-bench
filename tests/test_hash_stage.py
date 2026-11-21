"""Stage 1 PDQ hash matching and perturbation robustness tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tests.conftest import make_concentric_rgb, make_geometric_scene_rgb

from zesb.perturbations import center_crop, gaussian_blur, jpeg_compress, scale_resize
from zesb.schemas import FrameCandidate, VerdictStatus
from zesb.stages.hash_stage import (
    HashIndex,
    HashStageAdapter,
    compute_pdq,
    pdq_bits_to_hex,
    pdq_hex_to_bits,
)


async def test_pdq_exact_and_perturbed_match_and_distinct_rejection(tmp_path: Path) -> None:
    """Verify exact, JPEG, blur, scale, and slight-crop frames match within threshold."""
    base = make_geometric_scene_rgb(width=256, height=256)

    bits, quality = compute_pdq(base)
    assert quality >= 50
    hex_digest = pdq_bits_to_hex(bits)
    assert len(hex_digest) == 64
    assert (pdq_hex_to_bits(hex_digest) == bits).all()

    index_path = tmp_path / "hashes.jsonl"
    index_path.write_text(
        json.dumps({"pdq_hex": hex_digest, "categories": ["GRAPHIC_VIOLENCE"]}) + "\n",
        encoding="utf-8",
    )
    index = HashIndex.from_file(index_path)
    adapter = HashStageAdapter(index=index, distance_threshold=31, min_quality_score=50)

    exact_pred = await adapter.predict(
        FrameCandidate(frame_index=0, timestamp_sec=0.0, rgb_array=base)
    )
    assert exact_pred.verdict == VerdictStatus.SENSITIVE
    assert exact_pred.risk_score == 1.0
    assert exact_pred.predicted_categories == ["GRAPHIC_VIOLENCE"]

    for idx, transformed in enumerate(
        (
            jpeg_compress(base, quality=30),
            gaussian_blur(base, radius=1.5),
            scale_resize(base, factor=0.5),
            center_crop(base, fraction=0.98),
        ),
        start=1,
    ):
        pred = await adapter.predict(
            FrameCandidate(frame_index=idx, timestamp_sec=float(idx), rgb_array=transformed)
        )
        assert pred.verdict == VerdictStatus.SENSITIVE
        assert pred.risk_score >= 1.0 - (31.0 / 256.0)

    distinct = make_concentric_rgb(width=256, height=256, ring=16)
    distinct_pred = await adapter.predict(
        FrameCandidate(frame_index=10, timestamp_sec=10.0, rgb_array=distinct)
    )
    assert distinct_pred.verdict == VerdictStatus.SAFE


async def test_low_quality_flat_frame_is_ignored(tmp_path: Path) -> None:
    """Ignore flat degenerate frames with PDQ quality < 50 to avoid false collisions."""
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    bits, quality = compute_pdq(flat)
    assert quality < 50
    csv_path = tmp_path / "hashes.csv"
    csv_path.write_text(
        f"pdq_hex,categories\n{pdq_bits_to_hex(bits)},SENSITIVE\n",
        encoding="utf-8",
    )
    adapter = HashStageAdapter(index=HashIndex.from_file(csv_path))
    pred = await adapter.predict(FrameCandidate(frame_index=0, timestamp_sec=0.0, rgb_array=flat))
    assert pred.verdict == VerdictStatus.SAFE
    assert pred.risk_score == 0.0
