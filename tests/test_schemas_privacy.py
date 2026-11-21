"""Zero-exposure serialization and validation tests for schemas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from zesb.schemas import (
    CascadeResult,
    DatasetSample,
    FrameCandidate,
    GroundTruthState,
    MediaType,
    SanitizedErrorRecord,
    VerdictStatus,
)


def test_dataset_sample_rejects_path_traversal(tmp_path: Path) -> None:
    """Reject relative_path values that traverse outside media_root."""
    with pytest.raises(ValidationError):
        DatasetSample(
            sample_id="bad_1",
            relative_path="../secret.png",
            media_type=MediaType.IMAGE,
            ground_truth=GroundTruthState.SENSITIVE,
        )

    with pytest.raises(ValidationError):
        DatasetSample(
            sample_id="bad_2",
            relative_path="/etc/passwd",
            media_type=MediaType.IMAGE,
            ground_truth=GroundTruthState.SAFE,
        )

    valid = DatasetSample(
        sample_id="ok_1",
        relative_path="nested/frame.png",
        media_type=MediaType.IMAGE,
        ground_truth=GroundTruthState.SAFE,
    )
    resolved = valid.resolve_media_path(tmp_path)
    assert resolved == (tmp_path / "nested" / "frame.png").resolve()


def test_frame_candidate_excludes_rgb_array_from_serialization() -> None:
    """Ensure FrameCandidate dumps and repr never expose raw RGB arrays."""
    rgb = np.full((16, 16, 3), 255, dtype=np.uint8)
    bits = np.zeros(256, dtype=np.uint8)
    candidate = FrameCandidate(
        frame_index=2,
        timestamp_sec=1.5,
        pdq_bits=bits,
        rgb_array=rgb,
    )

    dumped = candidate.model_dump()
    assert "rgb_array" not in dumped
    dumped_json = candidate.model_dump_json()
    parsed = json.loads(dumped_json)
    assert "rgb_array" not in parsed
    assert "255" not in dumped_json
    assert "255" not in repr(candidate)


def test_sanitized_error_record_blocks_sensitive_previews() -> None:
    """Reject any non-null preview payload on SENSITIVE error records."""
    digest = "a" * 64
    with pytest.raises(
        ValidationError, match="Sensitive records must never contain image previews"
    ):
        SanitizedErrorRecord(
            sample_id="sens_1",
            media_sha256=digest,
            error_type="FALSE_NEGATIVE",
            ground_truth=GroundTruthState.SENSITIVE,
            risk_score=0.1,
            safe_blurred_jpeg_b64="ZmFrZV9qcGVn",
        )

    safe_fp = SanitizedErrorRecord(
        sample_id="safe_1",
        media_sha256=digest,
        error_type="FALSE_POSITIVE",
        ground_truth=GroundTruthState.SAFE,
        risk_score=0.92,
        safe_blurred_jpeg_b64="ZmFrZV9qcGVn",
    )
    assert safe_fp.safe_blurred_jpeg_b64 == "ZmFrZV9qcGVn"


def test_cascade_result_clamps_scores_and_forbids_extra_fields() -> None:
    """Verify CascadeResult clamps scores to [0, 1] and forbids relative_path."""
    digest = "b" * 64
    res = CascadeResult(
        sample_id="s_1",
        media_sha256=digest,
        ground_truth=GroundTruthState.SENSITIVE,
        verdict=VerdictStatus.SENSITIVE,
        risk_score=1.00001,
        triggered_stage="HASH",
        latency_ms=4.2,
    )
    assert res.risk_score == 1.0
    assert "relative_path" not in res.model_dump()

    with pytest.raises(ValidationError):
        CascadeResult(
            sample_id="s_2",
            media_sha256=digest,
            ground_truth=GroundTruthState.SENSITIVE,
            verdict=VerdictStatus.SENSITIVE,
            risk_score=0.5,
            triggered_stage="HASH",
            latency_ms=1.0,
            relative_path="secret/file.png",  # type: ignore[call-arg]
        )
