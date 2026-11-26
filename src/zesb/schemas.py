"""Pydantic v2 data contracts and zero-exposure validation rules."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

STRICT_CONFIG = ConfigDict(extra="forbid", frozen=True)


class MediaType(StrEnum):
    """Supported media asset types."""

    IMAGE = "IMAGE"
    VIDEO = "VIDEO"


class GroundTruthState(StrEnum):
    """Ground-truth safety label for a benchmark sample."""

    SAFE = "SAFE"
    SENSITIVE = "SENSITIVE"


class VerdictStatus(StrEnum):
    """Normalized outcome returned by a stage or the full cascade."""

    SENSITIVE = "SENSITIVE"
    SAFE = "SAFE"
    PROVIDER_BLOCKED = "PROVIDER_BLOCKED"
    MODEL_REFUSED = "MODEL_REFUSED"
    ERROR = "ERROR"


def _clamp_unit_score(value: float) -> float:
    """Validate finiteness and clamp a score into [0.0, 1.0]."""
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError("Score must be a finite float")
    return min(1.0, max(0.0, numeric))


class DatasetSample(BaseModel):
    """Single row of an input manifest.jsonl dataset."""

    model_config = STRICT_CONFIG

    sample_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    media_type: MediaType
    ground_truth: GroundTruthState
    categories: list[str] = Field(default_factory=list)
    hard_negative_tag: str | None = None

    @field_validator("relative_path")
    @classmethod
    def _reject_traversal(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("relative_path must not contain null bytes")
        normalized = value.replace("\\", "/")
        candidate = Path(normalized)
        if (
            candidate.is_absolute()
            or normalized.startswith("/")
            or (len(normalized) >= 2 and normalized[1] == ":")
        ):
            raise ValueError("relative_path must be relative to media_root")
        if ".." in candidate.parts:
            raise ValueError("relative_path must not contain parent traversal ('..')")
        return value

    def resolve_media_path(self, media_root: Path) -> Path:
        """Resolve relative_path inside media_root, rejecting escapes."""
        root_resolved = media_root.resolve()
        candidate = (root_resolved / self.relative_path).resolve()
        if not candidate.is_relative_to(root_resolved):
            raise ValueError(f"Sample {self.sample_id} path escapes media_root")
        return candidate


class FrameCandidate(BaseModel):
    """In-memory video frame or image candidate with non-serializable pixels."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    pdq_bits: NDArray[np.uint8] | None = None
    rgb_array: Annotated[NDArray[np.uint8], Field(exclude=True, repr=False)]

    @field_serializer("pdq_bits")
    def _serialize_pdq_bits(self, value: NDArray[np.uint8] | None) -> list[int] | None:
        if value is None:
            return None
        return [int(bit) for bit in value.tolist()]


class CascadeResult(BaseModel):
    """Unified prediction envelope omitting sensitive paths and raw rationales."""

    model_config = STRICT_CONFIG

    sample_id: str = Field(min_length=1)
    media_sha256: str = Field(min_length=64, max_length=64)
    ground_truth: GroundTruthState
    true_categories: list[str] = Field(default_factory=list)
    hard_negative_tag: str | None = None
    verdict: VerdictStatus
    risk_score: float
    predicted_categories: list[str] = Field(default_factory=list)
    triggered_stage: Literal["HASH", "CLASSIFIER", "VLM", "NONE"]
    peak_timestamps_sec: list[float] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    media_duration_sec: float = Field(default=0.0, ge=0.0)
    review_window_sec: float = Field(default=0.0, ge=0.0)

    @field_validator("risk_score", mode="before")
    @classmethod
    def _clamp_risk_score(cls, value: float) -> float:
        return _clamp_unit_score(value)


class SanitizedErrorRecord(BaseModel):
    """Sanitized failure record permitting blurred previews only for SAFE items."""

    model_config = STRICT_CONFIG

    sample_id: str = Field(min_length=1)
    media_sha256: str = Field(min_length=64, max_length=64)
    error_type: Literal[
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE",
        "BLOCKED_OR_REFUSED",
        "RUNTIME_ERROR",
    ]
    ground_truth: GroundTruthState
    risk_score: float
    predicted_categories: list[str] = Field(default_factory=list)
    true_categories: list[str] = Field(default_factory=list)
    hard_negative_tag: str | None = None
    safe_blurred_jpeg_b64: str | None = None

    @field_validator("risk_score", mode="before")
    @classmethod
    def _clamp_risk_score(cls, value: float) -> float:
        return _clamp_unit_score(value)

    @model_validator(mode="after")
    def _enforce_zero_exposure_preview(self) -> SanitizedErrorRecord:
        if (
            self.ground_truth == GroundTruthState.SENSITIVE
            and self.safe_blurred_jpeg_b64 is not None
        ):
            raise ValueError("Sensitive records must never contain image previews")
        return self


class StagePrediction(BaseModel):
    """Normalized prediction returned by any individual cascade stage."""

    model_config = STRICT_CONFIG

    verdict: VerdictStatus
    risk_score: float
    predicted_categories: list[str] = Field(default_factory=list)
    provider_code: str | None = None

    @field_validator("risk_score", mode="before")
    @classmethod
    def _clamp_risk_score(cls, value: float) -> float:
        return _clamp_unit_score(value)


class TaxonomyCategory(BaseModel):
    """Single category entry in a safety policy taxonomy."""

    model_config = STRICT_CONFIG

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class TaxonomyConfig(BaseModel):
    """Policy taxonomy specification loaded from YAML."""

    model_config = STRICT_CONFIG

    mode: Literal["binary", "multilabel"]
    categories: list[TaxonomyCategory] = Field(min_length=1)
    benign_lookalikes: list[str] = Field(default_factory=list)


class IngestConfig(BaseModel):
    """Video scene-sampling and perceptual deduplication settings."""

    model_config = STRICT_CONFIG

    scene_threshold: float = Field(default=27.0, gt=0.0)
    min_fps: float = Field(default=1.0, gt=0.0)
    dedup_hamming_threshold: int = Field(default=4, ge=0, le=64)


class CascadeConfig(BaseModel):
    """Routing thresholds and network guardrails for the three-stage cascade."""

    model_config = STRICT_CONFIG

    pdq_distance_threshold: int = Field(default=31, ge=0, le=256)
    pdq_min_quality: int = Field(default=50, ge=0, le=100)
    borderline_low: float = Field(default=0.25, ge=0.0, le=1.0)
    borderline_high: float = Field(default=0.90, ge=0.0, le=1.0)
    positive_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    window_sec: float = Field(default=5.0, gt=0.0)
    allow_non_private_hosts: bool = False

    @model_validator(mode="after")
    def _validate_borderline_band(self) -> CascadeConfig:
        if self.borderline_low > self.borderline_high:
            raise ValueError("borderline_low must not exceed borderline_high")
        return self


class ExposureConfig(BaseModel):
    """Human review exposure model used to compute the exposure budget."""

    model_config = STRICT_CONFIG

    hash_auto_action: bool = True
    targeted_video_review: bool = True
    recall_targets: list[float] = Field(default_factory=lambda: [0.90, 0.95, 0.99])

    @field_validator("recall_targets")
    @classmethod
    def _validate_recall_targets(cls, value: list[float]) -> list[float]:
        if any(not 0.0 < target <= 1.0 for target in value):
            raise ValueError("recall_targets must lie in (0.0, 1.0]")
        return sorted(set(value))


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration loaded from YAML."""

    model_config = STRICT_CONFIG

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    taxonomy: TaxonomyConfig
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    exposure: ExposureConfig = Field(default_factory=ExposureConfig)
