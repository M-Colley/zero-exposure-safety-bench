"""Three-stage cascade router and sliding-window video temporal pooling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from zesb.ingest import ingest_sample
from zesb.schemas import (
    BenchmarkConfig,
    CascadeResult,
    DatasetSample,
    FrameCandidate,
    StagePrediction,
    VerdictStatus,
)
from zesb.stages.base import StageAdapter

TriggeredStage = Literal["HASH", "CLASSIFIER", "VLM", "NONE"]

_STAGE_PRIORITY: dict[TriggeredStage, int] = {
    "NONE": 0,
    "CLASSIFIER": 1,
    "VLM": 2,
    "HASH": 3,
}


@dataclass(frozen=True)
class FrameOutcome:
    """Evaluation outcome for a single frame candidate."""

    timestamp_sec: float
    prediction: StagePrediction
    triggered_stage: TriggeredStage


def pool_video_outcomes(
    outcomes: list[FrameOutcome],
    window_sec: float,
    positive_threshold: float,
) -> tuple[VerdictStatus, float, list[str], TriggeredStage, list[float]]:
    """Aggregate frame predictions via sliding-window max pooling."""
    if not outcomes:
        return VerdictStatus.SAFE, 0.0, [], "NONE", []

    ordered = sorted(outcomes, key=lambda item: item.timestamp_sec)

    best_window_score = -1.0
    for idx, anchor in enumerate(ordered):
        end_t = anchor.timestamp_sec + window_sec
        window_max = anchor.prediction.risk_score
        for follower in ordered[idx + 1 :]:
            if follower.timestamp_sec > end_t:
                break
            window_max = max(window_max, follower.prediction.risk_score)
        best_window_score = max(best_window_score, window_max)

    video_risk_score = min(1.0, max(0.0, best_window_score))

    category_union: set[str] = set()
    peak_timestamps: list[float] = []
    has_sensitive = False
    has_blocked = False
    has_refused = False
    has_error = False
    best_stage: TriggeredStage = "NONE"

    for item in ordered:
        pred = item.prediction
        if pred.verdict == VerdictStatus.SENSITIVE:
            has_sensitive = True
        elif pred.verdict == VerdictStatus.PROVIDER_BLOCKED:
            has_blocked = True
        elif pred.verdict == VerdictStatus.MODEL_REFUSED:
            has_refused = True
        elif pred.verdict == VerdictStatus.ERROR:
            has_error = True

        if pred.risk_score >= positive_threshold or pred.verdict in {
            VerdictStatus.SENSITIVE,
            VerdictStatus.PROVIDER_BLOCKED,
            VerdictStatus.MODEL_REFUSED,
        }:
            category_union.update(pred.predicted_categories)

        if abs(pred.risk_score - video_risk_score) <= 1e-9:
            peak_timestamps.append(item.timestamp_sec)
            if _STAGE_PRIORITY[item.triggered_stage] > _STAGE_PRIORITY[best_stage]:
                best_stage = item.triggered_stage

    if best_stage == "NONE":
        for item in ordered:
            if _STAGE_PRIORITY[item.triggered_stage] > _STAGE_PRIORITY[best_stage]:
                best_stage = item.triggered_stage

    if has_sensitive or video_risk_score >= positive_threshold:
        verdict = (
            VerdictStatus.PROVIDER_BLOCKED
            if has_blocked and not has_sensitive
            else VerdictStatus.MODEL_REFUSED
            if has_refused and not has_sensitive
            else VerdictStatus.SENSITIVE
        )
    elif has_blocked:
        verdict = VerdictStatus.PROVIDER_BLOCKED
    elif has_refused:
        verdict = VerdictStatus.MODEL_REFUSED
    elif has_error and all(o.prediction.verdict == VerdictStatus.ERROR for o in ordered):
        verdict = VerdictStatus.ERROR
    else:
        verdict = VerdictStatus.SAFE

    return (
        verdict,
        video_risk_score,
        sorted(category_union),
        best_stage,
        peak_timestamps,
    )


class CascadeRunner:
    """Executes Stage 1 -> Stage 2 -> Stage 3 cascade across images and videos."""

    def __init__(
        self,
        config: BenchmarkConfig,
        hash_stage: StageAdapter,
        classifier_stage: StageAdapter | None = None,
        vlm_stage: StageAdapter | None = None,
    ) -> None:
        self.config = config
        self.hash_stage = hash_stage
        self.classifier_stage = classifier_stage
        self.vlm_stage = vlm_stage

    async def evaluate_frame(self, frame: FrameCandidate) -> FrameOutcome:
        """Route a single frame through Stage 1, conditional Stage 2, and Stage 3."""
        cfg = self.config.cascade
        s1 = await self.hash_stage.predict(frame)
        if s1.verdict in {
            VerdictStatus.SENSITIVE,
            VerdictStatus.PROVIDER_BLOCKED,
            VerdictStatus.MODEL_REFUSED,
        }:
            return FrameOutcome(
                timestamp_sec=frame.timestamp_sec,
                prediction=s1,
                triggered_stage="HASH",
            )

        if self.classifier_stage is None:
            return FrameOutcome(
                timestamp_sec=frame.timestamp_sec,
                prediction=s1,
                triggered_stage="NONE",
            )

        s2 = await self.classifier_stage.predict(frame)
        if s2.verdict in {
            VerdictStatus.PROVIDER_BLOCKED,
            VerdictStatus.MODEL_REFUSED,
            VerdictStatus.ERROR,
        }:
            return FrameOutcome(
                timestamp_sec=frame.timestamp_sec,
                prediction=s2,
                triggered_stage="CLASSIFIER",
            )

        if s2.risk_score > cfg.borderline_high:
            return FrameOutcome(
                timestamp_sec=frame.timestamp_sec,
                prediction=StagePrediction(
                    verdict=VerdictStatus.SENSITIVE,
                    risk_score=s2.risk_score,
                    predicted_categories=s2.predicted_categories,
                ),
                triggered_stage="CLASSIFIER",
            )

        if s2.risk_score < cfg.borderline_low:
            return FrameOutcome(
                timestamp_sec=frame.timestamp_sec,
                prediction=StagePrediction(
                    verdict=VerdictStatus.SAFE,
                    risk_score=s2.risk_score,
                    predicted_categories=[],
                ),
                triggered_stage="CLASSIFIER",
            )

        if self.vlm_stage is None:
            return FrameOutcome(
                timestamp_sec=frame.timestamp_sec,
                prediction=s2,
                triggered_stage="CLASSIFIER",
            )

        s3 = await self.vlm_stage.predict(frame)
        return FrameOutcome(
            timestamp_sec=frame.timestamp_sec,
            prediction=s3,
            triggered_stage="VLM",
        )

    async def evaluate_sample(
        self,
        sample: DatasetSample,
        media_root: Path,
    ) -> CascadeResult:
        """Ingest one manifest sample, evaluate retained frames, and pool results."""
        start_ns = time.perf_counter_ns()
        sha256, candidates = ingest_sample(
            sample=sample,
            media_root=media_root,
            config=self.config.ingest,
        )

        outcomes: list[FrameOutcome] = []
        for frame in candidates:
            outcomes.append(await self.evaluate_frame(frame))

        verdict, risk_score, predicted_cats, triggered_stage, peak_ts = pool_video_outcomes(
            outcomes=outcomes,
            window_sec=self.config.cascade.window_sec,
            positive_threshold=self.config.cascade.positive_threshold,
        )
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        return CascadeResult(
            sample_id=sample.sample_id,
            media_sha256=sha256,
            ground_truth=sample.ground_truth,
            true_categories=list(sample.categories),
            hard_negative_tag=sample.hard_negative_tag,
            verdict=verdict,
            risk_score=risk_score,
            predicted_categories=predicted_cats,
            triggered_stage=triggered_stage,
            peak_timestamps_sec=peak_ts,
            latency_ms=elapsed_ms,
        )
