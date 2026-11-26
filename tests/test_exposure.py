"""Analytical verification of the human exposure budget and TVEC configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zesb.cascade import FrameOutcome, review_window_seconds
from zesb.cli import load_benchmark_config
from zesb.metrics import compute_benchmark_metrics, compute_human_exposure
from zesb.schemas import (
    CascadeResult,
    ExposureConfig,
    GroundTruthState,
    StagePrediction,
    VerdictStatus,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _res(
    sample_id: str,
    gt: GroundTruthState,
    verdict: VerdictStatus,
    score: float,
    stage: str = "CLASSIFIER",
    duration: float = 0.0,
    window: float = 0.0,
) -> CascadeResult:
    return CascadeResult(
        sample_id=sample_id,
        media_sha256="e" * 64,
        ground_truth=gt,
        verdict=verdict,
        risk_score=score,
        triggered_stage=stage,  # type: ignore[arg-type]
        latency_ms=1.0,
        media_duration_sec=duration,
        review_window_sec=window,
    )


def _fixture_results() -> list[CascadeResult]:
    sens, safe = GroundTruthState.SENSITIVE, GroundTruthState.SAFE
    return [
        _res("hash", sens, VerdictStatus.SENSITIVE, 1.0, stage="HASH", duration=60.0, window=5.0),
        _res("cls", sens, VerdictStatus.SENSITIVE, 0.95, duration=60.0, window=10.0),
        _res("missed", sens, VerdictStatus.SAFE, 0.2, duration=60.0),
        _res("error", sens, VerdictStatus.ERROR, 0.0, duration=60.0),
        _res("fp", safe, VerdictStatus.SENSITIVE, 0.7),
        _res("tn", safe, VerdictStatus.SAFE, 0.1),
    ]


def test_exposure_budget_default_policy() -> None:
    m = compute_human_exposure(_fixture_results())
    assert m.manual_exposure_items == 4
    assert m.manual_exposure_video_sec == pytest.approx(240.0)
    assert m.auto_actioned_items == 1
    assert m.review_items == 3  # cls, error, fp
    assert m.exposure_items == 2  # cls, error
    assert m.exposure_video_sec == pytest.approx(70.0)  # 10 s window + 60 s error fallback
    assert m.missed_sensitive_items == 1
    assert m.exposure_items_per_1000 == pytest.approx(2 / 6 * 1000)
    assert m.exposure_reduction_items == pytest.approx(0.5)
    assert m.exposure_reduction_video_sec == pytest.approx(1 - 70 / 240)


def test_exposure_budget_without_hash_action_or_targeted_review() -> None:
    cfg = ExposureConfig(hash_auto_action=False, targeted_video_review=False)
    m = compute_human_exposure(_fixture_results(), cfg)
    assert m.auto_actioned_items == 0
    assert m.exposure_items == 3
    assert m.exposure_video_sec == pytest.approx(180.0)


def test_exposure_at_recall_targets() -> None:
    m = compute_human_exposure(_fixture_results(), ExposureConfig(recall_targets=[0.75, 0.9]))
    # Hash and error items count as detected at every threshold (2/4 = 0.5).
    low = m.at_recall["0.750"]
    assert low.threshold == pytest.approx(0.95)
    assert low.achieved_recall == pytest.approx(0.75)
    assert (low.review_items, low.exposure_items) == (2, 2)  # error, cls
    high = m.at_recall["0.900"]
    assert high.threshold == pytest.approx(0.2)
    assert high.achieved_recall == pytest.approx(1.0)
    assert (high.review_items, high.exposure_items) == (4, 3)  # error, cls, fp, missed


def test_review_window_union() -> None:
    def outcome(t: float, score: float) -> FrameOutcome:
        verdict = VerdictStatus.SENSITIVE if score >= 0.5 else VerdictStatus.SAFE
        return FrameOutcome(
            timestamp_sec=t,
            prediction=StagePrediction(verdict=verdict, risk_score=score),
            triggered_stage="CLASSIFIER",
        )

    outcomes = [outcome(0.0, 0.1), outcome(2.0, 0.9), outcome(4.0, 0.8), outcome(20.0, 0.6)]
    # [2, 7] U [4, 9] U [20, 22 clipped by duration] = 7 + 2 = 9 s
    assert review_window_seconds(outcomes, 5.0, 0.5, duration_sec=22.0) == pytest.approx(9.0)
    assert review_window_seconds([], 5.0, 0.5, duration_sec=22.0) == 0.0


def test_benchmark_metrics_include_exposure() -> None:
    metrics = compute_benchmark_metrics(_fixture_results())
    assert metrics.human_exposure.exposure_items == 2


def test_tvec_config_loads() -> None:
    config = load_benchmark_config(REPO_ROOT / "configs" / "tvec_gifct.yaml")
    ids = {c.id for c in config.taxonomy.categories}
    assert "BRANDED_PUBLICATION" in ids
    assert "CSAM_CONFIRMED" not in ids
    assert config.exposure.recall_targets == [0.9, 0.95, 0.99]
    assert np.isclose(config.cascade.window_sec, 5.0)


def test_recall_targets_validated() -> None:
    with pytest.raises(ValueError):
        ExposureConfig(recall_targets=[0.0])
