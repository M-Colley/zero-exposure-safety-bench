"""Analytical verification of high-precision operating points and dual evaluation views."""

from __future__ import annotations

import pytest

from zesb.metrics import compute_benchmark_metrics
from zesb.schemas import CascadeResult, GroundTruthState, VerdictStatus


def _make_result(
    sample_id: str,
    gt: GroundTruthState,
    verdict: VerdictStatus,
    risk_score: float,
    tag: str | None = None,
    stage: str = "CLASSIFIER",
    latency_ms: float = 10.0,
) -> CascadeResult:
    return CascadeResult(
        sample_id=sample_id,
        media_sha256="c" * 64,
        ground_truth=gt,
        true_categories=["GRAPHIC_VIOLENCE"] if gt == GroundTruthState.SENSITIVE else [],
        hard_negative_tag=tag,
        verdict=verdict,
        risk_score=risk_score,
        predicted_categories=["GRAPHIC_VIOLENCE"] if verdict == VerdictStatus.SENSITIVE else [],
        triggered_stage=stage,  # type: ignore[arg-type]
        latency_ms=latency_ms,
    )


def test_analytical_operating_points_and_hard_negative_slices() -> None:
    """Verify Recall@99.9% Precision, Recall@0.1% FPR, and tag-stratified FPR."""
    results: list[CascadeResult] = []

    # 1,000 SAFE samples:
    # - 100 pediatric_dermatology: 2 false positives above 0.50 (scores 0.80, 0.75) -> FPR = 0.02
    # - 100 martial_arts: 0 false positives -> FPR = 0.00
    # - 800 general safe: scores <= 0.40
    for i in range(100):
        is_fp = i < 2
        results.append(
            _make_result(
                f"ped_{i}",
                GroundTruthState.SAFE,
                VerdictStatus.SENSITIVE if is_fp else VerdictStatus.SAFE,
                0.80 if i == 0 else (0.75 if i == 1 else 0.10 + i * 0.001),
                tag="pediatric_dermatology",
            )
        )
    for i in range(100):
        results.append(
            _make_result(
                f"ma_{i}",
                GroundTruthState.SAFE,
                VerdictStatus.SAFE,
                0.05 + i * 0.002,
                tag="martial_arts",
            )
        )
    for i in range(800):
        results.append(
            _make_result(
                f"safe_{i}",
                GroundTruthState.SAFE,
                VerdictStatus.SAFE,
                0.001 + (i % 40) * 0.005,
            )
        )

    # 10 SENSITIVE samples:
    # - 9 ranked strictly above max SAFE score (0.80): 0.91..0.99 (1 PROVIDER_BLOCKED at 1.0)
    # - 1 low-scoring miss at 0.30
    for i in range(8):
        results.append(
            _make_result(
                f"sens_{i}",
                GroundTruthState.SENSITIVE,
                VerdictStatus.SENSITIVE,
                0.91 + i * 0.01,
                stage="HASH" if i < 2 else "CLASSIFIER",
                latency_ms=5.0 if i < 2 else 20.0,
            )
        )
    results.append(
        _make_result(
            "sens_blocked",
            GroundTruthState.SENSITIVE,
            VerdictStatus.PROVIDER_BLOCKED,
            1.0,
            stage="CLASSIFIER",
        )
    )
    results.append(
        _make_result(
            "sens_miss",
            GroundTruthState.SENSITIVE,
            VerdictStatus.SAFE,
            0.30,
            stage="VLM",
        )
    )

    metrics = compute_benchmark_metrics(results)
    assert metrics.total_samples == 1010
    assert metrics.sensitive_samples == 10
    assert metrics.safe_samples == 1000

    # At score >= 0.91: TP=9, FP=0 -> Precision = 1.0 >= 0.999, FPR = 0.0 <= 0.001, Recall = 0.90
    assert metrics.effective_detection.recall_at_precision_999 == pytest.approx(0.90, abs=1e-6)
    assert metrics.effective_detection.recall_at_precision_990 == pytest.approx(0.90, abs=1e-6)
    assert metrics.effective_detection.recall_at_fpr_001 == pytest.approx(0.90, abs=1e-6)
    assert metrics.effective_detection.recall_at_fpr_010 == pytest.approx(1.00, abs=1e-6)

    # Dual views: blocked counts as missed in strict_taxonomy, positive in effective_detection
    assert metrics.strict_taxonomy.confusion_matrix.tp == 8
    assert metrics.strict_taxonomy.confusion_matrix.fn == 2
    assert metrics.strict_taxonomy.provider_block_rate == pytest.approx(1 / 1010, rel=1e-6)
    assert metrics.effective_detection.confusion_matrix.tp == 9
    assert metrics.effective_detection.confusion_matrix.fn == 1

    # Hard-negative slice FPR
    assert metrics.hard_negative_fpr_by_tag["pediatric_dermatology"] == pytest.approx(
        0.02, abs=1e-6
    )
    assert metrics.hard_negative_fpr_by_tag["martial_arts"] == pytest.approx(0.00, abs=1e-6)

    # Stage exit distribution
    assert metrics.stage_exit_distribution["HASH"] == 2
    assert metrics.stage_exit_distribution["VLM"] == 1
