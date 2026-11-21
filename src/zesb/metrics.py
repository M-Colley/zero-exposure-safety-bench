"""High-precision safety metrics, dual evaluation views, and slice stratification."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from zesb.schemas import CascadeResult, GroundTruthState, VerdictStatus

EFFECTIVE_POSITIVE_VERDICTS: frozenset[VerdictStatus] = frozenset(
    {
        VerdictStatus.SENSITIVE,
        VerdictStatus.PROVIDER_BLOCKED,
        VerdictStatus.MODEL_REFUSED,
    }
)


class ConfusionCounts(BaseModel):
    """Binary confusion matrix counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tn: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tp: int = Field(ge=0)


class StrictTaxonomyMetrics(BaseModel):
    """Metrics evaluating exact structured predictions without block/refusal credit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    binary_f1: float = Field(ge=0.0, le=1.0)
    confusion_matrix: ConfusionCounts
    micro_f1: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    per_category_pr_auc: dict[str, float]
    provider_block_rate: float = Field(ge=0.0, le=1.0)
    model_refusal_rate: float = Field(ge=0.0, le=1.0)
    error_rate: float = Field(ge=0.0, le=1.0)


class EffectiveDetectionMetrics(BaseModel):
    """Metrics treating blocks and refusals as positive detections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    f1: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    confusion_matrix: ConfusionCounts
    pr_auc: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)
    recall_at_precision_999: float = Field(ge=0.0, le=1.0)
    recall_at_precision_990: float = Field(ge=0.0, le=1.0)
    recall_at_fpr_001: float = Field(ge=0.0, le=1.0)
    recall_at_fpr_010: float = Field(ge=0.0, le=1.0)


class BenchmarkMetrics(BaseModel):
    """Complete benchmark evaluation metrics across both views and slices."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_samples: int = Field(ge=0)
    sensitive_samples: int = Field(ge=0)
    safe_samples: int = Field(ge=0)
    strict_taxonomy: StrictTaxonomyMetrics
    effective_detection: EffectiveDetectionMetrics
    hard_negative_fpr_by_tag: dict[str, float]
    stage_exit_distribution: dict[str, int]
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)


def _confusion_counts(y_true: list[int], y_pred: list[int]) -> ConfusionCounts:
    if not y_true:
        return ConfusionCounts(tn=0, fp=0, fn=0, tp=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    return ConfusionCounts(tn=tn, fp=fp, fn=fn, tp=tp)


def _recall_at_precision(
    y_true: list[int],
    y_score: list[float],
    min_precision: float,
) -> float:
    if len(set(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # Exclude the final synthetic point (precision=1.0, recall=0.0) from non-trivial matches
    valid_mask = precision[:-1] >= min_precision
    if not np.any(valid_mask):
        return 0.0
    return float(np.max(recall[:-1][valid_mask]))


def _recall_at_fpr(
    y_true: list[int],
    y_score: list[float],
    max_fpr: float,
) -> float:
    if len(set(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid_mask = fpr <= max_fpr
    if not np.any(valid_mask):
        return 0.0
    return float(np.max(tpr[valid_mask]))


def _safe_binary_pr_auc(y_true: list[int], y_score: list[float]) -> float:
    if len(set(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _safe_binary_roc_auc(y_true: list[int], y_score: list[float]) -> float:
    if len(set(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _compute_multilabel_metrics(
    results: list[CascadeResult],
    y_score: list[float],
) -> tuple[float, float, dict[str, float]]:
    """Compute micro F1, macro F1, and per-category PR-AUC for strict predictions."""
    all_categories: set[str] = set()
    for res in results:
        true_cats = (
            res.true_categories
            if res.true_categories
            else (["SENSITIVE"] if res.ground_truth == GroundTruthState.SENSITIVE else [])
        )
        all_categories.update(true_cats)
        if res.verdict == VerdictStatus.SENSITIVE:
            pred_cats = res.predicted_categories if res.predicted_categories else ["SENSITIVE"]
            all_categories.update(pred_cats)

    ordered_cats = sorted(all_categories)
    if not ordered_cats or not results:
        return 0.0, 0.0, {}

    cat_to_idx = {cat: idx for idx, cat in enumerate(ordered_cats)}
    y_true_mat = np.zeros((len(results), len(ordered_cats)), dtype=np.int64)
    y_pred_mat = np.zeros((len(results), len(ordered_cats)), dtype=np.int64)

    for row_idx, res in enumerate(results):
        true_cats = (
            res.true_categories
            if res.true_categories
            else (["SENSITIVE"] if res.ground_truth == GroundTruthState.SENSITIVE else [])
        )
        for cat in true_cats:
            y_true_mat[row_idx, cat_to_idx[cat]] = 1

        if res.verdict == VerdictStatus.SENSITIVE:
            pred_cats = res.predicted_categories if res.predicted_categories else ["SENSITIVE"]
            for cat in pred_cats:
                y_pred_mat[row_idx, cat_to_idx[cat]] = 1

    micro_f1 = float(f1_score(y_true_mat, y_pred_mat, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true_mat, y_pred_mat, average="macro", zero_division=0))

    per_cat_pr_auc: dict[str, float] = {}
    for cat, col_idx in cat_to_idx.items():
        col_true = y_true_mat[:, col_idx].tolist()
        if len(set(col_true)) < 2:
            per_cat_pr_auc[cat] = 0.0
            continue
        col_scores = [
            y_score[i]
            if (
                results[i].verdict == VerdictStatus.SENSITIVE
                and cat
                in (
                    results[i].predicted_categories
                    if results[i].predicted_categories
                    else ["SENSITIVE"]
                )
            )
            else 0.0
            for i in range(len(results))
        ]
        # Use continuous risk_score directly if only one category exists; otherwise gated score
        if len(ordered_cats) == 1:
            col_scores = y_score
        per_cat_pr_auc[cat] = float(average_precision_score(col_true, col_scores))

    return micro_f1, macro_f1, per_cat_pr_auc


def compute_benchmark_metrics(results: list[CascadeResult]) -> BenchmarkMetrics:
    """Compute dual-view metrics, operating points, and hard-negative FPR."""
    total = len(results)
    y_true = [1 if r.ground_truth == GroundTruthState.SENSITIVE else 0 for r in results]
    y_score = [float(r.risk_score) for r in results]

    strict_pred = [1 if r.verdict == VerdictStatus.SENSITIVE else 0 for r in results]
    effective_pred = [1 if r.verdict in EFFECTIVE_POSITIVE_VERDICTS else 0 for r in results]

    strict_cm = _confusion_counts(y_true, strict_pred)
    strict_bin_f1 = float(f1_score(y_true, strict_pred, zero_division=0)) if total > 0 else 0.0
    micro_f1, macro_f1, per_cat_pr_auc = _compute_multilabel_metrics(results, y_score)

    blocked_count = sum(1 for r in results if r.verdict == VerdictStatus.PROVIDER_BLOCKED)
    refused_count = sum(1 for r in results if r.verdict == VerdictStatus.MODEL_REFUSED)
    error_count = sum(1 for r in results if r.verdict == VerdictStatus.ERROR)

    eff_cm = _confusion_counts(y_true, effective_pred)
    eff_tp = eff_cm.tp
    eff_fp = eff_cm.fp
    eff_fn = eff_cm.fn
    eff_precision = eff_tp / (eff_tp + eff_fp) if (eff_tp + eff_fp) > 0 else 0.0
    eff_recall = eff_tp / (eff_tp + eff_fn) if (eff_tp + eff_fn) > 0 else 0.0
    eff_f1 = float(f1_score(y_true, effective_pred, zero_division=0)) if total > 0 else 0.0

    tag_totals: dict[str, int] = {}
    tag_fps: dict[str, int] = {}
    for res in results:
        if res.ground_truth == GroundTruthState.SAFE and res.hard_negative_tag:
            tag = res.hard_negative_tag
            tag_totals[tag] = tag_totals.get(tag, 0) + 1
            if res.verdict in EFFECTIVE_POSITIVE_VERDICTS:
                tag_fps[tag] = tag_fps.get(tag, 0) + 1

    hard_negative_fpr = {
        tag: tag_fps.get(tag, 0) / count for tag, count in sorted(tag_totals.items())
    }

    stage_dist: dict[str, int] = {"HASH": 0, "CLASSIFIER": 0, "VLM": 0, "NONE": 0}
    for res in results:
        stage_dist[res.triggered_stage] = stage_dist.get(res.triggered_stage, 0) + 1

    latencies = [res.latency_ms for res in results]
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    sensitive_count = sum(y_true)
    safe_count = total - sensitive_count

    return BenchmarkMetrics(
        total_samples=total,
        sensitive_samples=sensitive_count,
        safe_samples=safe_count,
        strict_taxonomy=StrictTaxonomyMetrics(
            binary_f1=strict_bin_f1,
            confusion_matrix=strict_cm,
            micro_f1=micro_f1,
            macro_f1=macro_f1,
            per_category_pr_auc=per_cat_pr_auc,
            provider_block_rate=blocked_count / total if total > 0 else 0.0,
            model_refusal_rate=refused_count / total if total > 0 else 0.0,
            error_rate=error_count / total if total > 0 else 0.0,
        ),
        effective_detection=EffectiveDetectionMetrics(
            f1=eff_f1,
            precision=eff_precision,
            recall=eff_recall,
            confusion_matrix=eff_cm,
            pr_auc=_safe_binary_pr_auc(y_true, y_score),
            roc_auc=_safe_binary_roc_auc(y_true, y_score),
            recall_at_precision_999=_recall_at_precision(y_true, y_score, 0.999),
            recall_at_precision_990=_recall_at_precision(y_true, y_score, 0.990),
            recall_at_fpr_001=_recall_at_fpr(y_true, y_score, 0.001),
            recall_at_fpr_010=_recall_at_fpr(y_true, y_score, 0.010),
        ),
        hard_negative_fpr_by_tag=hard_negative_fpr,
        stage_exit_distribution=stage_dist,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
    )
