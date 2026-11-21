"""Zero-exposure JSON/JSONL artifact writer and SAFE-only blurred preview generator."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import cv2
from PIL import Image, ImageFilter

from zesb.metrics import BenchmarkMetrics
from zesb.schemas import (
    CascadeResult,
    DatasetSample,
    GroundTruthState,
    MediaType,
    SanitizedErrorRecord,
    VerdictStatus,
)


def build_safe_blurred_preview_b64(sample: DatasetSample, media_root: Path) -> str | None:
    """Create a blurred 128x128 JPEG thumbnail strictly for SAFE samples."""
    if sample.ground_truth != GroundTruthState.SAFE:
        return None

    media_path = sample.resolve_media_path(media_root)
    if not media_path.is_file():
        return None

    if sample.media_type == MediaType.IMAGE:
        with Image.open(media_path) as img:
            rgb_img = img.convert("RGB")
    else:
        cap = cv2.VideoCapture(str(media_path))
        try:
            ok, bgr_frame = cap.read()
        finally:
            cap.release()
        if not ok or bgr_frame is None:
            return None
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_img = Image.fromarray(rgb_frame, mode="RGB")

    blurred = rgb_img.filter(ImageFilter.GaussianBlur(radius=20)).resize(
        (128, 128),
        resample=Image.Resampling.BILINEAR,
    )
    buffer = io.BytesIO()
    blurred.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_sanitized_error_records(
    results: list[CascadeResult],
    samples_by_id: dict[str, DatasetSample],
    media_root: Path,
    safe_previews: bool = False,
) -> list[SanitizedErrorRecord]:
    """Convert CascadeResults into zero-exposure error records."""
    records: list[SanitizedErrorRecord] = []
    for res in results:
        error_type: str | None = None
        if res.verdict == VerdictStatus.ERROR:
            error_type = "RUNTIME_ERROR"
        elif res.verdict in {VerdictStatus.PROVIDER_BLOCKED, VerdictStatus.MODEL_REFUSED}:
            error_type = "BLOCKED_OR_REFUSED"
        elif res.ground_truth == GroundTruthState.SAFE and res.verdict == VerdictStatus.SENSITIVE:
            error_type = "FALSE_POSITIVE"
        elif res.ground_truth == GroundTruthState.SENSITIVE and res.verdict == VerdictStatus.SAFE:
            error_type = "FALSE_NEGATIVE"

        if error_type is None:
            continue

        preview_b64: str | None = None
        if (
            safe_previews
            and error_type == "FALSE_POSITIVE"
            and res.ground_truth == GroundTruthState.SAFE
        ):
            sample = samples_by_id.get(res.sample_id)
            if sample is not None and sample.ground_truth == GroundTruthState.SAFE:
                preview_b64 = build_safe_blurred_preview_b64(sample, media_root)

        if error_type == "FALSE_POSITIVE":
            typed_err = "FALSE_POSITIVE"
        elif error_type == "FALSE_NEGATIVE":
            typed_err = "FALSE_NEGATIVE"
        elif error_type == "BLOCKED_OR_REFUSED":
            typed_err = "BLOCKED_OR_REFUSED"
        else:
            typed_err = "RUNTIME_ERROR"

        records.append(
            SanitizedErrorRecord(
                sample_id=res.sample_id,
                media_sha256=res.media_sha256,
                error_type=typed_err,
                ground_truth=res.ground_truth,
                risk_score=res.risk_score,
                predicted_categories=list(res.predicted_categories),
                true_categories=list(res.true_categories),
                hard_negative_tag=res.hard_negative_tag,
                safe_blurred_jpeg_b64=preview_b64,
            )
        )
    return records


def write_safe_preview_html(
    output_path: Path,
    error_records: list[SanitizedErrorRecord],
) -> Path:
    """Write an HTML review sheet containing only SAFE false-positive previews."""
    cards: list[str] = []
    for rec in error_records:
        if rec.ground_truth != GroundTruthState.SAFE or rec.safe_blurred_jpeg_b64 is None:
            continue
        sid = html.escape(rec.sample_id)
        tag = html.escape(rec.hard_negative_tag or "none")
        cards.append(
            "<div class='card'>"
            f"<img src='data:image/jpeg;base64,{rec.safe_blurred_jpeg_b64}' alt='SAFE FP'/>"
            f"<p><strong>{sid}</strong> | score={rec.risk_score:.3f} | tag={tag}</p>"
            "</div>"
        )
    body = "\n".join(cards) if cards else "<p>No SAFE false-positive previews.</p>"
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        "<title>ZESB Safe False-Positive Review</title></head>"
        f"<body><h1>SAFE False-Positive Blurred Previews</h1>{body}</body></html>"
    )
    output_path.write_text(doc, encoding="utf-8")
    return output_path


def write_reports(
    output_dir: Path,
    metrics: BenchmarkMetrics,
    error_records: list[SanitizedErrorRecord],
    safe_previews: bool = False,
) -> tuple[Path, Path]:
    """Write metrics.json and sanitized_errors.jsonl inside output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    errors_path = output_dir / "sanitized_errors.jsonl"

    metrics_path.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with errors_path.open("w", encoding="utf-8") as handle:
        for rec in error_records:
            handle.write(rec.model_dump_json() + "\n")

    if safe_previews:
        write_safe_preview_html(output_dir / "safe_false_positives.html", error_records)

    return metrics_path, errors_path
