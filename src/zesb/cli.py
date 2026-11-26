"""Typer CLI entrypoint for Zero-Exposure Safety Benchmark (`zesb`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from zesb.cascade import CascadeRunner
from zesb.ingest import ingest_sample, load_manifest
from zesb.metrics import compute_benchmark_metrics
from zesb.perturbations import DEFAULT_PERTURBATIONS
from zesb.report import build_sanitized_error_records, write_reports
from zesb.schemas import (
    BenchmarkConfig,
    CascadeResult,
    FrameCandidate,
    GroundTruthState,
    MediaType,
    VerdictStatus,
)
from zesb.stages.base import StageAdapter
from zesb.stages.classifier_stage import RestClassifierAdapter, VLLMLogprobAdapter
from zesb.stages.hash_stage import HashIndex, HashStageAdapter
from zesb.stages.vlm_stage import VLMVerifierAdapter

app = typer.Typer(
    name="zesb",
    help="Zero-Exposure Multimodal Safety Benchmark CLI.",
    no_args_is_help=True,
)


def load_benchmark_config(config_path: Path) -> BenchmarkConfig:
    """Load and validate a YAML benchmark configuration file."""
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return BenchmarkConfig.model_validate(raw)


async def _run_evaluation(
    manifest_path: Path,
    media_root: Path,
    config: BenchmarkConfig,
    output_dir: Path,
    hash_index_path: Path | None,
    classifier_url: str | None,
    classifier_type: str,
    classifier_model: str,
    vlm_url: str | None,
    vlm_model: str,
    safe_previews: bool,
    allow_non_private_hosts: bool,
) -> tuple[Path, Path, int]:
    samples = load_manifest(manifest_path)
    hash_index = HashIndex.from_file(hash_index_path) if hash_index_path else HashIndex.empty()
    hash_stage = HashStageAdapter(
        index=hash_index,
        distance_threshold=config.cascade.pdq_distance_threshold,
        min_quality_score=config.cascade.pdq_min_quality,
    )

    allow_external = allow_non_private_hosts or config.cascade.allow_non_private_hosts

    classifier_stage: StageAdapter | None = None
    if classifier_url:
        if classifier_type == "rest":
            classifier_stage = RestClassifierAdapter(
                endpoint_url=classifier_url,
                positive_threshold=config.cascade.positive_threshold,
                allow_non_private=allow_external,
            )
        elif classifier_type == "vllm":
            classifier_stage = VLLMLogprobAdapter(
                base_url=classifier_url,
                model=classifier_model,
                positive_threshold=config.cascade.positive_threshold,
                allow_non_private=allow_external,
            )
        else:
            raise ValueError(f"Unsupported classifier_type: {classifier_type}")

    vlm_stage: StageAdapter | None = None
    if vlm_url:
        vlm_stage = VLMVerifierAdapter(
            taxonomy=config.taxonomy,
            base_url=vlm_url,
            model=vlm_model,
            allow_non_private=allow_external,
        )

    runner = CascadeRunner(
        config=config,
        hash_stage=hash_stage,
        classifier_stage=classifier_stage,
        vlm_stage=vlm_stage,
    )

    results: list[CascadeResult] = []
    for sample in samples:
        results.append(await runner.evaluate_sample(sample=sample, media_root=media_root))

    metrics = compute_benchmark_metrics(results, config.exposure)
    samples_by_id = {sample.sample_id: sample for sample in samples}
    error_records = build_sanitized_error_records(
        results=results,
        samples_by_id=samples_by_id,
        media_root=media_root,
        safe_previews=safe_previews,
    )
    metrics_path, errors_path = write_reports(
        output_dir=output_dir,
        metrics=metrics,
        error_records=error_records,
        safe_previews=safe_previews,
    )
    return metrics_path, errors_path, len(results)


@app.command("evaluate")
def evaluate_command(
    manifest: Annotated[Path, typer.Option("--manifest", help="Path to manifest.jsonl.")],
    media_root: Annotated[Path, typer.Option("--media-root", help="Root directory for media.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Output directory.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="YAML benchmark configuration path."),
    ] = Path("configs/binary_default.yaml"),
    hash_index: Annotated[
        Path | None,
        typer.Option("--hash-index", help="Optional Stage 1 PDQ hash index (JSONL or CSV)."),
    ] = None,
    classifier_url: Annotated[
        str | None,
        typer.Option("--classifier-url", help="Optional Stage 2 classifier base URL."),
    ] = None,
    classifier_type: Annotated[
        str,
        typer.Option("--classifier-type", help="Stage 2 adapter type: vllm or rest."),
    ] = "vllm",
    classifier_model: Annotated[
        str,
        typer.Option("--classifier-model", help="Stage 2 model identifier."),
    ] = "shieldgemma-2-4b",
    vlm_url: Annotated[
        str | None,
        typer.Option("--vlm-url", help="Optional Stage 3 VLM base URL."),
    ] = None,
    vlm_model: Annotated[
        str,
        typer.Option("--vlm-model", help="Stage 3 VLM model identifier."),
    ] = "qwen2.5-vl-7b-instruct",
    safe_previews: Annotated[
        bool,
        typer.Option(
            "--safe-previews/--no-safe-previews",
            help="Emit blurred thumbnails exclusively for SAFE false positives.",
        ),
    ] = False,
    allow_non_private_hosts: Annotated[
        bool,
        typer.Option(
            "--allow-non-private-hosts/--no-allow-non-private-hosts",
            help="Permit non-private HTTP endpoints.",
        ),
    ] = False,
) -> None:
    """Evaluate a multimodal dataset manifest through the three-stage cascade."""
    cfg = load_benchmark_config(config)
    metrics_path, errors_path, count = asyncio.run(
        _run_evaluation(
            manifest_path=manifest,
            media_root=media_root,
            config=cfg,
            output_dir=output_dir,
            hash_index_path=hash_index,
            classifier_url=classifier_url,
            classifier_type=classifier_type,
            classifier_model=classifier_model,
            vlm_url=vlm_url,
            vlm_model=vlm_model,
            safe_previews=safe_previews,
            allow_non_private_hosts=allow_non_private_hosts,
        )
    )
    typer.echo(f"Evaluated {count} samples. Metrics: {metrics_path} Errors: {errors_path}")


async def _run_perturb_test(
    manifest_path: Path,
    media_root: Path,
    hash_index_path: Path,
    config: BenchmarkConfig,
    output_dir: Path,
) -> Path:
    samples = [s for s in load_manifest(manifest_path) if s.media_type == MediaType.IMAGE]
    index = HashIndex.from_file(hash_index_path)
    adapter = HashStageAdapter(
        index=index,
        distance_threshold=config.cascade.pdq_distance_threshold,
        min_quality_score=config.cascade.pdq_min_quality,
    )

    decoded: list[tuple[GroundTruthState, FrameCandidate]] = []
    for sample in samples:
        _, frames = ingest_sample(sample, media_root, config.ingest)
        if frames:
            decoded.append((sample.ground_truth, frames[0]))

    transforms: dict[str, object] = {"original": None, **DEFAULT_PERTURBATIONS}
    report: dict[str, dict[str, float | int]] = {}
    for name in transforms:
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        fn_transform = DEFAULT_PERTURBATIONS.get(name)
        for gt, frame in decoded:
            rgb = fn_transform(frame.rgb_array) if fn_transform is not None else frame.rgb_array
            candidate = FrameCandidate(
                frame_index=0,
                timestamp_sec=0.0,
                rgb_array=rgb,
            )
            pred = await adapter.predict(candidate)
            matched = pred.verdict == VerdictStatus.SENSITIVE
            if gt == GroundTruthState.SENSITIVE:
                if matched:
                    tp += 1
                else:
                    fn += 1
            else:
                if matched:
                    fp += 1
                else:
                    tn += 1

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        report[name] = {
            "recall": recall,
            "fpr": fpr,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "perturbation_metrics.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out_path


@app.command("perturb-test")
def perturb_test_command(
    manifest: Annotated[Path, typer.Option("--manifest", help="Path to manifest.jsonl.")],
    media_root: Annotated[Path, typer.Option("--media-root", help="Root directory for media.")],
    hash_index: Annotated[Path, typer.Option("--hash-index", help="PDQ hash index path.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Output directory.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="YAML benchmark configuration path."),
    ] = Path("configs/binary_default.yaml"),
) -> None:
    """Measure Stage 1 PDQ recall and false-positive rate across perturbations."""
    cfg = load_benchmark_config(config)
    out_path = asyncio.run(
        _run_perturb_test(
            manifest_path=manifest,
            media_root=media_root,
            hash_index_path=hash_index,
            config=cfg,
            output_dir=output_dir,
        )
    )
    typer.echo(f"Perturbation metrics written to {out_path}")
