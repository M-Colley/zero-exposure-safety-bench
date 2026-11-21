"""Tests for the zero-exposure Inspect AI bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import make_concentric_rgb, make_geometric_scene_rgb, write_synthetic_png

from zesb.cascade import CascadeRunner
from zesb.inspect_bridge import (
    build_sanitized_inspect_records,
    create_zesb_inspect_task,
    make_cascade_evaluator,
)
from zesb.schemas import (
    BenchmarkConfig,
    TaxonomyCategory,
    TaxonomyConfig,
    VerdictStatus,
)
from zesb.stages.hash_stage import HashIndex, HashStageAdapter, compute_pdq, pdq_bits_to_hex


def _binary_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="inspect_test",
        version="0.1.0",
        taxonomy=TaxonomyConfig(
            mode="binary",
            categories=[
                TaxonomyCategory(
                    id="SENSITIVE",
                    label="Sensitive",
                    description="Harmful content",
                )
            ],
        ),
    )


async def test_sanitized_inspect_records_omit_paths_and_evaluate_via_closure(
    tmp_path: Path,
) -> None:
    """Ensure Inspect sample payloads exclude relative_path and evaluate by sample_id."""
    media_root = tmp_path / "media"
    sens_rgb = make_geometric_scene_rgb(128, 128)
    safe_rgb = make_concentric_rgb(128, 128, ring=12)
    write_synthetic_png(media_root / "restricted_dir" / "secret_01.png", sens_rgb)
    write_synthetic_png(media_root / "benign_dir" / "family_01.png", safe_rgb)

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "item_sens",
                        "relative_path": "restricted_dir/secret_01.png",
                        "media_type": "IMAGE",
                        "ground_truth": "SENSITIVE",
                        "categories": ["SENSITIVE"],
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "item_safe",
                        "relative_path": "benign_dir/family_01.png",
                        "media_type": "IMAGE",
                        "ground_truth": "SAFE",
                        "hard_negative_tag": "family_vacation",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records, lookup = build_sanitized_inspect_records(manifest_path, media_root)
    assert len(records) == 2
    serialized = json.dumps([r.model_dump() for r in records])
    assert "restricted_dir" not in serialized
    assert "secret_01.png" not in serialized
    assert "benign_dir" not in serialized

    bits_sens, _ = compute_pdq(sens_rgb)
    index = HashIndex.from_hex_entries([(pdq_bits_to_hex(bits_sens), ["SENSITIVE"])])
    runner = CascadeRunner(
        config=_binary_config(),
        hash_stage=HashStageAdapter(index=index),
    )
    eval_fn = make_cascade_evaluator(runner=runner, samples_by_id=lookup, media_root=media_root)

    res_sens = await eval_fn("item_sens")
    res_safe = await eval_fn("item_safe")
    assert res_sens.verdict == VerdictStatus.SENSITIVE
    assert res_safe.verdict == VerdictStatus.SAFE


def test_create_zesb_inspect_task_raises_clean_import_error_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify clear ImportError guidance when optional inspect-ai extra is absent."""
    import importlib

    real_import = importlib.import_module

    def _block_inspect(name: str, package: str | None = None) -> object:
        if name.startswith("inspect_ai"):
            raise ModuleNotFoundError("No module named 'inspect_ai'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _block_inspect)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")

    with pytest.raises(ImportError, match="zero-exposure-safety-bench\\[inspect\\]"):
        create_zesb_inspect_task(
            manifest_path=manifest,
            media_root=tmp_path,
            config=_binary_config(),
        )
