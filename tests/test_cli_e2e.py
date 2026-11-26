"""End-to-end CLI benchmark execution and raw-byte zero-exposure leak audit."""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import (
    make_checkerboard_rgb,
    make_concentric_rgb,
    make_geometric_scene_rgb,
    write_multi_scene_mp4,
    write_synthetic_png,
)
from typer.testing import CliRunner

from zesb.cli import app
from zesb.stages.hash_stage import compute_pdq, pdq_bits_to_hex


def test_cli_evaluate_zero_exposure_leak_audit(tmp_path: Path) -> None:
    """Run zesb evaluate end-to-end and audit raw output bytes for zero leaks."""
    media_root = tmp_path / "media"
    output_dir = tmp_path / "out"

    # Sensitive image 1 (matched by PDQ index -> true positive)
    sens_img_1 = make_geometric_scene_rgb(256, 256)
    write_synthetic_png(
        media_root / "classified_vault" / "sensitive_payload_001.png",
        sens_img_1,
    )
    bits_sens_1, _ = compute_pdq(sens_img_1)

    # Sensitive image 2 (unindexed -> false negative in sanitized_errors.jsonl)
    sens_img_2 = make_checkerboard_rgb(256, 256, tile=24)
    write_synthetic_png(
        media_root / "classified_vault" / "sensitive_unindexed_002.png",
        sens_img_2,
    )

    # Sensitive multi-scene video (unindexed -> false negative in sanitized_errors.jsonl)
    write_multi_scene_mp4(
        media_root / "classified_vault" / "sensitive_clip_003.mp4",
        fps=30.0,
    )

    # Safe image 1 (unindexed -> true negative)
    safe_img_1 = make_concentric_rgb(256, 256, ring=16)
    write_synthetic_png(
        media_root / "benign_pool" / "safe_scene_001.png",
        safe_img_1,
    )

    # Safe image 2 (intentionally placed in PDQ index -> SAFE FALSE_POSITIVE for preview test)
    safe_fp_img = make_geometric_scene_rgb(256, 256).copy()
    safe_fp_img[0:20, 0:20] = (10, 200, 200)
    write_synthetic_png(
        media_root / "benign_pool" / "safe_hard_neg_002.png",
        safe_fp_img,
    )
    bits_safe_fp, _ = compute_pdq(safe_fp_img)

    hash_index_path = tmp_path / "pdq_index.jsonl"
    hash_index_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "pdq_hex": pdq_bits_to_hex(bits_sens_1),
                        "categories": ["SENSITIVE"],
                    }
                ),
                json.dumps(
                    {
                        "pdq_hex": pdq_bits_to_hex(bits_safe_fp),
                        "categories": ["SENSITIVE"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows = [
        {
            "sample_id": "s_img_tp",
            "relative_path": "classified_vault/sensitive_payload_001.png",
            "media_type": "IMAGE",
            "ground_truth": "SENSITIVE",
            "categories": ["SENSITIVE"],
        },
        {
            "sample_id": "s_img_fn",
            "relative_path": "classified_vault/sensitive_unindexed_002.png",
            "media_type": "IMAGE",
            "ground_truth": "SENSITIVE",
            "categories": ["SENSITIVE"],
        },
        {
            "sample_id": "s_vid_fn",
            "relative_path": "classified_vault/sensitive_clip_003.mp4",
            "media_type": "VIDEO",
            "ground_truth": "SENSITIVE",
            "categories": ["SENSITIVE"],
        },
        {
            "sample_id": "safe_tn",
            "relative_path": "benign_pool/safe_scene_001.png",
            "media_type": "IMAGE",
            "ground_truth": "SAFE",
            "hard_negative_tag": "family_vacation",
        },
        {
            "sample_id": "safe_fp",
            "relative_path": "benign_pool/safe_hard_neg_002.png",
            "media_type": "IMAGE",
            "ground_truth": "SAFE",
            "hard_negative_tag": "pediatric_dermatology",
        },
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(r) for r in manifest_rows) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--manifest",
            str(manifest_path),
            "--media-root",
            str(media_root),
            "--config",
            "configs/binary_default.yaml",
            "--hash-index",
            str(hash_index_path),
            "--output-dir",
            str(output_dir),
            "--safe-previews",
        ],
    )
    assert result.exit_code == 0, result.output

    metrics_file = output_dir / "metrics.json"
    errors_file = output_dir / "sanitized_errors.jsonl"
    preview_html = output_dir / "safe_false_positives.html"
    assert metrics_file.is_file()
    assert errors_file.is_file()
    assert preview_html.is_file()

    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics_data["total_samples"] == 5
    assert metrics_data["sensitive_samples"] == 3
    assert metrics_data["safe_samples"] == 2
    assert metrics_data["human_exposure"]["manual_exposure_items"] == 3
    assert metrics_data["human_exposure"]["manual_exposure_video_sec"] > 0.0
    assert metrics_data["hard_negative_fpr_by_tag"]["pediatric_dermatology"] == 1.0
    assert metrics_data["hard_negative_fpr_by_tag"]["family_vacation"] == 0.0

    # Parse sanitized_errors.jsonl and verify preview rules
    error_lines = [
        json.loads(line)
        for line in errors_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(error_lines) == 3  # s_img_fn, s_vid_fn, safe_fp
    for err in error_lines:
        if err["ground_truth"] == "SENSITIVE":
            assert err["safe_blurred_jpeg_b64"] is None
        else:
            assert err["error_type"] == "FALSE_POSITIVE"
            assert err["safe_blurred_jpeg_b64"] is not None

    # Raw-byte leak audit across all output artifacts and stdout/stderr
    forbidden_tokens = [
        b"classified_vault",
        b"sensitive_payload_001.png",
        b"sensitive_unindexed_002.png",
        b"sensitive_clip_003.mp4",
        b"benign_pool",
        b"safe_scene_001.png",
        b"safe_hard_neg_002.png",
        b"\xff\xd8\xff",
        b"\x89PNG",
    ]
    raw_streams = [
        metrics_file.read_bytes(),
        errors_file.read_bytes(),
        preview_html.read_bytes(),
        result.output.encode("utf-8"),
    ]
    for stream in raw_streams:
        for token in forbidden_tokens:
            assert token not in stream, f"Leaked forbidden token {token!r} in CLI output"

    html_text = preview_html.read_text(encoding="utf-8")
    assert "safe_fp" in html_text
    assert "s_img_fn" not in html_text
    assert "s_vid_fn" not in html_text


def test_cli_perturb_test_e2e(tmp_path: Path) -> None:
    """Run zesb perturb-test and verify perturbation_metrics.json structure."""
    media_root = tmp_path / "media"
    output_dir = tmp_path / "out_perturb"

    sens_img = make_geometric_scene_rgb(256, 256)
    write_synthetic_png(media_root / "sens.png", sens_img)
    bits_sens, _ = compute_pdq(sens_img)

    safe_img = make_concentric_rgb(256, 256, ring=16)
    write_synthetic_png(media_root / "safe.png", safe_img)

    hash_index_path = tmp_path / "hashes.jsonl"
    hash_index_path.write_text(
        json.dumps({"pdq_hex": pdq_bits_to_hex(bits_sens), "categories": ["SENSITIVE"]}) + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "s1",
                        "relative_path": "sens.png",
                        "media_type": "IMAGE",
                        "ground_truth": "SENSITIVE",
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "n1",
                        "relative_path": "safe.png",
                        "media_type": "IMAGE",
                        "ground_truth": "SAFE",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "perturb-test",
            "--manifest",
            str(manifest_path),
            "--media-root",
            str(media_root),
            "--hash-index",
            str(hash_index_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    out_file = output_dir / "perturbation_metrics.json"
    assert out_file.is_file()
    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["original"]["recall"] == 1.0
    assert report["original"]["fpr"] == 0.0
    assert report["jpeg_q30"]["recall"] == 1.0
    assert "center_crop_90" in report
    assert "horizontal_flip" in report
