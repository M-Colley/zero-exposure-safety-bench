"""Stage 2 logprob calibration, block/refusal handling, and Stage 3 routing tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest
from tests.conftest import make_geometric_scene_rgb

from zesb.cascade import CascadeRunner
from zesb.schemas import (
    BenchmarkConfig,
    CascadeConfig,
    FrameCandidate,
    TaxonomyCategory,
    TaxonomyConfig,
    VerdictStatus,
)
from zesb.stages.classifier_stage import VLLMLogprobAdapter, validate_endpoint_url
from zesb.stages.hash_stage import HashIndex, HashStageAdapter
from zesb.stages.vlm_stage import VLMVerifierAdapter


def _make_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="test_cfg",
        version="1.0",
        taxonomy=TaxonomyConfig(
            mode="binary",
            categories=[
                TaxonomyCategory(
                    id="SENSITIVE",
                    label="Sensitive",
                    description="Harmful material",
                )
            ],
            benign_lookalikes=["Pediatric dermatology"],
        ),
        cascade=CascadeConfig(borderline_low=0.25, borderline_high=0.90),
    )


def test_public_url_rejected_when_allowlist_active() -> None:
    """Ensure non-private hostnames and public IPs raise ValueError by default."""
    with pytest.raises(ValueError, match="rejected"):
        validate_endpoint_url("https://api.openai.com/v1", allow_non_private=False)

    with pytest.raises(ValueError, match="rejected"):
        validate_endpoint_url("http://8.8.8.8:8000/v1", allow_non_private=False)

    assert validate_endpoint_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    assert validate_endpoint_url("http://10.0.0.5:8000/v1") == "http://10.0.0.5:8000/v1"
    assert (
        validate_endpoint_url("https://api.openai.com/v1", allow_non_private=True)
        == "https://api.openai.com/v1"
    )


def test_offline_env_overrides_allow_non_private(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ZESB_OFFLINE_ONLY=1 blocks external URLs even when allow_non_private=True."""
    monkeypatch.setenv("ZESB_OFFLINE_ONLY", "1")
    with pytest.raises(ValueError, match="rejected"):
        validate_endpoint_url("https://api.openai.com/v1", allow_non_private=True)
    assert validate_endpoint_url("http://127.0.0.1:8000/v1", allow_non_private=True) == (
        "http://127.0.0.1:8000/v1"
    )


async def test_vllm_logprob_softmax_calibration() -> None:
    """Verify exact two-class softmax over highest matching positive and negative logprobs."""
    l_pos = -0.2
    l_neg = -1.7
    expected = math.exp(l_pos) / (math.exp(l_pos) + math.exp(l_neg))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cmpl_1",
                "object": "chat.completion",
                "created": 1,
                "model": "shieldgemma-2-4b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Yes"},
                        "finish_reason": "length",
                        "logprobs": {
                            "content": [
                                {
                                    "token": "Yes",
                                    "logprob": l_pos,
                                    "top_logprobs": [
                                        {"token": " Yes", "logprob": l_pos},
                                        {"token": "no", "logprob": l_neg},
                                        {"token": "maybe", "logprob": -5.0},
                                    ],
                                }
                            ]
                        },
                    }
                ],
            },
        )

    adapter = VLLMLogprobAdapter(
        base_url="http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(handler),
    )
    frame = FrameCandidate(
        frame_index=0,
        timestamp_sec=0.0,
        rgb_array=make_geometric_scene_rgb(64, 64),
    )
    pred = await adapter.predict(frame)
    assert pred.verdict == VerdictStatus.SENSITIVE
    assert pred.risk_score == pytest.approx(expected, rel=1e-6)


async def test_vllm_http_400_safety_filter_returns_provider_blocked() -> None:
    """Capture HTTP 400 safety blocks as PROVIDER_BLOCKED with risk_score 1.0."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Blocked by content_filter",
                    "type": "invalid_request_error",
                    "code": "PROHIBITED_CONTENT",
                }
            },
        )

    adapter = VLLMLogprobAdapter(
        base_url="http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(handler),
    )
    frame = FrameCandidate(
        frame_index=0,
        timestamp_sec=0.0,
        rgb_array=make_geometric_scene_rgb(64, 64),
    )
    pred = await adapter.predict(frame)
    assert pred.verdict == VerdictStatus.PROVIDER_BLOCKED
    assert pred.risk_score == 1.0


async def test_vllm_refusal_prefix_returns_model_refused() -> None:
    """Detect refusal text when logprobs are unavailable."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cmpl_2",
                "object": "chat.completion",
                "created": 1,
                "model": "shieldgemma-2-4b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I cannot assist with analyzing this image.",
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
            },
        )

    adapter = VLLMLogprobAdapter(
        base_url="http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(handler),
    )
    frame = FrameCandidate(
        frame_index=0,
        timestamp_sec=0.0,
        rgb_array=make_geometric_scene_rgb(64, 64),
    )
    pred = await adapter.predict(frame)
    assert pred.verdict == VerdictStatus.MODEL_REFUSED
    assert pred.risk_score == 1.0


async def test_cascade_routes_borderline_scores_to_stage3_vlm(tmp_path: Path) -> None:
    """Verify Stage 3 VLM is invoked only when Stage 2 falls in [0.25, 0.90]."""
    cfg = _make_config()
    vlm_calls = 0

    def clf_handler(request: httpx.Request) -> httpx.Response:
        # Borderline score: l_pos == l_neg -> p = 0.50
        return httpx.Response(
            200,
            json={
                "id": "cmpl_clf",
                "object": "chat.completion",
                "created": 1,
                "model": "shieldgemma-2-4b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Yes"},
                        "finish_reason": "length",
                        "logprobs": {
                            "content": [
                                {
                                    "token": "Yes",
                                    "logprob": -0.693,
                                    "top_logprobs": [
                                        {"token": "yes", "logprob": -0.693},
                                        {"token": "no", "logprob": -0.693},
                                    ],
                                }
                            ]
                        },
                    }
                ],
            },
        )

    def vlm_handler(request: httpx.Request) -> httpx.Response:
        nonlocal vlm_calls
        vlm_calls += 1
        return httpx.Response(
            200,
            json={
                "id": "cmpl_vlm",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen2.5-vl-7b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "verdict": "SAFE",
                                    "categories": [],
                                    "confidence": 0.94,
                                }
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    runner = CascadeRunner(
        config=cfg,
        hash_stage=HashStageAdapter(HashIndex.empty()),
        classifier_stage=VLLMLogprobAdapter(transport=httpx.MockTransport(clf_handler)),
        vlm_stage=VLMVerifierAdapter(
            taxonomy=cfg.taxonomy,
            transport=httpx.MockTransport(vlm_handler),
        ),
    )

    frame = FrameCandidate(
        frame_index=0,
        timestamp_sec=0.0,
        rgb_array=make_geometric_scene_rgb(64, 64),
    )
    outcome = await runner.evaluate_frame(frame)
    assert vlm_calls == 1
    assert outcome.triggered_stage == "VLM"
    assert outcome.prediction.verdict == VerdictStatus.SAFE
    assert outcome.prediction.risk_score == pytest.approx(0.06, rel=1e-6)
