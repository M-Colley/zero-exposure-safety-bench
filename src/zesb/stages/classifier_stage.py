"""Stage 2 calibrated classifier adapters (vLLM token logprobs and local REST)."""

from __future__ import annotations

import base64
import io
import ipaddress
import math
import os
from collections.abc import Iterable
from typing import Any, cast
from urllib.parse import urlparse

import httpx
import numpy as np
import openai
from numpy.typing import NDArray
from PIL import Image

from zesb.schemas import FrameCandidate, StagePrediction, VerdictStatus

POSITIVE_TOKENS: frozenset[str] = frozenset({"yes", "unsafe", "sensitive", "1", "true"})
NEGATIVE_TOKENS: frozenset[str] = frozenset({"no", "safe", "benign", "0", "false"})
REFUSAL_PREFIXES: tuple[str, ...] = ("i cannot", "i'm sorry", "as an ai")
SAFETY_BLOCK_CODES: tuple[str, ...] = ("content_filter", "prohibited_content", "safety")
DEFAULT_MISSING_LOGPROB = -20.0


def validate_endpoint_url(base_url: str, allow_non_private: bool = False) -> str:
    """Validate endpoint URL scheme and enforce private/loopback host allowlist."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid endpoint URL: {base_url}")

    offline_enforced = os.environ.get("ZESB_OFFLINE_ONLY", "").strip() == "1"
    if offline_enforced or not allow_non_private:
        host = parsed.hostname.lower()
        if host != "localhost":
            try:
                addr = ipaddress.ip_address(host)
            except ValueError as exc:
                raise ValueError(
                    f"Non-private hostname '{host}' rejected by offline/private host policy"
                ) from exc
            if not (addr.is_loopback or addr.is_private):
                raise ValueError(f"Public IP '{host}' rejected by offline/private host policy")
    return base_url.rstrip("/")


def encode_frame_png_bytes(rgb_array: NDArray[np.uint8]) -> bytes:
    """Encode an RGB uint8 array into ephemeral in-memory PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(rgb_array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def encode_frame_png_data_uri(rgb_array: NDArray[np.uint8]) -> str:
    """Encode an RGB uint8 array as an ephemeral base64 PNG data URI."""
    encoded = base64.b64encode(encode_frame_png_bytes(rgb_array)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def detect_provider_block(status_code: int, body_text: str) -> str | None:
    """Return a normalized block code if HTTP 400 indicates a provider safety filter."""
    if status_code != 400:
        return None
    lowered = body_text.lower()
    for code in SAFETY_BLOCK_CODES:
        if code in lowered:
            return code
    return None


def is_refusal_text(text: str | None) -> bool:
    """Return True when model text begins with a known safety refusal prefix."""
    if not text:
        return False
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in REFUSAL_PREFIXES)


def compute_binary_logprob_probability(l_pos: float, l_neg: float) -> float:
    """Compute numerically stable two-class softmax probability P(sensitive)."""
    max_lp = max(l_pos, l_neg)
    exp_pos = math.exp(l_pos - max_lp)
    exp_neg = math.exp(l_neg - max_lp)
    return exp_pos / (exp_pos + exp_neg)


def _extract_candidate_logprobs(
    token_entries: Iterable[tuple[str, float]],
) -> tuple[float | None, float | None]:
    """Select maximum positive and negative token logprobs from candidate tokens."""
    best_pos: float | None = None
    best_neg: float | None = None
    for raw_token, logprob in token_entries:
        token = raw_token.strip().lower()
        if token in POSITIVE_TOKENS:
            best_pos = logprob if best_pos is None else max(best_pos, logprob)
        elif token in NEGATIVE_TOKENS:
            best_neg = logprob if best_neg is None else max(best_neg, logprob)
    return best_pos, best_neg


def build_openai_client(
    base_url: str,
    api_key: str,
    http_client: httpx.AsyncClient,
) -> openai.AsyncOpenAI:
    """Construct AsyncOpenAI reusing an httpx AsyncClient."""
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=cast(Any, http_client),
        max_retries=0,
    )


class VLLMLogprobAdapter:
    """Stage 2 adapter querying an OpenAI-compatible vLLM endpoint with logprobs."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "shieldgemma-2-4b",
        api_key: str = "local-token",
        positive_threshold: float = 0.50,
        allow_non_private: bool = False,
        timeout_sec: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = validate_endpoint_url(base_url, allow_non_private=allow_non_private)
        self.model = model
        self.api_key = api_key
        self.positive_threshold = positive_threshold
        self.timeout_sec = timeout_sec
        self._transport = transport
        self._http_client = http_client

    async def _call_chat_completion(
        self,
        client: httpx.AsyncClient,
        data_uri: str,
    ) -> StagePrediction:
        sdk = build_openai_client(self.base_url, self.api_key, client)
        try:
            response = await sdk.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Does this image contain graphic violence or child abuse "
                                    "material? Answer strictly with Yes or No."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=10,
            )
        except openai.APIStatusError as exc:
            body_text = exc.response.text if exc.response is not None else str(exc.body or "")
            block_code = detect_provider_block(exc.status_code, body_text)
            if block_code is not None:
                return StagePrediction(
                    verdict=VerdictStatus.PROVIDER_BLOCKED,
                    risk_score=1.0,
                    predicted_categories=[],
                    provider_code=block_code,
                )
            return StagePrediction(
                verdict=VerdictStatus.ERROR,
                risk_score=0.0,
                predicted_categories=[],
                provider_code=f"http_{exc.status_code}",
            )

        if not response.choices:
            return StagePrediction(
                verdict=VerdictStatus.ERROR,
                risk_score=0.0,
                predicted_categories=[],
            )

        choice = response.choices[0]
        content_text = choice.message.content if choice.message else None
        token_pairs: list[tuple[str, float]] = []

        if choice.logprobs and choice.logprobs.content:
            first_tok = choice.logprobs.content[0]
            if first_tok.top_logprobs:
                token_pairs.extend(
                    (entry.token, float(entry.logprob)) for entry in first_tok.top_logprobs
                )
            else:
                token_pairs.append((first_tok.token, float(first_tok.logprob)))

        best_pos, best_neg = _extract_candidate_logprobs(token_pairs)
        if best_pos is None and best_neg is None:
            if is_refusal_text(content_text):
                return StagePrediction(
                    verdict=VerdictStatus.MODEL_REFUSED,
                    risk_score=1.0,
                    predicted_categories=[],
                )
            if content_text is not None:
                cleaned = content_text.strip().lower()
                if cleaned in POSITIVE_TOKENS:
                    return StagePrediction(
                        verdict=VerdictStatus.SENSITIVE,
                        risk_score=1.0,
                        predicted_categories=["SENSITIVE"],
                    )
                if cleaned in NEGATIVE_TOKENS:
                    return StagePrediction(
                        verdict=VerdictStatus.SAFE,
                        risk_score=0.0,
                        predicted_categories=[],
                    )
            return StagePrediction(
                verdict=VerdictStatus.ERROR,
                risk_score=0.0,
                predicted_categories=[],
            )

        l_pos = best_pos if best_pos is not None else DEFAULT_MISSING_LOGPROB
        l_neg = best_neg if best_neg is not None else DEFAULT_MISSING_LOGPROB
        p_sensitive = compute_binary_logprob_probability(l_pos, l_neg)
        is_sensitive = p_sensitive >= self.positive_threshold
        return StagePrediction(
            verdict=VerdictStatus.SENSITIVE if is_sensitive else VerdictStatus.SAFE,
            risk_score=p_sensitive,
            predicted_categories=["SENSITIVE"] if is_sensitive else [],
        )

    async def predict(self, frame: FrameCandidate) -> StagePrediction:
        """Score a single frame candidate via calibrated logprob softmax."""
        data_uri = encode_frame_png_data_uri(frame.rgb_array)
        if self._http_client is not None:
            return await self._call_chat_completion(self._http_client, data_uri)
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self.timeout_sec,
        ) as client:
            return await self._call_chat_completion(client, data_uri)


class RestClassifierAdapter:
    """Stage 2 adapter posting raw PNG bytes to a local classifier microservice."""

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:8001/classify",
        positive_threshold: float = 0.50,
        allow_non_private: bool = False,
        timeout_sec: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.endpoint_url = validate_endpoint_url(
            endpoint_url,
            allow_non_private=allow_non_private,
        )
        self.positive_threshold = positive_threshold
        self.timeout_sec = timeout_sec
        self._transport = transport
        self._http_client = http_client

    async def _post_image(self, client: httpx.AsyncClient, png_bytes: bytes) -> StagePrediction:
        response = await client.post(
            self.endpoint_url,
            content=png_bytes,
            headers={"Content-Type": "image/png"},
        )
        block_code = detect_provider_block(response.status_code, response.text)
        if block_code is not None:
            return StagePrediction(
                verdict=VerdictStatus.PROVIDER_BLOCKED,
                risk_score=1.0,
                predicted_categories=[],
                provider_code=block_code,
            )
        if response.status_code >= 400:
            return StagePrediction(
                verdict=VerdictStatus.ERROR,
                risk_score=0.0,
                predicted_categories=[],
                provider_code=f"http_{response.status_code}",
            )

        payload = response.json()
        raw_scores = payload.get("scores") or {}
        scores = {str(k): float(v) for k, v in raw_scores.items()}
        risk_score = float(payload.get("risk_score", max(scores.values(), default=0.0)))
        predicted_cats = [cat for cat, score in scores.items() if score >= self.positive_threshold]
        is_sensitive = risk_score >= self.positive_threshold
        return StagePrediction(
            verdict=VerdictStatus.SENSITIVE if is_sensitive else VerdictStatus.SAFE,
            risk_score=risk_score,
            predicted_categories=predicted_cats,
        )

    async def predict(self, frame: FrameCandidate) -> StagePrediction:
        """POST raw PNG bytes of a frame to the configured local REST classifier."""
        png_bytes = encode_frame_png_bytes(frame.rgb_array)
        if self._http_client is not None:
            return await self._post_image(self._http_client, png_bytes)
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self.timeout_sec,
        ) as client:
            return await self._post_image(client, png_bytes)
