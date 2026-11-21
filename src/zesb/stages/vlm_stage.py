"""Stage 3 local VLM structured contextual verifier."""

from __future__ import annotations

import json
from typing import Literal

import httpx
import openai
from pydantic import BaseModel, ConfigDict, Field

from zesb.schemas import FrameCandidate, StagePrediction, TaxonomyConfig, VerdictStatus
from zesb.stages.classifier_stage import (
    build_openai_client,
    detect_provider_block,
    encode_frame_png_data_uri,
    is_refusal_text,
    validate_endpoint_url,
)


class VLMJsonOutput(BaseModel):
    """Enforced structured output returned by the Stage 3 VLM verifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Literal["SENSITIVE", "SAFE"]
    categories: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


def build_taxonomy_system_prompt(taxonomy: TaxonomyConfig) -> str:
    """Render the active YAML policy taxonomy and benign lookalikes into a system prompt."""
    category_lines = [f"- {cat.id} ({cat.label}): {cat.description}" for cat in taxonomy.categories]
    lookalike_lines = [f"- {item}" for item in taxonomy.benign_lookalikes]
    sections = [
        "You are an automated safety classifier evaluating a single media frame.",
        "Classify the frame strictly against this policy taxonomy:",
        *category_lines,
    ]
    if lookalike_lines:
        sections.extend(
            [
                "Distinguish genuine policy violations from these benign lookalikes:",
                *lookalike_lines,
            ]
        )
    sections.append(
        "Return JSON matching the schema with verdict ('SENSITIVE' or 'SAFE'), "
        "matching category IDs, and confidence in [0.0, 1.0]."
    )
    return "\n".join(sections)


class VLMVerifierAdapter:
    """Stage 3 adapter enforcing JSON schema taxonomy verification via local vLLM."""

    def __init__(
        self,
        taxonomy: TaxonomyConfig,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen2.5-vl-7b-instruct",
        api_key: str = "local-token",
        allow_non_private: bool = False,
        timeout_sec: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.system_prompt = build_taxonomy_system_prompt(taxonomy)
        self.base_url = validate_endpoint_url(base_url, allow_non_private=allow_non_private)
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self._transport = transport
        self._http_client = http_client

    async def _verify_frame(
        self,
        client: httpx.AsyncClient,
        data_uri: str,
    ) -> StagePrediction:
        sdk = build_openai_client(self.base_url, self.api_key, client)
        schema = VLMJsonOutput.model_json_schema()
        try:
            response = await sdk.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Evaluate this frame and output structured JSON.",
                            },
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    },
                ],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "safety_verdict",
                        "strict": True,
                        "schema": schema,
                    },
                },
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

        raw_content = response.choices[0].message.content
        if is_refusal_text(raw_content):
            return StagePrediction(
                verdict=VerdictStatus.MODEL_REFUSED,
                risk_score=1.0,
                predicted_categories=[],
            )
        if not raw_content:
            return StagePrediction(
                verdict=VerdictStatus.ERROR,
                risk_score=0.0,
                predicted_categories=[],
            )

        try:
            parsed = VLMJsonOutput.model_validate(json.loads(raw_content))
        except Exception:
            return StagePrediction(
                verdict=VerdictStatus.ERROR,
                risk_score=0.0,
                predicted_categories=[],
            )

        if parsed.verdict == "SENSITIVE":
            return StagePrediction(
                verdict=VerdictStatus.SENSITIVE,
                risk_score=parsed.confidence,
                predicted_categories=list(parsed.categories),
            )
        return StagePrediction(
            verdict=VerdictStatus.SAFE,
            risk_score=1.0 - parsed.confidence,
            predicted_categories=[],
        )

    async def predict(self, frame: FrameCandidate) -> StagePrediction:
        """Run structured VLM taxonomy verification on an in-memory frame."""
        data_uri = encode_frame_png_data_uri(frame.rgb_array)
        if self._http_client is not None:
            return await self._verify_frame(self._http_client, data_uri)
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self.timeout_sec,
        ) as client:
            return await self._verify_frame(client, data_uri)
