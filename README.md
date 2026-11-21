# Zero-Exposure Safety Benchmark (`zesb`)

[![CI](https://github.com/M-Colley/zero-exposure-safety-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/M-Colley/zero-exposure-safety-bench/actions/workflows/ci.yml)

`zesb` benchmarks automated detection pipelines for graphic violence and child abuse material across images and videos. The package executes inside network-isolated environments without leaking file paths, raw pixel buffers, or image previews for sensitive samples.

## Architecture

The benchmark evaluates a three-stage safety cascade. Stage 1 matches 256-bit `PDQ` perceptual hashes against a local target index. Stage 2 scores unmatched frames with a calibrated classifier using local `vLLM` token logprobs or a local HTTP endpoint. Stage 3 routes borderline Stage 2 scores to a local vision-language model for structured taxonomy verification against benign lookalikes.

## Installation

`zesb` requires Python 3.11 or newer. Install the package and development dependencies with `uv`:

```bash
uv sync --dev
```

Install the optional UK AISI `Inspect AI` task bridge extra when integrating with `inspect_ai`:

```bash
uv pip install ".[inspect]"
```

On Linux hosts with `libavformat-dev` installed, you can also install the optional native `vpdq` extra:

```bash
uv pip install ".[linux-vpdq]"
```

## CLI Usage

Run an end-to-end cascade evaluation against a local manifest and media root:

```bash
zesb evaluate --manifest /input/manifest.jsonl --media-root /input/media --config configs/binary_default.yaml --output-dir /output
```

Measure Stage 1 `PDQ` recall degradation and false-positive rates under image perturbations:

```bash
zesb perturb-test --manifest /input/manifest.jsonl --media-root /input/media --hash-index /models/pdq_hashes.jsonl --output-dir /output
```

## Inspect AI Integration

`zesb.inspect_bridge.create_zesb_inspect_task` wraps `CascadeRunner` inside an `inspect_ai.Task` without embedding file paths or base64 media into `.eval` archives. Sample inputs store only opaque `sample_id` tokens and `SHA-256` digests while evaluating frames strictly in memory.

## Zero-Exposure Container Contract

Statutory clearinghouses and safety laboratories can build and run the rootless OCI image without network access. The container executes under `USER 65532:65532` and sets `ZESB_OFFLINE_ONLY=1`.

```bash
docker build -t zesb:0.1.0 .
docker run --rm --network=none --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m -v /path/to/input:/input:ro -v /path/to/models:/models:ro -v /path/to/output:/output:rw zesb:0.1.0 --manifest /input/manifest.jsonl --media-root /input/media --output-dir /output
```

Output artifacts in `/output` contain only aggregate metrics (`metrics.json`) and sanitized failure logs (`sanitized_errors.jsonl`). Records with `ground_truth == "SENSITIVE"` never include `relative_path`, raw model rationales, or image previews. Blurred JPEG thumbnails are generated exclusively for `SAFE` false-positive records when `--safe-previews` is explicitly passed.

## Verification and Testing

All repository tests generate synthetic images and multi-scene videos in temporary directories at runtime. Run the quality gates with:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src/zesb
uv run pytest -v
```

## License

This project is released under the [MIT License](LICENSE).
