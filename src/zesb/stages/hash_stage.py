"""Stage 1: Vectorized 256-bit PDQ perceptual hash matcher."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pdqhash
from numpy.typing import NDArray

from zesb.schemas import FrameCandidate, StagePrediction, VerdictStatus


def pdq_bits_to_hex(bits: NDArray[np.uint8]) -> str:
    """Pack a 256-element binary uint8 array into a 64-character hex digest."""
    if bits.shape != (256,):
        raise ValueError(f"Expected 256 PDQ bits, got shape {bits.shape}")
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes().hex()


def pdq_hex_to_bits(hex_digest: str) -> NDArray[np.uint8]:
    """Unpack a 64-character PDQ hex string into a 256-element uint8 bit vector."""
    cleaned = hex_digest.strip().lower()
    if len(cleaned) != 64:
        raise ValueError(f"PDQ hex digest must be 64 characters, got {len(cleaned)}")
    raw_bytes = bytes.fromhex(cleaned)
    bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))
    return bits.astype(np.uint8)


def compute_pdq(rgb_array: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], int]:
    """Compute 256-bit PDQ hash bits and quality score for an RGB uint8 image."""
    raw_bits, quality = pdqhash.compute(rgb_array)
    bits = np.asarray(raw_bits, dtype=np.uint8)
    return bits, int(quality)


@dataclass(frozen=True)
class HashMatch:
    """Nearest match found in the PDQ hash index."""

    distance: int
    categories: list[str]


class HashIndex:
    """Vectorized in-memory index of target 256-bit PDQ hashes."""

    def __init__(
        self,
        bit_matrix: NDArray[np.uint8],
        categories: list[list[str]],
    ) -> None:
        if bit_matrix.ndim != 2 or bit_matrix.shape[1] != 256:
            raise ValueError(f"Expected shape (N, 256), got {bit_matrix.shape}")
        if len(categories) != bit_matrix.shape[0]:
            raise ValueError("categories length must match bit_matrix row count")
        self.bit_matrix = bit_matrix.astype(np.uint8, copy=False)
        self.categories = categories

    @classmethod
    def empty(cls) -> HashIndex:
        """Create an empty PDQ hash index."""
        return cls(
            bit_matrix=np.empty((0, 256), dtype=np.uint8),
            categories=[],
        )

    @classmethod
    def from_hex_entries(cls, entries: list[tuple[str, list[str]]]) -> HashIndex:
        """Construct a HashIndex from (pdq_hex, categories) pairs."""
        if not entries:
            return cls.empty()
        rows = [pdq_hex_to_bits(hex_str) for hex_str, _ in entries]
        cats = [list(cat_list) for _, cat_list in entries]
        return cls(bit_matrix=np.vstack(rows), categories=cats)

    @classmethod
    def from_file(cls, path: Path) -> HashIndex:
        """Load target PDQ hex strings from a JSONL or CSV file."""
        suffix = path.suffix.lower()
        entries: list[tuple[str, list[str]]] = []

        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    obj = json.loads(stripped)
                    hex_val = str(obj.get("pdq_hex") or obj.get("hash") or "")
                    raw_cats = obj.get("categories") or []
                    cats = [str(c) for c in raw_cats]
                    entries.append((hex_val, cats))
            return cls.from_hex_entries(entries)

        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    return cls.empty()
                for row in reader:
                    hex_val = str(row.get("pdq_hex") or row.get("hash") or "").strip()
                    if not hex_val:
                        continue
                    raw_cat = (row.get("categories") or "").strip()
                    cats = [c.strip() for c in raw_cat.split("|") if c.strip()] if raw_cat else []
                    entries.append((hex_val, cats))
            return cls.from_hex_entries(entries)

        raise ValueError(f"Unsupported hash index format: {path.suffix}")

    def query(self, bits: NDArray[np.uint8]) -> HashMatch | None:
        """Return the minimum-Hamming-distance entry in the index, or None if empty."""
        if self.bit_matrix.shape[0] == 0:
            return None
        distances = np.bitwise_xor(self.bit_matrix, bits[None, :]).sum(axis=1)
        best_idx = int(np.argmin(distances))
        return HashMatch(
            distance=int(distances[best_idx]),
            categories=list(self.categories[best_idx]),
        )


class HashStageAdapter:
    """Stage 1 adapter performing vectorized PDQ perceptual hash matching."""

    def __init__(
        self,
        index: HashIndex,
        distance_threshold: int = 31,
        min_quality_score: int = 50,
    ) -> None:
        self.index = index
        self.distance_threshold = distance_threshold
        self.min_quality_score = min_quality_score

    async def predict(self, frame: FrameCandidate) -> StagePrediction:
        """Compute PDQ hash and flag matches within Hamming distance threshold."""
        if self.index.bit_matrix.shape[0] == 0:
            return StagePrediction(
                verdict=VerdictStatus.SAFE,
                risk_score=0.0,
                predicted_categories=[],
            )

        bits, quality = compute_pdq(frame.rgb_array)
        if quality < self.min_quality_score:
            return StagePrediction(
                verdict=VerdictStatus.SAFE,
                risk_score=0.0,
                predicted_categories=[],
            )

        match = self.index.query(bits)
        if match is None:
            return StagePrediction(
                verdict=VerdictStatus.SAFE,
                risk_score=0.0,
                predicted_categories=[],
            )

        if match.distance <= self.distance_threshold:
            risk_score = 1.0 - (match.distance / 256.0)
            return StagePrediction(
                verdict=VerdictStatus.SENSITIVE,
                risk_score=risk_score,
                predicted_categories=match.categories,
            )

        return StagePrediction(
            verdict=VerdictStatus.SAFE,
            risk_score=0.0,
            predicted_categories=[],
        )
