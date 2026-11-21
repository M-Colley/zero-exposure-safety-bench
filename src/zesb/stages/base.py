"""Abstract StageAdapter protocol for cascade detection stages."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from zesb.schemas import FrameCandidate, StagePrediction


@runtime_checkable
class StageAdapter(Protocol):
    """Asynchronous stage scorer operating on a single in-memory frame."""

    async def predict(self, frame: FrameCandidate) -> StagePrediction:
        """Score one frame candidate and return a normalized StagePrediction."""
        ...
