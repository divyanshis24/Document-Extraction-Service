"""The boundary between the job system and extraction.

StubExtractor makes no LLM calls and returns nulls. The real extractor replaces it behind
the same interface, without any change to the queue or the worker.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from app.jobs import ClaimedJob

Progress = Callable[[str], Awaitable[None]]


class ExtractionError(Exception):
    """A failure with a message that is safe to show to the client."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class ExtractionOutput:
    entries: list[dict]
    metadata: dict = field(default_factory=dict)
    # True when some fields could not be attempted, which reports the job as PARTIAL.
    partial: bool = False


class Extractor(Protocol):
    async def extract(self, job: ClaimedJob, document: bytes, progress: Progress) -> ExtractionOutput: ...


class StubExtractor:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self._delay = delay_seconds

    async def extract(self, job: ClaimedJob, document: bytes, progress: Progress) -> ExtractionOutput:
        await progress("EXTRACTING")
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        # PARTIAL, not SUCCESS: the stub has not actually read the document.
        return ExtractionOutput(entries=[{}], partial=True)
