"""Errors shared between the worker and everything it calls."""
from __future__ import annotations


class ExtractionError(Exception):
    """A failure with a message that is safe to show to the client."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message
