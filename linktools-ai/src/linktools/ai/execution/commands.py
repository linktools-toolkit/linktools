"""Validated lifecycle command values."""

from dataclasses import dataclass

from .models import RunSnapshot


@dataclass(frozen=True, slots=True)
class CompleteRun:
    snapshot: RunSnapshot
    owner: str
    fence: int
