"""Specification source and codec contracts."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar, runtime_checkable

from ..errors import SpecConflictError, SpecError, SpecNotFoundError, SpecParseError
from .revision import RevisionCache


T = TypeVar("T")


@runtime_checkable
class SpecSource(Protocol):
    async def revision(self) -> str: ...
    async def list_ids(self, suffix: str) -> tuple[str, ...]: ...
    async def read(self, path: str) -> str: ...


@runtime_checkable
class SpecCodec(Protocol, Generic[T]):
    def decode(self, item_id: str, raw: str) -> T: ...


__all__ = [
    "SpecCodec",
    "SpecConflictError",
    "SpecError",
    "SpecNotFoundError",
    "SpecParseError",
    "SpecSource",
    "RevisionCache",
]
