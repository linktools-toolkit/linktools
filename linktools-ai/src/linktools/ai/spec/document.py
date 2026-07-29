"""Persisted specification documents and revision changes."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpecDocumentInfo:
    path: str
    kind: str
    version: int
    etag: str
    active: bool = True


@dataclass(frozen=True, slots=True)
class SpecDocument:
    info: SpecDocumentInfo
    content: bytes


@dataclass(frozen=True, slots=True)
class SpecDocumentChange:
    revision: int
    path: str
    info: SpecDocumentInfo | None


__all__ = ["SpecDocument", "SpecDocumentChange", "SpecDocumentInfo"]
