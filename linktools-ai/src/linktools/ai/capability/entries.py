"""Capability entry domain values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapabilityEntryInfo:
    path: str
    kind: str
    version: int
    etag: str
    active: bool = True


@dataclass(frozen=True, slots=True)
class CapabilityEntry:
    info: CapabilityEntryInfo
    content: bytes


@dataclass(frozen=True, slots=True)
class CapabilityEntryChange:
    revision: int
    path: str
    info: CapabilityEntryInfo | None
