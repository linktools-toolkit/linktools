#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Storage-neutral artifact references for agent resources and trace archives."""

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from re import fullmatch
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

ArtifactDomain = Literal["capability", "session", "runtime", "trace"]
_ALLOWED_DOMAINS = {"capability", "session", "runtime", "trace"}
_SEGMENT_PATTERN = r"[A-Za-z0-9._:-]+"
_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


def _clean_path(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        raise ValueError("artifact path is required")
    path = PurePosixPath(text)
    if path.is_absolute():
        raise ValueError(f"absolute artifact path is not allowed: {value}")
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"path escape is not allowed: {value}")
    return "/".join(parts)


def _clean_segment(name: str, value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"artifact {name} is required")
    if text in (".", "..") or not fullmatch(_SEGMENT_PATTERN, text):
        raise ValueError(f"unsafe artifact {name}: {value}")
    return text


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"unsupported artifact metadata key: {type(key).__name__}")
            frozen[key] = _freeze_metadata(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata(item) for item in value)
    if isinstance(value, _JSON_SCALAR_TYPES):
        return value
    raise TypeError(f"unsupported artifact metadata value: {type(value).__name__}")


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [_thaw_metadata(item) for item in value]
    if isinstance(value, list):
        return [_thaw_metadata(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    domain: ArtifactDomain
    scope: str
    kind: str
    path: str
    version: "int | None" = None

    def __post_init__(self) -> None:
        domain = str(self.domain or "").strip()
        if domain not in _ALLOWED_DOMAINS:
            raise ValueError(f"unsupported artifact domain: {self.domain}")
        scope = _clean_segment("scope", self.scope)
        kind = _clean_segment("kind", self.kind)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", _clean_path(self.path))

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.scope}/{self.kind}/{self.path}"

    def as_dict(self) -> "dict[str, Any]":
        return {
            "domain": self.domain,
            "scope": self.scope,
            "kind": self.kind,
            "path": self.path,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ArtifactMeta:
    ref: ArtifactRef
    checksum: str
    size_bytes: int
    backend: str
    location: str
    status: str
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def as_dict(self) -> "dict[str, Any]":
        return {
            "ref": self.ref.as_dict(),
            "checksum": self.checksum,
            "size_bytes": int(self.size_bytes),
            "backend": self.backend,
            "location": self.location,
            "status": self.status,
            "metadata": _thaw_metadata(self.metadata),
        }


@runtime_checkable
class AgentArtifactStore(Protocol):
    async def get(self, ref: ArtifactRef) -> "bytes | None":
        ...

    async def put(self, ref: ArtifactRef, content: bytes, *, idempotency_key: str) -> ArtifactMeta:
        ...
