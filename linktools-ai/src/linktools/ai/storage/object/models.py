#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The storage kernel's core value types.

``StorageKey`` is the normalized, absolute, POSIX-style object key; ``ObjectInfo``
is the per-object metadata (with ``version`` -- per-key monotonic -- strictly
separated from ``commit_revision`` -- the backend namespace's transaction
watermark); ``StoredObject`` pairs info + bytes; the three-state lookup shape
(``Found``/``Masked``/``Missing``) models an overlay read; ``WriteOptions``
carries the CAS + idempotency gates; ``Depth`` bounds a listing. These are
pure, frozen, backend-agnostic value types -- the lowest layer of the storage
kernel, depending on nothing domain-specific."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Union


def _normalize_key(value: str) -> str:
    """Enforce the StorageKey rules: non-empty, absolute (leading ``/``),
    collapse consecutive ``/`` (incl. trailing), reject ``.``/``..``/NUL."""
    if value is None:
        raise ValueError("storage key is required")
    text = str(value)
    if not text:
        raise ValueError("storage key must not be empty")
    if "\x00" in text:
        raise ValueError(f"NUL byte not allowed in key: {value!r}")
    if not text.startswith("/"):
        raise ValueError(f"storage key must be absolute (leading '/'): {value!r}")
    segments = [seg for seg in text.split("/") if seg != ""]
    for seg in segments:
        if seg in (".", ".."):
            raise ValueError(f"path traversal not allowed in key: {value!r}")
    if not segments:
        return "/"
    return "/" + "/".join(segments)


@dataclass(frozen=True, slots=True)
class StorageKey:
    """A normalized, absolute, POSIX-style object key.

    Root is ``/``. Consecutive slashes collapse; ``.``/``..``/NUL are rejected.
    The value carries NO backend namespace -- it is purely a logical path. Use
    ``parent``/``name``/``join``/``is_under`` to navigate without
    re-normalizing by hand.
    """

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _normalize_key(self.value))

    def __str__(self) -> str:
        return self.value

    @property
    def is_root(self) -> bool:
        return self.value == "/"

    @property
    def _segments(self) -> "tuple[str, ...]":
        if self.is_root:
            return ()
        return tuple(self.value.strip("/").split("/"))

    @property
    def name(self) -> str:
        segs = self._segments
        return segs[-1] if segs else ""

    @property
    def parent(self) -> "StorageKey":
        # Root is its own parent: navigation bottoms out at the namespace root
        # rather than producing None, so callers can chain unconditionally.
        segs = self._segments
        if not segs or len(segs) == 1:
            return StorageKey("/")
        return StorageKey("/" + "/".join(segs[:-1]))

    def join(self, name: str) -> "StorageKey":
        """Append a single path segment under this key."""
        if self.is_root:
            return StorageKey(f"/{name}")
        return StorageKey(f"{self.value}/{name}")

    def is_under(self, ancestor: "StorageKey") -> bool:
        """True when this key is ``ancestor`` itself or a descendant of it."""
        if ancestor.is_root:
            return True
        return self.value == ancestor.value or self.value.startswith(
            ancestor.value + "/"
        )


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """Per-object metadata. ``version`` is the per-key monotonic version;
    ``commit_revision`` is the backend namespace's transaction watermark
    (None for backends without transactions). The two are strictly separate
    concepts and must never be conflated."""

    key: StorageKey
    etag: str
    version: int
    commit_revision: "int | None"
    content_type: "str | None"
    size: int
    modified_at: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)
    tombstoned: bool = False

    def __post_init__(self) -> None:
        # Freeze the mapping so the frozen dataclass is genuinely immutable.
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )


@dataclass(frozen=True, slots=True)
class StoredObject:
    """An object's metadata + its bytes."""

    info: ObjectInfo
    content: bytes


# --- three-state lookup (overlay read result) -------------------------------


@dataclass(frozen=True, slots=True)
class Found:
    """The lookup found a live object at the key."""

    object: StoredObject


@dataclass(frozen=True, slots=True)
class Masked:
    """The key exists in the layer but is masked (e.g. a primary tombstone
    hides an overlay value). Carries ``commit_revision`` so an overlay consumer
    can tell whether its cached view of the mask is stale."""

    key: StorageKey
    version: int
    commit_revision: "int | None"


class _MissingSentinel:
    """The lookup found nothing at the key (no live object, no mask). A
    singleton: ``Missing`` is the one instance."""

    __slots__ = ()
    _instance: "_MissingSentinel | None" = None

    def __new__(cls) -> "_MissingSentinel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Missing"

    def __bool__(self) -> bool:
        return False


Missing = _MissingSentinel()


LookupResult = Union[Found, Masked, _MissingSentinel]


@dataclass(frozen=True, slots=True)
class WriteOptions:
    """CAS + idempotency gates for a write, plus the metadata a put() call
    attaches to the object (the only place ``ObjectInfo.content_type`` /
    ``ObjectInfo.metadata`` can come from -- ``put()`` itself takes no
    separate parameters for them).

    - ``if_match``: an etag; the write proceeds only if the current object's
      etag matches (optimistic update).
    - ``if_none_match``: when truthy, the write proceeds only if the key does
      NOT exist (create-only).
    - ``idempotency_key``: a caller-supplied request key; a replay with the
      same key + content + content_type + metadata returns the original
      result with no version bump, while the same key + different content
      (or content_type/metadata) is a conflict.
    - ``content_type`` / ``metadata``: attached to the object; ignored by
      delete() and by move()'s target precondition checks (move carries the
      source object's own content_type/metadata forward unchanged).
    """

    if_match: "str | None" = None
    if_none_match: "bool | None" = None
    idempotency_key: "str | None" = None
    content_type: "str | None" = None
    metadata: "Mapping[str, object] | None" = None


class Depth(enum.Enum):
    """Listing depth bound. ``ZERO`` matches the prefix itself; ``ONE`` matches
    direct children; ``INFINITY`` matches all descendants."""

    ZERO = "zero"
    ONE = "one"
    INFINITY = "infinity"


@dataclass(frozen=True, slots=True)
class ObjectPage:
    """One page of a listing result."""

    items: "tuple[ObjectInfo, ...]"
    next_cursor: "str | None" = None


@dataclass(frozen=True, slots=True)
class ObjectVersionPage:
    """One page of a per-key version-history listing."""

    items: "tuple[ObjectInfo, ...]"
    next_cursor: "str | None" = None
