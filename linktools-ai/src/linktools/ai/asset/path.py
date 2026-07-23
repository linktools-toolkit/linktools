#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AssetPath: a normalized, absolute, POSIX-style asset path value type."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import InvalidAssetPathError

if TYPE_CHECKING:
    from .models import Depth


def _normalize(value: str) -> str:
    if value is None:
        raise InvalidAssetPathError("asset path is required")
    text = str(value).strip()
    if not text:
        raise InvalidAssetPathError("asset path must not be empty")
    if "\x00" in text:
        raise InvalidAssetPathError(f"NUL byte not allowed in path: {value!r}")
    if not text.startswith("/"):
        raise InvalidAssetPathError(f"asset path must be absolute: {value!r}")
    segments = [seg for seg in text.split("/") if seg != ""]
    if not segments:
        # "/" alone is the root namespace: a valid path whose direct children
        # a ONE-depth list enumerates and against which every path matches.
        return "/"
    for seg in segments:
        if seg in (".", ".."):
            raise InvalidAssetPathError(f"path traversal not allowed: {value!r}")
    return "/" + "/".join(segments)


@dataclass(frozen=True, slots=True)
class AssetPath:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _normalize(self.value))

    def __str__(self) -> str:
        return self.value

    @property
    def parts(self) -> "tuple[str, ...]":
        return tuple(self.value.strip("/").split("/"))

    @property
    def namespace(self) -> str:
        return self.parts[0]

    def child(self, name: str) -> "AssetPath":
        return AssetPath(f"{self.value}/{name}")

    def __truediv__(self, name: str) -> "AssetPath":
        return self.child(name)


def _relative_depth(base_value: str, candidate_value: str) -> "int | None":
    """String-level relative-depth computation. Split out from
    :func:`relative_asset_depth` so the root-namespace case (``base_value ==
    "/"``) is unit-testable without constructing an ``AssetPath("/")``, which
    normalization currently rejects."""
    if candidate_value == base_value:
        return 0
    prefix = "/" if base_value == "/" else base_value + "/"
    if not candidate_value.startswith(prefix):
        return None
    return candidate_value[len(prefix) :].count("/") + 1


def relative_asset_depth(base: AssetPath, candidate: AssetPath) -> "int | None":
    """Depth of ``candidate`` relative to ``base``: ``0`` when identical, ``1``
    for a direct child, ``2+`` for a deeper descendant, ``None`` when
    ``candidate`` is not under ``base``. Shared by every asset backend so
    Depth.ZERO/ONE/INFINITY resolve identically across Memory, Filesystem, and
    SqlAlchemy."""
    return _relative_depth(base.value, candidate.value)


def matches_asset_depth(
    base: AssetPath, candidate: AssetPath, depth: "Depth"
) -> bool:
    from .models import Depth

    relative = relative_asset_depth(base, candidate)
    if relative is None:
        return False
    if depth is Depth.ZERO:
        return relative == 0
    if depth is Depth.ONE:
        return relative <= 1
    return True
