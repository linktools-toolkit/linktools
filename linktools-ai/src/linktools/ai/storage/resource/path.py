#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResourcePath: a normalized, absolute, POSIX-style resource path value type."""

from dataclasses import dataclass

from ...errors import InvalidResourcePathError


def _normalize(value: str) -> str:
    if value is None:
        raise InvalidResourcePathError("resource path is required")
    text = str(value).strip()
    if not text:
        raise InvalidResourcePathError("resource path must not be empty")
    if "\x00" in text:
        raise InvalidResourcePathError(f"NUL byte not allowed in path: {value!r}")
    if not text.startswith("/"):
        raise InvalidResourcePathError(f"resource path must be absolute: {value!r}")
    segments = [seg for seg in text.split("/") if seg != ""]
    if not segments:
        raise InvalidResourcePathError(f"resource path must not be empty: {value!r}")
    for seg in segments:
        if seg in (".", ".."):
            raise InvalidResourcePathError(f"path traversal not allowed: {value!r}")
    return "/" + "/".join(segments)


@dataclass(frozen=True, slots=True)
class ResourcePath:
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

    def child(self, name: str) -> "ResourcePath":
        return ResourcePath(f"{self.value}/{name}")

    def __truediv__(self, name: str) -> "ResourcePath":
        return self.child(name)
