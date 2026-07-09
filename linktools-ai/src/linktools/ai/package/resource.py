#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package resource types (spec §13.4/§13.5): ResourceRef (a scoped path) plus
the pydantic content/listing models. ``sanitize_package_path`` enforces the
path sandbox -- parent traversal would let one package read another's files."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .scope import PackageScope


@dataclass(frozen=True, slots=True)
class ResourceRef:
    scope: "PackageScope | None"
    path: str


class ResourceInfo(BaseModel):
    path: str
    kind: "str | None" = None
    size_bytes: "int | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)


class ResourceContent(BaseModel):
    path: str
    content: "str | bytes"
    content_type: "str | None" = None
    size_bytes: "int | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)


class ResourceListResult(BaseModel):
    items: "list[ResourceInfo]"
    next_cursor: "str | None" = None


def sanitize_package_path(path: str) -> str:
    """Normalize a caller-supplied relative path and reject anything that
    escapes the package root. Absolute paths, drive letters, ``..`` segments,
    and null bytes are refused; the result is a POSIX-style relative path with
    no leading slash."""
    if path is None:
        return ""
    raw = str(path).replace("\\", "/")
    if "\x00" in raw:
        raise ValueError("path contains null byte")
    if raw.startswith("/"):
        # An absolute path is never a valid relative resource path.
        raise ValueError(f"absolute path rejected: {path!r}")
    segments: "list[str]" = []
    for part in raw.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            # Parent traversal would escape the package root.
            raise ValueError(f"path escapes package root: {path!r}")
        if len(part) > 1 and part[1] == ":":
            # Reject Windows drive-prefixed segments.
            raise ValueError(f"absolute/reject path segment: {part!r}")
        segments.append(part)
    return "/".join(segments)
