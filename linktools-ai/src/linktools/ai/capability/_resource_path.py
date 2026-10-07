#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical relative paths inside Capability-owned resource packages."""

from collections.abc import Iterable

from ..asset import AssetKey, validate_materialized_path
from ..errors import AIError, ErrorCode

SKILL_DECLARATION_FILES = ("SKILL.md", "skill.md")
_MCP_DECLARATION_FILES = frozenset({"mcp.json", "mcp.yaml"})


def require_resource_path(path: str) -> str:
    """Return one canonical package-relative POSIX resource path."""
    validate_materialized_path(path)
    return path


def validate_resource_path(path: str) -> None:
    try:
        require_resource_path(path)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error


def validate_resource_tree(paths: Iterable[str]) -> None:
    files = set(paths)
    for path in files:
        validate_resource_path(path)
        parts = path.split("/")
        if any("/".join(parts[:end]) in files for end in range(1, len(parts))):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def mcp_resource_path(key: AssetKey, root: AssetKey) -> "str | None":
    """Return the MCP package-relative resource path for one Asset key."""
    if not isinstance(key, AssetKey) or not isinstance(root, AssetKey):
        raise TypeError("MCP resource keys must be AssetKey values")
    prefix = f"{root.id}/"
    if key.kind != root.kind or not key.id.startswith(prefix):
        return None
    relative = key.id[len(prefix) :]
    try:
        relative = require_resource_path(relative)
    except ValueError as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if relative in _MCP_DECLARATION_FILES:
        return None
    return relative


__all__ = [
    "SKILL_DECLARATION_FILES",
    "mcp_resource_path",
    "validate_resource_path",
    "validate_resource_tree",
]
