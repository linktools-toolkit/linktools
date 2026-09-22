#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the file tree shared by MCP snapshot and materialization boundaries."""

from collections.abc import Iterable

from ..errors import AIError, ErrorCode


def validate_resource_path(path: str) -> None:
    if (
        not path
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def validate_resource_tree(paths: Iterable[str]) -> None:
    files = set(paths)
    for path in files:
        validate_resource_path(path)
        parts = path.split("/")
        if any("/".join(parts[:end]) in files for end in range(1, len(parts))):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


__all__ = ["validate_resource_path", "validate_resource_tree"]
