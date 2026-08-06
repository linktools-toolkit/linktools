#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic manifest digest helpers."""

from pathlib import Path

from linktools.ai.core.ids import canonical_sha256


def file_manifest(root: Path) -> 'dict[str, str]':
    return {
        path.relative_to(root).as_posix(): canonical_sha256({"size": path.stat().st_size, "sha256": path.read_bytes().hex()})
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def manifest_digest(manifest: 'dict[str, str]') -> str:
    return canonical_sha256(manifest)


__all__ = ["file_manifest", "manifest_digest"]
