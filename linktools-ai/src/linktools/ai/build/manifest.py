#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build-time source manifest creation."""

from pathlib import Path

from ..foundation.digest import sha256_digest


def build_source_manifest(source_root: "str | Path") -> "dict[str, str]":
    root = Path(source_root)
    return {str(path.relative_to(root)): sha256_digest(path.read_bytes()) for path in sorted(root.rglob("*.py"))}


__all__ = ["build_source_manifest"]
