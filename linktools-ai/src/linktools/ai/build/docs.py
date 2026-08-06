#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pinned documentation snapshot metadata."""

from pathlib import Path


def build_snapshot(path: "str | Path") -> str:
    return Path(path).read_text(encoding="utf-8")


class DocsSnapshotBuilder:
    """Read a checked-in upstream documentation snapshot."""

    def build(self, path: "str | Path") -> str:
        return build_snapshot(path)


__all__ = ["DocsSnapshotBuilder", "build_snapshot"]
