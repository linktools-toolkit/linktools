#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared filesystem helpers for the file-backed stores.

``_atomic_write`` and ``_validate_id_segment`` are used by every file store
(Run / Task / Evaluation / Swarm / Memory / Approval / Idempotency). They live
here so each store is a sibling importing a shared utility, rather than a child
reaching into FilesystemRunStore's private helpers."""

from pathlib import Path

from .atomic import atomic_write_bytes


def _validate_id_segment(value: str, *, kind: str) -> str:
    """Reject id segments that could escape their directory (path separators,
    '.' / '..') so a caller-controlled id never writes outside the store root."""
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    """Crash-safe atomic write (temp file + fsync + ``os.replace`` + fsync of
    the parent directory). Delegates to :func:`atomic_write_bytes`; kept as the
    historical name every file store already imports."""
    atomic_write_bytes(path, content)


__all__: "list[str]" = ["_atomic_write", "_validate_id_segment"]
