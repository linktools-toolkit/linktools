#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared filesystem helpers for the file-backed stores.

``_atomic_write`` and ``_validate_id_segment`` are used by every file store
(Run / Task / Evaluation / Swarm / Memory / Approval / Idempotency). They live
here so each store is a sibling importing a shared utility, rather than a child
reaching into FileRunStore's private helpers."""

import os
import tempfile
from pathlib import Path


def _validate_id_segment(value: str, *, kind: str) -> str:
    """Reject id segments that could escape their directory (path separators,
    '.' / '..') so a caller-controlled id never writes outside the store root."""
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    """Write ``content`` to ``path`` atomically: a temp file in the same dir,
    fsynced before the rename so an OS crash / power loss cannot leave a 0-byte
    file at the destination."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


__all__: "list[str]" = ["_atomic_write", "_validate_id_segment"]
