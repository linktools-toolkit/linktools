#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single crash-safe atomic-write helper every File store reuses.

Writes ``content`` to ``path`` via a same-directory temp file, fsyncs the file,
``os.replace``s it into place, then fsyncs the PARENT directory. The directory
fsync is what makes the rename durable after a power loss / OS crash: without
it the file's data may be flushed but the directory entry naming it can be
lost, leaving no file (or a stale one) at ``path`` on reboot.

Callers must NEVER use bare ``Path.write_text`` / ``write_bytes`` for store
metadata that must survive a crash -- route through
:func:`atomic_write_bytes` so durability + the directory fsync are not
forgotten. On filesystems / platforms where opening a directory for fsync is
not supported, the directory fsync is skipped (best-effort); the file fsync +
rename still hold."""

import os
import tempfile
from pathlib import Path

__all__ = ["atomic_write_bytes"]


def _fsync_directory(directory: Path) -> None:
    # fsync the directory entry so the rename is durable. Opening a directory
    # for fsync is POSIX-only; silently skip where unsupported (the file fsync
    # + rename still hold; only the strongest crash guarantee is relaxed).
    try:
        dfd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        # Some filesystems (e.g. network FS) reject directory fsync. The file
        # fsync already landed; skipping the dir fsync weakens but does not
        # remove the durability guarantee.
        pass
    finally:
        os.close(dfd)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Crash-safe write of ``content`` to ``path``: same-dir temp file, write +
    flush + fsync(file), ``os.replace``, then fsync(parent directory). The temp
    file is removed in ``finally`` on any failure (no-op on success once the
    rename ran). Cancellation / interrupts propagate naturally."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
