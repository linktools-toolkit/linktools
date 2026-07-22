#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Async wrappers over blocking file I/O.

Every filesystem primitive here (open, read, write, fsync, replace, unlink,
stat, list, hash) is a blocking syscall; running it directly inside an async
method stalls the event loop for the duration of the disk operation. These
helpers move each call onto a worker thread via :func:`asyncio.to_thread` so a
large artifact's disk I/O never blocks the loop, and keep the blocking surface
in one auditable place.
"""


import asyncio
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "async_read_bytes",
    "async_write_exclusive",
    "async_stat_size",
    "async_exists",
    "async_unlink",
    "async_list_files",
    "async_list_subdirs",
    "async_hash_file",
    "async_fsync_file",
    "async_fsync_directory",
    "async_makedirs",
    "async_mkstemp",
    "async_replace",
    "async_open_read",
    "async_read_chunk",
    "async_close",
    "async_mtime",
]


def _fsync_directory(directory: Path) -> None:
    # fsync the directory entry so a same-dir rename is durable after a crash.
    # Opening a directory for fsync is POSIX-only; skip where unsupported -- the
    # file fsync + rename still hold, only the strongest guarantee is relaxed.
    try:
        dfd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


async def async_exists(path: Path) -> bool:
    return await asyncio.to_thread(path.exists)


async def async_stat_size(path: Path) -> "int | None":
    """Return the file size in bytes, or ``None`` if it does not exist."""

    def _stat() -> "int | None":
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return None

    return await asyncio.to_thread(_stat)


async def async_read_bytes(path: Path) -> bytes:
    return await asyncio.to_thread(path.read_bytes)


async def async_write_exclusive(path: Path, content: bytes) -> None:
    """Create ``path`` with ``content`` using ``O_CREAT | O_EXCL`` (no clobber),
    fsync the file, then fsync the parent directory. Raises ``FileExistsError``
    if ``path`` already exists -- the caller resolves the create-only conflict."""

    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            # Exclusive create owns the file; remove the partial on any failure
            # so a retry is not blocked by a half-written file.
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            raise
        _fsync_directory(path.parent)

    await asyncio.to_thread(_write)


async def async_unlink(path: Path) -> bool:
    """Remove ``path``; return whether it existed."""

    def _unlink() -> bool:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    return await asyncio.to_thread(_unlink)


async def async_list_files(root: Path) -> "list[Path]":
    """Sorted regular files directly under ``root``. Empty list if ``root`` is
    absent."""

    def _list() -> "list[Path]":
        if not root.exists():
            return []
        return sorted(p for p in root.iterdir() if p.is_file())

    return await asyncio.to_thread(_list)


async def async_list_subdirs(root: Path) -> "list[Path]":
    """Sorted subdirectories directly under ``root``. Empty list if ``root`` is
    absent."""

    def _list() -> "list[Path]":
        if not root.exists():
            return []
        return sorted(p for p in root.iterdir() if p.is_dir())

    return await asyncio.to_thread(_list)


async def async_hash_file(path: Path, *, chunk_size: int = 64 * 1024) -> str:
    """SHA-256 of ``path`` computed in fixed-size chunks."""

    def _hash() -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    return await asyncio.to_thread(_hash)


async def async_fsync_file(file) -> None:
    await asyncio.to_thread(file.flush)
    await asyncio.to_thread(os.fsync, file.fileno())


async def async_fsync_directory(directory: Path) -> None:
    await asyncio.to_thread(_fsync_directory, directory)


async def async_makedirs(directory: Path) -> None:
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)


async def async_mkstemp(
    *, directory: Path, prefix: str, suffix: str
) -> "tuple[int, str]":
    return await asyncio.to_thread(
        tempfile.mkstemp, dir=str(directory), prefix=prefix, suffix=suffix
    )


async def async_replace(src: "str | Path", dst: "str | Path") -> None:
    await asyncio.to_thread(os.replace, src, dst)


async def async_open_read(path: Path):
    return await asyncio.to_thread(open, path, "rb")


async def async_read_chunk(file, chunk_size: int) -> bytes:
    return await asyncio.to_thread(file.read, chunk_size)


async def async_write_chunk(file, chunk: bytes) -> None:
    await asyncio.to_thread(file.write, chunk)


async def async_close(file) -> None:
    await asyncio.to_thread(file.close)


async def async_mtime(path: Path) -> datetime:
    return await asyncio.to_thread(
        lambda: datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    )
