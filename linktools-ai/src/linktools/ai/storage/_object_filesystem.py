#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem ObjectStore implementation."""

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._files import sync_directory, write_json_atomic
from ._lock import FilesystemMutationLock
from ._object import (
    ObjectStat,
    _CHUNK_SIZE,
    _finish_owned_task,
    _key_digest,
    _settle_task,
    _spool_file,
    _track_object_task,
    _validate_digest,
    _validate_key,
    _validate_put,
    _validate_store_id,
)

_logger = environ.get_logger("ai.storage.object")


class FilesystemObjectStore:
    """Store content-addressed objects directly below ``root``."""

    def __init__(self, root: str | Path, *, store_id: str = "builtin") -> None:
        _validate_store_id(store_id)
        self._root = Path(root).expanduser().resolve()
        self._store_id = store_id
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._offline_owner: asyncio.Task[Any] | None = None

    @property
    def store_id(self) -> str:
        return self._store_id

    def local_paths(self) -> tuple[Path, ...]:
        return (self._root,)

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(task for task in self._background_tasks if not task.done())

    def _paths(self, key: str) -> tuple[Path, Path]:
        digest = _key_digest(self.store_id, key).hex()
        root = self._root / digest[:2]
        return root / f"{digest}.bin", root / f"{digest}.json"

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
    ) -> ObjectStat:
        _validate_put(key, expected_size, expected_digest)
        owner = asyncio.current_task()
        offline_owned = owner is not None and owner is self._offline_owner
        task = asyncio.create_task(
            self._put_owned(
                key,
                chunks,
                expected_size=expected_size,
                expected_digest=expected_digest,
                acquire_lock=not offline_owned,
            ),
            name=f"filesystem-object-put-{_key_digest(self.store_id, key).hex()[:12]}",
        )
        _track_object_task(self._background_tasks, task, "filesystem object put")
        return await _finish_owned_task(task)

    async def _put_owned(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
        acquire_lock: bool,
    ) -> ObjectStat:
        temporary_root = self._root / ".tmp"
        await asyncio.to_thread(temporary_root.mkdir, parents=True, exist_ok=True)
        name = await asyncio.to_thread(_create_temp_file, temporary_root)
        try:
            size, digest = await _spool_file(chunks, name, expected_size)
            if digest != expected_digest or size != expected_size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await asyncio.to_thread(_sync_file, name)
            destination, metadata = self._paths(key)

            async def publish() -> None:
                nonlocal name
                duplicate = await asyncio.to_thread(
                    _publish_filesystem_object,
                    name,
                    destination,
                    metadata,
                    key,
                    expected_size,
                    expected_digest,
                )
                if duplicate:
                    _logger.debug(
                        "filesystem object duplicate accepted by metadata: key=%s",
                        key,
                    )
                else:
                    name = None

            if acquire_lock:
                async with FilesystemMutationLock(self._root / "object.lock"):
                    await publish()
            else:
                await publish()
            return ObjectStat(key, expected_digest, expected_size)
        finally:
            if name is not None:
                await asyncio.to_thread(name.unlink, missing_ok=True)

    async def stat(self, key: str) -> ObjectStat | None:
        _validate_key(key)
        destination, metadata = self._paths(key)
        return await asyncio.to_thread(
            _stat_filesystem_object,
            metadata,
            destination,
            key,
        )

    async def validate_integrity(self) -> None:
        await asyncio.to_thread(
            _validate_filesystem_objects,
            self._root,
            self._store_id,
        )

    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        lock = FilesystemMutationLock(self._root / "object.lock")
        await lock.__aenter__()
        owner = asyncio.current_task()
        if owner is None:
            await lock.__aexit__(None, None, None)
            raise RuntimeError("offline ObjectStore exclusivity requires an asyncio task")
        self._offline_owner = owner
        try:
            yield
        finally:
            self._offline_owner = None
            await lock.__aexit__(None, None, None)

    async def _list_objects(self) -> AsyncIterator[ObjectStat]:
        values = await asyncio.to_thread(
            _list_filesystem_objects,
            self._root,
            self._store_id,
        )
        for value in values:
            yield value

    def list_objects(self) -> AsyncIterator[ObjectStat]:
        return self._list_objects()

    async def delete_object(self, key: str, *, expected_digest: str) -> bool:
        _validate_key(key)
        owner = asyncio.current_task()
        offline_owned = owner is not None and owner is self._offline_owner
        task = asyncio.create_task(
            self._delete_owned(
                key,
                expected_digest=expected_digest,
                acquire_lock=not offline_owned,
            ),
            name=f"filesystem-object-delete-{_key_digest(self.store_id, key).hex()[:12]}",
        )
        _track_object_task(self._background_tasks, task, "filesystem object delete")
        return await _finish_owned_task(task)

    async def _delete_owned(
        self,
        key: str,
        *,
        expected_digest: str,
        acquire_lock: bool,
    ) -> bool:
        destination, metadata = self._paths(key)

        async def delete() -> bool:
            current = await asyncio.to_thread(
                _stat_filesystem_object,
                metadata,
                destination,
                key,
            )
            if current is None:
                return False
            if current.digest != expected_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await asyncio.to_thread(
                _delete_filesystem_object,
                destination,
                metadata,
            )

        if acquire_lock:
            async with FilesystemMutationLock(self._root / "object.lock"):
                return await delete()
        return await delete()

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        _validate_key(key)
        destination, metadata = self._paths(key)
        expected = await asyncio.to_thread(
            _read_filesystem_metadata,
            metadata,
            destination,
            key,
        )
        digest = hashlib.sha256()
        size = 0
        open_task = asyncio.create_task(asyncio.to_thread(destination.open, "rb"))
        _track_object_task(self._background_tasks, open_task, "filesystem object open")
        handle, cancelled = await _settle_task(open_task)
        if cancelled:
            close_task = asyncio.create_task(asyncio.to_thread(handle.close))
            _track_object_task(
                self._background_tasks,
                close_task,
                "filesystem object close cleanup",
            )
            await _finish_owned_task(close_task)
            raise asyncio.CancelledError
        try:
            while True:
                read_task = asyncio.create_task(asyncio.to_thread(handle.read, _CHUNK_SIZE))
                _track_object_task(
                    self._background_tasks,
                    read_task,
                    "filesystem object read",
                )
                chunk, cancelled = await _settle_task(read_task)
                if cancelled:
                    raise asyncio.CancelledError
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                yield chunk
            if expected.size != size or expected.digest != digest.hexdigest():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        finally:
            close_task = asyncio.create_task(asyncio.to_thread(handle.close))
            _track_object_task(
                self._background_tasks,
                close_task,
                "filesystem object close cleanup",
            )
            await _finish_owned_task(close_task)

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)


def _create_temp_file(root: Path) -> Path:
    import tempfile

    descriptor, name = tempfile.mkstemp(prefix="object-", dir=root)
    os.close(descriptor)
    return Path(name)


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _stat_filesystem_object(
    metadata: Path,
    destination: Path,
    key: str,
) -> ObjectStat | None:
    if not metadata.is_file() and not destination.is_file():
        return None
    return _read_filesystem_metadata(metadata, destination, key)


def _publish_filesystem_object(
    temporary: Path,
    destination: Path,
    metadata: Path,
    key: str,
    size: int,
    digest: str,
) -> bool:
    if destination.exists() or metadata.exists():
        current = _read_filesystem_metadata(metadata, destination, key)
        if current.digest != digest or current.size != size:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    write_json_atomic(
        metadata,
        {"key": key, "digest": digest, "size": size},
        fsync=True,
    )
    sync_directory(destination.parent)
    return False


def _validate_filesystem_objects(root: Path, store_id: str) -> None:
    expected_files: set[Path] = set()
    for metadata in root.glob("*/*.json"):
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
            key = value["key"]
            expected_digest = value["digest"]
            expected_size = int(value["size"])
            _validate_key(str(key))
            digest = _key_digest(store_id, str(key)).hex()
            destination = metadata.with_name(f"{digest}.bin")
            expected_metadata = metadata.with_name(f"{digest}.json")
            if metadata != expected_metadata:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stat = _read_filesystem_metadata(metadata, destination, str(key))
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if stat.digest != expected_digest or stat.size != expected_size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        actual_size, actual_digest = _hash_file(destination)
        if actual_size != expected_size or actual_digest != expected_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_files.update({metadata, destination})
    actual_files = {
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != root / "object.lock"
        and root / ".tmp" not in path.parents
    }
    if actual_files != expected_files:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _list_filesystem_objects(root: Path, store_id: str) -> tuple[ObjectStat, ...]:
    values: list[ObjectStat] = []
    for metadata in root.glob("*/*.json"):
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
            key = value["key"]
            destination = metadata.with_suffix(".bin")
            stat = _read_filesystem_metadata(metadata, destination, str(key))
            if _key_digest(store_id, str(key)).hex() != metadata.stem:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        values.append(stat)
    return tuple(values)


def _delete_filesystem_object(destination: Path, metadata: Path) -> bool:
    present = destination.exists() or metadata.exists()
    if not present:
        return False
    destination.unlink(missing_ok=True)
    metadata.unlink(missing_ok=True)
    sync_directory(destination.parent)
    return True


def _read_filesystem_metadata(
    metadata: Path,
    destination: Path,
    key: str,
) -> ObjectStat:
    if not metadata.is_file() or not destination.is_file():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("object metadata must be an object")  # noqa: TRY004
        if value.get("key") != key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        digest = str(value["digest"])
        size = int(value["size"])
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _validate_digest(digest)
    if size < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ObjectStat(key, digest, size)


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


__all__ = ["FilesystemObjectStore"]
