#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem ObjectStore implementation."""

import asyncio
import hashlib
import json
import os
import shutil
import stat as stat_module
import tempfile
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

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
_OBJECT_FORMAT_VERSION = 1
_DATA_NAME = "data"
_METADATA_NAME = "metadata.json"
_IGNORED_ROOT_NAMES = frozenset({".tmp", ".trash"})


class FilesystemObjectStore:
    """Store immutable objects as atomically published filesystem units."""

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

    def _path(self, key: str) -> Path:
        digest = _key_digest(self.store_id, key).hex()
        return self._root / digest[:2] / digest

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
        await asyncio.to_thread(_require_directory, temporary_root)
        temporary = await asyncio.to_thread(_create_temp_object, temporary_root)
        try:
            data_path = temporary / _DATA_NAME
            size, digest = await _spool_file(chunks, data_path, expected_size)
            if digest != expected_digest or size != expected_size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await asyncio.to_thread(_sync_file, data_path)
            await asyncio.to_thread(
                write_json_atomic,
                temporary / _METADATA_NAME,
                {
                    "version": _OBJECT_FORMAT_VERSION,
                    "key": key,
                    "digest": expected_digest,
                    "size": expected_size,
                },
                fsync=True,
            )
            destination = self._path(key)

            async def publish() -> None:
                nonlocal temporary
                await asyncio.to_thread(_cleanup_tombstones, self._root)
                duplicate = await asyncio.to_thread(
                    _publish_filesystem_object,
                    temporary,
                    destination,
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
                    temporary = None

            if acquire_lock:
                async with FilesystemMutationLock(self._root / "object.lock"):
                    await publish()
            else:
                await publish()
            return ObjectStat(key, expected_digest, expected_size)
        finally:
            if temporary is not None:
                await asyncio.to_thread(_remove_tree_if_exists, temporary)

    async def stat(self, key: str) -> ObjectStat | None:
        _validate_key(key)
        destination = self._path(key)
        return await asyncio.to_thread(
            _stat_filesystem_object,
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
        _validate_digest(expected_digest)
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
        destination = self._path(key)

        async def delete() -> bool:
            await asyncio.to_thread(_cleanup_tombstones, self._root)
            current = await asyncio.to_thread(
                _stat_filesystem_object,
                destination,
                key,
            )
            if current is None:
                return False
            if current.digest != expected_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await asyncio.to_thread(
                _delete_filesystem_object,
                self._root,
                destination,
            )

        if acquire_lock:
            async with FilesystemMutationLock(self._root / "object.lock"):
                return await delete()
        return await delete()

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        _validate_key(key)
        destination = self._path(key)
        expected = await asyncio.to_thread(
            _read_filesystem_metadata,
            destination,
            key,
        )
        data_path = destination / _DATA_NAME
        digest = hashlib.sha256()
        size = 0
        open_task = asyncio.create_task(asyncio.to_thread(data_path.open, "rb"))
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
                next_size = size + len(chunk)
                if next_size > expected.size:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                digest.update(chunk)
                size = next_size
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


def _create_temp_object(root: Path) -> Path:
    return Path(tempfile.mkdtemp(prefix="object-", dir=root))


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _stat_filesystem_object(
    destination: Path,
    key: str,
) -> ObjectStat | None:
    if not _path_present(destination):
        return None
    return _read_filesystem_metadata(destination, key)


def _publish_filesystem_object(
    temporary: Path,
    destination: Path,
    key: str,
    size: int,
    digest: str,
) -> bool:
    if _path_present(destination):
        current = _read_filesystem_metadata(destination, key)
        if current.digest != digest or current.size != size:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return True
    parent_missing = not _path_present(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_directory(destination.parent)
    if parent_missing:
        sync_directory(destination.parent.parent)
    published = False
    try:
        os.replace(temporary, destination)
        published = True
        sync_directory(destination.parent)
    except BaseException as error:
        if published:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        raise
    return False


def _validate_filesystem_objects(root: Path, store_id: str) -> None:
    for object_path in _filesystem_object_paths(root):
        stat = _read_filesystem_metadata(object_path)
        digest = _key_digest(store_id, stat.key).hex()
        if object_path.name != digest or object_path.parent.name != digest[:2]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        actual_size, actual_digest = _hash_file(object_path / _DATA_NAME)
        if actual_size != stat.size or actual_digest != stat.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _list_filesystem_objects(root: Path, store_id: str) -> tuple[ObjectStat, ...]:
    values: list[ObjectStat] = []
    for object_path in _filesystem_object_paths(root):
        value = _read_filesystem_metadata(object_path)
        digest = _key_digest(store_id, value.key).hex()
        if object_path.name != digest or object_path.parent.name != digest[:2]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values.append(value)
    return tuple(values)


def _delete_filesystem_object(root: Path, destination: Path) -> bool:
    if not _path_present(destination):
        return False
    trash_missing = not _path_present(root / ".trash")
    trash_root = root / ".trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    _require_directory(trash_root)
    if trash_missing:
        sync_directory(root)
    tombstone = trash_root / (
        f"{destination.parent.name}-{destination.name}-{uuid4().hex}"
    )
    moved = False
    try:
        os.replace(destination, tombstone)
        moved = True
        sync_directory(destination.parent)
        sync_directory(trash_root)
        shutil.rmtree(tombstone)
        sync_directory(trash_root)
        return True
    except BaseException as error:
        if moved:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        raise


def _cleanup_tombstones(root: Path) -> None:
    trash_root = root / ".trash"
    if not _path_present(trash_root):
        return
    _require_directory(trash_root)
    try:
        for path in tuple(trash_root.iterdir()):
            _remove_tree_if_exists(path)
        sync_directory(trash_root)
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error


def _read_filesystem_metadata(
    destination: Path,
    key: str | None = None,
) -> ObjectStat:
    _require_directory(destination)
    metadata = destination / _METADATA_NAME
    data_path = destination / _DATA_NAME
    _require_regular_file(metadata)
    _require_regular_file(data_path)
    names = {path.name for path in destination.iterdir()}
    if names != {_METADATA_NAME, _DATA_NAME}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "key",
        "digest",
        "size",
    }:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = value["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != _OBJECT_FORMAT_VERSION:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    stored_key = value["key"]
    digest = value["digest"]
    size = value["size"]
    if (
        not isinstance(stored_key, str)
        or not isinstance(digest, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        _validate_key(stored_key)
        _validate_digest(digest)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if key is not None and stored_key != key:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ObjectStat(stored_key, digest, size)


def _filesystem_object_paths(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    _require_directory(root)
    values: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda value: value.name):
        if child.name in _IGNORED_ROOT_NAMES:
            _require_directory(child)
            continue
        if child.name == "object.lock":
            _require_regular_file(child)
            continue
        if not _is_lower_hex(child.name, 2):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _require_directory(child)
        for object_path in sorted(child.iterdir(), key=lambda value: value.name):
            if (
                not _is_lower_hex(object_path.name, 64)
                or not object_path.name.startswith(child.name)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _require_directory(object_path)
            values.append(object_path)
    return tuple(values)


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _require_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not stat_module.S_ISREG(mode):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _require_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not stat_module.S_ISDIR(mode):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _is_lower_hex(value: str, length: int) -> bool:
    return (
        len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _remove_tree_if_exists(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


__all__ = ["FilesystemObjectStore"]
