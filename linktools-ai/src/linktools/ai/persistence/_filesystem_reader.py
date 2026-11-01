#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline ownership and metadata validation for legacy filesystem reads."""

import asyncio
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from ..errors import AIError, ErrorCode
from ..storage import FilesystemWriterLock
from ._filesystem_layout import _digest, _read_generation_value, _read_json

_ValueT = TypeVar("_ValueT")


async def _await_owned(task: asyncio.Task[_ValueT]) -> _ValueT:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    try:
        result = task.result()
    except BaseException as error:
        if cancellation is not None:
            raise cancellation from error
        raise
    if cancellation is not None:
        raise cancellation
    return result


async def _read_owned(read: Callable[[], _ValueT]) -> _ValueT:
    # A cancelled read must finish before the source's offline locks are released.
    return await _await_owned(asyncio.create_task(asyncio.to_thread(read)))


@dataclass(frozen=True, slots=True)
class _FilesystemMember:
    runtime_domain: str
    root: Path
    partition_key: bytes | None = None


@dataclass(frozen=True, slots=True)
class _FilesystemInventory:
    members: tuple[_FilesystemMember, ...]
    archive_scope: str | None


class _FilesystemOfflineSource:
    """Read one legacy group's metadata under its native writer locks."""

    def __init__(self, root: Path, *, namespace: str, tenant_id: str, scope_digest: str) -> None:
        self._root = root
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._scope_digest = scope_digest
        self._metadata_root = root / ".state-groups" / scope_digest
        self._group_lock = FilesystemWriterLock(self._metadata_root / "state.lock")
        self._owner: asyncio.Task[object] | None = None
        self._locks: list[FilesystemWriterLock] = []
        self._inventory: _FilesystemInventory | None = None
        self._close_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._owner is not None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async with self._close_lock:
            if self._owner is not None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            await _await_owned(asyncio.create_task(self._release()))

    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        if self._owner is not None or self._locks or self._close_lock.locked():
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        owner = asyncio.current_task()
        if owner is None:
            raise RuntimeError("offline reads require an asyncio task")
        self._owner = owner
        primary: BaseException | None = None
        try:
            await _await_owned(asyncio.create_task(self._acquire()))
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            self._inventory = None
            try:
                await _await_owned(asyncio.create_task(self._release()))
            except BaseException as error:
                if primary is not None:
                    raise primary from error
                raise
            finally:
                self._owner = None

    def describe(self) -> _FilesystemInventory:
        if self._owner is None or self._owner is not asyncio.current_task() or self._inventory is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._inventory

    async def _release(self) -> None:
        while self._locks:
            await self._locks[-1].release()
            self._locks.pop()

    async def _acquire(self) -> None:
        if not self._root.is_dir() or not self._metadata_root.is_dir():
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        await self._group_lock.acquire()
        self._locks.append(self._group_lock)
        # Membership can change until the group writer is excluded.
        before = await asyncio.to_thread(self._load_inventory)
        for member in sorted(before.members, key=lambda value: value.root.as_posix()):
            if not member.root.is_dir():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            lock = FilesystemWriterLock(member.root / "state.lock")
            await lock.acquire()
            self._locks.append(lock)
        after = await asyncio.to_thread(self._validate_locked_inventory, before)
        self._inventory = after

    def _load_inventory(self) -> _FilesystemInventory:
        try:
            if (self._root / f".txn-{self._scope_digest}" / "commit").exists():
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            actual = _read_json(self._metadata_root / "manifest.json")
            entries = actual.get("members")
            if not isinstance(entries, list):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            members: list[_FilesystemMember] = []
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"runtime_domain", "relative_path"}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                domain, relative = entry["runtime_domain"], entry["relative_path"]
                if not isinstance(domain, str) or not domain or domain == "group_archive" or not isinstance(relative, str):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                root = (self._root / relative).resolve()
                if (root == self._root or self._root not in root.parents
                        or root.relative_to(self._root).parts[0] == ".state-groups"
                        or root.relative_to(self._root).parts[0].startswith(".txn-")):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                key = None
                if domain.startswith("group_partition:"):
                    key = bytes.fromhex(domain.removeprefix("group_partition:"))
                    if len(key) != 32 or root != self._root / ".state-partitions" / self._scope_digest / key.hex():
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                members.append(_FilesystemMember(domain, root, key))
            if len({member.root for member in members}) != len(members) or len({member.runtime_domain for member in members}) != len(members):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            expected = {
                "format": "linktools-ai-state-group", "version": 1,
                "namespace_digest": _digest(self._namespace), "tenant_digest": _digest(self._tenant_id),
                "members": [{"runtime_domain": member.runtime_domain,
                             "relative_path": member.root.relative_to(self._root).as_posix()}
                            for member in sorted(members, key=lambda value: (value.runtime_domain, value.root.as_posix()))],
            }
            if actual != expected:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            archive_root = self._root / ".state-archive" / self._scope_digest
            archive_metadata = self._metadata_root / "archive_scope.json"
            scope = None
            if archive_root.exists() or archive_metadata.exists():
                if not archive_root.is_dir() or not archive_metadata.is_file():
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                scope = _read_json(archive_metadata).get("archive_scope_id")
                if not isinstance(scope, str) or len(scope) != 32 or any(c not in "0123456789abcdef" for c in scope):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                archive_root = archive_root.resolve()
                if (archive_root == self._root or self._root not in archive_root.parents
                        or archive_root.relative_to(self._root).parts[0] == ".state-groups"
                        or archive_root.relative_to(self._root).parts[0].startswith(".txn-")
                        or archive_root in {member.root for member in members}):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                members.append(_FilesystemMember("group_archive", archive_root))
            return _FilesystemInventory(tuple(members), scope)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _validate_locked_inventory(self, before: _FilesystemInventory) -> _FilesystemInventory:
        after = self._load_inventory()
        if before != after:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._check_journals(after.members)
        try:
            _read_generation_value(self._metadata_root / "generation")
            device = os.stat(self._root).st_dev
            for member in after.members:
                if not member.root.is_dir() or os.stat(member.root).st_dev != device:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                expected = {
                    "format": "linktools-ai-state", "layout_version": 1,
                    "namespace_digest": _digest(self._namespace), "tenant_digest": _digest(self._tenant_id),
                    "runtime_domain": member.runtime_domain,
                }
                if _read_json(member.root / "manifest.json") != expected:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                _read_generation_value(member.root / "generation")
            return after
        except (OSError, ValueError, TypeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _check_journals(self, members: tuple[_FilesystemMember, ...]) -> None:
        if any((member.root / ".txn" / "commit").exists() for member in members):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        member_paths = tuple(member.root.relative_to(self._root).as_posix() for member in members)
        try:
            for journal in self._root.iterdir():
                if (not journal.is_dir() or not journal.name.startswith(".txn-")
                        or journal.name == f".txn-{self._scope_digest}" or not (journal / "commit").is_file()):
                    continue
                plan = _read_json(journal / "plan.json")
                for item in tuple(plan.get("writes", ())) + tuple(plan.get("deletes", ())):
                    value = item.get("path") if isinstance(item, Mapping) else item
                    if not isinstance(value, str) or any(value == root or value.startswith(root + "/") for root in member_paths):
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
