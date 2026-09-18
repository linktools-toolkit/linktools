#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Granular, journaled filesystem implementation of StateStore."""

import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Literal, TypeVar

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import FilesystemJournal, FilesystemWriterLock, sync_directory
from ._codec import decode_alias, decode_fact, decode_operation, decode_record
from ._filesystem_layout import (
    _FactStreamInfo, _FilesystemCache, _FilesystemIndex, _RECORD_INDEX_MARKER,
    _RECORD_INDEX_VERSION, _build_record_index_nodes, _digest, _fact_item_path,
    _fact_meta_path, _fact_subject_path, _layout_digest, _layout_sequence_name,
    _read_fact_metadata, _read_generation_value, _read_json, _read_operation_ref,
    _read_record_index_node_path, _read_sequence_metadata, _read_subject_sequence,
    _record_index_marker_valid, _record_index_node_payload, _record_path,
    _require_layout_path, _sync_record_index_tree, _track_physical_task, _write_json, _write_text,
)
from ._filesystem_transaction import _FilesystemTransaction
from ._store import (
    StateCallback, StateGroupCallback, StateStorageGroup, StateTransaction, StoredOperation,
    StoredRecord, active_state_group_transaction, active_state_transaction, bind_state_scope,
    reset_state_transaction,
)

ValueT = TypeVar("ValueT")
_logger = environ.get_logger("ai.runtime.state.filesystem")
_CommitOutcome = Literal["committed", "not_committed", "unknown"]


class _FilesystemGroupTransaction:
    def __init__(
        self,
        group: "FilesystemStateStorageGroup",
        transactions: Mapping["FilesystemStateStore", StateTransaction],
    ) -> None:
        self._group = group
        self._transactions = transactions

    def transaction(self, store: "FilesystemStateStore") -> StateTransaction:
        if store.storage_group is not self._group:
            raise RuntimeError("store does not belong to this StateStorageGroup")
        try:
            return self._transactions[store]
        except KeyError as error:
            raise RuntimeError(
                "store was not enlisted in the StateStorageGroup transaction"
            ) from error


class FilesystemStateStorageGroup:
    """Own one journal and coordinate independent filesystem domain views."""

    def __init__(
        self,
        transaction_root: Path,
        *,
        namespace: str,
        tenant_id: str,
        scope_digest: str,
        standalone: bool = False,
    ) -> None:
        self._transaction_root = transaction_root.resolve()
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._scope_digest = scope_digest
        self._standalone = standalone
        self._members: list[FilesystemStateStore] = []
        self._mutation_lock = asyncio.Lock()
        self._group_lock = FilesystemWriterLock(self._metadata_root / "state.lock")
        self._journal = FilesystemJournal(
            self._transaction_root if not standalone else transaction_root,
            error_code=ErrorCode.STORAGE_INTEGRITY_ERROR,
            transaction_name=".txn" if standalone else f".txn-{scope_digest}",
        )
        self._generation_path = (
            self._metadata_root / "generation"
            if not standalone
            else transaction_root / "generation"
        )
        self._initialized = False
        self._closed = False
        self._poisoned = False
        self._generation: int | None = None
        self._initialization_task: asyncio.Task[None] | None = None
        self._maintenance_tasks: set[asyncio.Task[None]] = set()
        self._pending_physical: set[asyncio.Task[None]] = set()

    @property
    def _metadata_root(self) -> Path:
        return self._transaction_root / ".state-groups" / self._scope_digest

    @property
    def transaction_root(self) -> Path:
        return self._transaction_root

    def add_member(self, store: "FilesystemStateStore") -> None:
        if self._initialized or self._closed:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if store._namespace != self._namespace or store._tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if any(
            member._runtime_domain == store._runtime_domain for member in self._members
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if store not in self._members:
            self._members.append(store)

    async def initialize(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._initialized:
            return
        task = self._initialization_task
        if task is None:
            task = asyncio.create_task(
                self._initialize_owned(),
                name=f"filesystem-initialize-{self._scope_digest}",
            )
            self._initialization_task = task
            self._maintenance_tasks.add(task)
            task.add_done_callback(self._initialization_done)
        await asyncio.shield(task)

    async def _initialize_owned(self) -> None:
        async with self._mutation_lock:
            if self._closed:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            if self._initialized:
                return
            await self._initialize_locked()

    def _initialization_done(self, task: asyncio.Task[None]) -> None:
        if self._initialization_task is task:
            self._initialization_task = None
        self._maintenance_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception(
                "filesystem initialization owner failed: scope=%s",
                self._scope_digest,
            )

    async def _initialize_locked(self) -> None:
        ordered = tuple(
            sorted(self._members, key=lambda member: member.root.as_posix())
        )
        acquired: list[FilesystemWriterLock] = []
        try:
            if self._standalone:
                if len(ordered) != 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await ordered[0]._writer_lock.acquire()
                acquired.append(ordered[0]._writer_lock)
            else:
                await self._group_lock.acquire()
                acquired.append(self._group_lock)
                for member in ordered:
                    await member._writer_lock.acquire()
                    acquired.append(member._writer_lock)
                await asyncio.to_thread(self._validate_roots_sync)
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True
            _logger.info(
                "filesystem StateStorageGroup initialized: scope=%s domains=%s",
                self._scope_digest,
                ",".join(member._runtime_domain for member in ordered),
            )
        except BaseException:
            for lock in reversed(acquired):
                await lock.release()
            raise

    async def close(self) -> None:
        if (
            self._closed
            or not self._members
            or not all(member._closed for member in self._members)
        ):
            return
        if any(not task.done() for task in self._maintenance_tasks):
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        async with self._mutation_lock:
            if self._closed:
                return
            if any(not task.done() for task in self._maintenance_tasks):
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            if any(not task.done() for task in self._pending_physical) or any(
                not task.done()
                for member in self._members
                for task in member._pending_physical
            ):
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            self._closed = True
            self._initialized = False
            locks = tuple(
                sorted(self._members, key=lambda value: value.root.as_posix())
            )
            for member in locks:
                await member._consistency_lock.acquire()
            for member in sorted(
                self._members, key=lambda value: value.root.as_posix(), reverse=True
            ):
                await member._writer_lock.release()
            for member in reversed(locks):
                member._consistency_lock.release()
            if not self._standalone:
                await self._group_lock.release()
        _logger.debug(
            "filesystem StateStorageGroup closed: scope=%s", self._scope_digest
        )

    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        if self._initialized or self._closed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        members = tuple(sorted(self._members, key=lambda value: value.root.as_posix()))
        acquired: list[FilesystemWriterLock] = []
        if not self._standalone:
            await self._group_lock.acquire()
            acquired.append(self._group_lock)
        try:
            for member in members:
                await member._writer_lock.acquire()
                acquired.append(member._writer_lock)
            yield
        finally:
            for lock in reversed(acquired):
                await lock.release()

    async def read(
        self, store: "FilesystemStateStore", fn: StateCallback[ValueT]
    ) -> ValueT:
        self._ensure_member(store)
        active = active_state_transaction(store)
        if active is not None:
            return await fn(active)
        started = monotonic()
        await store._consistency_lock.acquire()
        try:
            self._ensure_member(store)
        except BaseException:
            store._consistency_lock.release()
            raise
        _logger.debug(
            "filesystem member lock acquired: domain=%s member_lock_wait_ms=%.3f",
            store._runtime_domain,
            (monotonic() - started) * 1000,
        )
        transaction = _FilesystemTransaction(store.root, store._require_index())
        token = bind_state_scope(
            self,
            {store: transaction},
            writable=False,
        )
        try:
            readonly = active_state_transaction(store)
            if readonly is None:
                raise RuntimeError("read-only StateTransaction scope was not bound")
            return await fn(readonly)
        finally:
            reset_state_transaction(token)
            store._consistency_lock.release()

    async def mutate(
        self,
        stores: Sequence["FilesystemStateStore"],
        fn: StateGroupCallback[ValueT],
    ) -> ValueT:
        members = tuple(
            sorted(dict.fromkeys(stores), key=lambda value: value.root.as_posix())
        )
        if not members:
            raise ValueError("StateStorageGroup mutation requires a store")
        for store in members:
            self._ensure_member(store)
        active = active_state_transaction(members[0], writable=True)
        if active is not None:
            return await fn(active_state_group_transaction(self, members))
        mutation_started = monotonic()
        await self._mutation_lock.acquire()
        try:
            for store in members:
                self._ensure_member(store)
        except BaseException:
            self._mutation_lock.release()
            raise
        _logger.debug(
            "filesystem group mutation lock acquired: scope=%s "
            "group_mutation_wait_ms=%.3f",
            self._scope_digest,
            (monotonic() - mutation_started) * 1000,
        )
        locked: list[FilesystemStateStore] = []
        try:
            member_wait_started = monotonic()
            for store in members:
                await store._consistency_lock.acquire()
                locked.append(store)
            _logger.debug(
                "filesystem mutation members locked: scope=%s member_lock_wait_ms=%.3f",
                self._scope_digest,
                (monotonic() - member_wait_started) * 1000,
            )
            transaction_now = datetime.now(timezone.utc)
            try:
                transactions = {
                    store: _FilesystemTransaction(
                        store.root,
                        store._require_index(),
                        now=transaction_now,
                    )
                    for store in members
                }
                group_transaction = _FilesystemGroupTransaction(self, transactions)
                token = bind_state_scope(self, transactions)
                try:
                    result = await fn(group_transaction)
                finally:
                    reset_state_transaction(token)
                if any(
                    transaction.has_changes for transaction in transactions.values()
                ):
                    await self._commit(transactions)
                return result
            finally:
                for store in reversed(locked):
                    store._consistency_lock.release()
        finally:
            self._mutation_lock.release()

    async def validate_integrity(self) -> None:
        self._ensure_ready()
        task = asyncio.create_task(
            self._validate_integrity_owned(),
            name=f"filesystem-validate-{self._scope_digest}",
        )
        _track_physical_task(
            self._maintenance_tasks,
            task,
            f"filesystem integrity validation {self._scope_digest}",
        )
        await asyncio.shield(task)

    async def _validate_integrity_owned(self) -> None:
        async with self._mutation_lock:
            self._ensure_ready()
            ordered = tuple(
                sorted(self._members, key=lambda value: value.root.as_posix())
            )
            for member in ordered:
                await member._consistency_lock.acquire()
            try:
                for member in ordered:
                    member._ensure_ready()
                    await asyncio.to_thread(member._validate_integrity_sync)
            finally:
                for member in reversed(ordered):
                    member._consistency_lock.release()

    def _initialize_sync(self) -> None:
        if self._standalone:
            member = self._members[0]
            index, generation = member._initialize_sync()
            member._index = index
            member._index_generation = generation
            self._generation = generation
            return
        self._provision_group_sync()
        self._check_foreign_group_journals_sync()
        for member in self._members:
            member._provision()
            member._recover_sync()
        self._recover_sync()
        for member in self._members:
            member._ensure_record_index()
        for member in self._members:
            member._index = member._new_index()
            member._index_generation = member._generation()
        self._generation = self._read_generation()

    def _provision_group_sync(self) -> None:
        self._metadata_root.mkdir(parents=True, exist_ok=True)
        manifest = self._metadata_root / "manifest.json"
        expected = self._expected_manifest()
        if manifest.exists():
            try:
                actual = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if not self._manifest_matches(actual, expected):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            _write_json(manifest, expected)
            _write_text(self._metadata_root / "generation", "0")
            sync_directory(self._metadata_root)
        if not (self._metadata_root / "generation").is_file():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @staticmethod
    def _manifest_matches(
        actual: Mapping[str, object], expected: Mapping[str, object]
    ) -> bool:
        return isinstance(actual, Mapping) and actual == expected

    def _check_foreign_group_journals_sync(self) -> None:
        if self._standalone:
            return
        member_paths = tuple(
            member.root.relative_to(self._transaction_root).as_posix()
            for member in self._members
        )
        own_name = f".txn-{self._scope_digest}"
        try:
            journals = tuple(
                path
                for path in self._transaction_root.iterdir()
                if path.is_dir()
                and path.name.startswith(".txn-")
                and path.name != own_name
            )
            for journal in journals:
                if not (journal / "commit").is_file():
                    continue
                plan = _read_json(journal / "plan.json")
                paths = tuple(plan.get("writes", ())) + tuple(plan.get("deletes", ()))
                for item in paths:
                    value = item.get("path") if isinstance(item, Mapping) else item
                    if not isinstance(value, str):
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    if any(
                        value == root or value.startswith(root + "/")
                        for root in member_paths
                    ):
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        except AIError:
            raise
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error

    def _expected_manifest(self) -> dict[str, object]:
        return {
            "format": "linktools-ai-state-group",
            "version": 1,
            "namespace_digest": _digest(self._namespace),
            "tenant_digest": _digest(self._tenant_id),
            "members": [
                {
                    "runtime_domain": member._runtime_domain,
                    "relative_path": member.root.relative_to(
                        self._transaction_root
                    ).as_posix(),
                }
                for member in sorted(
                    self._members,
                    key=lambda value: (
                        value._runtime_domain,
                        value.root.relative_to(self._transaction_root).as_posix(),
                    ),
                )
            ],
        }

    def _validate_roots_sync(self) -> None:
        self._transaction_root.mkdir(parents=True, exist_ok=True)
        device = os.stat(self._transaction_root).st_dev
        paths = [member.root for member in self._members]
        if len(paths) != len(set(paths)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for path in paths:
            if path == self._transaction_root:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                relative = path.relative_to(self._transaction_root)
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if (
                not relative.parts
                or relative.parts[0] == ".state-groups"
                or relative.parts[0].startswith(".txn-")
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if os.path.commonpath((self._transaction_root, path)) != str(
                self._transaction_root
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            path.mkdir(parents=True, exist_ok=True)
            if os.stat(path).st_dev != device:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for index, left in enumerate(sorted(paths)):
            for right in sorted(paths)[index + 1 :]:
                if left in right.parents or right in left.parents:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _read_generation(self) -> int:
        return _read_generation_value(self._generation_path)

    def _write_generation(self, value: int) -> None:
        _write_text(self._generation_path, str(value))

    def _recover_sync(self) -> None:
        self._journal.recover(self._read_generation, self._write_generation)

    def _ensure_member(self, store: "FilesystemStateStore") -> None:
        if store.storage_group is not self:
            raise RuntimeError("store does not belong to this StateStorageGroup")
        self._ensure_ready()
        store._ensure_ready()

    def _ensure_ready(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._poisoned:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        if not self._initialized:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def _commit(
        self, transactions: Mapping["FilesystemStateStore", "_FilesystemTransaction"]
    ) -> None:
        if self._standalone:
            member, transaction = next(iter(transactions.items()))
            await member._commit(transaction)
            return
        base = self._generation
        if base is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        target = base + 1
        writes: dict[str, bytes] = {}
        deletes: set[str] = set()
        for store, transaction in transactions.items():
            prefix = (
                ""
                if self._standalone
                else store.root.relative_to(self._transaction_root).as_posix()
            )
            if not self._standalone:
                writes[f"{prefix}/generation"] = str(target).encode("utf-8")
            for relative, value in transaction.writes.items():
                writes[f"{prefix}/{relative}" if prefix else relative] = value
            for relative in transaction.deletes:
                deletes.add(f"{prefix}/{relative}" if prefix else relative)
        started = monotonic()
        physical = asyncio.create_task(
            asyncio.to_thread(self._commit_sync, writes, deletes, base, target),
            name=f"filesystem-group-commit-{self._scope_digest}",
        )
        _track_physical_task(
            self._pending_physical,
            physical,
            f"filesystem group commit {self._scope_digest}",
        )
        cancellation: asyncio.CancelledError | None = None
        error: BaseException | None = None
        try:
            await asyncio.shield(physical)
        except asyncio.CancelledError as cancellation_error:
            cancellation = cancellation_error
            if not physical.done():
                self._poisoned = True
                for member in self._members:
                    member._poisoned = True
                _logger.error(
                    "filesystem group mutation cancelled with unknown outcome: "
                    "scope=%s base=%s target=%s",
                    self._scope_digest,
                    base,
                    target,
                )
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from cancellation_error
            try:
                physical.result()
            except BaseException as commit_error:  # noqa: BLE001
                error = commit_error
        except BaseException as commit_error:  # noqa: BLE001
            error = commit_error
        if error is not None:
            outcome = await self._reconcile_commit(base, target)
            if outcome == "unknown":
                self._poisoned = True
                for member in self._members:
                    member._poisoned = True
                _logger.error(
                    "filesystem group mutation outcome unknown: scope=%s base=%s target=%s",
                    self._scope_digest,
                    base,
                    target,
                )
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error
            if outcome == "not_committed":
                if cancellation is not None:
                    raise cancellation
                raise error
        for store, transaction in transactions.items():
            store._apply_transaction(transaction, generation=target)
        self._generation = target
        _logger.debug(
            "state storage group committed: backend=filesystem domains=%s "
            "group_commit_ms=%.3f files=%s",
            ",".join(store._runtime_domain for store in transactions),
            (monotonic() - started) * 1000,
            len(writes) + len(deletes),
        )
        if cancellation is not None:
            raise cancellation

    def _commit_sync(
        self,
        writes: Mapping[str, bytes],
        deletes: Sequence[str],
        base: int,
        target: int,
    ) -> None:
        plan = self._journal.stage(
            writes, deletes, base_generation=base, target_generation=target
        )
        self._journal.publish(plan)
        self._write_generation(target)
        sync_directory(self._transaction_root)
        self._journal.complete()

    async def _reconcile_commit(self, base: int, target: int) -> _CommitOutcome:
        task = asyncio.create_task(
            asyncio.to_thread(self._reconcile_sync, base, target),
            name=f"filesystem-group-reconcile-{self._scope_digest}",
        )
        _track_physical_task(
            self._pending_physical,
            task,
            f"filesystem group reconcile {self._scope_digest}",
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if not task.done():
                self._poisoned = True
                for member in self._members:
                    member._poisoned = True
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error
            return task.result()

    def _reconcile_sync(self, base: int, target: int) -> _CommitOutcome:
        try:
            self._recover_sync()
            generation = self._read_generation()
        except BaseException:  # noqa: BLE001
            return "unknown"
        if generation == target:
            return "committed"
        if generation == base:
            return "not_committed"
        return "unknown"


class FilesystemStateStore:
    """A domain-local StateStore with crash-safe granular commits."""

    def __init__(
        self,
        root: str | Path,
        *,
        namespace: str,
        tenant_id: str,
        runtime_domain: str,
        _range_index: bool = False,
        group: FilesystemStateStorageGroup | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._namespace = namespace
        self._tenant_id = tenant_id
        if not isinstance(_range_index, bool):
            raise TypeError("_range_index must be bool")
        self._runtime_domain = runtime_domain
        self._range_index_enabled = _range_index
        self._writer_lock = FilesystemWriterLock(self._root / "state.lock")
        self._consistency_lock = asyncio.Lock()
        self._journal = FilesystemJournal(
            self._root,
            error_code=ErrorCode.STORAGE_INTEGRITY_ERROR,
        )
        self._closed = False
        self._poisoned = False
        self._initialized = False
        self._close_task: asyncio.Task[None] | None = None
        self._pending_physical: set[asyncio.Task[None]] = set()
        self._index: _FilesystemIndex | None = None
        self._index_generation: int | None = None
        self._storage_group = group or FilesystemStateStorageGroup(
            self._root,
            namespace=namespace,
            tenant_id=tenant_id,
            scope_digest=f"standalone-{runtime_domain}",
            standalone=True,
        )
        self._storage_group.add_member(self)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def storage_group(self) -> StateStorageGroup:
        return self._storage_group

    async def initialize(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        await self._storage_group.initialize()
        self._initialized = True
        _logger.info(
            "filesystem StateStore initialized: domain=%s root=%s",
            self._runtime_domain,
            self._root,
        )

    async def close(self) -> None:
        task = self._close_task
        retry = task is None
        if task is not None and task.done():
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                retry = True
            else:
                return
        if retry:
            self._closed = True
            self._initialized = False
            task = asyncio.create_task(self._close_inner())
            self._close_task = task
        if task is None:
            raise RuntimeError("filesystem close task was not created")
        await asyncio.shield(task)

    async def read(self, fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_ready()
        active = active_state_transaction(self)
        if active is not None:
            return await fn(active)
        return await self._storage_group.read(self, fn)

    async def mutate(self, fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_ready()
        active = active_state_transaction(self, writable=True)
        if active is not None:
            return await fn(active)
        return await self._storage_group.mutate(
            (self,), lambda group: fn(group.transaction(self))
        )

    async def validate_integrity(self) -> None:
        await self._storage_group.validate_integrity()

    def _ensure_ready(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._poisoned:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        if not self._initialized:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    def _require_index(self) -> _FilesystemIndex:
        if self._index is None or self._index_generation is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return self._index

    async def _close_inner(self) -> None:
        await self._storage_group.close()
        self._index = None
        self._index_generation = None
        _logger.debug("filesystem StateStore closed: domain=%s", self._runtime_domain)

    def _initialize_sync(self) -> tuple[_FilesystemIndex, int]:
        self._provision()
        self._recover_sync()
        self._ensure_record_index()
        index = self._new_index()
        generation = self._generation()
        return index, generation

    def _new_index(self) -> _FilesystemIndex:
        return _FilesystemIndex({}, {}, {}, {}, {}, _FilesystemCache(self._root))

    def _validate_integrity_sync(self) -> None:
        index = self._load_index()
        self._validate_index(index, decode_items=True)

    def _generation(self) -> int:
        if not self._root.exists():
            return 0
        if self._root.is_dir() and not any(self._root.iterdir()):
            return 0
        if self._root.is_dir() and all(
            path.name == "state.lock" for path in self._root.iterdir()
        ):
            return 0
        return _read_generation_value(self._root / "generation")

    def _expected_manifest(self) -> dict[str, str | int]:
        return {
            "format": "linktools-ai-state",
            "layout_version": 1,
            "namespace_digest": _digest(self._namespace),
            "tenant_digest": _digest(self._tenant_id),
            "runtime_domain": self._runtime_domain,
        }

    def _validate_existing_root(self) -> None:
        if not self._root.is_dir():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        manifest = self._root / "manifest.json"
        if not manifest.is_file():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            actual = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if actual != self._expected_manifest():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        generation = self._root / "generation"
        if not generation.is_file():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._generation()

    def _provision(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._root.is_dir():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        manifest = self._root / "manifest.json"
        if manifest.exists():
            self._validate_existing_root()
            return
        unexpected = [
            path for path in self._root.iterdir() if path.name != "state.lock"
        ]
        if unexpected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _write_json(manifest, self._expected_manifest())
        _write_text(self._root / "generation", "0")
        if self._range_index_enabled:
            marker = self._root / _RECORD_INDEX_MARKER
            _write_text(marker, _RECORD_INDEX_VERSION)
            sync_directory(marker.parent)
        sync_directory(self._root)

    def _ensure_record_index(self) -> None:
        index_root = self._root / "record-index"
        if not self._range_index_enabled:
            if index_root.exists():
                shutil.rmtree(index_root)
                sync_directory(self._root)
            return
        if _record_index_marker_valid(self._root):
            return
        if index_root.exists():
            shutil.rmtree(index_root)
            sync_directory(self._root)
        index_root.mkdir(parents=True, exist_ok=True)
        records: list[StoredRecord] = []
        for source in (self._root / "records").glob("*/*/*.json"):
            record = decode_record(_read_json(source))
            _require_layout_path(source, self._root, _record_path(record))
            records.append(record)
        nodes = _build_record_index_nodes(tuple(records))
        for relative, node in nodes.items():
            _write_json(self._root / relative, _record_index_node_payload(node))
        _sync_record_index_tree(index_root)
        marker = self._root / _RECORD_INDEX_MARKER
        _write_text(marker, _RECORD_INDEX_VERSION)
        sync_directory(index_root)
        sync_directory(self._root)

    def _load_index(self) -> _FilesystemIndex:
        try:
            records: dict[bytes, StoredRecord] = {}
            aliases: dict[bytes, bytes] = {}
            sequences: dict[bytes, int] = {}
            operations: dict[bytes, StoredOperation] = {}
            for path in (self._root / "records").glob("*/*/*.json"):
                value = decode_record(_read_json(path))
                _require_layout_path(
                    path,
                    self._root,
                    f"records/{value.kind}/{value.key_digest.hex()[:2]}/{value.key_digest.hex()}.json",
                )
                if value.key_digest in records:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                records[value.key_digest] = value
            for path in (self._root / "aliases").glob("*/*.json"):
                value = decode_alias(_read_json(path))
                _require_layout_path(
                    path,
                    self._root,
                    f"aliases/{value.alias_digest.hex()[:2]}/{value.alias_digest.hex()}.json",
                )
                if value.alias_digest in aliases:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                aliases[value.alias_digest] = value.record_key_digest
            for path in (self._root / "sequences").glob("*/*.json"):
                key, value = _read_sequence_metadata(path)
                _require_layout_path(
                    path, self._root, f"sequences/{key.hex()[:2]}/{key.hex()}.json"
                )
                if key in sequences:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                sequences[key] = value
            for path in (self._root / "operations/by-key").glob("*/*.json"):
                value = decode_operation(_read_json(path))
                key = value.key_digest.hex()
                _require_layout_path(
                    path, self._root, f"operations/by-key/{key[:2]}/{key}.json"
                )
                if value.key_digest in operations:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                operations[value.key_digest] = value
            streams: dict[bytes, _FactStreamInfo] = {}
            facts_root = self._root / "facts"
            for path in facts_root.glob("*/*/meta.json"):
                stream, owner, last_sequence = _read_fact_metadata(path)
                _require_layout_path(
                    path,
                    self._root,
                    f"facts/{stream.hex()[:2]}/{stream.hex()}/meta.json",
                )
                if stream in streams:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                subjects: dict[bytes, int] = {}
                for ref in path.parent.joinpath("subjects").glob("*.ref"):
                    subject = _layout_digest(ref.stem)
                    sequence = _read_subject_sequence(ref)
                    _require_layout_path(
                        ref, self._root, _fact_subject_path(self._root, stream, subject)
                    )
                    if subject in subjects or not 1 <= sequence <= last_sequence:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    subjects[subject] = sequence
                streams[stream] = _FactStreamInfo(
                    stream, owner, last_sequence, subjects
                )
            references: dict[tuple[bytes, int], bytes] = {}
            for path in (self._root / "operations/streams").glob("*/*/*.ref"):
                stream = _layout_digest(path.parent.name)
                sequence = _layout_sequence_name(path.stem)
                key = _read_operation_ref(path)
                _require_layout_path(
                    path,
                    self._root,
                    f"operations/streams/{stream.hex()[:2]}/{stream.hex()}/{sequence:020d}.ref",
                )
                references[(stream, sequence)] = key
            expected = {
                (value.stream_digest, value.sequence): value.key_digest
                for value in operations.values()
            }
            if references != expected:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            index = _FilesystemIndex(
                records,
                aliases,
                sequences,
                streams,
                operations,
                _FilesystemCache(self._root),
            )
            self._validate_index(index, decode_items=False)
            return index
        except AIError:
            raise
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _validate_index(self, index: _FilesystemIndex, *, decode_items: bool) -> None:
        if decode_items and _record_index_marker_valid(self._root):
            expected_nodes = _build_record_index_nodes(tuple(index.records.values()))
            expected_record_index = {_RECORD_INDEX_MARKER, *expected_nodes}
            actual_record_index = {
                path.relative_to(self._root).as_posix()
                for path in (self._root / "record-index").rglob("*")
                if path.is_file()
            }
            if actual_record_index != expected_record_index:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for relative, expected in expected_nodes.items():
                actual = _read_record_index_node_path(
                    self._root / relative,
                    expected_token=expected.token,
                )
                if actual != expected:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for key in index.aliases.values():
            if key not in index.records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for info in index.fact_streams.values():
            if info.owner_key_digest not in index.records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if info.last_sequence < 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if decode_items:
                latest: dict[bytes, int] = {}
                for sequence in range(1, info.last_sequence + 1):
                    fact = decode_fact(
                        _read_json(
                            _fact_item_path(self._root, info.stream_digest, sequence)
                        )
                    )
                    if (
                        fact.stream_digest != info.stream_digest
                        or fact.sequence != sequence
                        or fact.owner_key_digest != info.owner_key_digest
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if fact.subject_digest is not None:
                        latest[fact.subject_digest] = sequence
                if latest != info.subjects:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if decode_items:
            expected_fact_files = {
                _fact_meta_path(self._root, info.stream_digest)
                for info in index.fact_streams.values()
            }
            expected_fact_files.update(
                _fact_item_path(self._root, info.stream_digest, sequence)
                .relative_to(self._root)
                .as_posix()
                for info in index.fact_streams.values()
                for sequence in range(1, info.last_sequence + 1)
            )
            expected_fact_files.update(
                _fact_subject_path(self._root, info.stream_digest, subject)
                for info in index.fact_streams.values()
                for subject in info.subjects
            )
            actual_fact_files = {
                path.relative_to(self._root).as_posix()
                for path in (self._root / "facts").rglob("*")
                if path.is_file()
            }
            if actual_fact_files != expected_fact_files:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _commit(self, transaction: "_FilesystemTransaction") -> None:
        if not transaction.writes and not transaction.deletes:
            return
        started = monotonic()
        files_written = len(transaction.writes)
        files_deleted = len(transaction.deletes)
        bytes_written = sum(len(value) for value in transaction.writes.values())
        base = self._index_generation
        if base is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        target = base + 1
        physical = asyncio.create_task(
            asyncio.to_thread(self._commit_sync, transaction, base, target),
            name=f"filesystem-commit-{self._runtime_domain}",
        )
        _track_physical_task(
            self._pending_physical,
            physical,
            f"filesystem commit {self._runtime_domain}",
        )
        cancellation: asyncio.CancelledError | None = None
        physical_error: BaseException | None = None
        try:
            await asyncio.shield(physical)
        except asyncio.CancelledError as error:
            cancellation = error
            if not physical.done():
                self._poisoned = True
                _logger.error(
                    "filesystem mutation cancelled with unknown outcome: "
                    "domain=%s base=%s target=%s",
                    self._runtime_domain,
                    base,
                    target,
                )
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error
            try:
                physical.result()
            except BaseException as commit_error:  # noqa: BLE001
                physical_error = commit_error
        except BaseException as error:  # noqa: BLE001
            physical_error = error

        if physical_error is not None:
            outcome = await self._reconcile_commit(base, target)
            if outcome == "unknown":
                self._poisoned = True
                _logger.error(
                    "filesystem mutation outcome unknown: domain=%s base=%s target=%s",
                    self._runtime_domain,
                    base,
                    target,
                )
                if cancellation is not None:
                    raise cancellation
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from physical_error
            if outcome == "not_committed":
                if cancellation is not None:
                    raise cancellation
                raise physical_error
            _logger.warning(
                "filesystem mutation recovered after commit error: domain=%s generation=%s",
                self._runtime_domain,
                target,
            )
        self._apply_transaction(transaction, target)
        _logger.debug(
            "filesystem mutation committed: domain=%s generation=%s duration_ms=%.3f "
            "files_written=%s files_deleted=%s bytes_written=%s outcome=%s",
            self._runtime_domain,
            target,
            (monotonic() - started) * 1000,
            files_written,
            files_deleted,
            bytes_written,
            "recovered" if physical_error is not None else "committed",
        )
        if cancellation is not None:
            raise cancellation

    def _commit_sync(
        self,
        transaction: "_FilesystemTransaction",
        base: int,
        target: int,
    ) -> None:
        plan = self._journal.stage(
            transaction.writes,
            transaction.deletes,
            base_generation=base,
            target_generation=target,
        )
        self._journal.publish(plan)
        _write_text(self._root / "generation", str(target))
        sync_directory(self._root)
        self._journal.complete()

    async def _reconcile_commit(self, base: int, target: int) -> _CommitOutcome:
        task = asyncio.create_task(
            asyncio.to_thread(self._reconcile_commit_sync, base, target),
            name=f"filesystem-reconcile-{self._runtime_domain}",
        )
        _track_physical_task(
            self._pending_physical,
            task,
            f"filesystem reconcile {self._runtime_domain}",
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if not task.done():
                self._poisoned = True
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error
            return task.result()

    def _reconcile_commit_sync(self, base: int, target: int) -> _CommitOutcome:
        try:
            self._journal.recover(
                self._generation,
                lambda value: _write_text(self._root / "generation", str(value)),
            )
            generation = self._generation()
        except Exception:  # noqa: BLE001
            return "unknown"
        if generation == target:
            return "committed"
        if generation == base:
            return "not_committed"
        return "unknown"

    def _apply_transaction(
        self,
        transaction: "_FilesystemTransaction",
        generation: int | None,
    ) -> None:
        index = self._require_index()
        old_record_kinds = {
            key: index.records[key].kind
            for key in set(transaction.records.changes())
            | set(transaction.records.deleted())
            if key in index.records
        }
        transaction.records.apply_to(index.records)
        transaction.aliases.apply_to(index.aliases)
        transaction.sequences.apply_to(index.sequences)
        transaction.fact_streams.apply_to(index.fact_streams)
        transaction.operations.apply_to(index.operations)
        for key in transaction.records.deleted():
            index.cache.set_record(key, None, old_kind=old_record_kinds.get(key))
        for key, value in transaction.records.changes().items():
            index.cache.set_record(key, value, old_kind=old_record_kinds.get(key))
        for key in transaction.aliases.deleted():
            index.cache.set_alias(key, None)
        for key, value in transaction.aliases.changes().items():
            index.cache.set_alias(key, value)
        for key in transaction.sequences.deleted():
            index.cache.set_sequence(key, 0)
        for key, value in transaction.sequences.changes().items():
            index.cache.set_sequence(key, value)
        for key in transaction.fact_streams.deleted():
            index.cache.set_fact_stream(key, None)
        for key, value in transaction.fact_streams.changes().items():
            index.cache.set_fact_stream(key, value)
        for key in transaction.operations.deleted():
            index.cache.set_operation(key, None)
        for key, value in transaction.operations.changes().items():
            index.cache.set_operation(key, value)
        if _RECORD_INDEX_MARKER in transaction.deletes:
            index.cache.disable_record_index()
        if generation is not None:
            self._index_generation = generation

    def _recover_sync(self) -> None:
        self._journal.recover(
            self._generation,
            lambda target: _write_text(self._root / "generation", str(target)),
        )


__all__ = ["FilesystemStateStorageGroup", "FilesystemStateStore"]
