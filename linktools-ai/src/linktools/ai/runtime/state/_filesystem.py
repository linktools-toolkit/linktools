#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Granular, journaled filesystem implementation of StateStore."""

import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Literal, TypeVar

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import FilesystemJournal, FilesystemWriterLock, sync_directory
from ._codec import (
    decode_alias,
    decode_fact,
    decode_operation,
    decode_record,
    encode_alias,
    encode_fact,
    encode_operation,
    encode_record,
)
from ._store import (
    FactQuery,
    OperationQuery,
    RecordQuery,
    StateCallback,
    StateGroupCallback,
    StateStorageGroup,
    StateTransaction,
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    active_state_transaction,
    active_state_group_transaction,
    bind_state_scope,
    reset_state_transaction,
    validate_record_identity,
    validate_record_replacement,
)

ValueT = TypeVar("ValueT")
KeyT = TypeVar("KeyT")
MapValueT = TypeVar("MapValueT")
_logger = environ.get_logger("ai.runtime.state.filesystem")
_CommitOutcome = Literal["committed", "not_committed", "unknown"]


@dataclass(slots=True)
class _FactStreamInfo:
    stream_digest: bytes
    owner_key_digest: bytes
    last_sequence: int
    subjects: dict[bytes, int]


@dataclass(slots=True)
class _FilesystemIndex:
    records: dict[bytes, StoredRecord]
    aliases: dict[bytes, bytes]
    sequences: dict[bytes, int]
    fact_streams: dict[bytes, _FactStreamInfo]
    operations: dict[bytes, StoredOperation]


class _CowMap(MutableMapping[KeyT, MapValueT]):
    def __init__(self, base: Mapping[KeyT, MapValueT]) -> None:
        self._base = base
        self._changes: dict[KeyT, MapValueT] = {}
        self._deleted: set[KeyT] = set()

    def __getitem__(self, key: KeyT) -> MapValueT:
        if key in self._changes:
            return self._changes[key]
        if key in self._deleted:
            raise KeyError(key)
        return self._base[key]

    def __setitem__(self, key: KeyT, value: MapValueT) -> None:
        self._deleted.discard(key)
        self._changes[key] = value

    def __delitem__(self, key: KeyT) -> None:
        if key not in self and key not in self._changes:
            raise KeyError(key)
        self._changes.pop(key, None)
        self._deleted.add(key)

    def __iter__(self) -> Iterator[KeyT]:
        keys = set(self._base) | set(self._changes)
        return iter(keys - self._deleted)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def changes(self) -> Mapping[KeyT, MapValueT]:
        return self._changes

    def deleted(self) -> frozenset[KeyT]:
        return frozenset(self._deleted)

    def apply_to(self, target: MutableMapping[KeyT, MapValueT]) -> None:
        for key in self._deleted:
            target.pop(key, None)
        target.update(self._changes)


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
            raise RuntimeError("store was not enlisted in the StateStorageGroup transaction") from error


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
        self._operation_lock = asyncio.Lock()
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
        if any(member._runtime_domain == store._runtime_domain for member in self._members):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if store not in self._members:
            self._members.append(store)

    async def initialize(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._initialized:
            return
        ordered = tuple(sorted(self._members, key=lambda member: member.root.as_posix()))
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
                await _await_thread(self._validate_roots_sync)
            await _await_thread(self._initialize_sync)
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
        if self._closed or not self._members or not all(member._closed for member in self._members):
            return
        async with self._operation_lock:
            if self._closed:
                return
            self._closed = True
            self._initialized = False
            for member in sorted(self._members, key=lambda value: value.root.as_posix(), reverse=True):
                await member._writer_lock.release()
            if not self._standalone:
                await self._group_lock.release()
        _logger.debug("filesystem StateStorageGroup closed: scope=%s", self._scope_digest)

    async def read(self, store: "FilesystemStateStore", fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_member(store)
        active = active_state_transaction(store)
        if active is not None:
            return await fn(active)
        async with self._operation_lock:
            return await fn(_FilesystemTransaction(store.root, store._require_index()))

    async def mutate(
        self,
        stores: Sequence["FilesystemStateStore"],
        fn: StateGroupCallback[ValueT],
    ) -> ValueT:
        members = tuple(dict.fromkeys(stores))
        if not members:
            raise ValueError("StateStorageGroup mutation requires a store")
        for store in members:
            self._ensure_member(store)
        active = active_state_transaction(members[0])
        if active is not None:
            return await fn(active_state_group_transaction(self, members))
        async with self._operation_lock:
            transaction_now = datetime.now(timezone.utc)
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
            if any(transaction.has_changes for transaction in transactions.values()):
                await self._commit(transactions)
            return result

    async def validate_integrity(self) -> None:
        self._ensure_ready()
        async with self._operation_lock:
            for member in self._members:
                member._ensure_ready()
                await _await_thread(
                    lambda member=member: member._validate_index(
                        member._require_index(),
                        decode_items=True,
                    )
                )

    def _initialize_sync(self) -> None:
        if self._standalone:
            member = self._members[0]
            index, generation = member._initialize_sync()
            member._index = index
            member._index_generation = generation
            self._generation = generation
            return
        self._provision_group_sync()
        for member in self._members:
            member._provision()
            member._recover_sync()
        self._recover_sync()
        for member in self._members:
            member._index = member._load_index()
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
            if actual != expected:
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        else:
            _write_json(manifest, expected)
            _write_text(self._metadata_root / "generation", "0")
            sync_directory(self._metadata_root)
        if not (self._metadata_root / "generation").is_file():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _expected_manifest(self) -> dict[str, object]:
        return {
            "format": "linktools-ai-state-group",
            "version": 1,
            "transaction_root": self._transaction_root.as_posix(),
            "namespace_digest": _digest(self._namespace),
            "tenant_digest": _digest(self._tenant_id),
            "members": [
                {
                    "runtime_domain": member._runtime_domain,
                    "relative_path": member.root.relative_to(self._transaction_root).as_posix(),
                }
                for member in sorted(self._members, key=lambda value: value.root.as_posix())
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
            if not relative.parts or relative.parts[0] == ".state-groups" or relative.parts[0].startswith(".txn-"):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if os.path.commonpath((self._transaction_root, path)) != str(self._transaction_root):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            path.mkdir(parents=True, exist_ok=True)
            if os.stat(path).st_dev != device:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for index, left in enumerate(sorted(paths)):
            for right in sorted(paths)[index + 1 :]:
                if left in right.parents or right in left.parents:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _read_generation(self) -> int:
        try:
            return int(self._generation_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

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

    async def _commit(self, transactions: Mapping["FilesystemStateStore", "_FilesystemTransaction"]) -> None:
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
            prefix = "" if self._standalone else store.root.relative_to(self._transaction_root).as_posix()
            for relative, value in transaction.writes.items():
                writes[f"{prefix}/{relative}" if prefix else relative] = value
            for relative in transaction.deletes:
                deletes.add(f"{prefix}/{relative}" if prefix else relative)
        started = monotonic()
        physical = asyncio.create_task(
            asyncio.to_thread(self._commit_sync, writes, deletes, base, target),
            name=f"filesystem-group-commit-{self._scope_digest}",
        )
        cancellation: asyncio.CancelledError | None = None
        error: BaseException | None = None
        try:
            await asyncio.shield(physical)
        except asyncio.CancelledError as cancellation_error:
            cancellation = cancellation_error
            try:
                await asyncio.shield(physical)
            except BaseException as commit_error:
                error = commit_error
        except BaseException as commit_error:
            error = commit_error
        if error is not None:
            outcome = await _await_thread(lambda: self._reconcile_sync(base, target))
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
            store._apply_transaction(transaction, generation=None)
        self._generation = target
        _logger.debug(
            "state storage group committed: backend=filesystem domains=%s duration_ms=%.3f files=%s",
            ",".join(store._runtime_domain for store in transactions),
            (monotonic() - started) * 1000,
            len(writes) + len(deletes),
        )
        if cancellation is not None:
            raise cancellation

    def _commit_sync(self, writes: Mapping[str, bytes], deletes: Sequence[str], base: int, target: int) -> None:
        plan = self._journal.stage(writes, deletes, base_generation=base, target_generation=target)
        self._journal.publish(plan)
        self._write_generation(target)
        sync_directory(self._transaction_root)
        self._journal.complete()

    def _reconcile_sync(self, base: int, target: int) -> _CommitOutcome:
        try:
            self._recover_sync()
            generation = self._read_generation()
        except BaseException:
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
        group: FilesystemStateStorageGroup | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._runtime_domain = runtime_domain
        self._writer_lock = FilesystemWriterLock(self._root / "state.lock")
        self._operation_lock = asyncio.Lock()
        self._journal = FilesystemJournal(
            self._root,
            error_code=ErrorCode.STORAGE_INTEGRITY_ERROR,
        )
        self._closed = False
        self._poisoned = False
        self._initialized = False
        self._close_task: asyncio.Task[None] | None = None
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
        if self._close_task is None:
            self._closed = True
            self._initialized = False
            self._close_task = asyncio.create_task(self._close_inner())
        task = self._close_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise

    async def read(self, fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_ready()
        active = active_state_transaction(self)
        if active is not None:
            return await fn(active)
        return await self._storage_group.read(self, fn)

    async def mutate(self, fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_ready()
        active = active_state_transaction(self)
        if active is not None:
            return await fn(active)
        return await self._storage_group.mutate((self,), lambda group: fn(group.transaction(self)))

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
        index = self._load_index()
        generation = self._generation()
        self._validate_index(index, decode_items=False)
        return index, generation

    def _generation(self) -> int:
        if not self._root.exists():
            return 0
        if self._root.is_dir() and not any(self._root.iterdir()):
            return 0
        if self._root.is_dir() and all(path.name == "state.lock" for path in self._root.iterdir()):
            return 0
        try:
            return int((self._root / "generation").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _expected_manifest(self) -> dict[str, str | int]:
        return {
            "format": "linktools-ai-state",
            "layout_version": 3,
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
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
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
        unexpected = [path for path in self._root.iterdir() if path.name != "state.lock"]
        if unexpected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _write_json(manifest, self._expected_manifest())
        _write_text(self._root / "generation", "0")
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
                raw = _read_json(path)
                key = bytes.fromhex(str(raw["key"]))
                _require_layout_path(path, self._root, f"sequences/{key.hex()[:2]}/{key.hex()}.json")
                value = int(raw["value"])
                if len(key) != 32 or value < 0 or key in sequences:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                sequences[key] = value
            for path in (self._root / "operations/by-key").glob("*/*.json"):
                value = decode_operation(_read_json(path))
                key = value.key_digest.hex()
                _require_layout_path(path, self._root, f"operations/by-key/{key[:2]}/{key}.json")
                if value.key_digest in operations:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                operations[value.key_digest] = value
            streams: dict[bytes, _FactStreamInfo] = {}
            facts_root = self._root / "facts"
            for path in facts_root.glob("*/*/meta.json"):
                raw = _read_json(path)
                stream = bytes.fromhex(str(raw["stream"]))
                owner = bytes.fromhex(str(raw["owner_key"]))
                _require_layout_path(path, self._root, f"facts/{stream.hex()[:2]}/{stream.hex()}/meta.json")
                if len(stream) != 32 or len(owner) != 32 or stream in streams:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                last_sequence = int(raw["last_sequence"])
                if last_sequence < 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                subjects: dict[bytes, int] = {}
                for ref in path.parent.joinpath("subjects").glob("*.ref"):
                    if ref.stem == "none":
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    subject = bytes.fromhex(ref.stem)
                    sequence = int(_read_json(ref)["sequence"])
                    if len(subject) != 32 or subject in subjects or not 1 <= sequence <= last_sequence:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    subjects[subject] = sequence
                streams[stream] = _FactStreamInfo(stream, owner, last_sequence, subjects)
            references: dict[tuple[bytes, int], bytes] = {}
            for path in (self._root / "operations/streams").glob("*/*/*.ref"):
                stream = bytes.fromhex(path.parent.name)
                sequence = int(path.stem)
                key = bytes.fromhex(str(_read_json(path)["key"]))
                _require_layout_path(
                    path,
                    self._root,
                    f"operations/streams/{stream.hex()[:2]}/{stream.hex()}/{sequence:020d}.ref",
                )
                references[(stream, sequence)] = key
            expected = {
                (value.stream_digest, value.sequence): value.key_digest for value in operations.values()
            }
            if references != expected:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            index = _FilesystemIndex(records, aliases, sequences, streams, operations)
            self._validate_index(index, decode_items=False)
            return index
        except AIError:
            raise
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _validate_index(self, index: _FilesystemIndex, *, decode_items: bool) -> None:
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
                    fact = decode_fact(_read_json(_fact_item_path(self._root, info.stream_digest, sequence)))
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
                _fact_item_path(self._root, info.stream_digest, sequence).relative_to(self._root).as_posix()
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
        cancellation: asyncio.CancelledError | None = None
        physical_error: BaseException | None = None
        try:
            await asyncio.shield(physical)
        except asyncio.CancelledError as error:
            cancellation = error
            try:
                await asyncio.shield(physical)
            except BaseException as commit_error:
                physical_error = commit_error
        except BaseException as error:
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
        return await _await_thread(lambda: self._reconcile_commit_sync(base, target))

    def _reconcile_commit_sync(self, base: int, target: int) -> _CommitOutcome:
        try:
            self._journal.recover(
                self._generation,
                lambda value: _write_text(self._root / "generation", str(value)),
            )
            generation = self._generation()
        except Exception:
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
        transaction.records.apply_to(index.records)
        transaction.aliases.apply_to(index.aliases)
        transaction.sequences.apply_to(index.sequences)
        transaction.fact_streams.apply_to(index.fact_streams)
        transaction.operations.apply_to(index.operations)
        if generation is not None:
            self._index_generation = generation

    def _recover_sync(self) -> None:
        self._journal.recover(
            self._generation,
            lambda target: _write_text(self._root / "generation", str(target)),
        )


class _FilesystemTransaction:
    def __init__(
        self,
        root: Path,
        index: _FilesystemIndex,
        *,
        now: datetime | None = None,
    ) -> None:
        self._root = root
        self.records = _CowMap(index.records)
        self.aliases = _CowMap(index.aliases)
        self.sequences = _CowMap(index.sequences)
        self.operations = _CowMap(index.operations)
        self.fact_streams = _CowMap(index.fact_streams)
        self._owned_fact_streams: set[bytes] = set()
        self._facts: dict[tuple[bytes, int], StoredFact] = {}
        self._deleted_facts: set[tuple[bytes, int]] = set()
        self.guarded_record_keys: set[bytes] = set()
        self._now = now
        self.writes: dict[str, bytes] = {}
        self.deletes: set[str] = set()

    @property
    def has_changes(self) -> bool:
        return bool(
            self.records.changes()
            or self.records.deleted()
            or self.aliases.changes()
            or self.aliases.deleted()
            or self.sequences.changes()
            or self.sequences.deleted()
            or self.fact_streams.changes()
            or self.fact_streams.deleted()
            or self.operations.changes()
            or self.operations.deleted()
            or self.writes
            or self.deletes
        )

    async def now(self) -> datetime:
        if self._now is None:
            self._now = datetime.now(timezone.utc)
        return self._now

    async def validate_integrity(self) -> None:
        for key in self.aliases.values():
            if key not in self.records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for info in self.fact_streams.values():
            if info.owner_key_digest not in self.records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def get_record(self, key: bytes) -> StoredRecord | None:
        return self.records.get(key)

    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        return {key: self.records[key] for key in keys if key in self.records}

    async def insert_record(self, record: StoredRecord) -> None:
        validate_record_identity(record)
        if record.key_digest in self.records:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self.records[record.key_digest] = record
        self.guarded_record_keys.add(record.key_digest)
        self._write(_record_path(record), encode_record(record))

    async def guard_record(self, key: bytes, *, expected_storage_version: int) -> StoredRecord | None:
        current = self.records.get(key)
        if key in self.guarded_record_keys:
            return current
        if current is None or current.storage_version != expected_storage_version:
            return None
        guarded = StoredRecord(
            current.key_digest,
            current.partition_digest,
            current.scope_digest,
            current.parent_digest,
            current.kind,
            current.sort_key,
            current.state,
            expected_storage_version + 1,
            current.lease_owner,
            current.lease_fence,
            current.lease_expires_at,
            current.data,
        )
        self.records[key] = guarded
        self.guarded_record_keys.add(key)
        self._write(_record_path(guarded), encode_record(guarded))
        return guarded

    async def replace_record(self, record: StoredRecord, *, expected_storage_version: int) -> bool:
        current = self.records.get(record.key_digest)
        if current is None or current.storage_version != expected_storage_version:
            return False
        validate_record_replacement(current, record)
        validate_record_identity(record)
        if record.storage_version != expected_storage_version + 1:
            raise ValueError("replacement must increment storage_version exactly once")
        self.records[record.key_digest] = record
        self.guarded_record_keys.add(record.key_digest)
        self._write(_record_path(record), encode_record(record))
        return True

    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool:
        current = self.records.get(key)
        if current is None or current.storage_version != expected_storage_version:
            return False
        if lease_fence < 0 or lease_expires_at is not None and lease_expires_at.tzinfo is None:
            raise ValueError("record lease is invalid")
        updated = StoredRecord(
            current.key_digest,
            current.partition_digest,
            current.scope_digest,
            current.parent_digest,
            current.kind,
            current.sort_key,
            current.state,
            expected_storage_version + 1,
            lease_owner,
            lease_fence,
            lease_expires_at,
            current.data,
        )
        self.records[key] = updated
        self.guarded_record_keys.add(key)
        self._write(_record_path(updated), encode_record(updated))
        return True

    async def delete_record(self, key: bytes, *, expected_storage_version: int | None = None) -> bool:
        current = self.records.get(key)
        if current is None:
            return False
        expected = current.storage_version if expected_storage_version is None else expected_storage_version
        if await self.guard_record(key, expected_storage_version=expected) is None:
            return False
        await self.delete_fact_streams(key)
        for alias, record_key in tuple(self.aliases.items()):
            if record_key == key:
                del self.aliases[alias]
                self._delete(_alias_path(self._root, alias))
        del self.records[key]
        self.guarded_record_keys.discard(key)
        self._delete(_record_path(current))
        return True

    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        values = [record for record in self.records.values() if _matches_record(record, query)]
        values.sort(key=lambda record: (record.sort_key, record.key_digest))
        if query.after_sort_key is not None and query.after_key_digest is not None:
            values = [
                record
                for record in values
                if (record.sort_key, record.key_digest) > (query.after_sort_key, query.after_key_digest)
            ]
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def resolve_alias(self, alias: bytes) -> bytes | None:
        return (await self.resolve_aliases((alias,))).get(alias)

    async def resolve_aliases(self, aliases: Sequence[bytes]) -> Mapping[bytes, bytes]:
        values = {alias: self.aliases[alias] for alias in dict.fromkeys(aliases) if alias in self.aliases}
        for value in values.values():
            if value not in self.records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values

    async def insert_alias(self, alias: StoredAlias) -> None:
        current = self.aliases.get(alias.alias_digest)
        if current is not None and current != alias.record_key_digest:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if alias.record_key_digest not in self.guarded_record_keys:
            raise RuntimeError("alias owner must be guarded in the current transaction")
        self.aliases[alias.alias_digest] = alias.record_key_digest
        self._write(_alias_path(self._root, alias.alias_digest), encode_alias(alias))

    async def get_sequence(self, key: bytes) -> int:
        return self.sequences.get(key, 0)

    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]:
        return {key: self.sequences.get(key, 0) for key in keys}

    async def next_sequence(self, key: bytes) -> int:
        value = self.sequences.get(key, 0) + 1
        self.sequences[key] = value
        self._write(_sequence_path(self._root, key), {"key": key.hex(), "value": value})
        return value

    async def reserve_sequence(self, key: bytes, count: int) -> int:
        if count < 1:
            raise ValueError("sequence reservation count must be positive")
        value = self.sequences.get(key, 0) + count
        self.sequences[key] = value
        self._write(_sequence_path(self._root, key), {"key": key.hex(), "value": value})
        return value

    async def advance_sequence(self, key: bytes, expected: int) -> int:
        if self.sequences.get(key, 0) != expected:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await self.next_sequence(key)

    async def delete_sequence(self, key: bytes) -> None:
        self.sequences.pop(key, None)
        self._delete(_sequence_path(self._root, key))

    async def insert_fact(self, fact: StoredFact) -> None:
        if fact.owner_key_digest not in self.guarded_record_keys:
            raise RuntimeError("fact owner must be guarded in the current transaction")
        info = self._own_fact_stream(fact.stream_digest)
        if info is None:
            info = _FactStreamInfo(fact.stream_digest, fact.owner_key_digest, 0, {})
            self.fact_streams[fact.stream_digest] = info
            self._owned_fact_streams.add(fact.stream_digest)
        if info.owner_key_digest != fact.owner_key_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = (fact.stream_digest, fact.sequence)
        if fact.sequence <= info.last_sequence and key not in self._deleted_facts:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if fact.sequence != info.last_sequence + 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        info.last_sequence = fact.sequence
        self._deleted_facts.discard(key)
        self._facts[key] = fact
        self._write(_fact_item_path(self._root, fact.stream_digest, fact.sequence), encode_fact(fact))
        self._sync_fact_stream(info, subject=fact.subject_digest, sequence=fact.sequence)

    async def insert_facts(self, facts: Sequence[StoredFact]) -> None:
        for fact in facts:
            await self.insert_fact(fact)

    async def list_facts(self, query: FactQuery) -> tuple[StoredFact, ...]:
        info = self.fact_streams.get(query.stream_digest)
        if info is None:
            return ()
        if query.latest:
            if query.subject_digest is None:
                latest = info.last_sequence
            else:
                latest = info.subjects.get(query.subject_digest)
            if latest is None or query.after_sequence is not None and latest <= query.after_sequence:
                return ()
            await self._load_facts(info, (latest,))
            value = self._facts.get((info.stream_digest, latest))
            return () if value is None else (value,)
        if query.latest_per_subject:
            sequences = tuple(
                sequence
                for sequence in info.subjects.values()
                if query.after_sequence is None or sequence > query.after_sequence
            )
            await self._load_facts(info, sequences)
            values = [
                self._facts.get((info.stream_digest, sequence))
                for sequence in sequences
            ]
            values = [value for value in values if value is not None]
            values.sort(key=lambda value: value.sequence)
            if query.limit is not None:
                values = values[: query.limit]
            return tuple(values)
        start = 1 if query.after_sequence is None else query.after_sequence + 1
        end = info.last_sequence + 1 if query.limit is None else min(
            info.last_sequence + 1,
            start + query.limit,
        )
        sequences = range(start, end)
        await self._load_facts(info, tuple(sequences))
        values = [self._facts.get((info.stream_digest, sequence)) for sequence in sequences]
        values = [value for value in values if value is not None]
        if query.subject_digest is not None:
            values = [value for value in values if value.subject_digest == query.subject_digest]
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def delete_fact_streams(self, owner_key: bytes) -> None:
        for stream, source in tuple(self.fact_streams.items()):
            if source.owner_key_digest != owner_key:
                continue
            info = self._own_fact_stream(stream)
            if info is None:
                continue
            for sequence in range(1, info.last_sequence + 1):
                self._deleted_facts.add((info.stream_digest, sequence))
                self._delete(_fact_item_path(self._root, info.stream_digest, sequence))
            info.last_sequence = 0
            self._sync_fact_stream(info)
            del self.fact_streams[stream]

    async def insert_operation(self, value: StoredOperation) -> None:
        if value.key_digest in self.operations or any(
            item.stream_digest == value.stream_digest and item.sequence == value.sequence
            for item in self.operations.values()
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self.operations[value.key_digest] = value
        self._write(_operation_path(self._root, value), encode_operation(value))
        self._write(_operation_ref_path(self._root, value), {"key": value.key_digest.hex()})

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        return self.operations.get(key)

    async def replace_operation(self, value: StoredOperation, *, expected_state: str) -> bool:
        current = self.operations.get(value.key_digest)
        if current is None or current.state != expected_state:
            return False
        self.operations[value.key_digest] = value
        self._write(_operation_path(self._root, value), encode_operation(value))
        return True

    async def list_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        values = [
            item
            for item in self.operations.values()
            if (query.stream_digest is None or item.stream_digest == query.stream_digest)
            and (query.states is None or item.state in query.states)
            and (query.through_sequence is None or item.sequence <= query.through_sequence)
        ]
        values.sort(key=lambda item: (item.sequence, item.key_digest))
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def delete_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        values = await self.list_operations(query)
        for value in values:
            del self.operations[value.key_digest]
            self._delete(_operation_path(self._root, value))
            self._delete(_operation_ref_path(self._root, value))
        return values

    async def _load_facts(self, info: _FactStreamInfo, sequences: Sequence[int]) -> None:
        missing = tuple(
            sequence
            for sequence in sequences
            if (info.stream_digest, sequence) not in self._facts
            and (info.stream_digest, sequence) not in self._deleted_facts
        )
        if not missing:
            return
        values = await asyncio.to_thread(_read_fact_batch, self._root, info.stream_digest, missing)
        self._facts.update(values)

    def _own_fact_stream(self, stream: bytes) -> _FactStreamInfo | None:
        info = self.fact_streams.get(stream)
        if info is None or stream in self._owned_fact_streams:
            return info
        owned = _FactStreamInfo(
            info.stream_digest,
            info.owner_key_digest,
            info.last_sequence,
            dict(info.subjects),
        )
        self.fact_streams[stream] = owned
        self._owned_fact_streams.add(stream)
        return owned

    def _sync_fact_stream(
        self,
        info: _FactStreamInfo,
        *,
        subject: bytes | None = None,
        sequence: int | None = None,
    ) -> None:
        if info.last_sequence == 0:
            self._delete(_fact_meta_path(self._root, info.stream_digest))
            for subject_digest in tuple(info.subjects):
                self._delete(_fact_subject_path(self._root, info.stream_digest, subject_digest))
            info.subjects.clear()
            return
        if sequence is None:
            raise ValueError("fact stream update requires a sequence")
        if subject is not None and info.subjects.get(subject, 0) < sequence:
            info.subjects[subject] = sequence
        self._write(
            _fact_meta_path(self._root, info.stream_digest),
            {
                "stream": info.stream_digest.hex(),
                "owner_key": info.owner_key_digest.hex(),
                "last_sequence": info.last_sequence,
            },
        )
        if subject is not None:
            self._write(
                _fact_subject_path(self._root, info.stream_digest, subject),
                {"sequence": info.subjects[subject]},
            )

    def _write(self, relative: str | Path, value: Mapping[str, object] | bytes) -> None:
        relative = _relative_path(self._root, relative)
        self.deletes.discard(relative)
        self.writes[relative] = value if isinstance(value, bytes) else _json_bytes(value)

    def _delete(self, relative: str | Path) -> None:
        relative = _relative_path(self._root, relative)
        self.writes.pop(relative, None)
        self.deletes.add(relative)


def _matches_record(record: StoredRecord, query: RecordQuery) -> bool:
    return (
        (query.partition_digest is None or record.partition_digest == query.partition_digest)
        and (query.scope_digest is None or record.scope_digest == query.scope_digest)
        and (query.parent_digest is None or record.parent_digest == query.parent_digest)
        and (query.states is None or record.state in query.states)
    )


def _relative_path(root: Path, value: str | Path) -> str:
    if isinstance(value, Path):
        return value.relative_to(root).as_posix()
    return value


def _record_path(record: StoredRecord) -> str:
    key = record.key_digest.hex()
    return f"records/{record.kind}/{key[:2]}/{key}.json"


def _alias_path(root: Path, alias: bytes) -> str:
    value = alias.hex()
    return f"aliases/{value[:2]}/{value}.json"


def _sequence_path(root: Path, key: bytes) -> str:
    value = key.hex()
    return f"sequences/{value[:2]}/{value}.json"


def _fact_directory(stream: bytes) -> str:
    value = stream.hex()
    return f"facts/{value[:2]}/{value}"


def _fact_meta_path(root: Path, stream: bytes) -> str:
    return f"{_fact_directory(stream)}/meta.json"


def _fact_item_path(root: Path, stream: bytes, sequence: int) -> Path:
    return root / _fact_directory(stream) / "items" / f"{sequence:020d}.json"


def _fact_subject_path(root: Path, stream: bytes, subject: bytes) -> str:
    return f"{_fact_directory(stream)}/subjects/{subject.hex()}.ref"


def _operation_path(root: Path, value: StoredOperation) -> str:
    key = value.key_digest.hex()
    return f"operations/by-key/{key[:2]}/{key}.json"


def _operation_ref_path(root: Path, value: StoredOperation) -> str:
    stream = value.stream_digest.hex()
    return f"operations/streams/{stream[:2]}/{stream}/{value.sequence:020d}.ref"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON root must be an object")
    return value


def _read_fact_batch(
    root: Path,
    stream: bytes,
    sequences: Sequence[int],
) -> dict[tuple[bytes, int], StoredFact]:
    return {
        (stream, sequence): decode_fact(_read_json(_fact_item_path(root, stream, sequence)))
        for sequence in sequences
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))
    _sync_file(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    _sync_file(path)


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _require_layout_path(path: Path, root: Path, expected: str) -> None:
    try:
        actual = path.relative_to(root).as_posix()
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if actual != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _await_thread(fn: Callable[[], ValueT]) -> ValueT:
    task = asyncio.create_task(asyncio.to_thread(fn))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise


__all__ = ["FilesystemStateStorageGroup", "FilesystemStateStore"]
