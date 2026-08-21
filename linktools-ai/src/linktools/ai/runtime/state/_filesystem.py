#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Granular, journaled filesystem implementation of StateStore."""

import asyncio
import hashlib
import json
import os
from bisect import bisect_right
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
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
    FactScanCursor,
    OperationQuery,
    OperationScanCursor,
    RecordQuery,
    RecordScanCursor,
    RecordReplacement,
    StateCallback,
    StateGroupCallback,
    StateStorageGroup,
    StateTransaction,
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    active_state_group_transaction,
    active_state_transaction,
    bind_state_scope,
    reset_state_transaction,
    validate_operation_replacement,
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
    subjects_loaded: bool = True


@dataclass(frozen=True, slots=True)
class _RecordQueryIndexKey:
    kind: str | None
    partition_digest: bytes | None
    scope_digest: bytes | None
    parent_digest: bytes | None
    states: frozenset[str] | None


class _FilesystemCache:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._records: dict[bytes, StoredRecord | None] = {}
        self._aliases: dict[bytes, bytes | None] = {}
        self._sequences: dict[bytes, int] = {}
        self._fact_streams: dict[bytes, _FactStreamInfo | None] = {}
        self._operations: dict[bytes, StoredOperation | None] = {}
        self._record_kinds: tuple[str, ...] | None = None
        self._loaded_record_kinds: set[str] = set()
        self._aliases_complete = False
        self._fact_streams_complete = False
        self._operations_complete = False
        self._cache_hits = 0
        self._cache_misses = 0
        self._record_kind_scans = 0
        self._business_files_read = 0
        self._record_cache_generation = 0
        self._record_query_indexes: dict[
            _RecordQueryIndexKey,
            tuple[int, tuple[tuple[str, bytes], ...]],
        ] = {}

    def get_record(self, key: bytes) -> StoredRecord | None:
        if key in self._records:
            self._cache_hits += 1
            return self._records[key]
        self._cache_misses += 1
        value_hex = key.hex()
        matches: list[StoredRecord] = []
        for kind in self._record_kind_names():
            path = self._root / "records" / kind / value_hex[:2] / f"{value_hex}.json"
            if not path.is_file():
                continue
            value = decode_record(_read_json(path))
            if value.key_digest != key or value.kind != kind:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._business_files_read += 1
            matches.append(value)
        if len(matches) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if matches:
            self._records[key] = matches[0]
            return matches[0]
        self._records[key] = None
        return None

    def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        kinds = (query.kind,) if query.kind is not None else self._record_kind_names()
        for kind in kinds:
            self._load_record_kind(kind)
        index_key = _RecordQueryIndexKey(
            query.kind,
            query.partition_digest,
            query.scope_digest,
            query.parent_digest,
            query.states,
        )
        cached = self._record_query_indexes.get(index_key)
        if cached is None or cached[0] != self._record_cache_generation:
            ordered = tuple(
                (value.sort_key, value.key_digest)
                for value in sorted(
                    (
                        value
                        for value in self._records.values()
                        if isinstance(value, StoredRecord)
                        and value.kind in kinds
                        and _matches_record(value, query)
                    ),
                    key=lambda record: (record.sort_key, record.key_digest),
                )
            )
            cached = (self._record_cache_generation, ordered)
            self._record_query_indexes[index_key] = cached
            self._log_summary("record_query_index_build", len(ordered))
        ordered = cached[1]
        start = 0
        if query.after_sort_key is not None and query.after_key_digest is not None:
            start = bisect_right(ordered, (query.after_sort_key, query.after_key_digest))
        selected = ordered[start:] if query.limit is None else ordered[start : start + query.limit]
        return tuple(self._records[key] for _, key in selected if self._records.get(key) is not None)

    def get_alias(self, alias: bytes) -> bytes | None:
        if alias in self._aliases:
            self._cache_hits += 1
            return self._aliases[alias]
        self._cache_misses += 1
        value = None
        path = self._root / _alias_path(self._root, alias)
        if path.is_file():
            decoded = decode_alias(_read_json(path))
            if decoded.alias_digest != alias:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = decoded.record_key_digest
            self._business_files_read += 1
        self._aliases[alias] = value
        return value

    def list_aliases(self) -> tuple[tuple[bytes, bytes], ...]:
        if self._aliases_complete:
            return tuple(
                (alias, record_key)
                for alias, record_key in self._aliases.items()
                if record_key is not None
            )
        values: list[tuple[bytes, bytes]] = []
        for path in (self._root / "aliases").glob("*/*.json"):
            value = decode_alias(_read_json(path))
            self._aliases[value.alias_digest] = value.record_key_digest
            self._business_files_read += 1
            values.append((value.alias_digest, value.record_key_digest))
        self._aliases_complete = True
        return tuple(values)

    def get_sequence(self, key: bytes) -> int:
        if key in self._sequences:
            self._cache_hits += 1
            return self._sequences[key]
        self._cache_misses += 1
        value = 0
        path = self._root / _sequence_path(self._root, key)
        if path.is_file():
            stored_key, value = _read_sequence_metadata(path)
            if stored_key != key:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._business_files_read += 1
        self._sequences[key] = value
        return value

    def get_fact_stream(self, stream: bytes) -> _FactStreamInfo | None:
        if stream in self._fact_streams:
            self._cache_hits += 1
            return self._fact_streams[stream]
        self._cache_misses += 1
        path = self._root / _fact_meta_path(self._root, stream)
        if not path.is_file():
            self._fact_streams[stream] = None
            return None
        _require_layout_path(path, self._root, _fact_meta_path(self._root, stream))
        stored_stream, owner, last_sequence = _read_fact_metadata(path)
        if stored_stream != stream:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = _FactStreamInfo(stream, owner, last_sequence, {}, False)
        self._fact_streams[stream] = value
        self._business_files_read += 1
        return value

    def load_fact_subjects(self, info: _FactStreamInfo) -> None:
        if info.subjects_loaded:
            return
        subjects: dict[bytes, int] = {}
        root = self._root / _fact_directory(info.stream_digest) / "subjects"
        for ref in root.glob("*.ref"):
            subject = _layout_digest(ref.stem)
            _require_layout_path(
                ref,
                self._root,
                _fact_subject_path(self._root, info.stream_digest, subject),
            )
            sequence = _read_subject_sequence(ref)
            if not 1 <= sequence <= info.last_sequence:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            subjects[subject] = sequence
        info.subjects.update(subjects)
        info.subjects_loaded = True
        self._business_files_read += len(subjects)

    def list_fact_streams(self) -> tuple[_FactStreamInfo, ...]:
        if self._fact_streams_complete:
            return tuple(value for value in self._fact_streams.values() if value is not None)
        values: list[_FactStreamInfo] = []
        for path in (self._root / "facts").glob("*/*/meta.json"):
            stream, _owner, _last_sequence = _read_fact_metadata(path)
            info = self.get_fact_stream(stream)
            if info is not None:
                values.append(info)
        self._fact_streams_complete = True
        return tuple(values)

    def get_operation(self, key: bytes) -> StoredOperation | None:
        if key in self._operations:
            self._cache_hits += 1
            return self._operations[key]
        self._cache_misses += 1
        value = None
        key_hex = key.hex()
        path = self._root / "operations" / "by-key" / key_hex[:2] / f"{key_hex}.json"
        if path.is_file():
            value = decode_operation(_read_json(path))
            if value.key_digest != key:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._business_files_read += 1
        self._operations[key] = value
        return value

    def list_operations(self) -> tuple[StoredOperation, ...]:
        if self._operations_complete:
            return tuple(value for value in self._operations.values() if value is not None)
        values: list[StoredOperation] = []
        for path in (self._root / "operations/by-key").glob("*/*.json"):
            value = decode_operation(_read_json(path))
            self._operations[value.key_digest] = value
            self._business_files_read += 1
            values.append(value)
        self._operations_complete = True
        return tuple(values)

    def get_operation_by_stream_sequence(
        self,
        stream_digest: bytes,
        sequence: int,
    ) -> StoredOperation | None:
        stream = stream_digest.hex()
        path = self._root / "operations" / "streams" / stream[:2] / stream / f"{sequence:020d}.ref"
        if not path.is_file():
            return None
        try:
            _require_layout_path(
                path,
                self._root,
                f"operations/streams/{stream[:2]}/{stream}/{sequence:020d}.ref",
            )
            key = _read_operation_ref(path)
        except AIError:
            raise
        operation = self.get_operation(key)
        if (
            operation is None
            or operation.stream_digest != stream_digest
            or operation.sequence != sequence
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return operation

    def list_operation_stream(self, stream_digest: bytes) -> tuple[StoredOperation, ...]:
        stream = stream_digest.hex()
        root = self._root / "operations" / "streams" / stream[:2] / stream
        values: list[StoredOperation] = []
        if not root.is_dir():
            return ()
        for path in sorted(root.glob("*.ref")):
            sequence = _layout_sequence_name(path.stem)
            value = self.get_operation_by_stream_sequence(stream_digest, sequence)
            if value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.append(value)
        return tuple(values)

    def scan_records(self) -> tuple[StoredRecord, ...]:
        for kind in self._record_kind_names():
            self._load_record_kind(kind)
        return tuple(value for value in self._records.values() if value is not None)

    def scan_records_page(
        self,
        *,
        after: RecordScanCursor | None,
        limit: int,
    ) -> tuple[StoredRecord, ...]:
        _require_scan_limit(limit)
        values: list[StoredRecord] = []
        for kind in self._record_kind_names():
            if after is not None and kind < after.kind:
                continue
            root = self._root / "records" / kind
            if not root.is_dir():
                continue
            for shard in sorted(path for path in root.iterdir() if path.is_dir()):
                for path in sorted(shard.glob("*.json")):
                    key_hex = path.stem
                    key = _layout_digest(key_hex)
                    if after is not None and (kind, key) <= (
                        after.kind,
                        after.key_digest,
                    ):
                        continue
                    value = decode_record(_read_json(path))
                    _require_layout_path(
                        path,
                        self._root,
                        f"records/{kind}/{key_hex[:2]}/{key_hex}.json",
                    )
                    if value.kind != kind or value.key_digest != key:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    self._records[key] = value
                    self._business_files_read += 1
                    values.append(value)
                    if len(values) == limit:
                        return tuple(values)
        return tuple(values)

    def scan_facts(self) -> tuple[StoredFact, ...]:
        values: list[StoredFact] = []
        for info in self.list_fact_streams():
            loaded = _read_fact_batch(
                self._root,
                info.stream_digest,
                tuple(range(1, info.last_sequence + 1)),
            )
            values.extend(loaded.values())
        return tuple(values)

    def scan_facts_page(
        self,
        *,
        after: FactScanCursor | None,
        limit: int,
    ) -> tuple[StoredFact, ...]:
        _require_scan_limit(limit)
        facts_root = self._root / "facts"
        if not facts_root.is_dir():
            return ()
        values: list[StoredFact] = []
        for shard in sorted(path for path in facts_root.iterdir() if path.is_dir()):
            for stream_dir in sorted(
                path for path in shard.iterdir() if path.is_dir()
            ):
                stream = _layout_digest(stream_dir.name)
                if after is not None and stream < after.stream_digest:
                    continue
                meta_path = stream_dir / "meta.json"
                if not meta_path.is_file():
                    continue
                _require_layout_path(
                    meta_path,
                    self._root,
                    _fact_meta_path(self._root, stream),
                )
                info = self.get_fact_stream(stream)
                if info is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                first = 1
                if after is not None and stream == after.stream_digest:
                    first = after.sequence + 1
                if first > info.last_sequence:
                    continue
                sequences = tuple(
                    range(
                        first,
                        min(
                            info.last_sequence + 1,
                            first + limit - len(values),
                        ),
                    )
                )
                batch = _read_fact_batch(self._root, info.stream_digest, sequences)
                for sequence in sequences:
                    try:
                        value = batch[(info.stream_digest, sequence)]
                    except KeyError as error:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    self._business_files_read += 1
                    values.append(value)
                    if len(values) == limit:
                        return tuple(values)
        return tuple(values)

    def scan_operations(self) -> tuple[StoredOperation, ...]:
        return self.list_operations()

    def scan_operations_page(
        self,
        *,
        after: OperationScanCursor | None,
        limit: int,
    ) -> tuple[StoredOperation, ...]:
        _require_scan_limit(limit)
        values: list[StoredOperation] = []
        root = self._root / "operations" / "by-key"
        if not root.is_dir():
            return ()
        for shard in sorted(path for path in root.iterdir() if path.is_dir()):
            for path in sorted(shard.glob("*.json")):
                key_hex = path.stem
                key = _layout_digest(key_hex)
                if after is not None and key <= after.key_digest:
                    continue
                value = decode_operation(_read_json(path))
                _require_layout_path(
                    path,
                    self._root,
                    f"operations/by-key/{key_hex[:2]}/{key_hex}.json",
                )
                if value.key_digest != key:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._operations[key] = value
                self._business_files_read += 1
                values.append(value)
                if len(values) == limit:
                    return tuple(values)
        return tuple(values)

    def _record_kind_names(self) -> tuple[str, ...]:
        if self._record_kinds is None:
            root = self._root / "records"
            self._record_kinds = (
                tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
                if root.is_dir()
                else ()
            )
            self._record_kind_scans += 1
        return self._record_kinds

    def _load_record_kind(self, kind: str) -> None:
        if kind in self._loaded_record_kinds:
            self._cache_hits += 1
            return
        self._cache_misses += 1
        root = self._root / "records" / kind
        for path in root.glob("*/*.json"):
            value = decode_record(_read_json(path))
            if value.kind != kind:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _require_layout_path(
                path,
                self._root,
                f"records/{kind}/{value.key_digest.hex()[:2]}/{value.key_digest.hex()}.json",
            )
            self._business_files_read += 1
            self._records[value.key_digest] = value
        self._loaded_record_kinds.add(kind)
        self._record_kind_scans += 1

    def set_record(
        self,
        key: bytes,
        value: StoredRecord | None,
        *,
        old_kind: str | None = None,
    ) -> None:
        affected_kinds = {kind for kind in (old_kind, value.kind if value is not None else None) if kind is not None}
        if affected_kinds:
            self._record_query_indexes = {
                index_key: index
                for index_key, index in self._record_query_indexes.items()
                if index_key.kind is not None and index_key.kind not in affected_kinds
            }
        self._record_cache_generation += 1
        self._records[key] = value
        if isinstance(value, StoredRecord) and self._record_kinds is not None:
            self._record_kinds = tuple(sorted(set(self._record_kinds) | {value.kind}))

    def set_alias(self, key: bytes, value: bytes | None) -> None:
        self._aliases[key] = value

    def set_sequence(self, key: bytes, value: int) -> None:
        self._sequences[key] = value

    def set_fact_stream(self, key: bytes, value: _FactStreamInfo | None) -> None:
        self._fact_streams[key] = value

    def set_operation(self, key: bytes, value: StoredOperation | None) -> None:
        self._operations[key] = value

    def _log_summary(self, event: str, value: int) -> None:
        _logger.debug(
            "filesystem cache summary: event=%s value=%s cache_hit=%s cache_miss=%s "
            "record_kind_scan=%s business_files_read=%s",
            event,
            value,
            self._cache_hits,
            self._cache_misses,
            self._record_kind_scans,
            self._business_files_read,
        )


@dataclass(slots=True)
class _FilesystemIndex:
    records: dict[bytes, StoredRecord]
    aliases: dict[bytes, bytes]
    sequences: dict[bytes, int]
    fact_streams: dict[bytes, _FactStreamInfo]
    operations: dict[bytes, StoredOperation]
    cache: _FilesystemCache


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
        async with self._mutation_lock:
            if self._initialized:
                return
            await self._initialize_locked()

    async def _initialize_locked(self) -> None:
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
        async with self._mutation_lock:
            if self._closed:
                return
            self._closed = True
            self._initialized = False
            locks = tuple(sorted(self._members, key=lambda value: value.root.as_posix()))
            for member in locks:
                await member._consistency_lock.acquire()
            for member in sorted(self._members, key=lambda value: value.root.as_posix(), reverse=True):
                await member._writer_lock.release()
            for member in reversed(locks):
                member._consistency_lock.release()
            if not self._standalone:
                await self._group_lock.release()
        _logger.debug("filesystem StateStorageGroup closed: scope=%s", self._scope_digest)

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

    async def read(self, store: "FilesystemStateStore", fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_member(store)
        active = active_state_transaction(store)
        if active is not None:
            return await fn(active)
        started = monotonic()
        await store._consistency_lock.acquire()
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
        members = tuple(sorted(dict.fromkeys(stores), key=lambda value: value.root.as_posix()))
        if not members:
            raise ValueError("StateStorageGroup mutation requires a store")
        for store in members:
            self._ensure_member(store)
        active = active_state_transaction(members[0], writable=True)
        if active is not None:
            return await fn(active_state_group_transaction(self, members))
        mutation_started = monotonic()
        await self._mutation_lock.acquire()
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
                "filesystem mutation members locked: scope=%s "
                "member_lock_wait_ms=%.3f",
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
                if any(transaction.has_changes for transaction in transactions.values()):
                    await self._commit(transactions)
                return result
            finally:
                for store in reversed(locked):
                    store._consistency_lock.release()
        finally:
            self._mutation_lock.release()

    async def validate_integrity(self) -> None:
        self._ensure_ready()
        async with self._mutation_lock:
            ordered = tuple(sorted(self._members, key=lambda value: value.root.as_posix()))
            for member in ordered:
                await member._consistency_lock.acquire()
            try:
                for member in ordered:
                    member._ensure_ready()
                    await _await_thread(member._validate_integrity_sync)
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
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        else:
            _write_json(manifest, expected)
            _write_text(self._metadata_root / "generation", "0")
            sync_directory(self._metadata_root)
        if not (self._metadata_root / "generation").is_file():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @staticmethod
    def _manifest_matches(actual: Mapping[str, object], expected: Mapping[str, object]) -> bool:
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
                if path.is_dir() and path.name.startswith(".txn-") and path.name != own_name
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
                    if any(value == root or value.startswith(root + "/") for root in member_paths):
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        except AIError:
            raise
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error

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
        self._consistency_lock = asyncio.Lock()
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
        active = active_state_transaction(self, writable=True)
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
        if self._root.is_dir() and all(path.name == "state.lock" for path in self._root.iterdir()):
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
                key, value = _read_sequence_metadata(path)
                _require_layout_path(path, self._root, f"sequences/{key.hex()[:2]}/{key.hex()}.json")
                if key in sequences:
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
                stream, owner, last_sequence = _read_fact_metadata(path)
                _require_layout_path(path, self._root, f"facts/{stream.hex()[:2]}/{stream.hex()}/meta.json")
                if stream in streams:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                subjects: dict[bytes, int] = {}
                for ref in path.parent.joinpath("subjects").glob("*.ref"):
                    subject = _layout_digest(ref.stem)
                    sequence = _read_subject_sequence(ref)
                    _require_layout_path(ref, self._root, _fact_subject_path(self._root, stream, subject))
                    if subject in subjects or not 1 <= sequence <= last_sequence:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    subjects[subject] = sequence
                streams[stream] = _FactStreamInfo(stream, owner, last_sequence, subjects)
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
                (value.stream_digest, value.sequence): value.key_digest for value in operations.values()
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
        old_record_kinds = {
            key: index.records[key].kind
            for key in set(transaction.records.changes()) | set(transaction.records.deleted())
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
        self._cache = index.cache
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
        aliases = dict(await asyncio.to_thread(self._cache.list_aliases))
        aliases.update(self.aliases.changes())
        for alias in self.aliases.deleted():
            aliases.pop(alias, None)
        for key in aliases.values():
            if await self.get_record(key) is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        streams = {
            info.stream_digest: info
            for info in await asyncio.to_thread(self._cache.list_fact_streams)
        }
        streams.update(self.fact_streams.changes())
        for stream in self.fact_streams.deleted():
            streams.pop(stream, None)
        for info in streams.values():
            if await self.get_record(info.owner_key_digest) is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def get_record(self, key: bytes) -> StoredRecord | None:
        if key in self.records.deleted():
            return None
        if key in self.records:
            return self.records[key]
        return await asyncio.to_thread(self._cache.get_record, key)

    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        unique_keys = tuple(dict.fromkeys(keys))
        if not unique_keys:
            return {}
        cached = await asyncio.to_thread(
            lambda: {key: self._cache.get_record(key) for key in unique_keys}
        )
        values = {
            key: value
            for key, value in cached.items()
            if value is not None
        }
        values.update(
            key_value
            for key_value in self.records.changes().items()
            if key_value[0] in unique_keys
        )
        for key in self.records.deleted():
            values.pop(key, None)
        return values

    async def insert_record(self, record: StoredRecord) -> None:
        await self.insert_records((record,))

    async def insert_records(self, records: Sequence[StoredRecord]) -> None:
        values = tuple(records)
        keys = [record.key_digest for record in values]
        if len(keys) != len(set(keys)):
            raise ValueError("insert_records contains duplicate keys")
        if not values:
            return
        if await self.get_records(keys):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        for record in sorted(values, key=lambda value: value.key_digest):
            validate_record_identity(record)
            self.records[record.key_digest] = record
            self.guarded_record_keys.add(record.key_digest)
            self._write(_record_path(record), encode_record(record))

    async def guard_record(self, key: bytes, *, expected_storage_version: int) -> StoredRecord | None:
        if (
            isinstance(expected_storage_version, bool)
            or not isinstance(expected_storage_version, int)
            or expected_storage_version < 0
        ):
            raise ValueError("expected_storage_version must be a non-negative integer")
        current = await self.get_record(key)
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
        try:
            await self.replace_records((RecordReplacement(record, expected_storage_version),))
        except AIError as error:
            if error.code is ErrorCode.STORAGE_CONFLICT:
                return False
            raise
        return True

    async def replace_records(self, replacements: Sequence[RecordReplacement]) -> None:
        values = tuple(replacements)
        keys = [replacement.record.key_digest for replacement in values]
        if len(keys) != len(set(keys)):
            raise ValueError("replace_records contains duplicate keys")
        if not values:
            return
        current_values = await self.get_records(keys)
        candidates: list[StoredRecord] = []
        for replacement in sorted(values, key=lambda value: value.record.key_digest):
            current = current_values.get(replacement.record.key_digest)
            if current is None or current.storage_version != replacement.expected_storage_version:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            validate_record_replacement(current, replacement.record)
            validate_record_identity(replacement.record)
            if replacement.record.storage_version != replacement.expected_storage_version + 1:
                raise ValueError("replacement must increment storage_version exactly once")
            candidates.append(replacement.record)
        for record in candidates:
            self.records[record.key_digest] = record
            self.guarded_record_keys.add(record.key_digest)
            self._write(_record_path(record), encode_record(record))

    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool:
        if (
            isinstance(expected_storage_version, bool)
            or not isinstance(expected_storage_version, int)
            or expected_storage_version < 0
            or isinstance(lease_fence, bool)
            or not isinstance(lease_fence, int)
            or lease_fence < 0
        ):
            raise ValueError("record lease integer fields are invalid")
        current = await self.get_record(key)
        if current is None or current.storage_version != expected_storage_version:
            return False
        if lease_expires_at is not None and lease_expires_at.tzinfo is None:
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
        if expected_storage_version is not None and (
            isinstance(expected_storage_version, bool)
            or not isinstance(expected_storage_version, int)
            or expected_storage_version < 0
        ):
            raise ValueError("expected_storage_version must be a non-negative integer or None")
        current = await self.get_record(key)
        if current is None:
            return False
        expected = current.storage_version if expected_storage_version is None else expected_storage_version
        if await self.guard_record(key, expected_storage_version=expected) is None:
            return False
        await self.delete_fact_streams(key)
        aliases = dict(await asyncio.to_thread(self._cache.list_aliases))
        aliases.update(self.aliases.changes())
        for alias, record_key in tuple(aliases.items()):
            if record_key == key:
                self.aliases[alias] = record_key
                del self.aliases[alias]
                self._delete(_alias_path(self._root, alias))
        del self.records[key]
        self.guarded_record_keys.discard(key)
        self._delete(_record_path(current))
        return True

    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        cache_query = query
        if query.limit is not None and (
            self.records.changes() or self.records.deleted()
        ):
            pending = len(self.records.changes()) + len(self.records.deleted())
            cache_query = replace(
                query,
                limit=(
                    query.limit + pending
                    if query.limit + pending <= 1000
                    else None
                ),
            )
        values = {
            record.key_digest: record
            for record in await asyncio.to_thread(
                self._cache.list_records,
                cache_query,
            )
        }
        values.update(self.records.changes())
        for key in self.records.deleted():
            values.pop(key, None)
        values = [record for record in values.values() if _matches_record(record, query)]
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

    async def scan_records(self) -> tuple[StoredRecord, ...]:
        values = {
            record.key_digest: record
            for record in await asyncio.to_thread(self._cache.scan_records)
        }
        values.update(self.records.changes())
        for key in self.records.deleted():
            values.pop(key, None)
        return tuple(values.values())

    async def scan_records_page(
        self,
        *,
        after: RecordScanCursor | None,
        limit: int,
    ) -> tuple[StoredRecord, ...]:
        _require_scan_limit(limit)
        local = sorted(
            (
                value
                for key, value in self.records.changes().items()
                if key not in self.records.deleted()
                and (after is None or (value.kind, key) > (after.kind, after.key_digest))
            ),
            key=lambda value: (value.kind, value.key_digest),
        )
        local_index = 0
        base_page: tuple[StoredRecord, ...] = ()
        base_index = 0
        base_after = after
        base_done = False
        result: list[StoredRecord] = []
        while len(result) < limit:
            if base_index >= len(base_page) and not base_done:
                base_page = await asyncio.to_thread(
                    self._cache.scan_records_page,
                    after=base_after,
                    limit=limit,
                )
                base_index = 0
                if not base_page:
                    base_done = True
                else:
                    last = base_page[-1]
                    base_after = RecordScanCursor(last.kind, last.key_digest)
                    base_done = len(base_page) < limit
            base_value = (
                None if base_index >= len(base_page) else base_page[base_index]
            )
            local_value = None if local_index >= len(local) else local[local_index]
            if base_value is None and local_value is None:
                break
            if local_value is None or (
                base_value is not None
                and (base_value.kind, base_value.key_digest)
                < (local_value.kind, local_value.key_digest)
            ):
                value = base_value
                base_index += 1
            else:
                value = local_value
                local_index += 1
                if (
                    base_value is not None
                    and (base_value.kind, base_value.key_digest)
                    == (value.kind, value.key_digest)
                ):
                    base_index += 1
            if value is not None and value.key_digest not in self.records.deleted():
                result.append(value)
        return tuple(result)

    async def resolve_alias(self, alias: bytes) -> bytes | None:
        return (await self.resolve_aliases((alias,))).get(alias)

    async def resolve_aliases(self, aliases: Sequence[bytes]) -> Mapping[bytes, bytes]:
        unique_aliases = tuple(dict.fromkeys(aliases))
        if not unique_aliases:
            return {}
        cached = await asyncio.to_thread(
            lambda: {alias: self._cache.get_alias(alias) for alias in unique_aliases}
        )
        values = {alias: value for alias, value in cached.items() if value is not None}
        values.update(
            alias_value
            for alias_value in self.aliases.changes().items()
            if alias_value[0] in unique_aliases
        )
        for alias in self.aliases.deleted():
            values.pop(alias, None)
        records = await self.get_records(tuple(values.values()))
        if len(records) != len(set(values.values())):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values

    async def insert_alias(self, alias: StoredAlias) -> None:
        await self.insert_aliases((alias,))

    async def insert_aliases(self, aliases: Sequence[StoredAlias]) -> None:
        values = tuple(sorted(aliases, key=lambda value: value.alias_digest))
        if not values:
            return
        existing = await self.resolve_aliases(tuple(value.alias_digest for value in values))
        for alias in values:
            current = existing.get(alias.alias_digest)
            if current is not None and current != alias.record_key_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if alias.record_key_digest not in self.guarded_record_keys:
                raise RuntimeError("alias owner must be guarded in the current transaction")
            self.aliases[alias.alias_digest] = alias.record_key_digest
            self._write(_alias_path(self._root, alias.alias_digest), encode_alias(alias))

    async def get_sequence(self, key: bytes) -> int:
        if key in self.sequences.deleted():
            return 0
        if key in self.sequences:
            return self.sequences[key]
        return await asyncio.to_thread(self._cache.get_sequence, key)

    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]:
        unique_keys = tuple(dict.fromkeys(keys))
        if not unique_keys:
            return {}
        cached = await asyncio.to_thread(
            lambda: {key: self._cache.get_sequence(key) for key in unique_keys}
        )
        cached.update(self.sequences.changes())
        for key in self.sequences.deleted():
            cached[key] = 0
        return cached

    async def next_sequence(self, key: bytes) -> int:
        value = await self.get_sequence(key) + 1
        self.sequences[key] = value
        self._write(_sequence_path(self._root, key), {"key": key.hex(), "value": value})
        return value

    async def reserve_sequence(self, key: bytes, count: int) -> int:
        return (await self.reserve_sequences({key: count}))[key]

    async def reserve_sequences(self, requests: Mapping[bytes, int]) -> Mapping[bytes, int]:
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in requests.values()
        ):
            raise ValueError("sequence reservation count must be a positive integer")
        current = await self.get_sequences(tuple(requests))
        values = {key: current[key] + requests[key] for key in sorted(requests)}
        for key, value in values.items():
            self.sequences[key] = value
            self._write(_sequence_path(self._root, key), {"key": key.hex(), "value": value})
        return values

    async def advance_sequence(self, key: bytes, expected: int) -> int:
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise ValueError("expected sequence must be a non-negative integer")
        if await self.get_sequence(key) != expected:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await self.next_sequence(key)

    async def delete_sequence(self, key: bytes) -> None:
        await self.delete_sequences((key,))

    async def delete_sequences(self, keys: Sequence[bytes]) -> None:
        for key in sorted(set(keys)):
            if key in self.sequences:
                del self.sequences[key]
            else:
                self.sequences[key] = 0
                del self.sequences[key]
            self._delete(_sequence_path(self._root, key))

    async def insert_fact(self, fact: StoredFact) -> None:
        if fact.owner_key_digest not in self.guarded_record_keys:
            raise RuntimeError("fact owner must be guarded in the current transaction")
        info = await self._own_fact_stream(fact.stream_digest)
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
        if query.stream_digest in self.fact_streams.deleted():
            return ()
        info = self.fact_streams.get(query.stream_digest)
        if info is None and query.stream_digest not in self.fact_streams:
            info = await asyncio.to_thread(self._cache.get_fact_stream, query.stream_digest)
        if info is None:
            return ()
        if query.latest:
            if query.subject_digest is None:
                latest = info.last_sequence
            else:
                await self._load_fact_subjects(info)
                latest = info.subjects.get(query.subject_digest)
            if latest is None or query.after_sequence is not None and latest <= query.after_sequence:
                return ()
            await self._load_facts(info, (latest,))
            value = self._facts.get((info.stream_digest, latest))
            return () if value is None else (value,)
        if query.latest_per_subject:
            await self._load_fact_subjects(info)
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

    async def scan_facts(self) -> tuple[StoredFact, ...]:
        values = {
            (fact.stream_digest, fact.sequence): fact
            for fact in await asyncio.to_thread(self._cache.scan_facts)
        }
        values.update(self._facts)
        for key in self._deleted_facts:
            values.pop(key, None)
        return tuple(values.values())

    async def scan_facts_page(
        self,
        *,
        after: FactScanCursor | None,
        limit: int,
    ) -> tuple[StoredFact, ...]:
        _require_scan_limit(limit)
        local = sorted(
            (
                value
                for key, value in self._facts.items()
                if key not in self._deleted_facts
                and (
                    after is None
                    or (value.stream_digest, value.sequence)
                    > (after.stream_digest, after.sequence)
                )
            ),
            key=lambda value: (value.stream_digest, value.sequence),
        )
        local_index = 0
        base_page: tuple[StoredFact, ...] = ()
        base_index = 0
        base_after = after
        base_done = False
        result: list[StoredFact] = []
        while len(result) < limit:
            if base_index >= len(base_page) and not base_done:
                base_page = await asyncio.to_thread(
                    self._cache.scan_facts_page,
                    after=base_after,
                    limit=limit,
                )
                base_index = 0
                if not base_page:
                    base_done = True
                else:
                    last = base_page[-1]
                    base_after = FactScanCursor(
                        last.stream_digest,
                        last.sequence,
                    )
                    base_done = len(base_page) < limit
            base_value = (
                None if base_index >= len(base_page) else base_page[base_index]
            )
            local_value = None if local_index >= len(local) else local[local_index]
            if base_value is None and local_value is None:
                break
            if local_value is None or (
                base_value is not None
                and (base_value.stream_digest, base_value.sequence)
                < (local_value.stream_digest, local_value.sequence)
            ):
                value = base_value
                base_index += 1
            else:
                value = local_value
                local_index += 1
                if (
                    base_value is not None
                    and (base_value.stream_digest, base_value.sequence)
                    == (value.stream_digest, value.sequence)
                ):
                    base_index += 1
            if value is not None and (
                value.stream_digest,
                value.sequence,
            ) not in self._deleted_facts:
                result.append(value)
        return tuple(result)

    async def delete_fact_streams(self, owner_key: bytes) -> None:
        sources = {
            info.stream_digest: info
            for info in await asyncio.to_thread(self._cache.list_fact_streams)
        }
        sources.update(self.fact_streams.changes())
        for stream in self.fact_streams.deleted():
            sources.pop(stream, None)
        for stream, source in tuple(sources.items()):
            if source.owner_key_digest != owner_key:
                continue
            info = await self._own_fact_stream(stream)
            if info is None:
                continue
            for sequence in range(1, info.last_sequence + 1):
                self._deleted_facts.add((info.stream_digest, sequence))
                self._delete(_fact_item_path(self._root, info.stream_digest, sequence))
            info.last_sequence = 0
            self._sync_fact_stream(info)
            del self.fact_streams[stream]

    async def insert_operation(self, value: StoredOperation) -> None:
        existing = await self.get_operation(value.key_digest)
        stream_existing = await self._get_operation_by_stream_sequence(
            value.stream_digest,
            value.sequence,
        )
        if existing is not None or stream_existing is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self.operations[value.key_digest] = value
        self._write(_operation_path(self._root, value), encode_operation(value))
        self._write(_operation_ref_path(self._root, value), {"key": value.key_digest.hex()})

    async def _get_operation_by_stream_sequence(
        self,
        stream_digest: bytes,
        sequence: int,
    ) -> StoredOperation | None:
        for operation in self.operations.changes().values():
            if operation.stream_digest == stream_digest and operation.sequence == sequence:
                return operation
        value = await asyncio.to_thread(
            self._cache.get_operation_by_stream_sequence,
            stream_digest,
            sequence,
        )
        if value is not None and value.key_digest in self.operations.deleted():
            return None
        return value

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        if key in self.operations.deleted():
            return None
        if key in self.operations:
            return self.operations[key]
        return await asyncio.to_thread(self._cache.get_operation, key)

    async def replace_operation(self, value: StoredOperation, *, expected_state: str) -> bool:
        current = await self.get_operation(value.key_digest)
        if current is None or current.state != expected_state:
            return False
        validate_operation_replacement(current, value)
        self.operations[value.key_digest] = value
        self._write(_operation_path(self._root, value), encode_operation(value))
        return True

    async def list_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        if query.stream_digest is None:
            source = await asyncio.to_thread(self._cache.list_operations)
        else:
            source = await asyncio.to_thread(
                self._cache.list_operation_stream,
                query.stream_digest,
            )
        values_by_key = {item.key_digest: item for item in source}
        values_by_key.update(self.operations.changes())
        for key in self.operations.deleted():
            values_by_key.pop(key, None)
        values = [
            item
            for item in values_by_key.values()
            if (query.stream_digest is None or item.stream_digest == query.stream_digest)
            and (query.states is None or item.state in query.states)
            and (query.through_sequence is None or item.sequence <= query.through_sequence)
            and (query.compactable is None or item.compactable == query.compactable)
        ]
        values.sort(key=lambda item: (item.sequence, item.key_digest))
        if query.limit is not None:
            values = values[: query.limit]
        return tuple(values)

    async def scan_operations(self) -> tuple[StoredOperation, ...]:
        values = {
            item.key_digest: item
            for item in await asyncio.to_thread(self._cache.scan_operations)
        }
        values.update(self.operations.changes())
        for key in self.operations.deleted():
            values.pop(key, None)
        return tuple(values.values())

    async def scan_operations_page(
        self,
        *,
        after: OperationScanCursor | None,
        limit: int,
    ) -> tuple[StoredOperation, ...]:
        _require_scan_limit(limit)
        local = sorted(
            (
                value
                for key, value in self.operations.changes().items()
                if key not in self.operations.deleted()
                and (after is None or key > after.key_digest)
            ),
            key=lambda value: value.key_digest,
        )
        local_index = 0
        base_page: tuple[StoredOperation, ...] = ()
        base_index = 0
        base_after = after
        base_done = False
        result: list[StoredOperation] = []
        while len(result) < limit:
            if base_index >= len(base_page) and not base_done:
                base_page = await asyncio.to_thread(
                    self._cache.scan_operations_page,
                    after=base_after,
                    limit=limit,
                )
                base_index = 0
                if not base_page:
                    base_done = True
                else:
                    last = base_page[-1]
                    base_after = OperationScanCursor(last.key_digest)
                    base_done = len(base_page) < limit
            base_value = (
                None if base_index >= len(base_page) else base_page[base_index]
            )
            local_value = None if local_index >= len(local) else local[local_index]
            if base_value is None and local_value is None:
                break
            if local_value is None or (
                base_value is not None
                and base_value.key_digest < local_value.key_digest
            ):
                value = base_value
                base_index += 1
            else:
                value = local_value
                local_index += 1
                if (
                    base_value is not None
                    and base_value.key_digest == value.key_digest
                ):
                    base_index += 1
            if value is not None and value.key_digest not in self.operations.deleted():
                result.append(value)
        return tuple(result)

    async def delete_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        values = await self.list_operations(query)
        for value in values:
            self.operations[value.key_digest] = value
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

    async def _own_fact_stream(self, stream: bytes) -> _FactStreamInfo | None:
        if stream in self.fact_streams.deleted():
            return None
        info = self.fact_streams.get(stream)
        if info is None and stream not in self.fact_streams:
            info = await asyncio.to_thread(self._cache.get_fact_stream, stream)
        if info is None or stream in self._owned_fact_streams:
            return info
        await self._load_fact_subjects(info)
        owned = _FactStreamInfo(
            info.stream_digest,
            info.owner_key_digest,
            info.last_sequence,
            dict(info.subjects),
            info.subjects_loaded,
        )
        self.fact_streams[stream] = owned
        self._owned_fact_streams.add(stream)
        return owned

    async def _load_fact_subjects(self, info: _FactStreamInfo) -> None:
        if info.subjects_loaded:
            return
        await asyncio.to_thread(self._cache.load_fact_subjects, info)

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
        and (query.kind is None or record.kind == query.kind)
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


def _require_layout_keys(value: Mapping[str, object], expected: frozenset[str]) -> None:
    if set(value.keys()) != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _layout_digest(value: object) -> bytes:
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        result = bytes.fromhex(value)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(result) != 32 or result.hex() != value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _layout_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _layout_positive_int(value: object) -> int:
    result = _layout_nonnegative_int(value)
    if result < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _layout_sequence_name(value: str) -> int:
    if len(value) != 20 or not value.isascii() or not value.isdigit():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    sequence = int(value)
    if sequence < 1 or f"{sequence:020d}" != value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return sequence


def _read_sequence_metadata(path: Path) -> tuple[bytes, int]:
    try:
        raw = _read_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _require_layout_keys(raw, frozenset({"key", "value"}))
    return _layout_digest(raw["key"]), _layout_nonnegative_int(raw["value"])


def _read_fact_metadata(path: Path) -> tuple[bytes, bytes, int]:
    try:
        raw = _read_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _require_layout_keys(raw, frozenset({"stream", "owner_key", "last_sequence"}))
    return (
        _layout_digest(raw["stream"]),
        _layout_digest(raw["owner_key"]),
        _layout_positive_int(raw["last_sequence"]),
    )


def _read_subject_sequence(path: Path) -> int:
    try:
        raw = _read_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _require_layout_keys(raw, frozenset({"sequence"}))
    return _layout_positive_int(raw["sequence"])


def _read_operation_ref(path: Path) -> bytes:
    try:
        raw = _read_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _require_layout_keys(raw, frozenset({"key"}))
    return _layout_digest(raw["key"])


def _read_generation_value(path: Path) -> int:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not raw.isascii() or not raw.isdigit():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    value = int(raw)
    if value < 0 or str(value) != raw:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _read_fact_batch(
    root: Path,
    stream: bytes,
    sequences: Sequence[int],
) -> dict[tuple[bytes, int], StoredFact]:
    values: dict[tuple[bytes, int], StoredFact] = {}
    for sequence in sequences:
        try:
            fact = decode_fact(_read_json(_fact_item_path(root, stream, sequence)))
        except FileNotFoundError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if fact.stream_digest != stream or fact.sequence != sequence:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values[(stream, sequence)] = fact
    return values


def _require_scan_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("scan page limit must be positive")


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
