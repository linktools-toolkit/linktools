#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL implementation of the backend-neutral Runtime StateStore."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from time import time
from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4

from linktools.core import environ

from ..core import canonical_json_bytes, validate_resource_id
from ..errors import AIError, ErrorCode
from ..storage import (
    CacheBackend,
    IntegrityViolationKind,
    SqlStorageContext,
    create_sql_storage_context,
)
from ._plan import PersistenceDomain
from ._archive_read import ArchiveNodeRead, ArchiveNodeSlot
from ._record_owner import RECORD_OWNER_TABLES, record_owner_table, record_owner_edge, record_owner_rows, record_owner_values
from ._schema import build_runtime_sql_metadata
from ._record_cache import StateRecordCache
from .state import state_content_members
from .state import (
    FactQuery,
    FactScanCursor,
    GroupRecordReader,
    OperationQuery,
    OperationScanCursor,
    RecordLeaseTransition, RecordQuery,
    RecordReplacement,
    RecordScanCursor,
    StateCallback,
    StateGroupCallback,
    StateGroupTransaction,
    StateStorageGroup,
    StateStore,
    StateTransaction,
    StateTransactionNestingError,
    StoredFact,
    StoredOperation,
    StoredRecord,
    active_state_group_transaction,
    active_state_scope,
    is_active_state_transaction,
    active_state_transaction,
    bind_state_scope,
    reset_state_transaction,
    validate_stream_key,
    validate_store_key,
    standalone_store_key,
    STATE_GROUP_METADATA_KEY,
    STATE_GROUP_ARCHIVE_KEY,
    validate_relation_key,
    validate_record_key,
    record_key,
    validate_operation_replacement,
    validate_record_identity,
    validate_record_replacement,
)

from ._run_counters import RUN_COUNTER_COLUMNS, run_counter_address, run_counter_streams, run_counter_page_start

if TYPE_CHECKING:
    from ._message_root import NativeMessageRoot
    from ..storage import SqlValue
    from sqlalchemy import MetaData, Table
    from sqlalchemy.engine import Result
    from sqlalchemy.sql import ColumnElement, CompoundSelect, Executable, Select
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

ValueT = TypeVar("ValueT")
_logger = environ.get_logger("ai.runtime.state.sql")
_MAINTENANCE_PAGE_SIZE = 128
_RECORD_DELETE_BATCH_SIZE = 192
# One partition bind plus a locator and raw identity per key fits SQLite 999.
_RAW_KEY_BATCH_SIZE = 499
# Each immutable record contributes seventeen bound values.
_IMMUTABLE_RECORD_BATCH_SIZE = 48
# One store and two stream predicates leave 996 SQLite sequence binds.
_FACT_READ_BATCH_SIZE = 996
# Seven CASE assignments and seven identity/CAS binds plus the IN key
# require at most 22 binds per row and one shared store bind.
_RECORD_REPLACE_BATCH_SIZE = 40
# Store, scope, and kind predicates leave 996 SQLite subject binds.
_RECORD_SUBJECT_READ_BATCH_SIZE = 995
_GROUP_RECORD_BIND_BUDGET = 999
_ARCHIVE_READ_BIND_BUDGET = 999
# MySQL repeats fixed predicates/JSON paths across eight branches plus direct content.
_ARCHIVE_READ_FIXED_BINDS = 145


class _SqlGroupTransaction:
    def __init__(
        self,
        group: "SqlStateStorageGroup",
        transactions: Mapping[StateStore, "_SqlTransaction"],
    ) -> None:
        self._group = group
        self._transactions = transactions
        self._native_message_rows: dict[object, tuple[Mapping[str, object], ...]] = {}

    def transaction(self, store: StateStore) -> "_SqlTransaction":
        if store.storage_group is not self._group:
            raise RuntimeError("store does not belong to this StateStorageGroup")
        try:
            return self._transactions[store]
        except KeyError as error:
            raise RuntimeError(
                "store was not enlisted in the StateStorageGroup transaction"
            ) from error

    async def get_records(
        self, requests: Mapping[StateStore, Sequence[bytes]],
    ) -> Mapping[StateStore, Mapping[bytes, StoredRecord]]:
        active_state_group_transaction(self._group, tuple(requests), writable=False)
        active = active_state_scope()
        if active is None or active.record_reader is not self:
            raise StateTransactionNestingError("group record reader is outside its transaction")
        unique: dict[StateStore, tuple[bytes, ...]] = {}
        transactions: dict[StateStore, _SqlTransaction] = {}
        for store, keys in requests.items():
            for key in keys:
                validate_record_key(key)
            unique[store] = tuple(dict.fromkeys(keys))
            transactions[store] = self.transaction(store)
        missing = {
            store: tuple(key for key in keys if key not in transactions[store]._record_cache)
            for store, keys in unique.items()
        }
        for store, keys in tuple(missing.items()):
            transaction = transactions[store]
            payload_keys = tuple(key for key in keys if transaction._is_payload_key(key))
            if payload_keys:
                await transaction.get_records(payload_keys)
                missing[store] = tuple(key for key in keys if key not in payload_keys)
        for batch in self._record_read_batches(missing):
            first = transactions[next(iter(batch))]
            statement = self._record_read_statement(batch)
            compiled = statement.compile(
                dialect=first._context.engine.dialect,
                compile_kwargs={"render_postcompile": True},
            )
            count = len(compiled.params) if compiled.positiontup is None else len(compiled.positiontup)
            if count > _GROUP_RECORD_BIND_BUDGET:
                raise ValueError("group record read exceeds the SQL parameter budget")
            rows = (await first._execute(statement)).mappings().all()
            records: dict[bytes, dict[bytes, StoredRecord]] = {}
            for row in rows:
                record = _record_from_row(row)
                records.setdefault(row["store_key"], {})[record.key] = record
            for store, keys in batch.items():
                transaction = transactions[store]
                found = records.get(transaction._store_key, {})
                transaction._record_cache.update((key, found.get(key)) for key in keys)
        return {
            store: {key: value for key in keys
                    if (value := transactions[store]._record_cache[key]) is not None}
            for store, keys in unique.items()
        }

    async def read_content_references(
        self, *, owner_store: StateStore, owner_keys: Sequence[bytes], content_store: StateStore,
        content_scope_key: bytes, content_paths: Sequence[tuple[str, ...]],
        additional_records: Mapping[StateStore, Sequence[bytes]] | None = None,
        owner_query: RecordQuery | None = None,
        native_root_paths: Sequence[tuple[str, ...]] = (),
    ) -> Mapping[StateStore, Mapping[bytes, StoredRecord]] | None:
        from ._archive import archive_scope_parts, validate_archive_content_identity
        from ._content_reference_sql import content_reference_batch_size, content_reference_statement
        from ._step_payload_owner import payload_record_from_row

        active_state_group_transaction(self._group, (owner_store, content_store, *(additional_records or {})), writable=False)
        active = active_state_scope()
        if active is None or active.record_reader is not self:
            raise StateTransactionNestingError("content source reader is outside its transaction")
        owner, content = self.transaction(owner_store), self.transaction(content_store)
        dialect = owner._context.dialect.name
        if dialect not in {"sqlite", "mysql", "postgresql"}:
            return None
        paths = tuple(dict.fromkeys(content_paths))
        native_paths = (() if active.writable else tuple(dict.fromkeys(native_root_paths)))
        if (len(native_paths) > 4 or any(not path or len(path) > 16
                or any(not isinstance(part, str) or not part for part in path) for path in native_paths)):
            raise ValueError("native content requires bounded declared root paths")
        if (not paths or len(paths) > 16 or any(not path or len(path) > 16
                or any(not isinstance(part, str) or not part for part in path) for path in paths)):
            raise ValueError("content references require bounded explicit JSON paths")
        keys = tuple(dict.fromkeys(validate_record_key(key) for key in owner_keys))
        if owner_query is not None and (keys or additional_records or owner_query.kind != "tool_operation"
                or owner_query.scope_key is None and owner_query.parent_key is None):
            raise ValueError("content owner queries require one scoped tool selection")
        namespace, tenant = archive_scope_parts(content_scope_key)
        required = []
        for key in keys:
            if key not in owner._record_cache:
                required.append(key)
                continue
            record = owner._record_cache[key]
            if record is None:
                continue
            native_missing = False
            for path in native_paths:
                from ._message_root import NativeMessageRoot

                value = record.data
                for part in path:
                    value = value.get(part) if isinstance(value, Mapping) else None
                if value is None:
                    continue
                try:
                    root = NativeMessageRoot.from_json(value)
                except (TypeError, ValueError):
                    continue  # Typed decoding rejects malformed references.
                if (len(root.slices) == 1 and root.slices[0].start == root.slices[0].count - 1
                        and root.slices[0].end == root.slices[0].count
                        and root not in self._native_message_rows):
                    native_missing = True
                    break
            if native_missing:
                required.append(key)
                continue
            for path in paths:
                value = record.data
                for part in path:
                    value = value.get(part) if isinstance(value, Mapping) else None
                if not isinstance(value, str):
                    continue
                try:
                    identity = validate_archive_content_identity(value)
                except AIError:
                    continue
                target = record_key(namespace, tenant, "archive", "archive_content", (identity,))
                if target not in content._record_cache:
                    required.append(key)
                    break
        extras = {store: tuple(dict.fromkeys(validate_record_key(key) for key in requested
                    if key not in self.transaction(store)._record_cache))
                  for store, requested in (additional_records or {}).items()}
        extras = {store: requested for store, requested in extras.items() if requested}
        if sum(map(len, extras.values())) > 32:
            await self.get_records(extras)
            extras = {}
        extra_stores = {self.transaction(store)._store_key: store for store in extras}
        found: dict[StateStore, dict[bytes, StoredRecord]] = {owner_store: {}, content_store: {}}
        found.update({store: {} for store in extras if store not in found})
        size = max(1, content_reference_batch_size(len(paths), native_path_count=len(native_paths))
                   - (sum(map(len, extras.values())) * 3 + 5) // 6)
        offsets = (0,) if owner_query is not None else range(0, len(required), size)
        for offset in offsets:
            batch = tuple(required[offset:offset + size])
            statement = content_reference_statement(
                self._group.metadata, dialect=dialect, owner_store_key=owner._store_key, owner_keys=batch,
                content_store_key=content._store_key, content_scope_key=content_scope_key, content_paths=paths,
                typed_payloads=(content._store_key == STATE_GROUP_ARCHIVE_KEY
                    and any(path[-1] == "$execution_payload" for path in paths)),
                additional_records={self.transaction(store)._store_key: requested for store, requested in extras.items()} if offset == 0 else None,
                owner_statement=(_record_query_statement(self._group.metadata, owner._store_key, owner_query)
                                 if owner_query is not None else None),
                native_root_paths=native_paths,
            )
            captured_rows = tuple((await owner._execute(statement)).mappings())
            # UNION order is unspecified; seed owner rows before decoding source locators.
            for row in captured_rows:
                if row["source_kind"] == 0:
                    if row["store_key"] != owner._store_key or owner_query is None and row["key"] not in batch:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    record = _record_from_row(row)
                    owner._record_cache[record.key] = record
                    found[owner_store][record.key] = record
            for row in captured_rows:
                kind = row["source_kind"]
                if kind == 0:
                    continue
                if kind == 4:
                    from ._message_root import NativeMessageRoot
                    from ..steps import native_store_key

                    if row["store_key"] != owner._store_key or owner_query is None and row["key"] not in batch:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    slot = row["native_slot"]
                    if type(slot) is not int or not 0 <= slot < len(native_paths):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    record = found[owner_store].get(row["key"])
                    if record is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    value = record.data
                    for part in native_paths[slot]:
                        value = value.get(part) if isinstance(value, Mapping) else None
                    try:
                        root = NativeMessageRoot.from_json(value)
                    except (TypeError, ValueError):
                        # Ordinary typed decoding still rejects malformed references.
                        continue
                    opened = self._group.opened_archive
                    if (opened is None or root.group_id != opened[0]
                            or root.store_key != native_store_key(namespace, tenant)):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if len(root.slices) != 1:
                        continue
                    item = root.slices[0]
                    if item.start != item.count - 1 or item.end != item.count:
                        continue
                    self._native_message_rows[root] = (() if row["native_sha256"] is None else ({
                        name.removeprefix("native_"): value for name, value in row.items()
                        if name.startswith("native_") and name != "native_slot"},))
                    continue
                if kind == 3:
                    target_store = extra_stores.get(row["store_key"])
                    if target_store is None or row["key"] not in extras[target_store]:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    transaction = self.transaction(target_store)
                    record = _record_from_row(row)
                elif kind == 2:
                    record = payload_record_from_row({key.removeprefix("typed_"): value
                        for key, value in row.items() if key.startswith("typed_")})
                    target_store, transaction = content_store, content
                else:
                    target_store, transaction = (owner_store, owner) if kind == 0 else (content_store, content)
                    if kind not in {0, 1} or row["store_key"] != transaction._store_key:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    record = _record_from_row(row)
                    if kind == 0 and owner_query is None and record.key not in batch:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                transaction._record_cache[record.key] = record
                found[target_store][record.key] = record
            for key in batch:
                owner._record_cache.setdefault(key, None)
        if required:
            for store, requested in extras.items():
                for key in requested:
                    self.transaction(store)._record_cache.setdefault(key, None)
        elif extras:
            await self.get_records(extras)
        found[owner_store].update((key, record) for key in keys if (record := owner._record_cache.get(key)) is not None)
        for record in tuple(found[owner_store].values()):
            for path in paths:
                value = record.data
                for part in path:
                    value = value.get(part) if isinstance(value, Mapping) else None
                if not isinstance(value, str):
                    continue
                try:
                    identity = validate_archive_content_identity(value)
                except AIError:
                    continue
                target = record_key(namespace, tenant, "archive", "archive_content", (identity,))
                if (source := content._record_cache.get(target)) is not None:
                    found[content_store][target] = source
        return found

    def captured_native_message_rows(
        self, context: "SqlStorageContext", roots: Sequence["NativeMessageRoot"],
    ) -> tuple[Mapping[str, object], ...] | None:
        active = active_state_scope()
        if (active is None or active.writable or active.record_reader is not self
                or context is not self._group._context):
            return None
        if any(root not in self._native_message_rows for root in roots):
            return None
        rows = []
        seen = set()
        for request, root in enumerate(roots):
            # The existing range decoder aliases repeated identical slices.
            if root.slices[0] in seen:
                continue
            seen.add(root.slices[0])
            rows.extend(dict(row, request=request, span=0, position=0)
                        for row in self._native_message_rows[root])
        return tuple(rows)

    @classmethod
    def _record_read_batches(
        cls, requests: Mapping[StateStore, Sequence[bytes]],
    ) -> Iterator[Mapping[StateStore, tuple[bytes, ...]]]:
        batch: dict[StateStore, tuple[bytes, ...]] = {}
        used = 0
        for store, keys in requests.items():
            offset = 0
            while offset < len(keys):
                if _GROUP_RECORD_BIND_BUDGET - used < 3:
                    yield batch
                    batch, used = {}, 0
                count = min(len(keys) - offset, (_GROUP_RECORD_BIND_BUDGET - used - 1) // 2)
                batch[store] = tuple(keys[offset:offset + count])
                offset += count
                used += 2 * count + 1
                if offset < len(keys):
                    yield batch
                    batch, used = {}, 0
        if batch:
            yield batch

    def _record_read_statement(
        self, requests: Mapping[StateStore, Sequence[bytes]],
    ) -> "Select":
        from sqlalchemy import and_, or_, select

        table = record_owner_rows(self._group.metadata, keys=tuple(key for keys in requests.values() for key in keys))
        return select(table).where(or_(*(and_(
            table.c.store_key == self.transaction(store)._store_key,
            _keys_match(table.c.key, tuple(validate_record_key(key) for key in keys)),
        ) for store, keys in requests.items())))


def _payload_table(metadata: "MetaData") -> "Table":
    """Resolve the shared content owner without implicit schema provisioning."""
    from ._step_schema import build_step_sql_metadata

    table = metadata.tables.get("ai_step_messages")
    return build_step_sql_metadata().tables["ai_step_messages"] if table is None else table


class SqlStateStorageGroup:
    """Own one SQL context and one physical transaction for its logical stores.

    ``native_content_index`` declares provisioned native owner tables; it never
    probes or creates them. Binding a native partition makes the same promise.
    """

    def __init__(
        self,
        context: SqlStorageContext,
        metadata: "MetaData",
        *,
        owns_context: bool = False,
        read_only: bool = False,
        uses_archive: bool = False,
        native_content_index: bool = False,
        archive_cache: CacheBackend | None = None,
    ) -> None:
        self._context = context
        self._metadata = metadata
        self._owns_context = owns_context
        self._read_only = read_only
        self._mutation_lock = (
            asyncio.Lock() if context.dialect.single_writer else None
        )
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._initialized = False
        self._members: dict[bytes, SqlStateStore] = {}
        self._archive_cache = archive_cache
        self._archive_member: SqlStateStore | None = None
        self._opened_archive: tuple[str, SqlStateStore] | None = None
        self._archive_scope_initialized = False
        self._archive_on_mutation = uses_archive
        self._owned_members: set[SqlStateStore] = set()
        self._native_content_index = native_content_index

    def bind_native_partition(self, store_key: bytes) -> None:
        """Enable publication validation for explicitly provisioned Native tables."""
        from ..steps import native_store_locator
        native_store_locator(store_key)
        self._native_content_index = True

    def add_member(self, store: "SqlStateStore") -> None:
        self._members.setdefault(store.store_key, store)

    async def open_partition(
        self, key: bytes, *, create: bool = False,
    ) -> "SqlStateStore | None":
        validate_store_key(key)
        if active_state_scope() is not None:
            raise StateTransactionNestingError("open_partition must run outside a state callback")
        await self.initialize()
        if create and self._read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        existing = self._members.get(key)
        if existing is not None:
            return existing
        if not create:
            from sqlalchemy import func, literal, select, union

            statements = [
                select(self._metadata.tables[name].c.store_key).where(
                    self._metadata.tables[name].c.store_key == key,
                )
                for name in (*RECORD_OWNER_TABLES, "ai_state_facts",
                             "ai_state_counters", "ai_state_operations")
            ]
            if key == STATE_GROUP_ARCHIVE_KEY:
                messages = _payload_table(self._metadata)
                statements.append(select(literal(key)).where(func.substr(messages.c.store_key, 1, 14) == b"\x00\x0cstep_payload"))
            async with self._session() as session:
                result = await self._context.run_statement(session.execute(union(*statements).limit(1)))
                if result.first() is None:
                    return None
        return await self._open_partition(key)

    @property
    def opened_archive(self) -> tuple[str, "SqlStateStore"] | None:
        return self._opened_archive

    async def open_archive(
        self, *, create: bool = False,
    ) -> tuple[str, "SqlStateStore"] | None:
        if active_state_scope() is not None:
            raise StateTransactionNestingError("open_archive must run outside a state callback")
        await self.initialize()
        if create and self._read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        metadata_key = STATE_GROUP_METADATA_KEY
        metadata_store = await self._open_partition(metadata_key)
        if create and self._archive_on_mutation and not self._archive_scope_initialized:
            # The mutation already establishes and confirms the archive identity.
            # Do not repeat its absence/content checks in separate read snapshots.
            async def prepare(_group: StateGroupTransaction) -> None:
                pass
            await self.mutate((metadata_store,), prepare)
            if self._opened_archive is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return self._opened_archive
        key = record_key("", "", "archive", "archive_scope", ("archive_scope_id",))
        record = await metadata_store.read(lambda tx: tx.get_record(key))
        archive_key = STATE_GROUP_ARCHIVE_KEY
        if record is None:
            archive = await self._open_partition(archive_key)
            if await archive.read(lambda tx: tx.scan_records_page(after=None, limit=1)):
                record = await metadata_store.read(lambda tx: tx.get_record(key))
                if record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record is None and create:
            async def initialize_scope(tx: StateTransaction) -> StoredRecord:
                existing = await tx.get_record(key)
                if existing is not None:
                    return existing
                value = StoredRecord(
                    key=key, scope_key=None, parent_key=None,
                    kind="archive_scope", sort_key=b"archive_scope", state=None,
                    storage_revision=0, lease_owner=None, lease_fence=0,
                    lease_expires_at=None, data={"archive_scope_id": uuid4().hex},
                )
                await tx.insert_record(value)
                return value
            try:
                record = await metadata_store.mutate(initialize_scope)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                record = await metadata_store.read(lambda tx: tx.get_record(key))
                if record is None:
                    raise
        if record is None:
            return None
        scope = record.data.get("archive_scope_id")
        if not isinstance(scope, str) or len(scope) != 32 or any(c not in "0123456789abcdef" for c in scope):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        archive_key = STATE_GROUP_ARCHIVE_KEY
        self._archive_member = await self._open_partition(archive_key)
        self._archive_scope_initialized = True
        self._opened_archive = scope, self._archive_member
        return self._opened_archive

    async def _initialize_archive_scope(
        self, session: "AsyncSession", clock: "_SqlTransactionClock",
    ) -> tuple[str, bool]:
        """Establish resource identity in the first owner mutation's commit."""
        metadata_key = STATE_GROUP_METADATA_KEY
        key = record_key("", "", "archive", "archive_scope", ("archive_scope_id",))
        metadata = _SqlTransaction(
            session, self._metadata, self._context, metadata_key, clock=clock,
        )
        record = await metadata.get_record(key)
        if record is not None:
            scope = record.data.get("archive_scope_id")
            if not isinstance(scope, str) or len(scope) != 32 or any(c not in "0123456789abcdef" for c in scope):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return scope, True
        from sqlalchemy import func, literal, select, union_all

        # Only existence matters here; never hydrate an orphan's stored body.
        statements = [select(literal(1)).select_from(self._metadata.tables[name]).where(
            self._metadata.tables[name].c.store_key == STATE_GROUP_ARCHIVE_KEY)
            for name in RECORD_OWNER_TABLES]
        messages = _payload_table(self._metadata)
        statements.append(select(literal(1)).select_from(messages).where(
            func.substr(messages.c.store_key, 1, 14) == b"\x00\x0cstep_payload"))
        occupied = await self._context.run_statement(session.execute(union_all(*statements).limit(1)))
        if occupied.first() is not None:
            # Under READ COMMITTED another writer can publish between the two
            # reads. Recheck identity before treating visible content as orphaned.
            metadata = _SqlTransaction(
                session, self._metadata, self._context, metadata_key, clock=clock,
            )
            record = await metadata.get_record(key)
            scope = None if record is None else record.data.get("archive_scope_id")
            if not isinstance(scope, str) or len(scope) != 32 or any(c not in "0123456789abcdef" for c in scope):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return scope, True
        record = StoredRecord(
            key=key, scope_key=None, parent_key=None,
            kind="archive_scope", sort_key=b"archive_scope", state=None,
            storage_revision=0, lease_owner=None, lease_fence=0,
            lease_expires_at=None, data={"archive_scope_id": uuid4().hex},
        )
        # Another first writer may already have established this resource's
        # immutable identity. Its winning value must never be overwritten.
        await self._context.run_statement(self._context.dialect.insert_ignore_conflict(
            session, table=self._metadata.tables["ai_state_records"],
            values={**_record_values(record), "store_key": metadata_key},
            index_elements=("store_key", "key_locator"),
        ))

        # A competing first writer can win after our initial absence read.
        # Expose only its committed identity, never our rollback-prone candidate.
        metadata = _SqlTransaction(session, self._metadata, self._context, metadata_key, clock=clock)
        winner = await metadata.get_record(key)
        scope = None if winner is None else winner.data.get("archive_scope_id")
        if not isinstance(scope, str) or len(scope) != 32 or any(c not in "0123456789abcdef" for c in scope):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return scope, scope != record.data["archive_scope_id"]

    async def _open_partition(self, key: bytes) -> "SqlStateStore":
        store = self._members.get(key)
        if store is None or store._closed:
            store = SqlStateStore(
                self._context.engine, metadata=self._metadata,
                context=self._context, group=self, store_key=key,
                runtime_domain="group_internal",
                cache=self._archive_cache if key == STATE_GROUP_ARCHIVE_KEY else None,
            )
            self._members[key] = store
            self._owned_members.add(store)
        await store.initialize()
        return store

    async def scan_stores(self) -> tuple["SqlStateStore", ...]:
        from sqlalchemy import func, literal, select, union

        await self.initialize()
        statements = [
            select(self._metadata.tables[name].c.store_key)
            for name in (*RECORD_OWNER_TABLES, "ai_state_facts",
                         "ai_state_counters", "ai_state_operations")
        ]
        messages = _payload_table(self._metadata)
        statements.append(select(literal(STATE_GROUP_ARCHIVE_KEY)).where(
            func.substr(messages.c.store_key, 1, 14) == b"\x00\x0cstep_payload"))
        async with self._session() as session:
            result = await self._context.run_statement(session.execute(union(*statements)))
            keys = tuple(_row_store_key(value) for value in result.scalars())
        return tuple([await self._open_partition(key) for key in keys])

    @property
    def context(self) -> SqlStorageContext:
        return self._context

    @property
    def metadata(self) -> "MetaData":
        return self._metadata

    async def initialize(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._initialized:
            return
        await self._context.initialize()
        if not self._read_only:
            await self._context.require_write_access()
        self._initialized = True
        _logger.info(
            "SQL StateStorageGroup initialized: dialect=%s tables=%s",
            self._context.dialect.name,
            len(self._metadata.tables),
        )

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                for store in self._owned_members:
                    await store.close()
                if self._owns_context:
                    await self._context.close()
            except BaseException:
                self._closed = False
                raise
            self._initialized = False
        _logger.debug("SQL StateStorageGroup closed")

    async def read(self, store: "SqlStateStore", fn: StateCallback[ValueT]) -> ValueT:
        return await self.read_many(
            (store,), lambda group: fn(group.transaction(store))
        )

    async def get_records(
        self, requests: Mapping[StateStore, Sequence[bytes]],
    ) -> Mapping[StateStore, Mapping[bytes, StoredRecord]]:
        """Read only within an existing group callback, without opening a scope."""
        transaction = active_state_group_transaction(self, tuple(requests), writable=False)
        if not isinstance(transaction, GroupRecordReader):
            raise StateTransactionNestingError("no active group record reader")
        return await transaction.get_records(requests)

    async def read_many(
        self,
        stores: Sequence["SqlStateStore"],
        fn: StateGroupCallback[ValueT],
    ) -> ValueT:
        members = state_content_members((*stores, *((self._archive_member,) if self._archive_member else ())))
        if not stores:
            raise ValueError("StateStorageGroup read requires a store")
        for store in members:
            self._ensure_member(store)
        active = active_state_transaction(members[0])
        if active is not None:
            return await fn(active_state_group_transaction(self, members, writable=False))
        async with self._session() as session:
            async with session.begin():
                await self._context.dialect.begin_consistent_read(session)
                clock = _SqlTransactionClock(session, self._context)
                transactions = self._create_transactions(members, session, clock)
                group_transaction = _SqlGroupTransaction(self, transactions)
                token = bind_state_scope(self, transactions, writable=False, record_reader=group_transaction)
                try:
                    return await fn(active_state_group_transaction(self, members, writable=False))
                finally:
                    reset_state_transaction(token)

    def _create_transactions(
        self, stores: Sequence["SqlStateStore"], session: "AsyncSession",
        clock: "_SqlTransactionClock",
        observers: list[Callable[[bool], None]] | None = None,
    ) -> dict["SqlStateStore", "_SqlTransaction"]:
        partitions: dict[bytes, _SqlTransaction] = {}
        transactions: dict[SqlStateStore, _SqlTransaction] = {}
        for store in stores:
            key = store.store_key
            if key not in partitions:
                partitions[key] = _SqlTransaction(
                    session, self._metadata, self._context, key, clock=clock, observers=observers,
                    native_content_index=self._native_content_index,
                )
            transactions[store] = partitions[key]
        return transactions

    async def mutate(
        self,
        stores: Sequence["SqlStateStore"],
        fn: StateGroupCallback[ValueT],
    ) -> ValueT:
        if self._read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        if not stores:
            raise ValueError("StateStorageGroup mutation requires a store")
        for store in stores:
            self._ensure_member(store)
        # Enlist the logical archive before the callback. Its existing identity
        # is discovered inside the mutation snapshot, including a competing
        # writer's first publication; mutation callbacks are never replayed.
        if active_state_scope() is None and self._archive_on_mutation and self._archive_member is None:
            self._archive_member = await self._open_partition(STATE_GROUP_ARCHIVE_KEY)
        members = state_content_members((*stores, *((self._archive_member,) if self._archive_member else ())))
        active = active_state_transaction(members[0], writable=True)
        if active is not None:
            return await fn(active_state_group_transaction(self, members))

        captured_at = 0.0
        captures: dict[SqlStateStore, tuple[StoredRecord, ...]] = {}
        capture_tokens: dict[SqlStateStore, object] = {}
        changed: set[SqlStateStore] = set()
        confirmed = False
        pending_archive_scope: str | None = None
        observers: list[Callable[[bool], None]] = []
        aborted_observers: list[Callable[[bool], None]] = []

        async def execute(session: "AsyncSession") -> ValueT:
            nonlocal captured_at, captures, capture_tokens, changed, pending_archive_scope
            aborted_observers.extend(observers)
            observers.clear()
            captured_at = time()
            capture_tokens = {store: store._reads.capture_token for store in members}
            clock = _SqlTransactionClock(session, self._context)
            if self._archive_on_mutation and not self._archive_scope_initialized:
                pending_archive_scope, already_committed = await self._initialize_archive_scope(session, clock)
                if already_committed:
                    self._opened_archive = pending_archive_scope, self._archive_member
            transactions = self._create_transactions(members, session, clock, observers)
            group_transaction = _SqlGroupTransaction(self, transactions)
            token = bind_state_scope(self, transactions, record_reader=group_transaction)
            try:
                result = await fn(group_transaction)
                for transaction in transactions.values():
                    if transaction._mutation_error is not None:
                        raise transaction._mutation_error
                if self._native_content_index:
                    from ._native_content_index import flush_owners
                    await flush_owners(self._context, session,
                                       group_id=None if self._opened_archive is None else self._opened_archive[0])
                captures = {
                    store: tuple(record for key, record in transaction._record_cache.items()
                                 if key in transaction._guarded_record_revisions
                                 and record is not None and record.kind in {"session", "execution"})
                    for store, transaction in transactions.items()
                }
                changed_partitions = {
                    transaction._store_key for transaction in transactions.values()
                    if transaction._guarded_record_revisions or transaction._deleted_records or transaction._payloads_published
                }
                changed = {store for store in members if store.store_key in changed_partitions}
                return result
            finally:
                reset_state_transaction(token)

        lock = self._mutation_lock
        if lock is not None:
            await lock.acquire()
        try:
            from ._native_content_index import run_native_mutation
            result = await run_native_mutation(self._context, execute, domain="runtime.state.group")
            confirmed = True
            if pending_archive_scope is not None:
                self._opened_archive = pending_archive_scope, self._archive_member
            if self._archive_on_mutation:
                self._archive_scope_initialized = True
            return result
        except asyncio.CancelledError:
            raise
        except AIError:
            raise
        except Exception as error:
            from sqlalchemy.exc import (
                DBAPIError,
                DataError,
                IntegrityError,
                InterfaceError,
                OperationalError,
                ProgrammingError,
                SQLAlchemyError,
                TimeoutError as SqlAlchemyTimeoutError,
            )

            internal_details = {"phase": "runtime_storage_sql_mutation"}
            if isinstance(error, IntegrityError):
                violation = self._context.dialect.classify_integrity_error(error)
                code = (
                    ErrorCode.STORAGE_CONFLICT
                    if violation is IntegrityViolationKind.UNIQUE_CONFLICT
                    else ErrorCode.INTERNAL_ERROR
                )
            elif isinstance(error, (ProgrammingError, DataError)):
                code = ErrorCode.INTERNAL_ERROR
            elif isinstance(
                error,
                (InterfaceError, OperationalError, SqlAlchemyTimeoutError),
            ):
                code = ErrorCode.STORAGE_UNAVAILABLE
            elif isinstance(error, DBAPIError):
                code = (
                    ErrorCode.STORAGE_UNAVAILABLE
                    if error.connection_invalidated is True
                    else ErrorCode.INTERNAL_ERROR
                )
            elif isinstance(error, SQLAlchemyError):
                code = ErrorCode.INTERNAL_ERROR
            else:
                code = ErrorCode.INTERNAL_ERROR

            _logger.error(
                "SQL Runtime storage mutation error mapped: error_type=%s code=%s",
                type(error).__name__,
                code.value,
            )
            if code is ErrorCode.INTERNAL_ERROR:
                raise AIError(
                    code,
                    retryable=False,
                    safe_details=internal_details,
                ) from error
            raise AIError(code) from error
        finally:
            if lock is not None:
                lock.release()
            try:
                for store in members:
                    if confirmed and store not in changed:
                        continue
                    previous = await store._reads.invalidate()
                    # A delayed source return must not acquire a newer capture boundary.
                    if confirmed and previous is capture_tokens.get(store):
                        for record in captures.get(store, ()):
                            store._reads.publish_record(record, captured_at=captured_at)
            finally:
                for observer in (observers if confirmed else (*aborted_observers, *observers)):
                    try:
                        observer(confirmed)
                    except Exception:
                        # Cache maintenance cannot change the durable commit outcome.
                        _logger.warning("SQL mutation observer failed", exc_info=True)
                observers.clear()

    @asynccontextmanager
    async def _session(self):
        session = self._context.sessions()
        try:
            yield session
        finally:
            await session.close()

    def _ensure_member(self, store: "SqlStateStore") -> None:
        if store.storage_group is not self:
            raise RuntimeError("store does not belong to this StateStorageGroup")
        if store._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if not store._initialized:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if not self._initialized:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)


class SqlStateStore:
    """Optimistic SQL StateStore backed by the five primitive tables.

    Standalone native owners opt into ``native_content_index`` after explicit
    provisioning. Stores sharing a group use that group's owner declaration.
    """

    def __init__(
        self,
        engine: "AsyncEngine",
        *,
        metadata: "MetaData | None" = None,
        context: "SqlStorageContext | None" = None,
        runtime_domain: PersistenceDomain | str = PersistenceDomain.CONVERSATION,
        group: SqlStateStorageGroup | None = None,
        cache: CacheBackend | None = None,
        store_key: bytes | None = None,
        native_content_index: bool = False,
    ) -> None:
        resolved_context = context or create_sql_storage_context(engine)
        self._metadata = (
            metadata
            if metadata is not None
            else build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
        )
        self._runtime_domain = runtime_domain
        self._store_key = standalone_store_key(self.partition) if store_key is None else validate_store_key(store_key)
        self._owns_group = group is None
        self._storage_group = group or SqlStateStorageGroup(
            resolved_context,
            self._metadata,
            owns_context=context is None,
            native_content_index=native_content_index,
        )
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._initialized = False
        self._reads = StateRecordCache(self, cache, scope=f"sql:{self._store_key.hex()}")
        self._content_dependencies: tuple[StateStore, ...] = ()
        self._storage_group.add_member(self)

    @property
    def content_dependencies(self) -> tuple[StateStore, ...]:
        return self._content_dependencies

    def bind_content_dependency(self, source: StateStore) -> None:
        if source.storage_group is not self.storage_group:
            raise ValueError("content dependency must share its owning group")
        if active_state_scope() is not None:
            raise StateTransactionNestingError("content dependency must be bound before transaction entry")
        if source is not self and source not in self._content_dependencies:
            self._content_dependencies = (*self._content_dependencies, source)

    @property
    def record_cache_enabled(self) -> bool:
        """Report the configured ordinary-read policy, without owning new state."""
        return self._reads.enabled

    @property
    def context(self) -> SqlStorageContext:
        return self._storage_group.context

    @property
    def storage_group(self) -> StateStorageGroup:
        return self._storage_group

    @property
    def runtime_domain(self) -> PersistenceDomain | str:
        return self._runtime_domain

    @property
    def partition(self) -> str:
        return self._runtime_domain.value if isinstance(self._runtime_domain, PersistenceDomain) else self._runtime_domain

    @property
    def store_key(self) -> bytes:
        return self._store_key

    async def initialize(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        await self._storage_group.initialize()
        self._initialized = True

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                await self._reads.close()
                if self._owns_group:
                    await self._storage_group.close()
            except BaseException:
                self._closed = False
                raise
            self._initialized = False

    async def read_record(self, key: bytes) -> StoredRecord | None:
        """Read ordinary committed metadata with bounded cache freshness."""
        self._storage_group._ensure_member(self)
        return await self._reads.read_record(key)

    async def read_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        """Read ordinary metadata; explicit transaction scopes stay source-only."""
        self._storage_group._ensure_member(self)
        return await self._reads.read_records(keys)

    async def refresh_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        """Refresh an ordinary batch without changing authoritative read scopes."""
        self._storage_group._ensure_member(self)
        return await self._reads.refresh_records(keys)

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
        self._ensure_ready()
        record_cursor: RecordScanCursor | None = None
        while True:
            records = await self.read(
                lambda transaction, cursor=record_cursor: transaction.scan_records_page(
                    after=cursor,
                    limit=_MAINTENANCE_PAGE_SIZE,
                )
            )
            if not records:
                break
            last = records[-1]
            record_cursor = RecordScanCursor(last.kind, last.key)

        async with self._storage_group._session() as session:
            transaction = _SqlTransaction(
                session,
                self._metadata,
                self.context,
                self._store_key,
            )
            from sqlalchemy import and_, select

            records = record_owner_rows(self._metadata)
            facts = transaction._table("ai_state_facts")
            sequences = transaction._table("ai_state_counters")
            sequence_rows = (
                (
                    await transaction._execute(
                        select(sequences.c.stream_key, sequences.c.stream_locator, sequences.c.value).where(
                            sequences.c.store_key == transaction._store_key
                        )
                    )
                )
                .mappings()
                .all()
            )
            for row in sequence_rows:
                _validate_locators(row, "stream_key")
                _row_stream_key(row["stream_key"])
                _row_nonnegative_int(row["value"])
            possible_orphan_keys = (await transaction._execute(
                select(facts.c.owner_key).distinct()
                .select_from(
                    facts.outerjoin(
                        records,
                        and_(
                            facts.c.owner_locator == records.c.key_locator,
                            facts.c.owner_key == records.c.key,
                            records.c.store_key == transaction._store_key,
                        ),
                    )
                )
                .where(
                    facts.c.store_key == transaction._store_key,
                    records.c.id.is_(None),
                )
            )).scalars().all()
            missing_owners = await transaction.get_records(tuple(possible_orphan_keys))
            orphan_fact = next((key for key in possible_orphan_keys if key not in missing_owners), None)
            if orphan_fact is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        fact_cursor: FactScanCursor | None = None
        previous_stream: bytes | None = None
        previous_sequence: int | None = None
        while True:
            values = await self.read(
                lambda transaction, cursor=fact_cursor: transaction.scan_facts_page(
                    after=cursor,
                    limit=_MAINTENANCE_PAGE_SIZE,
                )
            )
            if not values:
                break
            for value in values:
                if value.stream_key != previous_stream:
                    if value.sequence != 1:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                elif (
                    previous_sequence is None or value.sequence != previous_sequence + 1
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                previous_stream = value.stream_key
                previous_sequence = value.sequence
            last = values[-1]
            fact_cursor = FactScanCursor(last.stream_key, last.sequence)

        operation_cursor: OperationScanCursor | None = None
        while True:
            values = await self.read(
                lambda transaction, cursor=operation_cursor: (
                    transaction.scan_operations_page(
                        after=cursor,
                        limit=_MAINTENANCE_PAGE_SIZE,
                    )
                )
            )
            if not values:
                break
            operation_cursor = OperationScanCursor(values[-1].key)
        _logger.info("SQL StateStore integrity validated")

    def _ensure_ready(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if not self._initialized:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)


class _SqlTransactionClock:
    """Keep one database timestamp within one physical transaction."""

    def __init__(self, session: "AsyncSession", context: SqlStorageContext) -> None:
        self._session = session
        self._context = context
        self._now: datetime | None = None

    async def now(self) -> datetime:
        if self._now is None:
            self._now = await self._context.run_statement(
                self._context.dialect.database_now(self._session)
            )
        return self._now


class _SqlTransaction:
    def __init__(
        self,
        session: "AsyncSession",
        metadata: "MetaData",
        context: SqlStorageContext,
        store_key: bytes,
        *,
        clock: _SqlTransactionClock | None = None,
        observers: list[Callable[[bool], None]] | None = None,
        native_content_index: bool = False,
    ) -> None:
        validate_store_key(store_key)
        self._session = session
        self._metadata = metadata
        self._context = context
        self._store_key = store_key
        self._guarded_record_revisions: dict[bytes, int] = {}
        self._payloads_published = False
        self._deleted_records = False
        self._record_cache: dict[bytes, StoredRecord | None] = {}
        self._sequence_cache: dict[bytes, int] = {}
        self._clock = clock if clock is not None else _SqlTransactionClock(session, context)
        self._mutation_error: BaseException | None = None
        self._observers = observers
        self._native_content_index = native_content_index

    def cached_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord | None]:
        if not is_active_state_transaction(self):
            raise StateTransactionNestingError("record cache requires its active storage transaction")
        return {key: self._record_cache[key] for key in keys if key in self._record_cache}

    async def read_sql_rows(
        self, context: SqlStorageContext, statement: "Select",
    ) -> tuple[Mapping[str, object], ...]:
        from sqlalchemy.sql import Select, visitors
        from sqlalchemy.sql.dml import UpdateBase
        from sqlalchemy.sql.elements import TextClause
        from sqlalchemy.sql.selectable import ForUpdateArg

        if (not isinstance(statement, Select) or context.engine is not self._context.engine
                or context.closed or self._context.closed or not is_active_state_transaction(self)
                or any(isinstance(node, (UpdateBase, TextClause, ForUpdateArg))
                       for node in visitors.iterate(statement))):
            raise StateTransactionNestingError("native rows require a read-only SQL plan in their active boundary")
        return tuple(dict(row) for row in (await self._execute(statement)).mappings())

    async def read_joined_records(
        self, context: SqlStorageContext, statement: "Select", *, keys: Sequence[bytes] = (),
        sequence_keys: Sequence[bytes] = (),
    ) -> tuple[tuple[StoredRecord, Mapping[str, object]], ...]:
        from sqlalchemy.sql import Select, visitors
        from sqlalchemy.sql.dml import UpdateBase
        from sqlalchemy.sql.elements import TextClause
        from sqlalchemy.sql.selectable import ForUpdateArg

        if (not isinstance(statement, Select) or context.engine is not self._context.engine
                or context.closed or self._context.closed or not is_active_state_transaction(self)
                or any(isinstance(node, (UpdateBase, TextClause, ForUpdateArg))
                       for node in visitors.iterate(statement))):
            raise StateTransactionNestingError("joined records require a read-only SQL plan in their active boundary")
        missing_sequences = tuple(dict.fromkeys(
            key for key in map(validate_stream_key, sequence_keys) if key not in self._sequence_cache
        ))
        rows = (await self._execute(statement)).mappings().all()
        result: list[tuple[StoredRecord, Mapping[str, object]]] = []
        sequences: dict[bytes, int] = {}
        for row in rows:
            if "key" not in row and {"store_key", "sha256", "message", "compression", "raw_size", "gc_revision"}.issubset(row):
                from ._step_payload_owner import payload_record_from_row
                if self._store_key != STATE_GROUP_ARCHIVE_KEY:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                record = payload_record_from_row(row)
            else:
                if row["store_key"] != self._store_key:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                record = _record_from_row(row)
            if missing_sequences:
                for key, column in zip(run_counter_streams(record.key, store_key=self._store_key), RUN_COUNTER_COLUMNS):
                    if key in missing_sequences:
                        if column not in row:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        value = row[column]
                        sequences[key] = 0 if value is None else _row_nonnegative_int(value)
            self._record_cache[record.key] = record
            result.append((record, row))
        self._sequence_cache.update({key: sequences.get(key, 0) for key in missing_sequences})
        for key in keys:
            self._record_cache.setdefault(key, None)
        return tuple(result)

    async def read_joined_facts(
        self, context: SqlStorageContext, statement: "Select",
    ) -> tuple[tuple[StoredFact, Mapping[str, "SqlValue"]], ...]:
        from sqlalchemy.sql import Select, visitors
        from sqlalchemy.sql.dml import UpdateBase
        from sqlalchemy.sql.elements import TextClause
        from sqlalchemy.sql.selectable import ForUpdateArg

        if (not isinstance(statement, Select) or context.engine is not self._context.engine
                or context.closed or self._context.closed
                or context.dialect.name != self._context.dialect.name
                or not is_active_state_transaction(self)
                or any(isinstance(node, (UpdateBase, TextClause, ForUpdateArg))
                       for node in visitors.iterate(statement))):
            raise StateTransactionNestingError(
                "joined facts require a read-only SQL plan in their active boundary"
            )
        rows = (await self._execute(statement)).mappings().all()
        result: list[tuple[StoredFact, Mapping[str, "SqlValue"]]] = []
        for row in rows:
            if row["store_key"] != self._store_key:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result.append((_fact_from_row(row), row))
        return tuple(result)

    def observe_mutation(self, callback: Callable[[bool], None]) -> None:
        if self._observers is None or (not is_active_state_transaction(self) or active_state_scope() is None or not active_state_scope().writable):
            raise StateTransactionNestingError("commit observer requires its active writable SQL boundary")
        self._observers.append(callback)

    async def run_sql(
        self, context: SqlStorageContext,
        callback: Callable[["AsyncSession"], Awaitable[ValueT]],
    ) -> ValueT:
        active = active_state_scope()
        if (context.engine is not self._context.engine or context.closed or self._context.closed
                or context.dialect.name != self._context.dialect.name
                or active is None or not active.writable
                or active.task is not asyncio.current_task()
                or not any(transaction is self for transaction in active.transactions.values())):
            raise StateTransactionNestingError(
                "prepared SQL requires its own active writable storage group transaction"
            )
        try:
            return await callback(self._session)
        except BaseException as error:
            self._abort_mutation(error)
            raise

    def _stage_native_owner(
        self, kind: str, identity: Sequence[str | int | bytes], data: object | None, *,
        state: object = None, inserting: bool = False,
    ) -> None:
        if self._native_content_index:
            from ._native_content_index import stage_owner
            stage_owner(self._session, kind, self._store_key, identity, data, state=state, inserting=inserting)
        elif data is not None:
            from ._archive_closure import iter_native_message_roots
            if next(iter_native_message_roots(data), None) is not None:
                error = AIError(
                    ErrorCode.STORAGE_CAPABILITY_MISSING,
                    "Native content owners require an explicitly provisioned native content tables",
                )
                self._abort_mutation(error)
                raise error

    def _native_record_owner_data(self, record: StoredRecord) -> object:
        from .captures import CAPTURE_RECORD_KINDS, capture_retention_value

        # Content rows own payload objects, never native message roots.
        return (None if record.kind == "archive_content" else
                capture_retention_value(record) if record.kind in CAPTURE_RECORD_KINDS else record.data)

    def _stage_native_record_owner(
        self, record: StoredRecord, *, inserting: bool = False, previous: StoredRecord | None = None,
    ) -> None:
        data = self._native_record_owner_data(record)
        if previous is not None:
            from ._archive_closure import iter_native_message_roots
            # The successful owner CAS validates the transaction's old revision.
            # An update without Native roots needs no publication validation.
            if (next(iter_native_message_roots(self._native_record_owner_data(previous)), None) is None
                    and next(iter_native_message_roots(data), None) is None):
                return
        self._stage_native_owner(record_owner_edge(record.key), (validate_record_key(record.key),), data, state=record.state, inserting=inserting)



    def _abort_mutation(self, error: BaseException) -> None:
        if self._mutation_error is None:
            self._mutation_error = error

    async def _execute(
        self,
        statement: "Executable",
        parameters: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    ) -> "Result[tuple[object, ...]]":
        try:
            return await self._context.run_statement(
                self._session.execute(statement, parameters)
            )
        except asyncio.CancelledError as error:
            # Statement cancellation can arrive after the database applied the write.
            if getattr(statement, "is_dml", False):
                self._abort_mutation(error)
            raise

    async def now(self) -> datetime:
        return await self._clock.now()

    async def get_record(self, key: bytes) -> StoredRecord | None:
        from sqlalchemy import select

        validate_record_key(key)
        if key in self._record_cache:
            return self._record_cache[key]
        if self._is_payload_key(key):
            value = (await self._payload_owner().read((key,))).get(key)
            self._record_cache[key] = value
            return value
        table = record_owner_rows(self._metadata, keys=(key,))
        row = (
            (
                await self._execute(
                    select(table).where(
                        table.c.store_key == self._store_key,
                        _locator_column(table.c.key) == _key_locator(validate_record_key(key)),
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            self._record_cache[key] = None
            self._sequence_cache.update({stream: 0 for stream in run_counter_streams(key, store_key=self._store_key)})
            return None
        if row is not None and row["key"] != key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = _record_from_row(row)
        self._record_cache[key] = value
        return value

    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        from sqlalchemy import select

        if not keys:
            return {}

        unique_keys = tuple(dict.fromkeys(validate_record_key(key) for key in keys))
        missing = tuple(key for key in unique_keys if key not in self._record_cache)
        payload_keys = tuple(key for key in missing if self._is_payload_key(key))
        if payload_keys:
            found = await self._payload_owner().read(payload_keys)
            self._record_cache.update((key, found.get(key)) for key in payload_keys)
            missing = tuple(key for key in missing if key not in payload_keys)
        table = record_owner_rows(self._metadata, keys=missing or unique_keys)
        rows = ()
        if missing:
            rows = (
                (
                    await self._execute(
                        select(table).where(
                            table.c.store_key == self._store_key,
                            _locator_column(table.c.key).in_(tuple(_key_locator(key) for key in missing))
                        )
                    )
                )
                .mappings()
                .all()
            )
        if any(row["key"] not in missing for row in rows):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values = tuple(_record_from_row(row) for row in rows)
        self._record_cache.update({value.key: value for value in values})
        for key in missing:
            self._record_cache.setdefault(key, None)
        return {
            key: value
            for key in unique_keys
            if (value := self._record_cache[key]) is not None
        }

    async def read_record_subjects(
        self, *, scope_key: bytes, kind: str, subject_ids: Sequence[str],
    ) -> tuple[StoredRecord, ...]:
        validate_relation_key(scope_key)
        subjects = tuple(dict.fromkeys(subject_ids))
        if not subjects:
            return ()
        from sqlalchemy import LargeBinary, and_, cast, select

        subject_values = tuple(validate_resource_id(subject).encode("utf-8") for subject in subjects)
        table = record_owner_rows(self._metadata, kind=kind)
        records: list[StoredRecord] = []
        for offset in range(0, len(subjects), _RECORD_SUBJECT_READ_BATCH_SIZE):
            # Keep selective scope/subject discovery separate from partition lookup.
            # The binary cast prevents partition equality from being pushed into
            # this stage and selecting a broad store-prefix index on fresh stats.
            matches = select(table.c.store_key, table.c.key_locator, table.c.key).where(
                _key_matches(table.c.scope_key, scope_key),
                table.c.subject_id.in_(subject_values[offset:offset + _RECORD_SUBJECT_READ_BATCH_SIZE]),
            ).distinct().subquery("subject_matches")
            rows = (await self._execute(select(table).select_from(matches.join(table, and_(
                table.c.store_key == cast(matches.c.store_key, LargeBinary),
                table.c.key_locator == matches.c.key_locator,
                table.c.key == matches.c.key,
            ))).where(table.c.store_key == self._store_key, table.c.kind == kind))).mappings().all()
            records.extend(_record_from_row(row) for row in rows)
        records.sort(key=lambda record: (record.sort_key, record.key))
        self._record_cache.update((record.key, record) for record in records)
        return tuple(records)

    async def read_archive_frontiers(
        self, *, scope_key: bytes, requests: Sequence[ArchiveNodeRead],
        content_keys: Sequence[bytes] = (),
    ) -> Mapping[bytes, StoredRecord] | None:
        dialect = self._context.engine.dialect
        if dialect.name not in {"sqlite", "mysql", "postgresql"}:
            return None
        validate_relation_key(scope_key)
        merged: dict[bytes, ArchiveNodeRead] = {}
        for request in requests:
            previous = merged.get(request.key)
            merged[request.key] = request if previous is None else ArchiveNodeRead(
                request.key, request.metadata_slots | previous.metadata_slots,
                request.content_slots | previous.content_slots,
            )
        all_direct = tuple(dict.fromkeys(content_keys))
        payload_keys = tuple(key for key in all_direct if self._is_payload_key(key))
        direct = tuple(key for key in all_direct if key not in payload_keys)
        for key in direct:
            validate_record_key(key)
        result: dict[bytes, StoredRecord] = dict(await self.get_records(payload_keys))
        for nodes, contents in self._archive_read_batches(tuple(merged.values()), direct):
            statement = self._archive_read_statement(scope_key, nodes, contents)
            compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
            parameter_count = len(compiled.params) if compiled.positiontup is None else len(compiled.positiontup)
            if parameter_count > _ARCHIVE_READ_BIND_BUDGET:
                raise ValueError("archive read exceeds the SQL parameter budget")
            rows = (await self._execute(statement)).mappings().all()
            for row in rows:
                # An absent optional edge contributes no record to the mapping.
                if row["key"] is None:
                    continue
                record = _record_from_row(row)
                result[record.key] = self._record_cache[record.key] = record
        if self._store_key == STATE_GROUP_ARCHIVE_KEY and any(item.content_slots for item in requests):
            from ._archive import archive_scope_parts
            namespace, tenant_id = archive_scope_parts(scope_key)
            nodes = {row.sort_key.decode(): row for row in result.values() if row.kind == "archive_sequence"}
            needed = []
            for request in requests:
                current = result.get(request.key)
                if current is None:
                    continue
                for slot, field in ((ArchiveNodeSlot.SELF, None), (ArchiveNodeSlot.LEFT, "left"),
                                    (ArchiveNodeSlot.RIGHT, "right"), (ArchiveNodeSlot.NEXT, "next")):
                    if not request.content_slots & slot:
                        continue
                    node = current if field is None else nodes.get(current.data.get(field))
                    identity = None if node is None else node.data.get("content")
                    if isinstance(identity, str) and identity.startswith("execution-payload:"):
                        needed.append(record_key(namespace, tenant_id, "archive", "archive_content", (identity,)))
            result.update(await self.get_records(tuple(dict.fromkeys(needed))))
        return result

    @classmethod
    def _archive_read_batches(
        cls, requests: Sequence[ArchiveNodeRead], content_keys: Sequence[bytes],
    ) -> Iterator[tuple[tuple[ArchiveNodeRead, ...], tuple[bytes, ...]]]:
        owner_count = 1
        available = _ARCHIVE_READ_BIND_BUDGET - owner_count * _ARCHIVE_READ_FIXED_BINDS
        nodes: list[ArchiveNodeRead] = []
        contents: list[bytes] = []
        used = 0
        for request in requests:
            # Each physical owner lookup repeats the bounded identity query.
            cost = owner_count * 2 * (int(request.metadata_slots).bit_count() + int(request.content_slots).bit_count())
            if cost > available:
                raise ValueError("archive read parameter budget cannot fit one node")
            if used + cost > available:
                yield tuple(nodes), tuple(contents)
                nodes, contents, used = [], [], 0
            nodes.append(request)
            used += cost
        for key in content_keys:
            if used + 2 > available:
                yield tuple(nodes), tuple(contents)
                nodes, contents, used = [], [], 0
            contents.append(key)
            used += 2
        if nodes or contents:
            yield tuple(nodes), tuple(contents)

    def _payload_field(self, column: "ColumnElement", field: str) -> "ColumnElement":
        from sqlalchemy import JSON, cast, type_coerce

        # The JSON scalar retains canonical text; parse only archive string links.
        encoded = type_coerce(column, JSON)[()].as_string()
        payload = (cast(encoded, JSON) if self._context.dialect.name == "postgresql"
                   else type_coerce(encoded, JSON))
        return payload[field].as_string()

    def _archive_link(self, column: "ColumnElement", field: str) -> "ColumnElement":
        from sqlalchemy import LargeBinary, cast, func
        from sqlalchemy.dialects.mysql import BINARY

        text = self._payload_field(column, field)
        if self._context.dialect.name == "postgresql":
            return func.convert_to(text, "UTF8", type_=LargeBinary())
        return cast(text, BINARY() if self._context.dialect.name == "mysql" else LargeBinary())

    def _archive_read_statement(
        self, scope_key: bytes, requests: Sequence[ArchiveNodeRead], content_keys: Sequence[bytes],
    ) -> "Select | CompoundSelect":
        from sqlalchemy import and_, func, literal, select, union

        table = self._table("ai_state_records")
        current = table.alias("archive_current")
        current_scope = and_(current.c.store_key == self._store_key,
                             _key_matches(current.c.scope_key, scope_key), current.c.kind == "archive_sequence")
        branches = []
        for slot, field in ((ArchiveNodeSlot.SELF, None), (ArchiveNodeSlot.LEFT, "left"),
                            (ArchiveNodeSlot.RIGHT, "right"), (ArchiveNodeSlot.NEXT, "next")):
            metadata_keys = tuple(validate_record_key(item.key) for item in requests if item.metadata_slots & slot)
            body_keys = tuple(validate_record_key(item.key) for item in requests if item.content_slots & slot)
            if not metadata_keys and not body_keys:
                continue
            identity = current.c.sort_key if field is None else self._archive_link(current.c.payload_json, field)
            if metadata_keys:
                branches.append(select(literal("archive_sequence").label("kind"), identity.label("identity")).where(
                    current_scope, _keys_match(current.c.key, metadata_keys),
                ))
            if not body_keys:
                continue
            node, joined = current, current
            if field is not None:
                node = table.alias("archive_" + field)
                joined = current.outerjoin(node, and_(
                    node.c.store_key == self._store_key, _key_matches(node.c.scope_key, scope_key),
                    node.c.kind == "archive_sequence", node.c.sort_prefix == func.substr(identity, 1, 128),
                    node.c.sort_key == identity,
                ))
            branches.append(select(literal("archive_content").label("kind"),
                                   self._archive_link(node.c.payload_json, "content").label("identity"))
                            .select_from(joined).where(current_scope, _keys_match(current.c.key, body_keys)))
        selections = []
        if branches:
            keys = (branches[0] if len(branches) == 1 else union(*branches)).subquery("archive_record_keys")
            # Metadata-only frontiers cannot reference payload owners.
            owners = ("ai_state_records",)
            for owner in owners:
                targets = record_owner_rows(self._metadata, owner=owner)
                lookup = select(targets).select_from(keys.outerjoin(targets, and_(
                    targets.c.store_key == self._store_key, _key_matches(targets.c.scope_key, scope_key),
                    targets.c.kind == keys.c.kind, targets.c.sort_prefix == func.substr(keys.c.identity, 1, 128),
                    targets.c.sort_key == keys.c.identity,
                )))
                selections.append(lookup)
        # Direct content addresses already identify their physical owner. Do not
        # rediscover them through an unindexed content-identity join.
        for owner in dict.fromkeys(record_owner_table(key, store_key=self._store_key) for key in content_keys):
            owner_keys = tuple(validate_record_key(key) for key in content_keys
                               if record_owner_table(key, store_key=self._store_key) == owner)
            content = record_owner_rows(self._metadata, keys=owner_keys)
            selections.append(select(content).where(
                content.c.store_key == self._store_key, _key_matches(content.c.scope_key, scope_key),
                content.c.kind == "archive_content", _keys_match(content.c.key, owner_keys),
            ))
        return selections[0] if len(selections) == 1 else union(*selections)

    async def insert_record(self, record: StoredRecord) -> None:
        await self.insert_records((record,))

    async def insert_records(self, records: Sequence[StoredRecord]) -> None:
        values = tuple(records)
        owners = tuple(dict.fromkeys(record_owner_table(record.key, store_key=self._store_key) for record in values))
        if len(owners) > 1:
            try:
                for owner in owners:
                    batch = tuple(record for record in values if record_owner_table(record.key, store_key=self._store_key) == owner)
                    await self.insert_records(batch)
            except BaseException as error:
                self._abort_mutation(error)
                raise
            return
        if owners == ("ai_step_messages",):
            try:
                if len({record.key for record in values}) != len(values):
                    raise ValueError("duplicate payload keys")
                await self._payload_owner().insert(values, immutable=False)
            except BaseException as error:
                self._abort_mutation(error)
                raise
            for record in values:
                self._record_cache[record.key] = record
                self._guarded_record_revisions[record.key] = record.storage_revision
            return
        keys = [record.key for record in values]
        if len(keys) != len(set(keys)):
            raise ValueError("insert_records contains duplicate keys")
        if not values:
            return
        for record in values:
            validate_record_identity(record)
        from sqlalchemy import insert

        await self._execute(
            insert(self._table(owners[0])).values(
                [
                    {
                        **record_owner_values(owners[0], _record_values(record)),
                        "store_key": self._store_key,
                    }
                    for record in sorted(values, key=lambda value: value.key)
                ]
            )
        )
        for record in values:
            self._stage_native_record_owner(record, inserting=True)
        self._log_batch("insert_records", len(values), 1)
        for record in values:
            self._guarded_record_revisions[record.key] = record.storage_revision
            self._record_cache[record.key] = record

    async def insert_immutable_records(self, records: Sequence[StoredRecord]) -> None:
        values = tuple(sorted(records, key=lambda record: record.key))
        owners = tuple(dict.fromkeys(record_owner_table(record.key, store_key=self._store_key) for record in values))
        if len(owners) > 1:
            try:
                for owner in owners:
                    batch = tuple(record for record in values if record_owner_table(record.key, store_key=self._store_key) == owner)
                    await self.insert_immutable_records(batch)
            except BaseException as error:
                self._abort_mutation(error)
                raise
            return
        if owners == ("ai_step_messages",):
            try:
                if len({record.key for record in values}) != len(values):
                    raise ValueError("duplicate payload keys")
                inserted = await self._payload_owner().insert(values, immutable=True)
                # A mixed/unknown insertion count still invalidates this partition,
                # without inventing a physical revision for existing payloads.
                self._payloads_published = bool(values)
            except BaseException as error:
                self._abort_mutation(error)
                raise
            for record in values:
                if record.key in inserted:
                    self._record_cache[record.key] = record
                    self._guarded_record_revisions[record.key] = record.storage_revision
                else:
                    # An ignored candidate is not an observation of the winner.
                    # Retain an actual row only when this transaction already
                    # inserted/guarded its revision; publication cannot change it.
                    cached = self._record_cache.get(record.key)
                    if (cached is None or self._guarded_record_revisions.get(record.key)
                            != cached.storage_revision):
                        self._record_cache.pop(record.key, None)
            return
        try:
            if len({record.key for record in values}) != len(values):
                raise ValueError("insert_immutable_records contains duplicate keys")
            for record in values:
                validate_record_identity(record)
            for offset in range(0, len(values), _IMMUTABLE_RECORD_BATCH_SIZE):
                rows = [{**record_owner_values(owners[0], _record_values(record)), "store_key": self._store_key}
                        for record in values[offset:offset + _IMMUTABLE_RECORD_BATCH_SIZE]]
                await self._context.run_statement(self._context.dialect.insert_immutable_many(
                    self._session, table=self._table(owners[0]), rows=rows,
                    index_elements=("store_key", "key_locator"), conflict_column="storage_revision",
                ))
        except BaseException as error:
            self._abort_mutation(error)
            raise
        for record in values:
            self._stage_native_record_owner(record, inserting=True)
            self._guarded_record_revisions[record.key] = record.storage_revision
            self._record_cache[record.key] = record

    async def guard_record(
        self, key: bytes, *, expected_storage_revision: int,
    ) -> StoredRecord | None:
        await self.get_record(key)
        revision = await self.guard_record_revision(key, expected_storage_revision=expected_storage_revision)
        return None if revision is None else await self.get_record(key)

    async def guard_record_revision(
        self, key: bytes, *, expected_storage_revision: int,
    ) -> int | None:
        if (
            isinstance(expected_storage_revision, bool)
            or not isinstance(expected_storage_revision, int)
            or expected_storage_revision < 0
        ):
            raise ValueError("expected_storage_revision must be a non-negative integer")
        if key in self._guarded_record_revisions:
            return self._guarded_record_revisions[key]
        from sqlalchemy import update

        if self._is_payload_key(key):
            if not await self._payload_owner().guard(key, expected_storage_revision):
                self._record_cache.pop(key, None)
                return None
            revision = expected_storage_revision + 1
            self._guarded_record_revisions[key] = revision
            cached = self._record_cache.get(key)
            if cached is not None and cached.storage_revision == expected_storage_revision:
                self._record_cache[key] = replace(cached, storage_revision=revision)
            else:
                self._record_cache.pop(key, None)
            return revision
        table = self._table(record_owner_table(key, store_key=self._store_key))
        result = await self._execute(
            update(table).where(
                table.c.store_key == self._store_key,
                _key_matches(table.c.key, validate_record_key(key)),
                table.c.storage_revision == expected_storage_revision,
            ).values(storage_revision=expected_storage_revision + 1)
        )
        if result.rowcount != 1:
            self._record_cache.pop(key, None)
            return None
        revision = expected_storage_revision + 1
        self._guarded_record_revisions[key] = revision
        cached = self._record_cache.get(key)
        if cached is not None and cached.storage_revision == expected_storage_revision:
            self._record_cache[key] = replace(cached, storage_revision=revision)
        else:
            self._record_cache.pop(key, None)
        return revision

    async def replace_record(
        self, record: StoredRecord, *, expected_storage_revision: int
    ) -> bool:
        try:
            await self.replace_records(
                (RecordReplacement(record, expected_storage_revision),)
            )
        except AIError as error:
            if error.code is ErrorCode.STORAGE_CONFLICT:
                return False
            raise
        return True

    async def replace_records(self, replacements: Sequence[RecordReplacement]) -> None:
        values = tuple(replacements)
        owners = tuple(dict.fromkeys(record_owner_table(record.record.key, store_key=self._store_key) for record in values))
        if len(owners) > 1:
            try:
                for owner in owners:
                    batch = tuple(record for record in values if record_owner_table(record.record.key, store_key=self._store_key) == owner)
                    await self.replace_records(batch)
            except BaseException as error:
                self._abort_mutation(error)
                raise
            return
        if owners == ("ai_step_messages",):
            try:
                if len({item.record.key for item in values}) != len(values):
                    raise ValueError("duplicate payload replacements")
                for replacement in values:
                    current = await self.get_record(replacement.record.key)
                    if current is None or replacement.record.storage_revision != replacement.expected_storage_revision + 1:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    validate_record_replacement(current, replacement.record)
                    if not await self._payload_owner().replace(replacement.record, replacement.expected_storage_revision):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    self._record_cache[replacement.record.key] = replacement.record
                    self._guarded_record_revisions[replacement.record.key] = replacement.record.storage_revision
            except BaseException as error:
                self._abort_mutation(error)
                raise
            return
        keys = [replacement.record.key for replacement in values]
        if len(keys) != len(set(keys)):
            raise ValueError("replace_records contains duplicate keys")
        if not values:
            return
        ordered = tuple(sorted(values, key=lambda value: value.record.key))
        for replacement in ordered:
            validate_record_identity(replacement.record)
            if (
                replacement.record.storage_revision
                != replacement.expected_storage_revision + 1
            ):
                raise ValueError(
                    "replacement must increment storage_revision exactly once"
                )
        from sqlalchemy import and_, bindparam, case, func, literal, or_, update

        table = self._table(owners[0])
        prepared = tuple(record_owner_values(owners[0], _record_values(value.record)) for value in ordered)
        columns = ("state", "storage_revision", "lease_owner", "lease_fence",
                   "lease_expires_at", "payload_json", "payload_bytes")
        identity_columns = tuple(name for name in ("scope_key", "parent_key", "subject_id", "kind", "sort_key") if name in table.c)
        columns = tuple(name for name in columns if name in table.c)
        executemany = self._context.dialect.prefer_update_executemany
        statement_count = 0
        for offset in range(0, len(ordered), _RECORD_REPLACE_BATCH_SIZE):
            batch = ordered[offset:offset + _RECORD_REPLACE_BATCH_SIZE]
            rows = prepared[offset:offset + _RECORD_REPLACE_BATCH_SIZE]
            parameters = None
            if executemany:
                statement = update(table).where(
                    table.c.store_key == self._store_key,
                    table.c.key_locator == bindparam("_replacement_locator"),
                    table.c.key == bindparam("_replacement_key"),
                    table.c.storage_revision == bindparam("_replacement_expected_storage_revision"),
                    *self._record_identity_conditions(table, {
                        column: bindparam(f"_identity_{column}") for column in identity_columns
                    }),
                ).values(**{column: bindparam(f"_replacement_{column}") for column in columns},
                         updated_at=func.current_timestamp())
                parameters = [
                    {"_replacement_key": row["key"], "_replacement_locator": row["key_locator"],
                     "_replacement_expected_storage_revision": value.expected_storage_revision,
                     **{f"_identity_{column}": row[column] for column in identity_columns},
                     **{f"_replacement_{column}": row[column] for column in columns}}
                    for row, value in zip(rows, batch)
                ]
            else:
                cas_conditions = [and_(
                    _key_matches(table.c.key, row["key"]),
                    table.c.storage_revision == value.expected_storage_revision,
                    *self._record_identity_conditions(table, row),
                ) for row, value in zip(rows, batch)]
                updates = {
                    column: rows[0][column] if len(rows) == 1 else case(
                        {row["key"]: literal(row[column], type_=table.c[column].type)
                         for row in rows},
                        value=table.c.key,
                        else_=table.c[column],
                    )
                    for column in columns
                }
                statement = update(table).where(
                    table.c.store_key == self._store_key,
                    _keys_match(table.c.key, [row["key"] for row in rows]),
                    or_(*cas_conditions),
                ).values(**updates, updated_at=func.current_timestamp())
            try:
                result = await self._execute(statement, parameters)
            except BaseException as error:
                self._abort_mutation(error)
                raise
            statement_count += len(batch) if executemany else 1
            if result.rowcount != len(batch):
                error = AIError(ErrorCode.STORAGE_CONFLICT)
                # A single failed CAS has not changed any record and can be retried.
                if len(ordered) > 1 or result.rowcount != 0:
                    self._abort_mutation(error)
                # Only a failed conditional write needs source inspection. Do not
                # classify failure from a potentially stale transaction-local view.
                for value in batch:
                    self._record_cache.pop(value.record.key, None)
                current = await self.get_records(tuple(value.record.key for value in batch))
                for value in batch:
                    existing = current.get(value.record.key)
                    if existing is not None and existing.storage_revision == value.expected_storage_revision:
                        validate_record_replacement(existing, value.record)
                raise error
        for replacement in ordered:
            record = replacement.record
            previous = self._record_cache.get(record.key)
            if previous is not None and previous.storage_revision != replacement.expected_storage_revision:
                previous = None
            self._stage_native_record_owner(record, previous=previous)
        self._log_batch("replace_records", len(ordered), statement_count)
        for replacement in ordered:
            self._guarded_record_revisions[replacement.record.key] = replacement.record.storage_revision
            self._record_cache[replacement.record.key] = replacement.record

    async def transition_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_revision: int,
        transition: RecordLeaseTransition,
    ) -> StoredRecord | None:
        if (isinstance(expected_storage_revision, bool)
                or not isinstance(expected_storage_revision, int)
                or expected_storage_revision < 0):
            raise ValueError("expected_storage_revision must be a non-negative integer")
        current = await self.get_record(key)
        if current is None or current.storage_revision != expected_storage_revision:
            return None
        protected = key in self._guarded_record_revisions
        now = await self._context.run_statement(
            self._context.dialect.database_current_time(self._session)
        )
        if transition.apply(current, now) is None:
            return None
        guarded = await self.guard_record(key, expected_storage_revision=expected_storage_revision)
        if guarded is None:
            return None
        try:
            if not protected:
                # A new CAS may wait past expiry. Already protected records were
                # sampled after their earlier write, not from tx.now() or the UPDATE.
                now = await self._context.run_statement(
                    self._context.dialect.database_current_time(self._session)
                )
            candidate = transition.apply(guarded, now)
            if candidate is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if transition.operation in ("guard", "guard_available"):
                return guarded
            if not await self.update_record_lease(
                key, expected_storage_revision=guarded.storage_revision,
                lease_owner=candidate.lease_owner, lease_fence=candidate.lease_fence,
                lease_expires_at=candidate.lease_expires_at,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        except BaseException as error:
            self._abort_mutation(error)
            raise
        return candidate

    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_revision: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool:
        from sqlalchemy import func, update

        if (
            isinstance(expected_storage_revision, bool)
            or not isinstance(expected_storage_revision, int)
            or expected_storage_revision < 0
            or isinstance(lease_fence, bool)
            or not isinstance(lease_fence, int)
            or lease_fence < 0
        ):
            raise ValueError("record lease integer fields are invalid")
        if lease_expires_at is not None and lease_expires_at.tzinfo is None:
            raise ValueError("record lease is invalid")
        if self._is_payload_key(key):
            if lease_owner is not None or lease_fence != 0 or lease_expires_at is not None:
                raise ValueError("immutable payload content cannot acquire a lease")
            current = await self.get_record(key)
            if current is None:
                return False
            return await self.replace_record(replace(current, storage_revision=expected_storage_revision + 1),
                                             expected_storage_revision=expected_storage_revision)
        table = self._table(record_owner_table(key, store_key=self._store_key))
        result = await self._execute(
            update(table)
            .where(
                table.c.store_key == self._store_key,
                _key_matches(table.c.key, validate_record_key(key)),
                table.c.storage_revision == expected_storage_revision,
            )
            .values(**record_owner_values(table.name, {
                "lease_owner": lease_owner, "lease_fence": lease_fence,
                "lease_expires_at": lease_expires_at,
                "storage_revision": expected_storage_revision + 1,
                "updated_at": func.current_timestamp(),
            }))
        )
        if result.rowcount == 1:
            self._guarded_record_revisions[key] = expected_storage_revision + 1
            cached = self._record_cache.get(key)
            if cached is not None:
                self._record_cache[key] = replace(
                    cached,
                    lease_owner=lease_owner,
                    lease_fence=lease_fence,
                    lease_expires_at=lease_expires_at,
                    storage_revision=expected_storage_revision + 1,
                )
        return result.rowcount == 1

    async def delete_records(self, keys: Sequence[bytes]) -> None:
        """A failed batch requires a new transaction, even if its error is caught."""
        unique_keys = tuple(dict.fromkeys(keys))
        owners = tuple(dict.fromkeys(record_owner_table(record, store_key=self._store_key) for record in unique_keys))
        if len(owners) > 1:
            try:
                for owner in owners:
                    batch = tuple(record for record in unique_keys if record_owner_table(record, store_key=self._store_key) == owner)
                    await self.delete_records(batch)
            except BaseException as error:
                self._abort_mutation(error)
                raise
            return
        if not unique_keys:
            return
        if owners == ("ai_step_messages",):
            try:
                for key in unique_keys:
                    await self.delete_record(key)
            except BaseException as error:
                self._abort_mutation(error)
                raise
            return
        from sqlalchemy import delete, tuple_, update

        table = self._table(owners[0])
        facts = self._table("ai_state_facts")
        deleted_keys: set[bytes] = set()
        statement_count = 0
        try:
            for offset in range(0, len(unique_keys), _RECORD_DELETE_BATCH_SIZE):
                current = await self.get_records(
                    unique_keys[offset : offset + _RECORD_DELETE_BATCH_SIZE]
                )
                if not current:
                    continue
                cas_conditions = tuple(
                    (validate_record_key(key), record.storage_revision)
                    for key, record in sorted(current.items())
                )
                record_keys = tuple(key for key, _revision in cas_conditions)
                # Win the existing owner CAS before retiring dependent facts.
                # Fact writers must advance this same revision before insertion.
                guarded = await self._execute(update(table).where(
                    table.c.store_key == self._store_key,
                    _keys_match(table.c.key, record_keys),
                    tuple_(table.c.key, table.c.storage_revision).in_(cas_conditions),
                ).values(storage_revision=table.c.storage_revision + 1))
                if guarded.rowcount != len(current):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                cas_conditions = tuple((key, revision + 1) for key, revision in cas_conditions)
                # Deleting the CAS-matched owners fences concurrent fact writers.
                result = await self._execute(
                    delete(table).where(
                        table.c.store_key == self._store_key,
                        _keys_match(table.c.key, record_keys),
                        tuple_(table.c.key, table.c.storage_revision).in_(cas_conditions),
                    )
                )
                if result.rowcount != len(current):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await self._execute(
                    delete(facts).where(
                        facts.c.store_key == self._store_key,
                        _keys_match(facts.c.owner_key, record_keys),
                    )
                )
                for key in current:
                    self._stage_native_owner(record_owner_edge(key), (validate_record_key(key),), None)
                deleted_keys.update(current)
                statement_count += 2
        except BaseException as error:
            # A caught partial deletion must still abort the owning transaction.
            self._abort_mutation(error)
            raise
        self._deleted_records = self._deleted_records or bool(deleted_keys)
        for key in deleted_keys:
            self._guarded_record_revisions.pop(key, None)
            self._record_cache[key] = None
            self._sequence_cache.update({stream: 0 for stream in run_counter_streams(key, store_key=self._store_key)})
        self._log_batch("delete_records", len(deleted_keys), statement_count)

    async def delete_record(
        self, key: bytes, *, expected_storage_revision: int | None = None
    ) -> bool:
        if expected_storage_revision is not None and (
            isinstance(expected_storage_revision, bool)
            or not isinstance(expected_storage_revision, int)
            or expected_storage_revision < 0
        ):
            raise ValueError(
                "expected_storage_revision must be a non-negative integer or None"
            )
        from sqlalchemy import delete

        current = await self.get_record(key)
        if current is None:
            return False
        expected = (
            current.storage_revision
            if expected_storage_revision is None
            else expected_storage_revision
        )
        guarded = await self.guard_record(key, expected_storage_revision=expected)
        if guarded is None:
            return False
        if self._is_payload_key(key):
            try:
                facts = self._table("ai_state_facts")
                await self._execute(delete(facts).where(facts.c.store_key == self._store_key,
                                                       _key_matches(facts.c.owner_key, key)))
                if not await self._payload_owner().delete(key, guarded.storage_revision):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            except BaseException as error:
                self._abort_mutation(error)
                raise
            self._deleted_records = True
            self._guarded_record_revisions.pop(key, None)
            self._record_cache[key] = None
            self._sequence_cache.update({stream: 0 for stream in run_counter_streams(key, store_key=self._store_key)})
            return True
        table = self._table(record_owner_table(key, store_key=self._store_key))
        statement = delete(table).where(
            table.c.store_key == self._store_key,
            _key_matches(table.c.key, validate_record_key(key)),
            table.c.storage_revision == guarded.storage_revision,
        )
        children_changed = False
        try:
            children_changed = True
            await self._execute(
                delete(self._table("ai_state_facts")).where(
                    self._table("ai_state_facts").c.store_key == self._store_key,
                    _key_matches(self._table("ai_state_facts").c.owner_key, validate_record_key(key)),
                )
            )
            result = await self._execute(statement)
        except BaseException as error:
            if children_changed:
                self._abort_mutation(error)
            raise
        if result.rowcount != 1:
            self._abort_mutation(AIError(ErrorCode.STORAGE_CONFLICT))
        if result.rowcount == 1:
            self._stage_native_owner(record_owner_edge(key), (validate_record_key(key),), None)
            self._deleted_records = True
            self._guarded_record_revisions.pop(key, None)
            self._record_cache[key] = None
            self._sequence_cache.update({stream: 0 for stream in run_counter_streams(key, store_key=self._store_key)})
        return result.rowcount == 1

    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        statement = _record_query_statement(self._metadata, self._store_key, query)
        rows = (await self._execute(statement)).mappings().all()
        values = tuple(_record_from_row(row) for row in rows)
        if self._store_key == STATE_GROUP_ARCHIVE_KEY:
            extra = await self._payload_owner().scan(query=query, limit=query.limit)
            values = tuple(sorted((*values, *extra), key=lambda row: (row.sort_key, row.key)))
            if query.limit is not None:
                values = values[:query.limit]
        self._record_cache.update({value.key: value for value in values})
        return values

    async def scan_records(self) -> tuple[StoredRecord, ...]:
        from sqlalchemy import select

        table = record_owner_rows(self._metadata)
        rows = (
            await self._execute(
                select(table).where(table.c.store_key == self._store_key)
            )
        ).mappings().all()
        values = tuple(_record_from_row(row) for row in rows)
        if self._store_key == STATE_GROUP_ARCHIVE_KEY:
            values += await self._payload_owner().scan()
        return values

    async def scan_records_page(
        self,
        *,
        after: RecordScanCursor | None,
        limit: int,
    ) -> tuple[StoredRecord, ...]:
        from sqlalchemy import select

        _require_scan_limit(limit)

        table = record_owner_rows(self._metadata)
        statement = select(table).where(table.c.store_key == self._store_key)
        if after is not None:
            statement = statement.where(
                table.c.key > validate_record_key(after.key)
            )
        statement = statement.order_by(table.c.key).limit(limit)
        rows = (await self._execute(statement)).mappings().all()
        values = tuple(_record_from_row(row) for row in rows)
        if self._store_key == STATE_GROUP_ARCHIVE_KEY:
            extra = await self._payload_owner().scan(after=None if after is None else after.key, limit=limit)
            values = tuple(sorted((*values, *extra), key=lambda row: row.key)[:limit])
        self._record_cache.update({value.key: value for value in values})
        return values


    async def get_sequence(self, key: bytes) -> int:
        validate_stream_key(key)
        if key in self._sequence_cache:
            return self._sequence_cache[key]
        values = await self.get_sequences((key,))
        return values[key]

    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]:
        if not keys:
            return {}
        from sqlalchemy import select

        unique_keys = tuple(dict.fromkeys(validate_stream_key(key) for key in keys))
        missing = tuple(key for key in unique_keys if key not in self._sequence_cache)
        run_keys = tuple(key for key in missing if run_counter_address(key, store_key=self._store_key) is not None)
        if run_keys:
            self._sequence_cache.update(await self._read_run_sequences(run_keys))
        missing = tuple(key for key in missing if run_counter_address(key, store_key=self._store_key) is None)
        table = self._table("ai_state_counters")
        if missing:
            rows = (
                (
                    await self._execute(
                        select(table).where(
                            table.c.store_key == self._store_key,
                            _locator_column(table.c.stream_key).in_(tuple(_key_locator(key) for key in missing))
                        )
                    )
                )
                .mappings()
                .all()
            )
            if any(row["stream_key"] not in missing for row in rows):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for row in rows:
                _validate_locators(row, "stream_key")
            self._sequence_cache.update(
                {
                    _row_stream_key(row["stream_key"]): _row_nonnegative_int(row["value"])
                    for row in rows
                }
            )
            for key in missing:
                self._sequence_cache.setdefault(key, 0)
        return {key: self._sequence_cache[key] for key in unique_keys}

    async def _read_run_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]:
        from sqlalchemy import select
        table = self._table("ai_step_run_admissions")
        addresses = {key: run_counter_address(key, store_key=self._store_key) for key in keys}
        owner_keys = tuple(dict.fromkeys(address[0] for address in addresses.values() if address is not None))
        rows = []
        for offset in range(0, len(owner_keys), _RAW_KEY_BATCH_SIZE):
            batch = owner_keys[offset:offset + _RAW_KEY_BATCH_SIZE]
            rows.extend((await self._execute(select(table.c.key, table.c.key_locator, *(
                table.c[name] for name in RUN_COUNTER_COLUMNS
            )).where(table.c.store_key == self._store_key,
                     table.c.key_locator.in_(tuple(_key_locator(key) for key in batch))))).mappings())
        by_key = {}
        for row in rows:
            if row["key"] not in owner_keys:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _validate_locators(row, "key")
            by_key[row["key"]] = row
        result = {}
        for key, address in addresses.items():
            assert address is not None
            row = by_key.get(address[0])
            value = None if row is None else row[address[1]]
            result[key] = 0 if value is None else _row_nonnegative_int(value)
        return result

    async def _scan_run_sequences(self) -> Mapping[bytes, int]:
        from sqlalchemy import or_, select
        table = self._table("ai_step_run_admissions")
        rows = (await self._execute(select(table.c.key, table.c.key_locator, *(
            table.c[name] for name in RUN_COUNTER_COLUMNS
        )).where(table.c.store_key == self._store_key,
                 or_(*(table.c[name].is_not(None) for name in RUN_COUNTER_COLUMNS))))).mappings()
        result = {}
        for row in rows:
            _validate_locators(row, "key")
            streams = run_counter_streams(row["key"], store_key=self._store_key)
            if not streams:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for key, column in zip(streams, RUN_COUNTER_COLUMNS, strict=True):
                if row[column] is not None:
                    result[key] = _row_nonnegative_int(row[column])
        return result

    async def scan_sequences(self) -> Mapping[bytes, int]:
        from sqlalchemy import select
        table = self._table("ai_state_counters")
        rows = (await self._execute(select(table).where(table.c.store_key == self._store_key))).mappings()
        values = {}
        for row in rows:
            _row_store_key(row["store_key"])
            _validate_locators(row, "stream_key")
            key = _row_stream_key(row["stream_key"])
            if run_counter_address(key, store_key=self._store_key) is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values[key] = _row_nonnegative_int(row["value"])
        values.update(await self._scan_run_sequences())
        return dict(sorted(values.items()))

    async def scan_sequences_page(
        self, *, after: bytes | None, limit: int,
    ) -> Mapping[bytes, int]:
        from sqlalchemy import select
        _require_scan_limit(limit)
        if after is not None:
            validate_stream_key(after)
        table = self._table("ai_state_counters")
        query = select(table).where(table.c.store_key == self._store_key)
        if after is not None:
            query = query.where(table.c.stream_key > after)
        values = {}
        for row in (await self._execute(query.order_by(table.c.stream_key).limit(limit))).mappings():
            _validate_locators(row, "stream_key")
            key = _row_stream_key(row["stream_key"])
            if run_counter_address(key, store_key=self._store_key) is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values[key] = _row_nonnegative_int(row["value"])
        table = self._table("ai_step_run_admissions")
        for index, column in enumerate(RUN_COUNTER_COLUMNS):
            bound = run_counter_page_start(after, column)
            if bound is None:
                continue
            query = select(table.c.key, table.c.key_locator, table.c[column]).where(
                table.c.store_key == self._store_key, table.c[column].is_not(None))
            start, inclusive = bound
            if start is not None:
                query = query.where(table.c.key >= start if inclusive else table.c.key > start)
            for row in (await self._execute(query.order_by(table.c.key).limit(limit))).mappings():
                _validate_locators(row, "key")
                streams = run_counter_streams(row["key"], store_key=self._store_key)
                if not streams or (after is not None and streams[index] <= after):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values[streams[index]] = _row_nonnegative_int(row[column])
        return dict(sorted(values.items())[:limit])

    async def next_sequence(self, key: bytes) -> int:
        return await self.reserve_sequence(key, 1)

    async def reserve_sequence(self, key: bytes, count: int) -> int:
        return (await self.reserve_sequences({key: count}))[key]

    async def reserve_sequences(self, requests: Mapping[bytes, int]) -> Mapping[bytes, int]:
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 1
               for count in requests.values()):
            raise ValueError("sequence reservation count must be a positive integer")
        for key in requests:
            validate_stream_key(key)
        if not requests:
            return {}
        try:
            return await self._reserve_sequences(requests)
        except BaseException as error:
            self._abort_mutation(error)
            raise

    async def _reserve_sequences(self, requests: Mapping[bytes, int]) -> Mapping[bytes, int]:
        run_requests = {key: count for key, count in requests.items() if run_counter_address(key, store_key=self._store_key) is not None}
        if run_requests:
            from sqlalchemy import func, update
            table = self._table("ai_step_run_admissions")
            returning = self._context.dialect.name in {"sqlite", "postgresql"}
            owners: dict[bytes, dict[str, object]] = {}
            for key, count in run_requests.items():
                address = run_counter_address(key, store_key=self._store_key)
                assert address is not None
                owner, column = address
                owners.setdefault(owner, {})[column] = func.coalesce(table.c[column], 0) + count
            returned: dict[bytes, Mapping[str, object]] = {}
            for owner, values in sorted(owners.items()):
                statement = update(table).where(
                    table.c.store_key == self._store_key, _key_matches(table.c.key, owner),
                ).values(**values, updated_at=func.current_timestamp())
                if returning:
                    changed = await self._execute(statement.returning(
                        table.c.key, table.c.key_locator, *(table.c[column] for column in values),
                    ))
                    row = changed.mappings().one_or_none()
                    matched = row is not None
                    if row is not None:
                        _validate_locators(row, "key")
                        if row["key"] != owner:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        returned[owner] = row
                else:
                    matched = (await self._execute(statement)).rowcount == 1
                if not matched:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            if returning:
                result = {}
                for key in run_requests:
                    owner, column = run_counter_address(key, store_key=self._store_key)
                    result[key] = _row_nonnegative_int(returned[owner][column])
            else:
                result = dict(await self._read_run_sequences(tuple(run_requests)))
            self._sequence_cache.update(result)
            ordinary = {key: count for key, count in requests.items() if key not in run_requests}
            if ordinary:
                result.update(await self._reserve_sequences(ordinary))
            return {key: result[key] for key in requests}
        table = self._table("ai_state_counters")
        if len(requests) == 1:
            key, count = next(iter(requests.items()))
            value = await self._context.run_statement(
                self._context.dialect.upsert_increment(
                    self._session,
                    table=table,
                    values={"store_key": self._store_key, "stream_key": validate_stream_key(key),
                            "stream_locator": _key_locator(key)},
                    column="value",
                    index_elements=("store_key", "stream_locator"),
                    identity_columns=("stream_key",),
                    step=count,
                )
            )
            result = {key: _row_nonnegative_int(value)}
            self._sequence_cache.update(result)
            return result
        rows = [
            {
                    "store_key": self._store_key,
                "stream_key": validate_stream_key(key),
                "stream_locator": _key_locator(key),
                "value": requests[key],
            }
            for key in sorted(requests)
        ]
        values = await self._context.run_statement(self._context.dialect.upsert_increment_many(
            self._session,
            table=table,
            rows=rows,
            column="value",
            index_elements=("store_key", "stream_locator"),
            identity_columns=("stream_key",),
            returning_key="stream_key",
        ))
        result = {
            _row_stream_key(key): _row_nonnegative_int(value)
            for key, value in values.items()
        }
        self._sequence_cache.update(result)
        return {key: result[key] for key in requests}

    async def advance_sequence(self, key: bytes, expected: int) -> int:
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise ValueError("expected sequence must be a non-negative integer")
        from sqlalchemy import func, update

        address = run_counter_address(validate_stream_key(key), store_key=self._store_key)
        if address is not None:
            owner, column = address
            table = self._table("ai_step_run_admissions")
            result = await self._execute(update(table).where(
                table.c.store_key == self._store_key, _key_matches(table.c.key, owner),
                table.c[column] == expected,
            ).values({column: expected + 1, "updated_at": func.current_timestamp()}))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._sequence_cache[key] = expected + 1
            return expected + 1
        table = self._table("ai_state_counters")
        result = await self._execute(
            update(table)
            .where(
                table.c.store_key == self._store_key,
                _key_matches(table.c.stream_key, validate_stream_key(key)),
                table.c.value == expected,
            )
            .values(value=expected + 1, updated_at=func.current_timestamp())
        )
        if result.rowcount != 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._sequence_cache[key] = expected + 1
        return expected + 1

    async def delete_sequence(self, key: bytes) -> None:
        await self.delete_sequences((key,))

    async def delete_sequences(self, keys: Sequence[bytes]) -> None:
        values = tuple(sorted(set(keys)))
        if not values:
            return
        from sqlalchemy import delete

        from sqlalchemy import func, update
        run_keys = tuple(key for key in values if run_counter_address(key, store_key=self._store_key) is not None)
        if run_keys:
            table = self._table("ai_step_run_admissions")
            owners: dict[bytes, dict[str, None]] = {}
            for key in run_keys:
                owner, column = run_counter_address(key, store_key=self._store_key)
                owners.setdefault(owner, {})[column] = None
            for owner, columns in sorted(owners.items()):
                await self._execute(update(table).where(
                    table.c.store_key == self._store_key, _key_matches(table.c.key, owner),
                ).values(**columns, updated_at=func.current_timestamp()))
            self._sequence_cache.update({key: 0 for key in run_keys})
            values = tuple(key for key in values if key not in run_keys)
            if not values:
                return
        table = self._table("ai_state_counters")
        for offset in range(0, len(values), _RAW_KEY_BATCH_SIZE):
            batch = values[offset:offset + _RAW_KEY_BATCH_SIZE]
            await self._execute(delete(table).where(
                table.c.store_key == self._store_key,
                _keys_match(table.c.stream_key, tuple(validate_stream_key(key) for key in batch)),
            ))
        self._log_batch("delete_sequences", len(values), (len(values) + _RAW_KEY_BATCH_SIZE - 1) // _RAW_KEY_BATCH_SIZE)
        for key in values:
            self._sequence_cache[key] = 0

    async def insert_fact(self, fact: StoredFact) -> None:
        await self.insert_facts((fact,))

    async def insert_facts(self, facts: Sequence[StoredFact]) -> None:
        if not facts:
            return
        if any(
            fact.owner_key not in self._guarded_record_revisions for fact in facts
        ):
            raise RuntimeError("fact owner must be guarded in the current transaction")
        from sqlalchemy import insert

        await self._execute(
            insert(self._table("ai_state_facts")).values(
                [
                    {**_fact_values(fact), "store_key": self._store_key}
                    for fact in facts
                ]
            )
        )
        for fact in facts:
            self._stage_native_owner("fact", (validate_stream_key(fact.stream_key), fact.sequence), fact.data, state=fact.state, inserting=True)
        self._log_batch("insert_facts", len(facts), 1)

    async def get_facts(
        self, stream_key: bytes, sequences: Sequence[int],
    ) -> tuple[StoredFact, ...]:
        from sqlalchemy import select

        ordered = sorted(set(sequences))
        if not ordered:
            return ()
        table = self._table("ai_state_facts")
        facts: list[StoredFact] = []
        for offset in range(0, len(ordered), _FACT_READ_BATCH_SIZE):
            rows = (await self._execute(
                select(table).where(
                    table.c.store_key == self._store_key,
                    _key_matches(table.c.stream_key, validate_stream_key(stream_key)),
                    table.c.sequence.in_(ordered[offset:offset + _FACT_READ_BATCH_SIZE]),
                ).order_by(table.c.sequence)
            )).mappings().all()
            facts.extend(_fact_from_row(row) for row in rows)
        return tuple(facts)

    async def list_facts(self, query: FactQuery) -> tuple[StoredFact, ...]:
        from sqlalchemy import func, select

        table = self._table("ai_state_facts")
        conditions = [
            table.c.store_key == self._store_key,
            _key_matches(table.c.stream_key, validate_stream_key(query.stream_key)),
        ]
        if query.after_sequence is not None:
            conditions.append(table.c.sequence > query.after_sequence)
        if query.subject_id is not None:
            conditions.append(table.c.subject_id == query.subject_id.encode("utf-8"))
        if query.latest_per_subject:
            latest_sequences = (
                select(func.max(table.c.sequence))
                .where(*conditions)
                .group_by(table.c.subject_id)
            )
            statement = (
                select(table)
                .where(
                    table.c.store_key == self._store_key,
                    _key_matches(table.c.stream_key, validate_stream_key(query.stream_key)),
                    table.c.sequence.in_(latest_sequences),
                )
                .order_by(table.c.sequence)
            )
        else:
            if query.latest:
                statement = (
                    select(table).where(*conditions).order_by(table.c.sequence.desc())
                )
            else:
                statement = select(table).where(*conditions).order_by(table.c.sequence)
        if query.limit is not None:
            statement = statement.limit(1 if query.latest else query.limit)
        elif query.latest:
            statement = statement.limit(1)
        rows = (await self._execute(statement)).mappings().all()
        return tuple(_fact_from_row(row) for row in rows)

    async def scan_facts(self) -> tuple[StoredFact, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_facts")
        rows = (
            await self._execute(
                select(table).where(table.c.store_key == self._store_key)
            )
        ).mappings().all()
        return tuple(_fact_from_row(row) for row in rows)

    async def scan_facts_page(
        self,
        *,
        after: FactScanCursor | None,
        limit: int,
    ) -> tuple[StoredFact, ...]:
        from sqlalchemy import select

        _require_scan_limit(limit)
        from sqlalchemy import and_, or_

        table = self._table("ai_state_facts")
        conditions = [table.c.store_key == self._store_key]
        if after is not None:
            conditions.append(
                or_(
                    table.c.stream_key > validate_stream_key(after.stream_key),
                    and_(
                        _key_matches(table.c.stream_key, validate_stream_key(after.stream_key)),
                        table.c.sequence > after.sequence,
                    ),
                )
            )
        statement = (
            select(table)
            .where(*conditions)
            .order_by(table.c.stream_key, table.c.sequence)
            .limit(limit)
        )
        rows = (await self._execute(statement)).mappings().all()
        return tuple(_fact_from_row(row) for row in rows)

    async def delete_fact_streams(self, owner_key: bytes) -> None:
        from sqlalchemy import delete

        if owner_key not in self._guarded_record_revisions:
            current = await self.get_record(owner_key)
            if current is not None and await self.guard_record_revision(
                owner_key, expected_storage_revision=current.storage_revision,
            ) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

        await self._execute(
            delete(self._table("ai_state_facts")).where(
                self._table("ai_state_facts").c.store_key == self._store_key,
                _key_matches(self._table("ai_state_facts").c.owner_key, validate_record_key(owner_key)),
            )
        )

    async def insert_operation(self, value: StoredOperation) -> None:
        from sqlalchemy import insert

        await self._execute(
            insert(self._table("ai_state_operations")).values(
                {**_operation_values(value), "store_key": self._store_key}
            )
        )
        self._stage_native_owner("operation", (value.key,), value.data, state=value.state, inserting=True)

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        from sqlalchemy import select

        table = self._table("ai_state_operations")
        row = (
            (
                await self._execute(
                    select(table).where(
                        table.c.store_key == self._store_key,
                        _locator_column(table.c.operation_key) == _key_locator(key),
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is not None and row["operation_key"] != key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return None if row is None else _operation_from_row(row)

    async def replace_operation(
        self, value: StoredOperation, *, expected_state: str
    ) -> bool:
        current = await self.get_operation(value.key)
        if current is None or current.state != expected_state:
            return False
        validate_operation_replacement(current, value)
        from sqlalchemy import update

        table = self._table("ai_state_operations")
        result = await self._execute(
            update(table)
            .where(
                table.c.store_key == self._store_key,
                _key_matches(table.c.operation_key, value.key),
                table.c.state == expected_state,
            )
            .values(_operation_values(value, updating=True))
        )
        if result.rowcount == 1:
            self._stage_native_owner("operation", (value.key,), value.data, state=value.state)
        return result.rowcount == 1

    async def list_operations(
        self, query: OperationQuery
    ) -> tuple[StoredOperation, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_operations")
        conditions = [table.c.store_key == self._store_key]
        if query.stream_key is not None:
            conditions.append(_key_matches(table.c.stream_key, validate_stream_key(query.stream_key)))
        if query.states is not None:
            conditions.append(table.c.state.in_(tuple(query.states)))
        if query.after_sequence is not None:
            conditions.append(table.c.sequence > query.after_sequence)
        if query.through_sequence is not None:
            conditions.append(table.c.sequence <= query.through_sequence)
        if query.compactable is not None:
            conditions.append(table.c.compactable == query.compactable)
        statement = (
            select(table)
            .where(*conditions)
            .order_by(table.c.sequence, table.c.operation_key)
        )
        if query.limit is not None:
            statement = statement.limit(query.limit)
        rows = (await self._execute(statement)).mappings().all()
        return tuple(_operation_from_row(row) for row in rows)

    async def scan_operations(self) -> tuple[StoredOperation, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_operations")
        rows = (
            await self._execute(
                select(table).where(table.c.store_key == self._store_key)
            )
        ).mappings().all()
        return tuple(_operation_from_row(row) for row in rows)

    async def scan_operations_page(
        self,
        *,
        after: OperationScanCursor | None,
        limit: int,
    ) -> tuple[StoredOperation, ...]:
        from sqlalchemy import select

        _require_scan_limit(limit)

        table = self._table("ai_state_operations")
        statement = select(table).where(table.c.store_key == self._store_key)
        if after is not None:
            statement = statement.where(table.c.operation_key > after.key)
        statement = statement.order_by(table.c.operation_key).limit(limit)
        rows = (await self._execute(statement)).mappings().all()
        return tuple(_operation_from_row(row) for row in rows)

    async def delete_operations(
        self, query: OperationQuery
    ) -> tuple[StoredOperation, ...]:
        values = await self.list_operations(query)
        from sqlalchemy import delete, update

        if values:
            table = self._table("ai_state_operations")
            for value in values:
                guarded = await self._execute(update(table).where(
                    table.c.store_key == self._store_key, _key_matches(table.c.operation_key, value.key),
                    table.c.state == value.state,
                ).values(state=value.state))
                if guarded.rowcount != 1:
                    error = AIError(ErrorCode.STORAGE_CONFLICT)
                    self._abort_mutation(error)
                    raise error
            for offset in range(0, len(values), _RAW_KEY_BATCH_SIZE):
                keys = tuple(value.key for value in values[offset:offset + _RAW_KEY_BATCH_SIZE])
                await self._execute(delete(table).where(
                    table.c.store_key == self._store_key, _keys_match(table.c.operation_key, keys),
                ))
        for value in values:
            self._stage_native_owner("operation", (value.key,), None)
        return values

    def _record_identity_conditions(
        self, table: "Table", values: Mapping[str, object],
    ) -> tuple["ColumnElement[bool]", ...]:
        from sqlalchemy import cast, type_coerce
        from sqlalchemy.dialects.mysql import BINARY

        conditions = []
        for name in ("scope_key", "parent_key", "subject_id", "kind", "sort_key"):
            if name not in table.c:
                continue
            column = table.c[name]
            # MySQL text equality may ignore case or trailing spaces even when
            # the in-memory identity contract distinguishes those values.
            if self._context.dialect.name == "mysql" and name == "kind":
                # SQL compares bytes; Python parameters remain text, not BLOBs.
                column = type_coerce(cast(column, BINARY), column.type)
            conditions.append(column.is_not_distinct_from(values[name]))
        return tuple(conditions)

    def _is_payload_key(self, key: bytes) -> bool:
        return record_owner_table(key, store_key=self._store_key) == "ai_step_messages"

    def _payload_owner(self):
        from ._step_payload_sql import StepPayloadSql
        return StepPayloadSql(_payload_table(self._metadata), self._execute, self._context, self._session)

    def _table(self, name: str) -> "Table":
        try:
            return self._metadata.tables[name]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _log_batch(self, operation: str, batch_size: int, statement_count: int) -> None:
        _logger.debug(
            "SQL batch executed: backend=%s operation=%s batch_size=%s statement_count=%s",
            self._context.dialect.name,
            operation,
            batch_size,
            statement_count,
        )


def _record_values(
    record: StoredRecord, *, updating: bool = False
) -> dict[str, object]:
    values: dict[str, object] = {
        "key": validate_record_key(record.key),
        "key_locator": _key_locator(record.key),
        "scope_locator": _key_locator(record.scope_key),
        "parent_locator": _key_locator(record.parent_key),
        "sort_prefix": record.sort_key[:128],
        "scope_key": record.scope_key,
        "parent_key": record.parent_key,
        "subject_id": None
        if record.subject_id is None
        else record.subject_id.encode("utf-8"),
        "kind": record.kind,
        "sort_key": record.sort_key,
        "state": record.state,
        "storage_revision": record.storage_revision,
        "lease_owner": record.lease_owner,
        "lease_fence": record.lease_fence,
        "lease_expires_at": record.lease_expires_at,
        "payload_json": canonical_json_bytes(dict(record.data)).decode("utf-8"),
        "payload_bytes": record.payload_bytes,
    }
    if updating:
        from sqlalchemy import func

        values["updated_at"] = func.current_timestamp()
    return values


def _record_from_row(row: Mapping[str, object]) -> StoredRecord:
    try:
        _row_store_key(row["store_key"])
        _validate_locators(row, "key", "scope_key", "parent_key")
        if row["sort_prefix"] != row["sort_key"][:128]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        record = StoredRecord(
            _row_record_key(row["key"]),
            row["scope_key"],
            row["parent_key"],
            _row_string(row["kind"]),
            row["sort_key"],
            _row_optional_string(row["state"]),
            _row_nonnegative_int(row["storage_revision"]),
            _row_optional_string(row["lease_owner"]),
            _row_nonnegative_int(row["lease_fence"]),
            _datetime_or_none(row["lease_expires_at"]),
            _row_mapping(row["payload_json"]),
            subject_id=_row_subject_id(row["subject_id"]),
            payload_bytes=row["payload_bytes"],
        )
    except AIError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return record


def _fact_values(fact: StoredFact) -> dict[str, object]:
    return {
        "stream_key": validate_stream_key(fact.stream_key),
        "stream_locator": _key_locator(fact.stream_key),
        "owner_locator": _key_locator(fact.owner_key),
        "sequence": fact.sequence,
        "owner_key": validate_record_key(fact.owner_key),
        "kind": fact.kind,
        "subject_id": None
        if fact.subject_id is None
        else fact.subject_id.encode("utf-8"),
        "state": fact.state,
        "payload_json": canonical_json_bytes(dict(fact.data)).decode("utf-8"),
    }


def _fact_from_row(row: Mapping[str, object]) -> StoredFact:
    try:
        _row_store_key(row["store_key"])
        _validate_locators(row, "stream_key", "owner_key")
        return StoredFact(
            _row_stream_key(row["stream_key"]),
            _row_nonnegative_int(row["sequence"]),
            _row_record_key(row["owner_key"]),
            _row_string(row["kind"]),
            _row_subject_id(row["subject_id"]),
            _row_optional_string(row["state"]),
            _row_mapping(row["payload_json"]),
        )
    except AIError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _operation_values(
    value: StoredOperation, *, updating: bool = False
) -> dict[str, object]:
    values: dict[str, object] = {
        "operation_key": value.key,
        "operation_locator": _key_locator(value.key),
        "stream_locator": _key_locator(value.stream_key),
        "stream_key": validate_stream_key(value.stream_key),
        "sequence": value.sequence,
        "state": value.state,
        "compactable": value.compactable,
        "payload_json": canonical_json_bytes(dict(value.data)).decode("utf-8"),
    }
    if updating:
        from sqlalchemy import func

        values["updated_at"] = func.current_timestamp()
    return values


def _operation_from_row(row: Mapping[str, object]) -> StoredOperation:
    try:
        _row_store_key(row["store_key"])
        _validate_locators(row, "operation_key", "stream_key")
        return StoredOperation(
            row["operation_key"],
            _row_stream_key(row["stream_key"]),
            _row_nonnegative_int(row["sequence"]),
            _row_string(row["state"]),
            _row_bool(row["compactable"]),
            _row_mapping(row["payload_json"]),
        )
    except AIError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _key_locator(value: bytes | None) -> str | None:
    return None if value is None else hashlib.sha256(value).hexdigest()


def _locator_column(column: "ColumnElement") -> "ColumnElement":
    name = "key_locator" if column.name == "key" else column.name.removesuffix("_key") + "_locator"
    return column.table.c[name]


def _key_matches(column: "ColumnElement", value: bytes | None) -> "ColumnElement[bool]":
    from sqlalchemy import and_

    return and_(_locator_column(column) == _key_locator(value), column == value)


def _keys_match(column: "ColumnElement", values: Sequence[bytes]) -> "ColumnElement[bool]":
    from sqlalchemy import and_

    values = tuple(values)
    return and_(_locator_column(column).in_(tuple(_key_locator(value) for value in values)), column.in_(values))


def _validate_locators(row: Mapping[str, object], *names: str) -> None:
    for name in names:
        locator = "key_locator" if name == "key" else name.removesuffix("_key") + "_locator"
        raw = row[name]
        if raw is not None and type(raw) is not bytes:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if row[locator] != _key_locator(raw):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _sort_key_prefix_upper_bound(prefix: bytes) -> bytes | None:
    for index in range(len(prefix) - 1, -1, -1):
        value = prefix[index]
        if value < 255:
            return prefix[:index] + bytes((value + 1,))
    return None


def _row_stream_key(value: object) -> bytes:
    try:
        return validate_stream_key(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _row_record_key(value: object) -> bytes:
    try:
        return validate_record_key(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _row_store_key(value: object) -> bytes:
    try:
        return validate_store_key(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _row_subject_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, bytes):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.decode("utf-8")


def _row_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _row_string(value: object) -> str:
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _row_optional_string(value: object) -> str | None:
    return None if value is None else _row_string(value)


def _row_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _row_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    value = json.loads(value)
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(value)


def _datetime_or_none(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(value, datetime):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _require_scan_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("scan page limit must be positive")

def _record_query_statement(metadata: "MetaData", store_key: bytes, query: RecordQuery) -> "Select":
    from sqlalchemy import and_, or_, select

    table = record_owner_rows(metadata, kind=query.kind)
    conditions = [table.c.store_key == store_key]
    if query.scope_key is not None:
        conditions.append(_key_matches(table.c.scope_key, query.scope_key))
    if query.parent_key is not None:
        conditions.append(_key_matches(table.c.parent_key, query.parent_key))
    if query.kind is not None:
        conditions.append(table.c.kind == query.kind)
    if query.states is not None:
        conditions.append(table.c.state.in_(tuple(query.states)))
    if query.sort_key_prefix is not None:
        conditions.append(table.c.sort_prefix >= query.sort_key_prefix[:128])
        conditions.append(table.c.sort_key >= query.sort_key_prefix)
        prefix_end = _sort_key_prefix_upper_bound(query.sort_key_prefix)
        if prefix_end is not None:
            conditions.append(table.c.sort_key < prefix_end)
    if query.after_sort_key is not None:
        conditions.append(table.c.sort_prefix >= query.after_sort_key[:128])
    if query.after_sort_key is not None and query.after_key is not None:
        conditions.append(
            or_(
                table.c.sort_key > query.after_sort_key,
                and_(
                    table.c.sort_key == query.after_sort_key,
                    table.c.key > validate_record_key(query.after_key),
                ),
            )
        )
    elif query.after_sort_key is not None:
        conditions.append(table.c.sort_key > query.after_sort_key)
    statement = (
        select(table)
        .where(*conditions)
        .order_by(
            table.c.sort_prefix,
            table.c.sort_key,
            table.c.key,
        )
    )
    if query.limit is not None:
        statement = statement.limit(query.limit)
    return statement


__all__ = ["SqlStateStorageGroup", "SqlStateStore"]
