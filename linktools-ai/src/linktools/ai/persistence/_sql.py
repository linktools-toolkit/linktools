#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL implementation of the backend-neutral Runtime StateStore."""

import asyncio
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from time import time
from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ..storage import (
    CacheBackend,
    IntegrityViolationKind,
    SqlStorageContext,
    create_sql_storage_context,
)
from ._plan import PersistenceDomain
from ._archive_read import ArchiveNodeRead, ArchiveNodeSlot
from ._schema import build_runtime_sql_metadata
from ._record_cache import StateRecordCache
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
    StateStorageGroup,
    StateStore,
    StateTransaction,
    StateTransactionNestingError,
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    active_state_group_transaction,
    active_state_scope,
    active_state_transaction,
    bind_state_scope,
    reset_state_transaction,
    validate_operation_replacement,
    validate_record_identity,
    validate_record_replacement,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Table
    from sqlalchemy.engine import Result
    from sqlalchemy.sql import Executable, Select
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

ValueT = TypeVar("ValueT")
_logger = environ.get_logger("ai.runtime.state.sql")
_MAINTENANCE_PAGE_SIZE = 128
_RECORD_DELETE_BATCH_SIZE = 256
# Six CASE assignments plus key/CAS predicates stay below 999 SQLite binds.
_RECORD_REPLACE_BATCH_SIZE = 64
_GROUP_RECORD_BIND_BUDGET = 999
_ARCHIVE_READ_BIND_BUDGET = 999
# Fixed predicates and JSON-path bindings across all eight edge branches.
_ARCHIVE_READ_FIXED_BINDS = 96


class _SqlGroupTransaction:
    def __init__(
        self,
        group: "SqlStateStorageGroup",
        transactions: Mapping[StateStore, "_SqlTransaction"],
    ) -> None:
        self._group = group
        self._transactions = transactions
        record_caches: dict[bytes, dict[bytes, StoredRecord | None]] = {}
        for transaction in transactions.values():
            # Aliases of a physical partition must observe each other's writes.
            transaction._record_cache = record_caches.setdefault(
                transaction._store_digest, transaction._record_cache,
            )

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
                _hex(key)
            unique[store] = tuple(dict.fromkeys(keys))
            transactions[store] = self.transaction(store)
        missing = {
            store: tuple(key for key in keys if key not in transactions[store]._record_cache)
            for store, keys in unique.items()
        }
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
            records: dict[str, dict[bytes, StoredRecord]] = {}
            for row in rows:
                record = _record_from_row(row)
                records.setdefault(row["store_digest"], {})[record.key_digest] = record
            for store, keys in batch.items():
                transaction = transactions[store]
                found = records.get(transaction._store_hex, {})
                transaction._record_cache.update((key, found.get(key)) for key in keys)
        return {
            store: {key: value for key in keys
                    if (value := transactions[store]._record_cache[key]) is not None}
            for store, keys in unique.items()
        }

    @classmethod
    def _record_read_batches(
        cls, requests: Mapping[StateStore, Sequence[bytes]],
    ) -> Iterator[Mapping[StateStore, tuple[bytes, ...]]]:
        batch: dict[StateStore, tuple[bytes, ...]] = {}
        used = 0
        for store, keys in requests.items():
            offset = 0
            while offset < len(keys):
                if _GROUP_RECORD_BIND_BUDGET - used < 2:
                    yield batch
                    batch, used = {}, 0
                count = min(len(keys) - offset, _GROUP_RECORD_BIND_BUDGET - used - 1)
                batch[store] = tuple(keys[offset:offset + count])
                offset += count
                used += count + 1
                if offset < len(keys):
                    yield batch
                    batch, used = {}, 0
        if batch:
            yield batch

    def _record_read_statement(
        self, requests: Mapping[StateStore, Sequence[bytes]],
    ) -> "Select":
        from sqlalchemy import and_, or_, select

        table = self._group.metadata.tables["ai_state_records"]
        return select(table).where(or_(*(and_(
            table.c.store_digest == self.transaction(store)._store_hex,
            table.c.key_digest.in_(tuple(_hex(key) for key in keys)),
        ) for store, keys in requests.items())))


class SqlStateStorageGroup:
    """Own one SQL context and one physical transaction for its logical stores."""

    def __init__(
        self,
        context: SqlStorageContext,
        metadata: "MetaData",
        *,
        owns_context: bool = False,
        read_only: bool = False,
        uses_archive: bool = False,
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
        self._archive_member: SqlStateStore | None = None
        self._archive_scope_initialized = False
        self._archive_on_mutation = uses_archive
        self._owned_members: set[SqlStateStore] = set()

    def add_member(self, store: "SqlStateStore") -> None:
        self._members.setdefault(store.store_digest, store)

    async def open_partition(
        self, key: bytes, *, create: bool = False,
    ) -> "SqlStateStore | None":
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("partition key must be a 32-byte digest")
        if active_state_scope() is not None:
            raise StateTransactionNestingError("open_partition must run outside a state callback")
        await self.initialize()
        if create and self._read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        existing = self._members.get(key)
        if existing is not None:
            return existing
        if not create:
            from sqlalchemy import select, union

            statements = [
                select(self._metadata.tables[name].c.store_digest).where(
                    self._metadata.tables[name].c.store_digest == key.hex(),
                )
                for name in ("ai_state_records", "ai_state_aliases", "ai_state_facts",
                             "ai_state_sequences", "ai_state_operations")
            ]
            async with self._session() as session:
                result = await self._context.run_statement(session.execute(union(*statements).limit(1)))
                if result.first() is None:
                    return None
        return await self._open_partition(key)

    async def open_archive(
        self, *, create: bool = False,
    ) -> tuple[str, "SqlStateStore"] | None:
        if active_state_scope() is not None:
            raise StateTransactionNestingError("open_archive must run outside a state callback")
        await self.initialize()
        if create and self._read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        metadata_digest = hashlib.sha256(b"linktools-ai:state-group-metadata").digest()
        metadata_store = await self._open_partition(metadata_digest)
        key = hashlib.sha256(b"archive_scope_id").digest()
        record = await metadata_store.read(lambda tx: tx.get_record(key))
        archive_digest = hashlib.sha256(b"linktools-ai:state-group-archive").digest()
        if record is None:
            archive = await self._open_partition(archive_digest)
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
                    key_digest=key, scope_digest=None, parent_digest=None,
                    kind="archive_scope", sort_key="archive_scope", state=None,
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
        archive_digest = hashlib.sha256(b"linktools-ai:state-group-archive").digest()
        self._archive_member = await self._open_partition(archive_digest)
        self._archive_scope_initialized = True
        return scope, self._archive_member

    async def _initialize_archive_scope(
        self, session: "AsyncSession", clock: "_SqlTransactionClock",
    ) -> None:
        """Establish resource identity in the first owner mutation's commit."""
        metadata_digest = hashlib.sha256(b"linktools-ai:state-group-metadata").digest()
        key = hashlib.sha256(b"archive_scope_id").digest()
        metadata = _SqlTransaction(
            session, self._metadata, self._context, metadata_digest, clock=clock,
        )
        record = await metadata.get_record(key)
        if record is not None:
            scope = record.data.get("archive_scope_id")
            if not isinstance(scope, str) or len(scope) != 32 or any(c not in "0123456789abcdef" for c in scope):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        archive = _SqlTransaction(
            session, self._metadata, self._context,
            hashlib.sha256(b"linktools-ai:state-group-archive").digest(), clock=clock,
        )
        if await archive.scan_records_page(after=None, limit=1):
            # Under READ COMMITTED another writer can publish between the two
            # reads. Recheck identity before treating visible content as orphaned.
            metadata = _SqlTransaction(
                session, self._metadata, self._context, metadata_digest, clock=clock,
            )
            record = await metadata.get_record(key)
            scope = None if record is None else record.data.get("archive_scope_id")
            if not isinstance(scope, str) or len(scope) != 32 or any(c not in "0123456789abcdef" for c in scope):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        record = StoredRecord(
            key_digest=key, scope_digest=None, parent_digest=None,
            kind="archive_scope", sort_key="archive_scope", state=None,
            storage_revision=0, lease_owner=None, lease_fence=0,
            lease_expires_at=None, data={"archive_scope_id": uuid4().hex},
        )
        # Another first writer may already have established this resource's
        # immutable identity. Its winning value must never be overwritten.
        await self._context.run_statement(self._context.dialect.insert_ignore_conflict(
            session, table=self._metadata.tables["ai_state_records"],
            values={**_record_values(record), "store_digest": metadata_digest.hex()},
            index_elements=("store_digest", "key_digest"),
        ))

    async def _open_partition(self, digest: bytes) -> "SqlStateStore":
        store = self._members.get(digest)
        if store is None or store._closed:
            store = SqlStateStore(
                self._context.engine, metadata=self._metadata,
                context=self._context, group=self, store_digest=digest,
                runtime_domain="group_internal",
            )
            self._members[digest] = store
            self._owned_members.add(store)
        await store.initialize()
        return store

    async def scan_stores(self) -> tuple["SqlStateStore", ...]:
        from sqlalchemy import select, union

        await self.initialize()
        statements = [
            select(self._metadata.tables[name].c.store_digest)
            for name in ("ai_state_records", "ai_state_aliases", "ai_state_facts",
                         "ai_state_sequences", "ai_state_operations")
        ]
        async with self._session() as session:
            result = await self._context.run_statement(session.execute(union(*statements)))
            digests = tuple(bytes.fromhex(value) for value in result.scalars())
        return tuple([await self._open_partition(digest) for digest in digests])

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
        members = tuple(dict.fromkeys((*stores, *((self._archive_member,) if self._archive_member else ()))))
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
                transactions = {
                    store: _SqlTransaction(
                        session,
                        self._metadata,
                        self._context,
                        store.store_digest,
                        clock=clock,
                    )
                    for store in members
                }
                group_transaction = _SqlGroupTransaction(self, transactions)
                token = bind_state_scope(self, transactions, writable=False, record_reader=group_transaction)
                try:
                    return await fn(active_state_group_transaction(self, members, writable=False))
                finally:
                    reset_state_transaction(token)

    async def mutate(
        self,
        stores: Sequence["SqlStateStore"],
        fn: StateGroupCallback[ValueT],
    ) -> ValueT:
        if self._read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        members = tuple(dict.fromkeys((*stores, *((self._archive_member,) if self._archive_member else ()))))
        if not stores:
            raise ValueError("StateStorageGroup mutation requires a store")
        for store in members:
            self._ensure_member(store)
        active = active_state_transaction(members[0], writable=True)
        if active is not None:
            return await fn(active_state_group_transaction(self, members))

        captured_at = 0.0
        captures: dict[SqlStateStore, tuple[StoredRecord, ...]] = {}
        capture_tokens: dict[SqlStateStore, object] = {}
        changed: set[SqlStateStore] = set()
        confirmed = False

        async def execute(session: "AsyncSession") -> ValueT:
            nonlocal captured_at, captures, capture_tokens, changed
            captured_at = time()
            capture_tokens = {store: store._reads.capture_token for store in members}
            clock = _SqlTransactionClock(session, self._context)
            if self._archive_on_mutation and not self._archive_scope_initialized:
                await self._initialize_archive_scope(session, clock)
            transactions = {
                store: _SqlTransaction(
                    session,
                    self._metadata,
                    self._context,
                    store.store_digest,
                    clock=clock,
                )
                for store in members
            }
            group_transaction = _SqlGroupTransaction(self, transactions)
            token = bind_state_scope(self, transactions, record_reader=group_transaction)
            try:
                result = await fn(group_transaction)
                for transaction in transactions.values():
                    if transaction._mutation_error is not None:
                        raise transaction._mutation_error
                captures = {
                    store: tuple(record for key, record in transaction._record_cache.items()
                                 if key in transaction._guarded_record_keys
                                 and record is not None and record.kind in {"session", "execution"})
                    for store, transaction in transactions.items()
                }
                changed_partitions = {
                    transaction._store_digest for transaction in transactions.values()
                    if transaction._guarded_record_keys or transaction._deleted_records
                }
                changed = {store for store in members if store.store_digest in changed_partitions}
                return result
            finally:
                reset_state_transaction(token)

        lock = self._mutation_lock
        if lock is not None:
            await lock.acquire()
        try:
            domains = ",".join(store.partition for store in members)
            result = await self._context.run_mutation(
                execute,
                domain=f"runtime.state.group[{domains}]",
            )
            confirmed = True
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
            for store in members:
                if confirmed and store not in changed:
                    continue
                previous = await store._reads.invalidate()
                # A delayed source return must not acquire a newer capture boundary.
                if confirmed and previous is capture_tokens.get(store):
                    for record in captures.get(store, ()):
                        store._reads.publish_record(record, captured_at=captured_at)

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
    """Optimistic SQL StateStore backed by the five primitive tables."""

    def __init__(
        self,
        engine: "AsyncEngine",
        *,
        metadata: "MetaData | None" = None,
        context: "SqlStorageContext | None" = None,
        runtime_domain: PersistenceDomain | str = PersistenceDomain.CONVERSATION,
        group: SqlStateStorageGroup | None = None,
        cache: CacheBackend | None = None,
        store_digest: bytes | None = None,
    ) -> None:
        resolved_context = context or create_sql_storage_context(engine)
        self._metadata = (
            metadata
            if metadata is not None
            else build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
        )
        self._runtime_domain = runtime_domain
        self._store_digest = (
            hashlib.sha256(
                f"standalone:{self.partition}".encode("utf-8")
            ).digest()
            if store_digest is None
            else bytes(store_digest)
        )
        if len(self._store_digest) != 32:
            raise ValueError("store_digest must be a 32-byte digest")
        self._owns_group = group is None
        self._storage_group = group or SqlStateStorageGroup(
            resolved_context,
            self._metadata,
            owns_context=context is None,
        )
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._initialized = False
        self._reads = StateRecordCache(self, cache, scope=f"sql:{self._store_digest.hex()}")
        self._storage_group.add_member(self)

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
    def store_digest(self) -> bytes:
        return self._store_digest

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
            record_cursor = RecordScanCursor(last.kind, last.key_digest)

        async with self._storage_group._session() as session:
            transaction = _SqlTransaction(
                session,
                self._metadata,
                self.context,
                self._store_digest,
            )
            from sqlalchemy import and_, select

            records = transaction._table("ai_state_records")
            aliases = transaction._table("ai_state_aliases")
            facts = transaction._table("ai_state_facts")
            sequences = transaction._table("ai_state_sequences")
            sequence_rows = (
                (
                    await transaction._execute(
                        select(sequences.c.key_digest, sequences.c.value).where(
                            sequences.c.store_digest == transaction._store_hex
                        )
                    )
                )
                .mappings()
                .all()
            )
            for row in sequence_rows:
                _row_digest(row["key_digest"])
                _row_nonnegative_int(row["value"])
            alias_rows = (
                (
                    await transaction._execute(
                        select(aliases.c.alias_digest, aliases.c.record_key_digest).where(
                            aliases.c.store_digest == transaction._store_hex
                        )
                    )
                )
                .mappings()
                .all()
            )
            for row in alias_rows:
                _row_digest(row["alias_digest"])
                _row_digest(row["record_key_digest"])
            orphan_alias = (await transaction._execute(
                select(aliases.c.id)
                .select_from(
                    aliases.outerjoin(
                        records,
                        and_(
                            aliases.c.record_key_digest == records.c.key_digest,
                            records.c.store_digest == transaction._store_hex,
                        ),
                    )
                )
                .where(
                    aliases.c.store_digest == transaction._store_hex,
                    records.c.id.is_(None),
                )
                .limit(1)
            )).scalar()
            orphan_fact = (await transaction._execute(
                select(facts.c.id)
                .select_from(
                    facts.outerjoin(
                        records,
                        and_(
                            facts.c.owner_key_digest == records.c.key_digest,
                            records.c.store_digest == transaction._store_hex,
                        ),
                    )
                )
                .where(
                    facts.c.store_digest == transaction._store_hex,
                    records.c.id.is_(None),
                )
                .limit(1)
            )).scalar()
            if orphan_alias is not None or orphan_fact is not None:
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
                if value.stream_digest != previous_stream:
                    if value.sequence != 1:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                elif (
                    previous_sequence is None or value.sequence != previous_sequence + 1
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                previous_stream = value.stream_digest
                previous_sequence = value.sequence
            last = values[-1]
            fact_cursor = FactScanCursor(last.stream_digest, last.sequence)

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
            operation_cursor = OperationScanCursor(values[-1].key_digest)
        _logger.info(
            "SQL StateStore integrity validated: domain=%s", self.partition
        )

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
        store_digest: bytes,
        *,
        clock: _SqlTransactionClock | None = None,
    ) -> None:
        if not isinstance(store_digest, bytes) or len(store_digest) != 32:
            raise ValueError("store_digest must be a 32-byte digest")
        self._session = session
        self._metadata = metadata
        self._context = context
        self._store_digest = store_digest
        self._store_hex = store_digest.hex()
        self._guarded_record_keys: set[bytes] = set()
        self._deleted_records = False
        self._record_cache: dict[bytes, StoredRecord | None] = {}
        self._alias_cache: dict[bytes, bytes | None] = {}
        self._sequence_cache: dict[bytes, int] = {}
        self._clock = clock if clock is not None else _SqlTransactionClock(session, context)
        self._mutation_error: BaseException | None = None

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

        if key in self._record_cache:
            return self._record_cache[key]
        table = self._table("ai_state_records")
        row = (
            (
                await self._execute(
                    select(table).where(
                        table.c.store_digest == self._store_hex,
                        table.c.key_digest == _hex(key),
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            self._record_cache[key] = None
            return None
        value = _record_from_row(row)
        self._record_cache[key] = value
        return value

    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        if not keys:
            return {}
        from sqlalchemy import select

        unique_keys = tuple(dict.fromkeys(keys))
        missing = tuple(key for key in unique_keys if key not in self._record_cache)
        table = self._table("ai_state_records")
        rows = ()
        if missing:
            rows = (
                (
                    await self._execute(
                        select(table).where(
                            table.c.store_digest == self._store_hex,
                            table.c.key_digest.in_(tuple(_hex(key) for key in missing))
                        )
                    )
                )
                .mappings()
                .all()
            )
        values = tuple(_record_from_row(row) for row in rows)
        self._record_cache.update({value.key_digest: value for value in values})
        for key in missing:
            self._record_cache.setdefault(key, None)
        return {
            key: value
            for key in unique_keys
            if (value := self._record_cache[key]) is not None
        }

    async def read_archive_frontiers(
        self, *, scope_digest: bytes, requests: Sequence[ArchiveNodeRead],
        content_keys: Sequence[bytes] = (),
    ) -> Mapping[bytes, StoredRecord] | None:
        dialect = self._context.engine.dialect
        if dialect.name not in {"sqlite", "mysql", "postgresql"}:
            return None
        scope_hex = _hex(scope_digest)
        merged: dict[bytes, ArchiveNodeRead] = {}
        for request in requests:
            previous = merged.get(request.key_digest)
            merged[request.key_digest] = request if previous is None else ArchiveNodeRead(
                request.key_digest, request.metadata_slots | previous.metadata_slots,
                request.content_slots | previous.content_slots,
            )
        direct = tuple(dict.fromkeys(content_keys))
        for key in direct:
            _hex(key)
        result: dict[bytes, StoredRecord] = {}
        for nodes, contents in self._archive_read_batches(tuple(merged.values()), direct):
            statement = self._archive_read_statement(scope_hex, nodes, contents)
            compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
            parameter_count = len(compiled.params) if compiled.positiontup is None else len(compiled.positiontup)
            if parameter_count > _ARCHIVE_READ_BIND_BUDGET:
                raise ValueError("archive read exceeds the SQL parameter budget")
            rows = (await self._execute(statement)).mappings().all()
            for row in rows:
                # An absent optional edge contributes no record to the mapping.
                if row["key_digest"] is None:
                    continue
                record = _record_from_row(row)
                result[record.key_digest] = self._record_cache[record.key_digest] = record
        return result

    @classmethod
    def _archive_read_batches(
        cls, requests: Sequence[ArchiveNodeRead], content_keys: Sequence[bytes],
    ) -> Iterator[tuple[tuple[ArchiveNodeRead, ...], tuple[bytes, ...]]]:
        available = _ARCHIVE_READ_BIND_BUDGET - _ARCHIVE_READ_FIXED_BINDS
        nodes: list[ArchiveNodeRead] = []
        contents: list[bytes] = []
        used = 0
        for request in requests:
            # Each enabled branch contains the input key once, regardless of depth.
            cost = int(request.metadata_slots).bit_count() + int(request.content_slots).bit_count()
            if cost > available:
                raise ValueError("archive read parameter budget cannot fit one node")
            if used + cost > available:
                yield tuple(nodes), tuple(contents)
                nodes, contents, used = [], [], 0
            nodes.append(request)
            used += cost
        for key in content_keys:
            if used + 1 > available:
                yield tuple(nodes), tuple(contents)
                nodes, contents, used = [], [], 0
            contents.append(key)
            used += 1
        if nodes or contents:
            yield tuple(nodes), tuple(contents)

    def _archive_read_statement(
        self, scope_hex: str, requests: Sequence[ArchiveNodeRead], content_keys: Sequence[bytes],
    ) -> "Select":
        from sqlalchemy import and_, literal, select, union

        table = self._table("ai_state_records")
        current = table.alias("archive_current")
        current_scope = and_(current.c.store_digest == self._store_hex,
                             current.c.scope_digest == scope_hex, current.c.kind == "archive_sequence")
        branches = []
        for slot, field in ((ArchiveNodeSlot.SELF, None), (ArchiveNodeSlot.LEFT, "left"),
                            (ArchiveNodeSlot.RIGHT, "right"), (ArchiveNodeSlot.NEXT, "next")):
            metadata_keys = tuple(_hex(item.key_digest) for item in requests if item.metadata_slots & slot)
            body_keys = tuple(_hex(item.key_digest) for item in requests if item.content_slots & slot)
            if not metadata_keys and not body_keys:
                continue
            identity = current.c.sort_key if field is None else current.c.payload_json[field].as_string()
            if metadata_keys:
                branches.append(select(literal("archive_sequence").label("kind"), identity.label("identity")).where(
                    current_scope, current.c.key_digest.in_(metadata_keys),
                ))
            if not body_keys:
                continue
            node, joined = current, current
            if field is not None:
                node = table.alias("archive_" + field)
                joined = current.outerjoin(node, and_(
                    node.c.store_digest == self._store_hex, node.c.scope_digest == scope_hex,
                    node.c.kind == "archive_sequence", node.c.sort_key == identity,
                ))
            branches.append(select(literal("archive_content").label("kind"),
                                   node.c.payload_json["content"].as_string().label("identity"))
                            .select_from(joined).where(current_scope, current.c.key_digest.in_(body_keys)))
        if content_keys:
            content = table.alias("archive_direct_content")
            branches.append(select(content.c.kind.label("kind"), content.c.sort_key.label("identity")).where(
                content.c.store_digest == self._store_hex, content.c.scope_digest == scope_hex,
                content.c.kind == "archive_content", content.c.key_digest.in_(tuple(_hex(key) for key in content_keys)),
            ))
        keys = (branches[0] if len(branches) == 1 else union(*branches)).subquery("archive_record_keys")
        # Keep the bounded identity set on the driving side of the row lookup.
        return select(table).select_from(keys.outerjoin(table, and_(
            table.c.store_digest == self._store_hex, table.c.scope_digest == scope_hex,
            table.c.kind == keys.c.kind, table.c.sort_key == keys.c.identity,
        )))

    async def insert_record(self, record: StoredRecord) -> None:
        await self.insert_records((record,))

    async def insert_records(self, records: Sequence[StoredRecord]) -> None:
        values = tuple(records)
        keys = [record.key_digest for record in values]
        if len(keys) != len(set(keys)):
            raise ValueError("insert_records contains duplicate keys")
        if not values:
            return
        for record in values:
            validate_record_identity(record)
        from sqlalchemy import insert

        await self._execute(
            insert(self._table("ai_state_records")).values(
                [
                    {
                        **_record_values(record),
                        "store_digest": self._store_hex,
                    }
                    for record in sorted(values, key=lambda value: value.key_digest)
                ]
            )
        )
        self._log_batch("insert_records", len(values), 1)
        for record in values:
            self._guarded_record_keys.add(record.key_digest)
            self._record_cache[record.key_digest] = record

    async def guard_record(
        self,
        key: bytes,
        *,
        expected_storage_revision: int,
    ) -> StoredRecord | None:
        if (
            isinstance(expected_storage_revision, bool)
            or not isinstance(expected_storage_revision, int)
            or expected_storage_revision < 0
        ):
            raise ValueError("expected_storage_revision must be a non-negative integer")
        if key in self._guarded_record_keys:
            return await self.get_record(key)
        current = await self.get_record(key)
        if current is None or current.storage_revision != expected_storage_revision:
            return None
        from sqlalchemy import update

        table = self._table("ai_state_records")
        result = await self._execute(
            update(table)
            .where(
                table.c.store_digest == self._store_hex,
                table.c.key_digest == _hex(key),
                table.c.storage_revision == expected_storage_revision,
            )
            .values(storage_revision=expected_storage_revision + 1)
        )
        if result.rowcount != 1:
            return None
        guarded = replace(current, storage_revision=expected_storage_revision + 1)
        self._guarded_record_keys.add(key)
        self._record_cache[key] = guarded
        return guarded

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
        keys = [replacement.record.key_digest for replacement in values]
        if len(keys) != len(set(keys)):
            raise ValueError("replace_records contains duplicate keys")
        if not values:
            return
        current = await self.get_records(keys)
        ordered = tuple(sorted(values, key=lambda value: value.record.key_digest))
        for replacement in ordered:
            existing = current.get(replacement.record.key_digest)
            if (
                existing is None
                or existing.storage_revision != replacement.expected_storage_revision
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            validate_record_identity(replacement.record)
            validate_record_replacement(existing, replacement.record)
            if (
                replacement.record.storage_revision
                != replacement.expected_storage_revision + 1
            ):
                raise ValueError(
                    "replacement must increment storage_revision exactly once"
                )
        from sqlalchemy import bindparam, case, func, literal, tuple_, update

        table = self._table("ai_state_records")
        prepared = tuple(_record_values(value.record) for value in ordered)
        columns = ("state", "storage_revision", "lease_owner", "lease_fence",
                   "lease_expires_at", "payload_json")
        executemany = self._context.dialect.prefer_update_executemany
        statement_count = 0
        for offset in range(0, len(ordered), _RECORD_REPLACE_BATCH_SIZE):
            batch = ordered[offset:offset + _RECORD_REPLACE_BATCH_SIZE]
            rows = prepared[offset:offset + _RECORD_REPLACE_BATCH_SIZE]
            parameters = None
            if executemany:
                statement = update(table).where(
                    table.c.store_digest == self._store_hex,
                    table.c.key_digest == bindparam("_replacement_key"),
                    table.c.storage_revision == bindparam("_replacement_expected_storage_revision"),
                ).values(**{column: bindparam(f"_replacement_{column}") for column in columns},
                         updated_at=func.current_timestamp())
                parameters = [
                    {"_replacement_key": row["key_digest"],
                     "_replacement_expected_storage_revision": value.expected_storage_revision,
                     **{f"_replacement_{column}": row[column] for column in columns}}
                    for row, value in zip(rows, batch)
                ]
            else:
                cas_conditions = [(row["key_digest"], value.expected_storage_revision)
                                  for row, value in zip(rows, batch)]
                updates = {
                    column: rows[0][column] if len(rows) == 1 else case(
                        {row["key_digest"]: literal(row[column], type_=table.c[column].type)
                         for row in rows},
                        value=table.c.key_digest,
                        else_=table.c[column],
                    )
                    for column in columns
                }
                statement = update(table).where(
                    table.c.store_digest == self._store_hex,
                    table.c.key_digest.in_([row["key_digest"] for row in rows]),
                    tuple_(table.c.key_digest, table.c.storage_revision).in_(cas_conditions),
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
                raise error
        self._log_batch("replace_records", len(ordered), statement_count)
        for replacement in ordered:
            self._guarded_record_keys.add(replacement.record.key_digest)
            self._record_cache[replacement.record.key_digest] = replacement.record

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
        now = await self._context.run_statement(
            self._context.dialect.database_current_time(self._session)
        )
        if transition.apply(current, now) is None:
            return None
        guarded = await self.guard_record(key, expected_storage_revision=expected_storage_revision)
        if guarded is None:
            return None
        try:
            # The CAS may wait past expiry. Sample time in a separate statement
            # after its write protection is held, never from tx.now() or the UPDATE.
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
        table = self._table("ai_state_records")
        result = await self._execute(
            update(table)
            .where(
                table.c.store_digest == self._store_hex,
                table.c.key_digest == _hex(key),
                table.c.storage_revision == expected_storage_revision,
            )
            .values(
                lease_owner=lease_owner,
                lease_fence=lease_fence,
                lease_expires_at=lease_expires_at,
                storage_revision=expected_storage_revision + 1,
                updated_at=func.current_timestamp(),
            )
        )
        if result.rowcount == 1:
            self._guarded_record_keys.add(key)
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
        if not unique_keys:
            return
        from sqlalchemy import delete, tuple_

        table = self._table("ai_state_records")
        aliases = self._table("ai_state_aliases")
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
                    (_hex(key), record.storage_revision)
                    for key, record in sorted(current.items())
                )
                key_hexes = tuple(key for key, _revision in cas_conditions)
                # Deleting the CAS-matched owners fences concurrent alias/fact writers.
                result = await self._execute(
                    delete(table).where(
                        table.c.store_digest == self._store_hex,
                        table.c.key_digest.in_(key_hexes),
                        tuple_(table.c.key_digest, table.c.storage_revision).in_(cas_conditions),
                    )
                )
                if result.rowcount != len(current):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await self._execute(
                    delete(aliases).where(
                        aliases.c.store_digest == self._store_hex,
                        aliases.c.record_key_digest.in_(key_hexes),
                    )
                )
                await self._execute(
                    delete(facts).where(
                        facts.c.store_digest == self._store_hex,
                        facts.c.owner_key_digest.in_(key_hexes),
                    )
                )
                deleted_keys.update(current)
                statement_count += 3
        except BaseException as error:
            # A caught partial deletion must still abort the owning transaction.
            self._abort_mutation(error)
            raise
        self._deleted_records = self._deleted_records or bool(deleted_keys)
        for key in deleted_keys:
            self._guarded_record_keys.discard(key)
            self._record_cache[key] = None
        for alias, record_key in tuple(self._alias_cache.items()):
            if record_key in deleted_keys:
                self._alias_cache[alias] = None
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
        table = self._table("ai_state_records")
        statement = delete(table).where(
            table.c.store_digest == self._store_hex,
            table.c.key_digest == _hex(key),
            table.c.storage_revision == guarded.storage_revision,
        )
        children_changed = False
        try:
            await self._execute(
                delete(self._table("ai_state_aliases")).where(
                    self._table("ai_state_aliases").c.store_digest == self._store_hex,
                    self._table("ai_state_aliases").c.record_key_digest == _hex(key),
                )
            )
            children_changed = True
            await self._execute(
                delete(self._table("ai_state_facts")).where(
                    self._table("ai_state_facts").c.store_digest == self._store_hex,
                    self._table("ai_state_facts").c.owner_key_digest == _hex(key),
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
            self._deleted_records = True
            self._guarded_record_keys.discard(key)
            self._record_cache[key] = None
            for alias, record_key in tuple(self._alias_cache.items()):
                if record_key == key:
                    self._alias_cache[alias] = None
        return result.rowcount == 1

    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        from sqlalchemy import and_, or_, select

        table = self._table("ai_state_records")
        conditions = [table.c.store_digest == self._store_hex]
        if query.scope_digest is not None:
            conditions.append(table.c.scope_digest == _hex(query.scope_digest))
        if query.parent_digest is not None:
            conditions.append(table.c.parent_digest == _hex(query.parent_digest))
        if query.kind is not None:
            conditions.append(table.c.kind == query.kind)
        if query.states is not None:
            conditions.append(table.c.state.in_(tuple(query.states)))
        if query.sort_key_prefix is not None:
            conditions.append(table.c.sort_key >= query.sort_key_prefix)
            prefix_end = _sort_key_prefix_upper_bound(query.sort_key_prefix)
            if prefix_end is not None:
                conditions.append(table.c.sort_key < prefix_end)
        if query.after_sort_key is not None and query.after_key_digest is not None:
            conditions.append(
                or_(
                    table.c.sort_key > query.after_sort_key,
                    and_(
                        table.c.sort_key == query.after_sort_key,
                        table.c.key_digest > _hex(query.after_key_digest),
                    ),
                )
            )
        statement = (
            select(table)
            .where(*conditions)
            .order_by(
                table.c.sort_key,
                table.c.key_digest,
            )
        )
        if query.limit is not None:
            statement = statement.limit(query.limit)
        rows = (await self._execute(statement)).mappings().all()
        values = tuple(_record_from_row(row) for row in rows)
        self._record_cache.update({value.key_digest: value for value in values})
        return values

    async def scan_records(self) -> tuple[StoredRecord, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_records")
        rows = (
            await self._execute(
                select(table).where(table.c.store_digest == self._store_hex)
            )
        ).mappings().all()
        return tuple(_record_from_row(row) for row in rows)

    async def scan_records_page(
        self,
        *,
        after: RecordScanCursor | None,
        limit: int,
    ) -> tuple[StoredRecord, ...]:
        _require_scan_limit(limit)
        from sqlalchemy import select

        table = self._table("ai_state_records")
        statement = select(table).where(table.c.store_digest == self._store_hex)
        if after is not None:
            statement = statement.where(
                table.c.key_digest > _hex(after.key_digest)
            )
        statement = statement.order_by(table.c.key_digest).limit(limit)
        rows = (await self._execute(statement)).mappings().all()
        values = tuple(_record_from_row(row) for row in rows)
        self._record_cache.update({value.key_digest: value for value in values})
        return values

    async def resolve_alias(self, alias: bytes) -> bytes | None:
        return (await self.resolve_aliases((alias,))).get(alias)

    async def resolve_aliases(self, aliases: Sequence[bytes]) -> Mapping[bytes, bytes]:
        unique_aliases = tuple(dict.fromkeys(aliases))
        missing = tuple(
            alias for alias in unique_aliases if alias not in self._alias_cache
        )
        if missing:
            from sqlalchemy import and_, select

            table = self._table("ai_state_aliases")
            records = self._table("ai_state_records")
            rows = (
                (
                    await self._execute(
                        select(
                            table.c.alias_digest,
                            table.c.record_key_digest,
                            records.c.id,
                        )
                        .select_from(
                            table.outerjoin(
                                records,
                                and_(
                                    table.c.record_key_digest == records.c.key_digest,
                                    records.c.store_digest == self._store_hex,
                                ),
                            )
                        )
                        .where(
                            table.c.store_digest == self._store_hex,
                            table.c.alias_digest.in_(
                                tuple(_hex(alias) for alias in missing)
                            ),
                        )
                    )
                )
                .mappings()
                .all()
            )
            if any(row["id"] is None for row in rows):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values = {
                _row_digest(row["alias_digest"]): _row_digest(row["record_key_digest"])
                for row in rows
            }
            self._alias_cache.update(values)
            for alias in missing:
                self._alias_cache.setdefault(alias, None)
        values = {
            alias: value
            for alias in unique_aliases
            if (value := self._alias_cache[alias]) is not None
        }
        return values

    async def scan_aliases(self) -> tuple[StoredAlias, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_aliases")
        rows = (
            await self._execute(
                select(table)
                .where(table.c.store_digest == self._store_hex)
                .order_by(table.c.alias_digest)
            )
        ).mappings().all()
        return tuple(
            StoredAlias(
                _row_digest(row["alias_digest"]),
                _row_digest(row["record_key_digest"]),
            )
            for row in rows
        )

    async def scan_aliases_page(
        self,
        *,
        after: bytes | None,
        limit: int,
    ) -> tuple[StoredAlias, ...]:
        _require_scan_limit(limit)
        from sqlalchemy import select

        table = self._table("ai_state_aliases")
        statement = select(table).where(table.c.store_digest == self._store_hex)
        if after is not None:
            statement = statement.where(table.c.alias_digest > _hex(after))
        rows = (
            await self._execute(
                statement.order_by(table.c.alias_digest).limit(limit)
            )
        ).mappings().all()
        return tuple(
            StoredAlias(
                _row_digest(row["alias_digest"]),
                _row_digest(row["record_key_digest"]),
            )
            for row in rows
        )

    async def insert_alias(self, alias: StoredAlias) -> None:
        await self.insert_aliases((alias,))

    async def insert_aliases(self, aliases: Sequence[StoredAlias]) -> None:
        if not aliases:
            return
        by_alias: dict[bytes, StoredAlias] = {}
        for alias in aliases:
            current = by_alias.get(alias.alias_digest)
            if (
                current is not None
                and current.record_key_digest != alias.record_key_digest
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            by_alias[alias.alias_digest] = alias
        values = tuple(sorted(by_alias.values(), key=lambda value: value.alias_digest))
        if any(
            alias.record_key_digest not in self._guarded_record_keys for alias in values
        ):
            raise RuntimeError("alias owner must be guarded in the current transaction")
        existing = await self.resolve_aliases(
            tuple(alias.alias_digest for alias in values)
        )
        rows: list[dict[str, object]] = []
        for alias in values:
            current = existing.get(alias.alias_digest)
            if current is not None:
                if current != alias.record_key_digest:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                continue
            rows.append(
                {
                    "store_digest": self._store_hex,
                    "alias_digest": _hex(alias.alias_digest),
                    "record_key_digest": _hex(alias.record_key_digest),
                }
            )
        if rows:
            from sqlalchemy import insert

            await self._execute(
                insert(self._table("ai_state_aliases")).values(rows)
            )
        self._log_batch("insert_aliases", len(values), 1 if rows else 0)
        for alias in values:
            self._alias_cache[alias.alias_digest] = alias.record_key_digest

    async def get_sequence(self, key: bytes) -> int:
        if key in self._sequence_cache:
            return self._sequence_cache[key]
        values = await self.get_sequences((key,))
        return values[key]

    async def get_sequences(self, keys: Sequence[bytes]) -> Mapping[bytes, int]:
        if not keys:
            return {}
        from sqlalchemy import select

        unique_keys = tuple(dict.fromkeys(keys))
        missing = tuple(key for key in unique_keys if key not in self._sequence_cache)
        table = self._table("ai_state_sequences")
        if missing:
            rows = (
                (
                    await self._execute(
                        select(table).where(
                            table.c.store_digest == self._store_hex,
                            table.c.key_digest.in_(tuple(_hex(key) for key in missing))
                        )
                    )
                )
                .mappings()
                .all()
            )
            self._sequence_cache.update(
                {
                    _row_digest(row["key_digest"]): _row_nonnegative_int(row["value"])
                    for row in rows
                }
            )
            for key in missing:
                self._sequence_cache.setdefault(key, 0)
        return {key: self._sequence_cache[key] for key in unique_keys}

    async def scan_sequences(self) -> Mapping[bytes, int]:
        from sqlalchemy import select

        table = self._table("ai_state_sequences")
        rows = (
            await self._execute(
                select(table)
                .where(table.c.store_digest == self._store_hex)
                .order_by(table.c.key_digest)
            )
        ).mappings().all()
        return {
            _row_digest(row["key_digest"]): _row_nonnegative_int(row["value"])
            for row in rows
        }

    async def scan_sequences_page(
        self,
        *,
        after: bytes | None,
        limit: int,
    ) -> Mapping[bytes, int]:
        _require_scan_limit(limit)
        from sqlalchemy import select

        table = self._table("ai_state_sequences")
        statement = select(table).where(table.c.store_digest == self._store_hex)
        if after is not None:
            statement = statement.where(table.c.key_digest > _hex(after))
        rows = (
            await self._execute(
                statement.order_by(table.c.key_digest).limit(limit)
            )
        ).mappings().all()
        return {
            _row_digest(row["key_digest"]): _row_nonnegative_int(row["value"])
            for row in rows
        }

    async def next_sequence(self, key: bytes) -> int:
        return await self.reserve_sequence(key, 1)

    async def reserve_sequence(self, key: bytes, count: int) -> int:
        return (await self.reserve_sequences({key: count}))[key]

    async def reserve_sequences(
        self, requests: Mapping[bytes, int]
    ) -> Mapping[bytes, int]:
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in requests.values()
        ):
            raise ValueError("sequence reservation count must be a positive integer")
        if not requests:
            return {}
        table = self._table("ai_state_sequences")
        if len(requests) == 1:
            key, count = next(iter(requests.items()))
            value = await self._context.run_statement(
                self._context.dialect.upsert_increment(
                    self._session,
                    table=table,
                    values={"store_digest": self._store_hex, "key_digest": _hex(key)},
                    column="value",
                    index_elements=("store_digest", "key_digest"),
                    step=count,
                )
            )
            result = {key: _row_nonnegative_int(value)}
            self._sequence_cache.update(result)
            return result
        rows = [
            {
                    "store_digest": self._store_hex,
                "key_digest": _hex(key),
                "value": requests[key],
            }
            for key in sorted(requests)
        ]
        values = await self._context.run_statement(self._context.dialect.upsert_increment_many(
            self._session,
            table=table,
            rows=rows,
            column="value",
            index_elements=("store_digest", "key_digest"),
            returning_key="key_digest",
        ))
        result = {
            _row_digest(key): _row_nonnegative_int(value)
            for key, value in values.items()
        }
        self._sequence_cache.update(result)
        return {key: result[key] for key in requests}

    async def advance_sequence(self, key: bytes, expected: int) -> int:
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise ValueError("expected sequence must be a non-negative integer")
        from sqlalchemy import func, update

        table = self._table("ai_state_sequences")
        result = await self._execute(
            update(table)
            .where(
                table.c.store_digest == self._store_hex,
                table.c.key_digest == _hex(key),
                table.c.value == expected,
            )
            .values(value=expected + 1, updated_at=func.current_timestamp())
        )
        if result.rowcount != 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return expected + 1

    async def delete_sequence(self, key: bytes) -> None:
        await self.delete_sequences((key,))

    async def delete_sequences(self, keys: Sequence[bytes]) -> None:
        values = tuple(sorted(set(keys)))
        if not values:
            return
        from sqlalchemy import delete

        await self._execute(
            delete(self._table("ai_state_sequences")).where(
                self._table("ai_state_sequences").c.store_digest == self._store_hex,
                self._table("ai_state_sequences").c.key_digest.in_(
                    tuple(_hex(key) for key in values)
                ),
            )
        )
        self._log_batch("delete_sequences", len(values), 1)
        for key in values:
            self._sequence_cache[key] = 0

    async def insert_fact(self, fact: StoredFact) -> None:
        await self.insert_facts((fact,))

    async def insert_facts(self, facts: Sequence[StoredFact]) -> None:
        if not facts:
            return
        if any(
            fact.owner_key_digest not in self._guarded_record_keys for fact in facts
        ):
            raise RuntimeError("fact owner must be guarded in the current transaction")
        from sqlalchemy import insert

        await self._execute(
            insert(self._table("ai_state_facts")).values(
                [
                    {**_fact_values(fact), "store_digest": self._store_hex}
                    for fact in facts
                ]
            )
        )
        self._log_batch("insert_facts", len(facts), 1)

    async def list_facts(self, query: FactQuery) -> tuple[StoredFact, ...]:
        from sqlalchemy import func, select

        table = self._table("ai_state_facts")
        conditions = [
            table.c.store_digest == self._store_hex,
            table.c.stream_digest == _hex(query.stream_digest),
        ]
        if query.after_sequence is not None:
            conditions.append(table.c.sequence > query.after_sequence)
        if query.subject_digest is not None:
            conditions.append(table.c.subject_digest == _hex(query.subject_digest))
        if query.latest_per_subject:
            latest_sequences = (
                select(func.max(table.c.sequence))
                .where(*conditions)
                .group_by(table.c.subject_digest)
            )
            statement = (
                select(table)
                .where(
                    table.c.store_digest == self._store_hex,
                    table.c.stream_digest == _hex(query.stream_digest),
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
                select(table).where(table.c.store_digest == self._store_hex)
            )
        ).mappings().all()
        return tuple(_fact_from_row(row) for row in rows)

    async def scan_facts_page(
        self,
        *,
        after: FactScanCursor | None,
        limit: int,
    ) -> tuple[StoredFact, ...]:
        _require_scan_limit(limit)
        from sqlalchemy import and_, or_, select

        table = self._table("ai_state_facts")
        conditions = [table.c.store_digest == self._store_hex]
        if after is not None:
            conditions.append(
                or_(
                    table.c.stream_digest > _hex(after.stream_digest),
                    and_(
                        table.c.stream_digest == _hex(after.stream_digest),
                        table.c.sequence > after.sequence,
                    ),
                )
            )
        statement = (
            select(table)
            .where(*conditions)
            .order_by(table.c.stream_digest, table.c.sequence)
            .limit(limit)
        )
        rows = (await self._execute(statement)).mappings().all()
        return tuple(_fact_from_row(row) for row in rows)

    async def delete_fact_streams(self, owner_key: bytes) -> None:
        from sqlalchemy import delete

        await self._execute(
            delete(self._table("ai_state_facts")).where(
                self._table("ai_state_facts").c.store_digest == self._store_hex,
                self._table("ai_state_facts").c.owner_key_digest == _hex(owner_key),
            )
        )

    async def insert_operation(self, value: StoredOperation) -> None:
        from sqlalchemy import insert

        await self._execute(
            insert(self._table("ai_state_operations")).values(
                {**_operation_values(value), "store_digest": self._store_hex}
            )
        )

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        from sqlalchemy import select

        table = self._table("ai_state_operations")
        row = (
            (
                await self._execute(
                    select(table).where(
                        table.c.store_digest == self._store_hex,
                        table.c.key_digest == _hex(key),
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        return None if row is None else _operation_from_row(row)

    async def replace_operation(
        self, value: StoredOperation, *, expected_state: str
    ) -> bool:
        current = await self.get_operation(value.key_digest)
        if current is None or current.state != expected_state:
            return False
        validate_operation_replacement(current, value)
        from sqlalchemy import update

        table = self._table("ai_state_operations")
        result = await self._execute(
            update(table)
            .where(
                table.c.store_digest == self._store_hex,
                table.c.key_digest == _hex(value.key_digest),
                table.c.state == expected_state,
            )
            .values(_operation_values(value, updating=True))
        )
        return result.rowcount == 1

    async def list_operations(
        self, query: OperationQuery
    ) -> tuple[StoredOperation, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_operations")
        conditions = [table.c.store_digest == self._store_hex]
        if query.stream_digest is not None:
            conditions.append(table.c.stream_digest == _hex(query.stream_digest))
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
            .order_by(table.c.sequence, table.c.key_digest)
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
                select(table).where(table.c.store_digest == self._store_hex)
            )
        ).mappings().all()
        return tuple(_operation_from_row(row) for row in rows)

    async def scan_operations_page(
        self,
        *,
        after: OperationScanCursor | None,
        limit: int,
    ) -> tuple[StoredOperation, ...]:
        _require_scan_limit(limit)
        from sqlalchemy import select

        table = self._table("ai_state_operations")
        statement = select(table).where(table.c.store_digest == self._store_hex)
        if after is not None:
            statement = statement.where(table.c.key_digest > _hex(after.key_digest))
        statement = statement.order_by(table.c.key_digest).limit(limit)
        rows = (await self._execute(statement)).mappings().all()
        return tuple(_operation_from_row(row) for row in rows)

    async def delete_operations(
        self, query: OperationQuery
    ) -> tuple[StoredOperation, ...]:
        values = await self.list_operations(query)
        from sqlalchemy import delete

        if values:
            await self._execute(
                delete(self._table("ai_state_operations")).where(
                    self._table("ai_state_operations").c.store_digest == self._store_hex,
                    self._table("ai_state_operations").c.key_digest.in_(
                        tuple(_hex(value.key_digest) for value in values)
                    ),
                )
            )
        return values

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
        "key_digest": _hex(record.key_digest),
        "scope_digest": None
        if record.scope_digest is None
        else _hex(record.scope_digest),
        "parent_digest": None
        if record.parent_digest is None
        else _hex(record.parent_digest),
        "kind": record.kind,
        "sort_key": record.sort_key,
        "state": record.state,
        "storage_revision": record.storage_revision,
        "lease_owner": record.lease_owner,
        "lease_fence": record.lease_fence,
        "lease_expires_at": record.lease_expires_at,
        "payload_json": dict(record.data),
    }
    if updating:
        from sqlalchemy import func

        values["updated_at"] = func.current_timestamp()
    return values


def _record_from_row(row: Mapping[str, object]) -> StoredRecord:
    try:
        record = StoredRecord(
            _row_digest(row["key_digest"]),
            _row_digest_or_none(row["scope_digest"]),
            _row_digest_or_none(row["parent_digest"]),
            _row_string(row["kind"]),
            _row_string(row["sort_key"]),
            _row_optional_string(row["state"]),
            _row_nonnegative_int(row["storage_revision"]),
            _row_optional_string(row["lease_owner"]),
            _row_nonnegative_int(row["lease_fence"]),
            _datetime_or_none(row["lease_expires_at"]),
            _row_mapping(row["payload_json"]),
        )
        validate_record_identity(record)
    except AIError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return record


def _fact_values(fact: StoredFact) -> dict[str, object]:
    return {
        "stream_digest": _hex(fact.stream_digest),
        "sequence": fact.sequence,
        "owner_key_digest": _hex(fact.owner_key_digest),
        "kind": fact.kind,
        "subject_digest": None
        if fact.subject_digest is None
        else _hex(fact.subject_digest),
        "state": fact.state,
        "payload_json": dict(fact.data),
    }


def _fact_from_row(row: Mapping[str, object]) -> StoredFact:
    try:
        return StoredFact(
            _row_digest(row["stream_digest"]),
            _row_nonnegative_int(row["sequence"]),
            _row_digest(row["owner_key_digest"]),
            _row_string(row["kind"]),
            _row_digest_or_none(row["subject_digest"]),
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
        "key_digest": _hex(value.key_digest),
        "stream_digest": _hex(value.stream_digest),
        "sequence": value.sequence,
        "state": value.state,
        "compactable": value.compactable,
        "payload_json": dict(value.data),
    }
    if updating:
        from sqlalchemy import func

        values["updated_at"] = func.current_timestamp()
    return values


def _operation_from_row(row: Mapping[str, object]) -> StoredOperation:
    try:
        return StoredOperation(
            _row_digest(row["key_digest"]),
            _row_digest(row["stream_digest"]),
            _row_nonnegative_int(row["sequence"]),
            _row_string(row["state"]),
            _row_bool(row["compactable"]),
            _row_mapping(row["payload_json"]),
        )
    except AIError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _sort_key_prefix_upper_bound(prefix: str) -> str | None:
    for index in range(len(prefix) - 1, -1, -1):
        value = ord(prefix[index])
        if value < 127:
            return prefix[:index] + chr(value + 1)
    return None


def _hex(value: bytes) -> str:
    if not isinstance(value, bytes) or len(value) != 32:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.hex()


def _row_digest(value: object) -> bytes:
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        result = bytes.fromhex(value)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(result) != 32 or result.hex() != value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _row_digest_or_none(value: object) -> bytes | None:
    return None if value is None else _row_digest(value)


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


__all__ = ["SqlStateStorageGroup", "SqlStateStore"]
