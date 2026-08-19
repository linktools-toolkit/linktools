#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL implementation of the backend-neutral Runtime StateStore."""

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypeVar

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import (
    SqlStorageContext,
    create_sql_storage_context,
)
from ._plan import RuntimeDomain
from ._schema import build_runtime_sql_metadata
from ._store import (
    FactQuery,
    OperationQuery,
    RecordQuery,
    RecordReplacement,
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
    validate_operation_replacement,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Table
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

ValueT = TypeVar("ValueT")
_logger = environ.get_logger("ai.runtime.state.sql")


class _SqlGroupTransaction:
    def __init__(self, group: "SqlStateStorageGroup", transactions: Mapping["SqlStateStore", StateTransaction]) -> None:
        self._group = group
        self._transactions = transactions

    def transaction(self, store: "SqlStateStore") -> StateTransaction:
        if store.storage_group is not self._group:
            raise RuntimeError("store does not belong to this StateStorageGroup")
        try:
            return self._transactions[store]
        except KeyError as error:
            raise RuntimeError("store was not enlisted in the StateStorageGroup transaction") from error


class SqlStateStorageGroup:
    """Own one SQL context and one physical transaction for its logical stores."""

    def __init__(
        self,
        context: SqlStorageContext,
        metadata: "MetaData",
        *,
        owns_context: bool = False,
    ) -> None:
        self._context = context
        self._metadata = metadata
        self._owns_context = owns_context
        self._closed = False
        self._initialized = False

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
        await self._context.initialize(metadata=self._metadata)
        self._initialized = True
        _logger.info(
            "SQL StateStorageGroup initialized: dialect=%s tables=%s",
            self._context.dialect.name,
            len(self._metadata.tables),
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._initialized = False
        if self._owns_context:
            await self._context.close()
        _logger.debug("SQL StateStorageGroup closed")

    async def read(self, store: "SqlStateStore", fn: StateCallback[ValueT]) -> ValueT:
        self._ensure_member(store)
        active = active_state_transaction(store)
        if active is not None:
            return await fn(active)
        async with self._session() as session:
            return await fn(_SqlTransaction(session, self._metadata, self._context))

    async def mutate(
        self,
        stores: Sequence["SqlStateStore"],
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

        async def execute(session: "AsyncSession") -> ValueT:
            transaction = _SqlTransaction(session, self._metadata, self._context)
            transactions = {store: transaction for store in members}
            group_transaction = _SqlGroupTransaction(self, transactions)
            token = bind_state_scope(self, transactions)
            try:
                return await fn(group_transaction)
            finally:
                reset_state_transaction(token)

        try:
            domains = ",".join(store.runtime_domain.value for store in members)
            return await self._context.run_mutation(
                execute,
                domain=f"runtime.state.group[{domains}]",
            )
        except asyncio.CancelledError:
            raise
        except AIError:
            raise
        except BaseException as error:
            from sqlalchemy.exc import IntegrityError

            if isinstance(error, IntegrityError):
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

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
        runtime_domain: RuntimeDomain = RuntimeDomain.CONVERSATION,
        group: SqlStateStorageGroup | None = None,
    ) -> None:
        resolved_context = context or create_sql_storage_context(engine)
        self._metadata = (
            metadata if metadata is not None else build_runtime_sql_metadata(frozenset({RuntimeDomain.CONVERSATION}))
        )
        self._runtime_domain = runtime_domain
        self._owns_group = group is None
        self._storage_group = group or SqlStateStorageGroup(
            resolved_context,
            self._metadata,
            owns_context=context is None,
        )
        self._closed = False
        self._initialized = False

    @property
    def context(self) -> SqlStorageContext:
        return self._storage_group.context

    @property
    def storage_group(self) -> StateStorageGroup:
        return self._storage_group

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    async def initialize(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        await self._storage_group.initialize()
        self._initialized = True
        _logger.debug(
            "SQL StateStore initialized: dialect=%s tables=%s",
            self.context.dialect.name,
            len(self._metadata.tables),
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._initialized = False
        if self._owns_group:
            await self._storage_group.close()
        _logger.debug("SQL StateStore closed")

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
        self._ensure_ready()
        async with self._storage_group._session() as session:
            transaction = _SqlTransaction(session, self._metadata, self.context)
            from sqlalchemy import select

            records = transaction._table("ai_state_records")
            aliases = transaction._table("ai_state_aliases")
            facts = transaction._table("ai_state_facts")
            sequences = transaction._table("ai_state_sequences")
            record_rows = (await session.execute(select(records))).mappings().all()
            for row in record_rows:
                _record_from_row(row)
            sequence_rows = (await session.execute(select(sequences.c.value))).all()
            if any(int(row[0]) < 0 for row in sequence_rows):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            orphan_alias = await session.scalar(
                select(aliases.c.id)
                .select_from(aliases.outerjoin(records, aliases.c.record_key_digest == records.c.key_digest))
                .where(records.c.id.is_(None))
                .limit(1)
            )
            orphan_fact = await session.scalar(
                select(facts.c.id)
                .select_from(facts.outerjoin(records, facts.c.owner_key_digest == records.c.key_digest))
                .where(records.c.id.is_(None))
                .limit(1)
            )
            if orphan_alias is not None or orphan_fact is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            fact_rows = (await session.execute(select(facts))).mappings().all()
            for row in fact_rows:
                _fact_from_row(row)
            operation_table = transaction._table("ai_state_operations")
            operation_rows = (await session.execute(select(operation_table))).mappings().all()
            for row in operation_rows:
                _operation_from_row(row)
            table = transaction._table("ai_state_facts")
            rows = (await session.execute(select(table.c.stream_digest, table.c.sequence))).all()
            grouped: dict[str, list[int]] = {}
            for stream, sequence in rows:
                grouped.setdefault(str(stream), []).append(int(sequence))
            for values in grouped.values():
                if sorted(values) != list(range(1, max(values) + 1)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info("SQL StateStore integrity validated: domain=%s", self._runtime_domain.value)

    def _ensure_ready(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if not self._initialized:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

class _SqlTransaction:
    def __init__(
        self,
        session: "AsyncSession",
        metadata: "MetaData",
        context: SqlStorageContext,
    ) -> None:
        self._session = session
        self._metadata = metadata
        self._context = context
        self._guarded_record_keys: set[bytes] = set()
        self._record_cache: dict[bytes, StoredRecord | None] = {}
        self._alias_cache: dict[bytes, bytes | None] = {}
        self._sequence_cache: dict[bytes, int] = {}
        self._now: datetime | None = None

    async def now(self) -> datetime:
        if self._now is None:
            self._now = await self._context.dialect.database_now(self._session)
        return self._now

    async def get_record(self, key: bytes) -> StoredRecord | None:
        from sqlalchemy import select

        if key in self._record_cache:
            return self._record_cache[key]
        table = self._table("ai_state_records")
        row = (
            (await self._session.execute(select(table).where(table.c.key_digest == _hex(key)))).mappings().one_or_none()
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
                    await self._session.execute(
                        select(table).where(
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

        await self._session.execute(
            insert(self._table("ai_state_records")).values(
                [_record_values(record) for record in sorted(values, key=lambda value: value.key_digest)]
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
        expected_storage_version: int,
    ) -> StoredRecord | None:
        if key in self._guarded_record_keys:
            return await self.get_record(key)
        current = await self.get_record(key)
        if current is None or current.storage_version != expected_storage_version:
            return None
        from sqlalchemy import update

        table = self._table("ai_state_records")
        result = await self._session.execute(
            update(table)
            .where(
                table.c.key_digest == _hex(key),
                table.c.storage_version == expected_storage_version,
            )
            .values(storage_version=expected_storage_version + 1)
        )
        if result.rowcount != 1:
            return None
        guarded = replace(current, storage_version=expected_storage_version + 1)
        self._guarded_record_keys.add(key)
        self._record_cache[key] = guarded
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
        current = await self.get_records(keys)
        ordered = tuple(sorted(values, key=lambda value: value.record.key_digest))
        for replacement in ordered:
            existing = current.get(replacement.record.key_digest)
            if existing is None or existing.storage_version != replacement.expected_storage_version:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            validate_record_identity(replacement.record)
            validate_record_replacement(existing, replacement.record)
            if replacement.record.storage_version != replacement.expected_storage_version + 1:
                raise ValueError("replacement must increment storage_version exactly once")
        from sqlalchemy import bindparam, func, update

        table = self._table("ai_state_records")
        statement = (
            update(table)
            .where(
                table.c.key_digest == bindparam("_replacement_key_digest"),
                table.c.storage_version == bindparam("_replacement_expected_version"),
            )
            .values(
                partition_digest=bindparam("_replacement_partition_digest"),
                scope_digest=bindparam("_replacement_scope_digest"),
                parent_digest=bindparam("_replacement_parent_digest"),
                kind=bindparam("_replacement_kind"),
                sort_key=bindparam("_replacement_sort_key"),
                state=bindparam("_replacement_state"),
                storage_version=bindparam("_replacement_storage_version"),
                lease_owner=bindparam("_replacement_lease_owner"),
                lease_fence=bindparam("_replacement_lease_fence"),
                lease_expires_at=bindparam("_replacement_lease_expires_at"),
                payload_json=bindparam("_replacement_payload_json"),
                updated_at=func.current_timestamp(),
            )
        )
        parameters = [_record_replacement_values(replacement) for replacement in ordered]
        result = await self._session.execute(statement, parameters)
        self._log_batch("replace_records", len(parameters), 1)
        if result.rowcount != len(parameters):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        for replacement in ordered:
            self._guarded_record_keys.add(replacement.record.key_digest)
            self._record_cache[replacement.record.key_digest] = replacement.record

    async def update_record_lease(
        self,
        key: bytes,
        *,
        expected_storage_version: int,
        lease_owner: str | None,
        lease_fence: int,
        lease_expires_at: datetime | None,
    ) -> bool:
        from sqlalchemy import func, update

        if lease_fence < 0 or lease_expires_at is not None and lease_expires_at.tzinfo is None:
            raise ValueError("record lease is invalid")
        table = self._table("ai_state_records")
        result = await self._session.execute(
            update(table)
            .where(
                table.c.key_digest == _hex(key),
                table.c.storage_version == expected_storage_version,
            )
            .values(
                lease_owner=lease_owner,
                lease_fence=lease_fence,
                lease_expires_at=lease_expires_at,
                storage_version=expected_storage_version + 1,
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
                    storage_version=expected_storage_version + 1,
                )
        return result.rowcount == 1

    async def delete_record(self, key: bytes, *, expected_storage_version: int | None = None) -> bool:
        from sqlalchemy import delete

        current = await self.get_record(key)
        if current is None:
            return False
        expected = current.storage_version if expected_storage_version is None else expected_storage_version
        guarded = await self.guard_record(key, expected_storage_version=expected)
        if guarded is None:
            return False
        table = self._table("ai_state_records")
        statement = delete(table).where(
            table.c.key_digest == _hex(key),
            table.c.storage_version == guarded.storage_version,
        )
        await self._session.execute(
            delete(self._table("ai_state_aliases")).where(
                self._table("ai_state_aliases").c.record_key_digest == _hex(key)
            )
        )
        await self._session.execute(
            delete(self._table("ai_state_facts")).where(
                self._table("ai_state_facts").c.owner_key_digest == _hex(key)
            )
        )
        result = await self._session.execute(statement)
        if result.rowcount == 1:
            self._guarded_record_keys.discard(key)
            self._record_cache[key] = None
            for alias, record_key in tuple(self._alias_cache.items()):
                if record_key == key:
                    self._alias_cache[alias] = None
        return result.rowcount == 1

    async def list_records(self, query: RecordQuery) -> tuple[StoredRecord, ...]:
        from sqlalchemy import and_, or_, select

        table = self._table("ai_state_records")
        conditions = []
        if query.partition_digest is not None:
            conditions.append(table.c.partition_digest == _hex(query.partition_digest))
        if query.scope_digest is not None:
            conditions.append(table.c.scope_digest == _hex(query.scope_digest))
        if query.parent_digest is not None:
            conditions.append(table.c.parent_digest == _hex(query.parent_digest))
        if query.kind is not None:
            conditions.append(table.c.kind == query.kind)
        if query.states is not None:
            conditions.append(table.c.state.in_(tuple(query.states)))
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
        rows = (await self._session.execute(statement)).mappings().all()
        values = tuple(_record_from_row(row) for row in rows)
        self._record_cache.update({value.key_digest: value for value in values})
        return values

    async def scan_records(self) -> tuple[StoredRecord, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_records")
        rows = (await self._session.execute(select(table))).mappings().all()
        return tuple(_record_from_row(row) for row in rows)

    async def resolve_alias(self, alias: bytes) -> bytes | None:
        return (await self.resolve_aliases((alias,))).get(alias)

    async def resolve_aliases(self, aliases: Sequence[bytes]) -> Mapping[bytes, bytes]:
        unique_aliases = tuple(dict.fromkeys(aliases))
        missing = tuple(alias for alias in unique_aliases if alias not in self._alias_cache)
        if missing:
            from sqlalchemy import select

            table = self._table("ai_state_aliases")
            records = self._table("ai_state_records")
            rows = (
                (
                    await self._session.execute(
                        select(table.c.alias_digest, table.c.record_key_digest, records.c.id)
                        .select_from(
                            table.outerjoin(
                                records,
                                table.c.record_key_digest == records.c.key_digest,
                            )
                        )
                        .where(table.c.alias_digest.in_(tuple(_hex(alias) for alias in missing)))
                    )
                )
                .mappings()
                .all()
            )
            if any(row["id"] is None for row in rows):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values = {
                bytes.fromhex(str(row["alias_digest"])): _hex_or_none(row["record_key_digest"])
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

    async def insert_alias(self, alias: StoredAlias) -> None:
        await self.insert_aliases((alias,))

    async def insert_aliases(self, aliases: Sequence[StoredAlias]) -> None:
        if not aliases:
            return
        by_alias: dict[bytes, StoredAlias] = {}
        for alias in aliases:
            current = by_alias.get(alias.alias_digest)
            if current is not None and current.record_key_digest != alias.record_key_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            by_alias[alias.alias_digest] = alias
        values = tuple(sorted(by_alias.values(), key=lambda value: value.alias_digest))
        if any(alias.record_key_digest not in self._guarded_record_keys for alias in values):
            raise RuntimeError("alias owner must be guarded in the current transaction")
        existing = await self.resolve_aliases(tuple(alias.alias_digest for alias in values))
        rows: list[dict[str, object]] = []
        for alias in values:
            current = existing.get(alias.alias_digest)
            if current is not None:
                if current != alias.record_key_digest:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                continue
            rows.append(
                {
                    "alias_digest": _hex(alias.alias_digest),
                    "record_key_digest": _hex(alias.record_key_digest),
                }
            )
        if rows:
            from sqlalchemy import insert

            await self._session.execute(insert(self._table("ai_state_aliases")).values(rows))
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
                    await self._session.execute(
                        select(table).where(table.c.key_digest.in_(tuple(_hex(key) for key in missing)))
                    )
                )
                .mappings()
                .all()
            )
            self._sequence_cache.update(
                {bytes.fromhex(str(row["key_digest"])): int(row["value"]) for row in rows}
            )
            for key in missing:
                self._sequence_cache.setdefault(key, 0)
        return {key: self._sequence_cache[key] for key in unique_keys}

    async def next_sequence(self, key: bytes) -> int:
        return await self.reserve_sequence(key, 1)

    async def reserve_sequence(self, key: bytes, count: int) -> int:
        return (await self.reserve_sequences({key: count}))[key]

    async def reserve_sequences(self, requests: Mapping[bytes, int]) -> Mapping[bytes, int]:
        if any(count < 1 for count in requests.values()):
            raise ValueError("sequence reservation count must be positive")
        if not requests:
            return {}
        table = self._table("ai_state_sequences")
        rows = [
            {"key_digest": _hex(key), "value": requests[key]}
            for key in sorted(requests)
        ]
        values = await self._context.dialect.upsert_increment_many(
            self._session,
            table=table,
            rows=rows,
            column="value",
            index_elements=("key_digest",),
        )
        result = {bytes.fromhex(key): int(value) for key, value in values.items()}
        self._sequence_cache.update(result)
        return {key: result[key] for key in requests}

    async def advance_sequence(self, key: bytes, expected: int) -> int:
        from sqlalchemy import func, update

        table = self._table("ai_state_sequences")
        result = await self._session.execute(
            update(table)
            .where(table.c.key_digest == _hex(key), table.c.value == expected)
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

        await self._session.execute(
            delete(self._table("ai_state_sequences")).where(
                self._table("ai_state_sequences").c.key_digest.in_(tuple(_hex(key) for key in values))
            )
        )
        self._log_batch("delete_sequences", len(values), 1)
        for key in values:
            self._sequence_cache[key] = 0

    async def insert_fact(self, fact: StoredFact) -> None:
        from sqlalchemy import insert

        if fact.owner_key_digest not in self._guarded_record_keys:
            raise RuntimeError("fact owner must be guarded in the current transaction")
        await self._session.execute(insert(self._table("ai_state_facts")).values(_fact_values(fact)))

    async def insert_facts(self, facts: Sequence[StoredFact]) -> None:
        if not facts:
            return
        if any(fact.owner_key_digest not in self._guarded_record_keys for fact in facts):
            raise RuntimeError("fact owner must be guarded in the current transaction")
        from sqlalchemy import insert

        await self._session.execute(
            insert(self._table("ai_state_facts")).values([_fact_values(fact) for fact in facts])
        )
        self._log_batch("insert_facts", len(facts), 1)

    async def list_facts(self, query: FactQuery) -> tuple[StoredFact, ...]:
        from sqlalchemy import func, select

        table = self._table("ai_state_facts")
        conditions = [table.c.stream_digest == _hex(query.stream_digest)]
        if query.after_sequence is not None:
            conditions.append(table.c.sequence > query.after_sequence)
        if query.subject_digest is not None:
            conditions.append(table.c.subject_digest == _hex(query.subject_digest))
        if query.latest_per_subject:
            ranked = (
                select(
                    table,
                    func.row_number()
                    .over(
                        partition_by=table.c.subject_digest,
                        order_by=table.c.sequence.desc(),
                    )
                    .label("_subject_rank"),
                )
                .where(*conditions)
                .subquery()
            )
            statement = (
                select(ranked)
                .where(ranked.c._subject_rank == 1)
                .order_by(ranked.c.sequence)
            )
        else:
            if query.latest:
                statement = select(table).where(*conditions).order_by(table.c.sequence.desc())
            else:
                statement = select(table).where(*conditions).order_by(table.c.sequence)
        if query.limit is not None:
            statement = statement.limit(1 if query.latest else query.limit)
        elif query.latest:
            statement = statement.limit(1)
        rows = (await self._session.execute(statement)).mappings().all()
        return tuple(_fact_from_row(row) for row in rows)

    async def scan_facts(self) -> tuple[StoredFact, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_facts")
        rows = (await self._session.execute(select(table))).mappings().all()
        return tuple(_fact_from_row(row) for row in rows)

    async def delete_fact_streams(self, owner_key: bytes) -> None:
        from sqlalchemy import delete

        await self._session.execute(
            delete(self._table("ai_state_facts")).where(
                self._table("ai_state_facts").c.owner_key_digest == _hex(owner_key)
            )
        )

    async def insert_operation(self, value: StoredOperation) -> None:
        from sqlalchemy import insert

        await self._session.execute(insert(self._table("ai_state_operations")).values(_operation_values(value)))

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        from sqlalchemy import select

        table = self._table("ai_state_operations")
        row = (
            (await self._session.execute(select(table).where(table.c.key_digest == _hex(key)))).mappings().one_or_none()
        )
        return None if row is None else _operation_from_row(row)

    async def replace_operation(self, value: StoredOperation, *, expected_state: str) -> bool:
        current = await self.get_operation(value.key_digest)
        if current is None or current.state != expected_state:
            return False
        validate_operation_replacement(current, value)
        from sqlalchemy import update

        table = self._table("ai_state_operations")
        result = await self._session.execute(
            update(table)
            .where(table.c.key_digest == _hex(value.key_digest), table.c.state == expected_state)
            .values(_operation_values(value, updating=True))
        )
        return result.rowcount == 1

    async def list_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_operations")
        conditions = []
        if query.stream_digest is not None:
            conditions.append(table.c.stream_digest == _hex(query.stream_digest))
        if query.states is not None:
            conditions.append(table.c.state.in_(tuple(query.states)))
        if query.through_sequence is not None:
            conditions.append(table.c.sequence <= query.through_sequence)
        if query.compactable is not None:
            conditions.append(table.c.compactable == query.compactable)
        statement = select(table).where(*conditions).order_by(table.c.sequence, table.c.key_digest)
        if query.limit is not None:
            statement = statement.limit(query.limit)
        rows = (await self._session.execute(statement)).mappings().all()
        return tuple(_operation_from_row(row) for row in rows)

    async def scan_operations(self) -> tuple[StoredOperation, ...]:
        from sqlalchemy import select

        table = self._table("ai_state_operations")
        rows = (await self._session.execute(select(table))).mappings().all()
        return tuple(_operation_from_row(row) for row in rows)

    async def delete_operations(self, query: OperationQuery) -> tuple[StoredOperation, ...]:
        values = await self.list_operations(query)
        from sqlalchemy import delete

        if values:
            await self._session.execute(
                delete(self._table("ai_state_operations")).where(
                    self._table("ai_state_operations").c.key_digest.in_(
                        tuple(_hex(value.key_digest) for value in values)
                    )
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


def _record_values(record: StoredRecord, *, updating: bool = False) -> dict[str, object]:
    values: dict[str, object] = {
        "key_digest": _hex(record.key_digest),
        "partition_digest": _hex(record.partition_digest),
        "scope_digest": None if record.scope_digest is None else _hex(record.scope_digest),
        "parent_digest": None if record.parent_digest is None else _hex(record.parent_digest),
        "kind": record.kind,
        "sort_key": record.sort_key,
        "state": record.state,
        "storage_version": record.storage_version,
        "lease_owner": record.lease_owner,
        "lease_fence": record.lease_fence,
        "lease_expires_at": record.lease_expires_at,
        "payload_json": dict(record.data),
    }
    if updating:
        from sqlalchemy import func

        values["updated_at"] = func.current_timestamp()
    return values


def _record_replacement_values(replacement: RecordReplacement) -> dict[str, object]:
    record = replacement.record
    return {
        "_replacement_key_digest": _hex(record.key_digest),
        "_replacement_expected_version": replacement.expected_storage_version,
        "_replacement_partition_digest": _hex(record.partition_digest),
        "_replacement_scope_digest": None
        if record.scope_digest is None
        else _hex(record.scope_digest),
        "_replacement_parent_digest": None
        if record.parent_digest is None
        else _hex(record.parent_digest),
        "_replacement_kind": record.kind,
        "_replacement_sort_key": record.sort_key,
        "_replacement_state": record.state,
        "_replacement_storage_version": record.storage_version,
        "_replacement_lease_owner": record.lease_owner,
        "_replacement_lease_fence": record.lease_fence,
        "_replacement_lease_expires_at": record.lease_expires_at,
        "_replacement_payload_json": dict(record.data),
    }


def _record_from_row(row: Mapping[str, object]) -> StoredRecord:
    record = StoredRecord(
        bytes.fromhex(str(row["key_digest"])),
        bytes.fromhex(str(row["partition_digest"])),
        _hex_or_none(row["scope_digest"]),
        _hex_or_none(row["parent_digest"]),
        str(row["kind"]),
        str(row["sort_key"]),
        None if row["state"] is None else str(row["state"]),
        int(row["storage_version"]),
        None if row["lease_owner"] is None else str(row["lease_owner"]),
        int(row["lease_fence"]),
        _datetime_or_none(row["lease_expires_at"]),
        dict(row["payload_json"]),
    )
    try:
        validate_record_identity(record)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return record


def _fact_values(fact: StoredFact) -> dict[str, object]:
    return {
        "stream_digest": _hex(fact.stream_digest),
        "sequence": fact.sequence,
        "owner_key_digest": _hex(fact.owner_key_digest),
        "kind": fact.kind,
        "subject_digest": None if fact.subject_digest is None else _hex(fact.subject_digest),
        "state": fact.state,
        "payload_json": dict(fact.data),
    }


def _fact_from_row(row: Mapping[str, object]) -> StoredFact:
    return StoredFact(
        bytes.fromhex(str(row["stream_digest"])),
        int(row["sequence"]),
        bytes.fromhex(str(row["owner_key_digest"])),
        str(row["kind"]),
        _hex_or_none(row["subject_digest"]),
        None if row["state"] is None else str(row["state"]),
        dict(row["payload_json"]),
    )


def _operation_values(value: StoredOperation, *, updating: bool = False) -> dict[str, object]:
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
    return StoredOperation(
        bytes.fromhex(str(row["key_digest"])),
        bytes.fromhex(str(row["stream_digest"])),
        int(row["sequence"]),
        str(row["state"]),
        bool(row["compactable"]),
        dict(row["payload_json"]),
    )


def _hex(value: bytes) -> str:
    if not isinstance(value, bytes) or len(value) != 32:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.hex()


def _hex_or_none(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _datetime_or_none(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


__all__ = ["SqlStateStorageGroup", "SqlStateStore"]
