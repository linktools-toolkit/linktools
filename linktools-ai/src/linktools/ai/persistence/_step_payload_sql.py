#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL adapter between logical archive payloads and immutable Step bytes."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ._step_payload_owner import payload_address, payload_record_from_row, payload_record_values, payload_store_locator
from .state import RecordQuery, StoredRecord

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.sql import ColumnElement, Executable
    from sqlalchemy.engine import Result
    from sqlalchemy.ext.asyncio import AsyncSession
    from ..storage import SqlStorageContext


class StepPayloadSql:
    def __init__(self, table: "Table", execute: Callable[["Executable"], Awaitable["Result"]],
                 context: "SqlStorageContext", session: "AsyncSession") -> None:
        self.table = table
        self.execute = execute
        self.context = context
        self.session = session

    def condition(self, key: bytes) -> "ColumnElement[bool]":
        from sqlalchemy import and_
        address = payload_address(key)
        if address is None:
            raise ValueError("not a typed payload address")
        store, digest = address
        return and_(self.table.c.store_locator == payload_store_locator(store),
                    self.table.c.store_key == store, self.table.c.sha256 == digest)

    async def read(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        from sqlalchemy import or_, select
        result = {}
        for offset in range(0, len(keys), 300):
            batch = keys[offset:offset + 300]
            rows = (await self.execute(select(self.table).where(or_(*(self.condition(key) for key in batch))))).mappings()
            for row in rows:
                record = payload_record_from_row(row)
                if record.key not in batch:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                result[record.key] = record
        return result

    async def scan(self, *, after: bytes | None = None, limit: int | None = None,
                   query: RecordQuery | None = None) -> tuple[StoredRecord, ...]:
        from sqlalchemy import LargeBinary, and_, cast, func, literal, or_, select
        from sqlalchemy.dialects.mysql import BINARY
        from ._archive import archive_scope_parts
        from ._record_key import encode_key_components
        from ._step_payload_owner import payload_store_key
        table = self.table
        tag = encode_key_components(("step_payload",), 274)
        binary = BINARY() if self.context.dialect.name == "mysql" else LargeBinary()
        digest = cast(table.c.sha256, binary)

        def concatenate(*parts):
            if self.context.dialect.name == "mysql":
                return func.concat(*parts)
            value = parts[0]
            for part in parts[1:]:
                value = value.op("||")(part)
            return cast(value, LargeBinary())

        logical_key = concatenate(literal(encode_key_components(("record",), 768)),
            func.substr(table.c.store_key, len(tag) + 1),
            literal(encode_key_components(("archive", "archive_content"), 768) + b"\x00Rexecution-payload:"), digest)
        sort_key = concatenate(literal(b"execution-payload:"), digest)
        conditions = [func.substr(table.c.store_key, 1, len(tag)) == tag]
        if after is not None:
            conditions.append(logical_key > after)
        if query is not None:
            if query.kind not in (None, "archive_content") or query.parent_key is not None or query.states is not None:
                return ()
            if query.scope_key is not None:
                try:
                    store = payload_store_key(*archive_scope_parts(query.scope_key))
                except AIError:
                    return ()
                conditions += [table.c.store_locator == payload_store_locator(store), table.c.store_key == store]
            if query.sort_key_prefix is not None:
                conditions.append(func.substr(sort_key, 1, len(query.sort_key_prefix)) == query.sort_key_prefix)
            if query.after_sort_key is not None:
                conditions.append(sort_key > query.after_sort_key if query.after_key is None else or_(
                    sort_key > query.after_sort_key, and_(sort_key == query.after_sort_key, logical_key > query.after_key)))
        statement = select(table).where(*conditions).order_by(
            *((table.c.sha256, table.c.store_key) if query is not None else (table.c.store_key, table.c.sha256)))
        if limit is not None:
            statement = statement.limit(limit)
        return tuple(payload_record_from_row(row) for row in (await self.execute(statement)).mappings())

    async def insert(self, records: Sequence[StoredRecord], *, immutable: bool) -> set[bytes]:
        from sqlalchemy import insert
        inserted: set[bytes] = set()
        for offset in range(0, len(records), 64):
            values = [payload_record_values(record) for record in records[offset:offset + 64]]
            if immutable:
                count = await self.context.run_statement(self.context.dialect.insert_ignore_conflict_many(
                    self.session, table=self.table, rows=values,
                    index_elements=("store_locator", "sha256"),
                ))
                if count == len(values):
                    inserted.update(record.key for record in records[offset:offset + 64])
            else:
                await self.execute(insert(self.table).values(values))
                inserted.update(record.key for record in records[offset:offset + 64])
        return inserted

    async def guard(self, key: bytes, revision: int) -> bool:
        from sqlalchemy import update
        result = await self.execute(update(self.table).where(self.condition(key),
            self.table.c.gc_revision == revision).values(gc_revision=revision + 1))
        return result.rowcount == 1

    async def replace(self, record: StoredRecord, revision: int) -> bool:
        from sqlalchemy import update
        values = payload_record_values(record)
        result = await self.execute(update(self.table).where(self.condition(record.key),
            self.table.c.gc_revision == revision).values(**values))
        return result.rowcount == 1

    async def delete(self, key: bytes, revision: int) -> bool:
        from sqlalchemy import delete
        result = await self.execute(delete(self.table).where(self.condition(key),
            self.table.c.gc_revision == revision))
        return result.rowcount == 1


__all__ = ["StepPayloadSql"]
