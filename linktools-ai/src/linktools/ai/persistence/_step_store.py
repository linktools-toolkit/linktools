#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy adaptation of the public upstream StepStore protocol."""

import base64
import hashlib
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields, replace
from itertools import groupby
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary, WeakSet, ref

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
    ToolEffectRecord,
)

from linktools.core import environ

from ..core import validate_resource_id
from ..errors import AIError, ErrorCode
from ..storage import SqlStorageContext, create_sql_storage_context
from ._archive import ContentArchive, ContentRecord, PreparedArchive
from ..model.attachments import read_attachment_payload
from ._archive_closure import NativeMessageClosure, validate_native_message_content
from ._message_root import NativeMessageRoot, NativeMessageSequence, NativeMessageSlice
from ._step_message_content import (
    CommittedNativeMessages, NativeMessageContent, NativeMessageReceipt, PreparedMessagePayload,
    PreparedNativeMessages,
)
from ._transcript_staging import MessageArchive, MessageSnapshot
from ._payload import encode_payload_bytes, decode_payload_bytes
from ..model.messages import combine_model_message_payloads, decode_model_message_payloads, encode_model_messages
from ._native_checkpoint import NativeCheckpoint
from ._step_backup import NativeStepBackup
from ._step_maintenance import StepContentCleanupResult, purge_step_content
from ._step_schema import build_step_sql_metadata
from .state import SqlStateTransaction, SqlMutationObserver, StateStore, StateTransaction, SqlStateRowReader
from ..steps import (
    NativeLifecycleEvent, NativeStepSource, validate_native_step_ids, native_store_key,
    validate_native_store_key, validate_native_replay_key, encode_native_replay_key,
    decode_native_replay_key, encode_native_store_key, native_store_locator,
)

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.engine import RowMapping
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
    from sqlalchemy.sql import Select
    from sqlalchemy.sql.elements import ColumnElement

    from ..storage import SqlValue
    from ._snapshot import SnapshotExclusiveGuard


_logger = environ.get_logger("ai.persistence.steps")
_RUN_ADAPTER = TypeAdapter(RunRecord)
_EVENT_ADAPTER = TypeAdapter(StepEvent)
_LIFECYCLE_ADAPTER = TypeAdapter(NativeLifecycleEvent)
_SNAPSHOT_ADAPTER = TypeAdapter(ContinuableSnapshot)
_BATCH_SIZE = 128
_RUN_CACHE_SIZE = 128
_RUN_CACHE_BYTES = 1024 * 1024
_EVENT_RUN_FIELDS = ("conversation_id", "parent_run_id", "agent_name", "metadata")
_EVENT_SOURCE_COLUMNS = tuple(field.name for field in fields(StepEvent)) + (
    "outcome", "store_key", "run_fields", "body_sha256", "fields_sha256",
)


@dataclass(frozen=True, slots=True)
class _RunFields:
    """Detached native fields from a successful insert or an existing run read."""

    engine: "AsyncEngine"
    store_key: bytes = field(repr=False)
    run_id: str
    registration_id: str | None
    started_at: str
    values: "tuple[SqlValue, ...]"
    byte_size: int
    started_body: str | None = None
    terminal_bodies: tuple[str, ...] = ()

    @classmethod
    def from_row(cls, row: "Mapping[str, SqlValue]", *, engine: "AsyncEngine") -> "_RunFields":
        values = tuple(row[name] for name in _EVENT_RUN_FIELDS)
        byte_size = sum(len(value.encode("utf-8")) for value in (
            row["store_key"], row["run_id"], row["registration_id"], row["started_at"], *values,
        ) if isinstance(value, str)) + len(row["store_key"])
        return cls(engine, row["store_key"], row["run_id"], row["registration_id"], row["started_at"], values, byte_size)


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """Native boundary records and encoded messages prepared outside SQL."""

    run: RunRecord | None
    events: tuple[StepEvent, ...]
    snapshot: ContinuableSnapshot | None
    snapshot_values: "dict[str, SqlValue] | None"
    messages: PreparedNativeMessages | None
    event_keys: tuple[bytes, ...]
    managed_start: bool = False


@dataclass(frozen=True, slots=True)
class StepWriteReceipt:
    """Provisional body identities and run fields, confirmed by the outer transaction."""

    message_digests: tuple[str, ...]
    sequence_digests: tuple[str, ...]
    run_fields: _RunFields | None = None
    native_messages: NativeMessageReceipt | None = None
    lifecycle_keys: tuple[str, ...] = ()


class SqlStepStore:
    """Persist upstream records in a fixed namespace and tenant partition.

    The engine is borrowed. Provision production tables through the migration
    boundary; ``initialize(local=True)`` may create a local SQLite schema.
    Tool-effect methods delegate to the caller's existing durable effect owner.
    Stored bodies and sequence nodes remain after snapshot pruning because
    independent concurrent saves may reuse their immutable content.
    Confirmed message digests are cached only after commit. Direct deletion or
    replacement of message bodies requires quiescing and closing all adapters
    before maintenance. ``max_cached_messages=0`` disables this optimization.
    Exact run fields use a separate confirmed cache bounded by entry count and
    UTF-8 bytes. Unknown or evicted runs keep events inline without a pre-read.
    Direct run replacement also requires quiescing and closing every adapter.
    """

    _instances: WeakSet["SqlStepStore"] = WeakSet()

    def __init__(
        self,
        engine: "AsyncEngine | SqlStorageContext",
        *,
        namespace: str,
        tenant_id: str,
        tool_effect_store: StepStore | None = None,
        max_snapshots_per_run: int | None = None,
        max_cached_messages: int = 4096,
    ) -> None:
        store_key = self.scope_key(namespace=namespace, tenant_id=tenant_id)
        if max_snapshots_per_run is not None and (
            not isinstance(max_snapshots_per_run, int) or max_snapshots_per_run < 1
        ):
            raise ValueError("max_snapshots_per_run must be an int >= 1 or None")
        if not isinstance(max_cached_messages, int) or max_cached_messages < 0:
            raise ValueError("max_cached_messages must be an int >= 0")
        self._context = engine if isinstance(engine, SqlStorageContext) else create_sql_storage_context(engine)
        self._owns_context = not isinstance(engine, SqlStorageContext)
        self._metadata = build_step_sql_metadata()
        self._store_key = store_key
        self._store_locator = native_store_locator(store_key)
        self._tool_effect_store = tool_effect_store
        self._max_snapshots_per_run = max_snapshots_per_run
        self._initialized = False
        self._max_cached_messages = max_cached_messages
        self._confirmed_messages: OrderedDict[str, None] = OrderedDict()
        self._confirmed_sequences: OrderedDict[str, None] = OrderedDict()
        self._message_content = NativeMessageContent(self._context, store_key=store_key,
            max_cached_messages=max_cached_messages, max_cached_nodes=max_cached_messages)
        self._message_archive: ContentArchive | None = None
        self._message_proofs: OrderedDict[NativeMessageSequence, CommittedNativeMessages] = OrderedDict()
        self._message_integrity: OrderedDict[NativeMessageSequence, tuple[str, int]] = OrderedDict()
        self._message_snapshots: dict[NativeMessageSequence, MessageSnapshot] = {}
        self._prepared_messages: WeakKeyDictionary[PreparedNativeMessages, MessageSnapshot] = WeakKeyDictionary()
        self._message_inputs: OrderedDict[str, tuple[str, int, int, NativeMessageSequence]] = OrderedDict()
        self._frozen_messages = MessageArchive()
        self._confirmed_runs: OrderedDict[str, _RunFields] = OrderedDict()
        self._instances.add(self)
        self._receipt_sink: ContextVar[list[StepWriteReceipt] | None] = ContextVar("step_write_receipts", default=None)

    @classmethod
    def scope_key(cls, *, namespace: str, tenant_id: str) -> bytes:
        """Derive the native partition identity from its logical scope."""
        return native_store_key(namespace, tenant_id)

    async def initialize(self, *, local: bool = False) -> None:
        """Configure the borrowed backend, optionally creating local SQLite tables."""
        if self._initialized and not self._context.closed:
            return
        if local:
            if self._context.engine.dialect.name != "sqlite":
                raise ValueError("only a local SQLite store may initialize its schema")
            await self._context.initialize()

            async def create(session: "AsyncSession") -> None:
                connection = await session.connection()
                await connection.run_sync(self._metadata.create_all)

            await self._context.run_mutation(create, domain="steps.schema")
        await self._context.initialize()
        self._initialized = True

    async def close(self) -> None:
        """Release this adapter's context without closing a borrowed engine."""
        self._confirmed_messages.clear()
        self._confirmed_sequences.clear()
        self._confirmed_runs.clear()
        self.invalidate_message_content()
        if self._owns_context:
            await self._context.close()

    async def purge_unreferenced_content(
        self, *, exclusive: "SnapshotExclusiveGuard",
    ) -> StepContentCleanupResult:
        """Reclaim retired content after quiescing every user of these tables.

        Close all other adapters first so no confirmed-content cache survives
        this maintenance window. The guard must also drain this adapter's work.
        Snapshot roots, run records and replay tombstones are never retired here.
        Normal reads and writes do not invoke this complete offline scan.
        """
        async with exclusive.offline_exclusivity():
            await self.initialize()
            self._confirmed_messages.clear()
            self._confirmed_sequences.clear()
            try:
                result = await purge_step_content(self._context, self._metadata, self._store_key,
                    group=None if self._message_archive is None else self._message_archive.store.storage_group,
                    read_object=None if self._message_archive is None else self._message_archive.read_payload_object)
            finally:
                # Cancellation or a lost commit acknowledgement must not leave
                # the caller believing an old body still exists after reopening.
                self._confirmed_messages.clear()
                self.invalidate_context(self._context)
            _logger.info("native content reclaimed: sequences=%s messages=%s",
                         result.deleted_sequences, result.deleted_messages)
            return result

    async def register_run(self, record: RunRecord) -> None:
        await self.initialize()

        async def write(session: "AsyncSession") -> _RunFields:
            return await self._write_run(session, record)

        source = await self._context.run_mutation(write, domain="steps.run")
        receipt = StepWriteReceipt((), (), source)
        self.confirm_prepared_step(receipt)
        self._publish_receipt(receipt)

    async def register_run_receipt(self, record: RunRecord) -> StepWriteReceipt:
        """Observe the public registration operation, including custom overrides."""
        with self._capture_receipts() as receipts:
            await self.register_run(record)
        receipt = self._matching_receipt(receipts, run_id=record.run_id)
        self._publish_receipt(receipt)
        return receipt

    @contextmanager
    def _capture_receipts(self) -> Iterator[list[StepWriteReceipt]]:
        receipts: list[StepWriteReceipt] = []
        token = self._receipt_sink.set(receipts)
        try:
            yield receipts
        finally:
            self._receipt_sink.reset(token)

    def _publish_receipt(self, receipt: StepWriteReceipt) -> None:
        sink = self._receipt_sink.get()
        if sink is not None:
            sink.append(receipt)

    def _matching_receipt(self, receipts: Sequence[StepWriteReceipt], *, run_id: str | None) -> StepWriteReceipt:
        if run_id is None:
            return replace(receipts[-1], run_fields=None) if receipts else StepWriteReceipt((), ())
        for receipt in reversed(receipts):
            if receipt.run_fields is not None and receipt.run_fields.run_id == run_id:
                return receipt
        raise RuntimeError("native store override did not confirm the run registration")

    async def _write_run(
        self, session: "AsyncSession", record: RunRecord, *, replay: bool = False,
        started: StepEvent | None = None,
    ) -> _RunFields:
        from sqlalchemy import update

        start_key = None if started is None else encode_native_replay_key(started.idempotency_key)
        values = self._run_values(record.run_id) | self._record_values(record)
        values["managed_start_key"] = start_key
        result = await self._context.dialect.insert_ignore_conflict(
            session, table=self._table("runs"), values=values,
            index_elements=("store_locator", "run_id"),
        )
        if not result.inserted:
            if replay and record.registration_id is not None:
                source_names = terminal_names = _EVENT_SOURCE_COLUMNS
                query = self._run_replay_statement(record, started=started)
                row = (await session.execute(query)).mappings().first()
                if row is not None and row["registration_id"] == record.registration_id:
                    source = _RunFields.from_row(row, engine=self._context.engine)
                    if started is not None:
                        if row["managed_start_key"] is None:
                            # Only an explicitly metadata-only header admits its
                            # first start. Missing committed sources fail closed.
                            actual = self.run_from_receipt(StepWriteReceipt((), (), source))
                            if (actual != record or row["start_idempotency_key"] is not None
                                    or row["terminal_idempotency_key"] is not None
                                    or row["has_other_start"]):
                                raise ValueError("first run_started differs from its admitted registration")
                            table = self._table("runs")
                            changed = await session.execute(update(table).where(
                                *self._run_predicates(table, record.run_id),
                                table.c.managed_start_key.is_(None),
                            ).values(managed_start_key=start_key))
                            if changed.rowcount == 1:
                                return source
                            # RR cannot see the winner in this snapshot. The
                            # existing SQL boundary retries only after rollback.
                            from ..storage import SqlSnapshotConflict
                            raise SqlSnapshotConflict()
                        if row["managed_start_key"] != start_key or row["start_idempotency_key"] is None:
                            raise ValueError("managed start source is missing or has another identity")
                        event = self.decode_started_source(
                            {name: row["start_" + name] for name in source_names}, run=row,
                        )
                        source = self._with_started(source, event)
                        self._validate_started_replay(record, started, source)
                        if row["terminal_idempotency_key"] is not None:
                            terminal = self.decode_started_source(
                                {name: row["terminal_" + name] for name in terminal_names}, run=row,
                            )
                            if not isinstance(terminal, NativeLifecycleEvent):
                                raise ValueError("native terminal source outcome is corrupt")
                            source = self._with_terminal(source, terminal)
                    return source
            raise ValueError(
                f"run_id {record.run_id!r} is already in the store; run_id is single-shot"
            )
        return _RunFields.from_row(values, engine=self._context.engine)

    def _run_replay_statement(self, record: RunRecord, *, started: StepEvent | None) -> "Select":
        from sqlalchemy import and_, select

        table, events = self._table("runs"), self._table("events")
        query = select(table.c.store_key, table.c.run_id, table.c.registration_id, table.c.started_at,
                       table.c.managed_start_key,
                       *(table.c[name] for name in _EVENT_RUN_FIELDS))
        if started is not None:
            other_start = events.alias("other_start")
            query = query.add_columns(
                *(events.c[name].label("start_" + name) for name in _EVENT_SOURCE_COLUMNS),
                select(other_start.c.id).where(
                    other_start.c.store_locator == table.c.store_locator,
                    other_start.c.store_key == table.c.store_key,
                    other_start.c.run_id == table.c.run_id,
                    other_start.c.kind == "run_started",
                ).exists().label("has_other_start"),
            )
            joined = table.outerjoin(events, and_(
                events.c.idempotency_key == encode_native_replay_key(started.idempotency_key),
                events.c.store_locator == table.c.store_locator, events.c.store_key == table.c.store_key,
                events.c.run_id == table.c.run_id,
            ))
            terminal = events.alias("terminal")
            joined = joined.outerjoin(terminal, and_(terminal.c.idempotency_key == encode_native_replay_key(NativeLifecycleEvent.source_key()),
                terminal.c.store_locator == table.c.store_locator, terminal.c.store_key == table.c.store_key,
                terminal.c.run_id == table.c.run_id))
            query = query.add_columns(*(terminal.c[name].label("terminal_" + name) for name in _EVENT_SOURCE_COLUMNS))
            query = query.select_from(joined)
        return query.where(*self._run_predicates(table, record.run_id))

    async def _import_run(self, session: "AsyncSession", record: RunRecord) -> _RunFields:
        from sqlalchemy import select

        values = self._run_values(record.run_id) | self._record_values(record)
        table = self._table("runs")
        result = await self._context.dialect.insert_ignore_conflict(
            session, table=table, values=values, index_elements=("store_locator", "run_id"),
        )
        if result.inserted:
            return _RunFields.from_row(values, engine=self._context.engine)
        row = (await session.execute(select(table).where(*self._run_predicates(table, record.run_id)))).mappings().one()
        if _RUN_ADAPTER.validate_python(self._record_data(row)) != record:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return _RunFields.from_row(row, engine=self._context.engine)

    async def import_run(self, record: RunRecord) -> None:
        """Import metadata independently of native lifecycle-event publication."""
        await self.initialize()
        source = await self._context.run_mutation(
            lambda session: self._import_run(session, record), domain="steps.run_metadata",
        )
        self._confirm_run(source)

    async def import_run_in_transaction(self, transaction: StateTransaction, record: RunRecord) -> None:
        """Admit metadata in the consumer's actual publication, not a new commit."""
        if not isinstance(transaction, SqlStateTransaction):
            raise TypeError("native SQL metadata requires a SQL state transaction")
        await transaction.run_sql(self._context, lambda session: self._import_run(session, record))
        # Provisional metadata is not a committed native receipt or cache entry.

    async def get_run_in_transaction(self, transaction: StateTransaction, *, run_id: str) -> RunRecord | None:
        """Read canonical metadata without opening a second connection or confirming it."""
        from sqlalchemy import select

        if not isinstance(transaction, SqlStateTransaction):
            raise TypeError("native SQL metadata requires a SQL state transaction")
        validate_resource_id(run_id)
        table = self._table("runs")

        async def read(session: "AsyncSession") -> RunRecord | None:
            row = (await session.execute(select(table).where(*self._run_predicates(table, run_id)))).mappings().first()
            return None if row is None else _RUN_ADAPTER.validate_python(self._record_data(row))

        return await transaction.run_sql(self._context, read)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        from sqlalchemy import select

        await self.initialize()
        table = self._table("runs")
        async with self._context.sessions() as session:
            row = (await self._context.run_statement(
                session.execute(select(table).where(*self._run_predicates(table, run_id)))
            )).mappings().first()
        if row is None:
            return None
        record = _RUN_ADAPTER.validate_python(self._record_data(row))
        self._confirm_run(_RunFields.from_row(row, engine=self._context.engine))
        return record

    async def get_runs(self, run_ids: Sequence[str]) -> Mapping[str, RunRecord]:
        """Resolve a bounded set of run registrations without scanning other runs."""
        from sqlalchemy import select

        identifiers = tuple(dict.fromkeys(run_ids))
        for identity in identifiers:
            validate_resource_id(identity)
        if not identifiers:
            return {}
        await self.initialize()
        table = self._table("runs")
        result = {}
        async with self._context.sessions() as session:
            for offset in range(0, len(identifiers), 128):
                rows = (await self._context.run_statement(session.execute(select(table).where(
                    table.c.store_locator == self._store_locator, table.c.store_key == self._store_key,
                    table.c.run_id.in_(identifiers[offset:offset + 128]),
                )))).mappings().all()
                for row in rows:
                    result[row["run_id"]] = _RUN_ADAPTER.validate_python(self._record_data(row))
                    self._confirm_run(_RunFields.from_row(row, engine=self._context.engine))
        return result

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None,
    ) -> list[RunRecord]:
        from sqlalchemy import select

        await self.initialize()
        table = self._table("runs")
        query = select(table).where(table.c.store_locator == self._store_locator, table.c.store_key == self._store_key)
        if parent_run_id is not None:
            validate_resource_id(parent_run_id)
            query = query.where(table.c.parent_run_id == parent_run_id)
        if conversation_id is not None:
            validate_resource_id(conversation_id)
            query = query.where(table.c.conversation_id == conversation_id)
        async with self._context.sessions() as session:
            rows = (await self._context.run_statement(session.execute(query.order_by(table.c.id)))).mappings().all()
        records = [_RUN_ADAPTER.validate_python(self._record_data(row)) for row in rows]
        for row in rows:
            self._confirm_run(_RunFields.from_row(row, engine=self._context.engine))
        return sorted(records, key=lambda r: r.started_at)

    async def append_event(self, event: StepEvent) -> None:
        await self.initialize()
        replay_keys = {id(event): encode_native_replay_key(event.idempotency_key)}

        async def write(session: "AsyncSession") -> _RunFields | None:
            if isinstance(event, NativeLifecycleEvent):
                return await self._write_lifecycle_event(session, event, source=self._confirmed_runs.get(event.run_id))
            await self._write_events(session, (event,), replay_keys=replay_keys)
            return None

        source = await self._context.run_mutation(write, domain="steps.event")
        if source is not None:
            self.confirm_prepared_step(StepWriteReceipt((), (), source))

    async def _write_events(
        self, session: "AsyncSession", events: Sequence[StepEvent], *, source: _RunFields | None = None,
        started: StepEvent | None = None, replay_keys: Mapping[int, bytes] | None = None,
    ) -> None:
        for (keyed, managed), group in groupby(
            events, key=lambda event: (event.idempotency_key is not None, event is started),
        ):
            rows = []
            for event in group:
                row = self._run_values(event.run_id) | self._event_values(event)
                known = source if source is not None and source.run_id == event.run_id else self._confirmed_runs.get(event.run_id)
                if event is started:
                    row["body_sha256"] = _digest(self._started_body(event))
                    row["fields_sha256"] = _digest(self._started_fields_body(event))
                row["run_fields"] = 0
                if known is not None:
                    for bit, (name, value) in enumerate(zip(_EVENT_RUN_FIELDS, known.values)):
                        # NULL stays an explicit event value, never an inheritance marker.
                        candidate = json.dumps(row[name]) if name == "metadata" else row[name]
                        if candidate is not None and candidate == value:
                            row[name] = None
                            row["run_fields"] |= 1 << bit
                row["idempotency_key"] = (encode_native_replay_key(event.idempotency_key) if replay_keys is None
                                          else replay_keys[id(event)])
                rows.append(row)
            # Private occurrence keys retain insertion occurrences; managed sources
            # must fail on conflict until their exact body is proven.
            batch_size = _BATCH_SIZE if keyed and not managed else 1
            for offset in range(0, len(rows), batch_size):
                if not keyed or managed:
                    await session.execute(self._table("events").insert().values(rows[offset]))
                    continue
                batch = rows[offset:offset + batch_size]
                inserted = await self._context.dialect.insert_ignore_conflict_many(
                    session, table=self._table("events"), rows=batch,
                    index_elements=("store_locator", "run_id", "idempotency_key"),
                )
                if inserted != len(batch):
                    from sqlalchemy import or_, select
                    table = self._table("events")
                    identities = or_(*((table.c.run_id == row["run_id"])
                                      & (table.c.idempotency_key == row["idempotency_key"]) for row in batch))
                    found = (await session.execute(select(table.c.run_id, table.c.idempotency_key, table.c.store_key).where(
                        table.c.store_locator == self._store_locator, identities,
                    ))).all()
                    expected = {(row["run_id"], row["idempotency_key"]) for row in batch}
                    if ({(row.run_id, row.idempotency_key) for row in found} != expected
                            or any(row.store_key != self._store_key for row in found)):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _write_lifecycle_event(
        self, session: "AsyncSession", event: NativeLifecycleEvent, *, source: _RunFields | None,
    ) -> _RunFields:
        if source is None or source.run_id != event.run_id or source.started_body is None:
            raise ValueError("managed terminal requires its confirmed registration and start source")
        body = self._started_body(event)
        previous = next((value for value in source.terminal_bodies
                         if self.decode_event(json.loads(value)).idempotency_key == event.idempotency_key), None)
        if previous is not None:
            if previous != body:
                raise ValueError("native lifecycle replay body differs from its source")
            return source
        # The registration receipt proves this attempt. A conflicting terminal
        # without body proof fails closed rather than adding a speculative read.
        await self._write_events(session, (event,), source=source, started=event)
        return self._with_terminal(source, event)

    async def list_events(
        self, *, run_id: str, started_receipt: StepWriteReceipt | None = None,
    ) -> list[StepEvent]:
        """Read events, optionally verifying a provisional start in the same query.

        UNKNOWN readback must establish the physical locator and stored integrity
        proof, not merely equality of the reconstructed event. Reading does not
        confirm the supplied receipt or advance the committed source cache.
        """
        from sqlalchemy import and_, case, cast, or_, select, Text

        expected = () if started_receipt is None else self.events_from_receipt(started_receipt)
        if any(event.run_id != run_id for event in expected):
            raise ValueError("lifecycle receipt belongs to another run")
        await self.initialize()
        table, runs = self._table("events"), self._table("runs")
        columns = {field.name: table.c[field.name] for field in fields(StepEvent)}
        columns["metadata"] = cast(table.c.metadata, Text)
        columns["outcome"] = table.c.outcome
        invalid = [table.c.run_fields.bitwise_and((1 << len(_EVENT_RUN_FIELDS)) - 1) != table.c.run_fields,
                   and_(table.c.run_fields != 0, runs.c.id.is_(None))]
        for bit, name in enumerate(_EVENT_RUN_FIELDS):
            referenced = table.c.run_fields.bitwise_and(1 << bit) != 0
            columns[name] = case((referenced, runs.c[name]), else_=columns[name]).label(name)
            invalid.append(and_(referenced, or_(table.c[name].is_not(None), runs.c[name].is_(None))))
        source_columns = () if not expected else (
            table.c.store_key,
        )
        registration_columns = ()
        if expected:
            proof_key = table.c.idempotency_key == encode_native_replay_key(expected[0].idempotency_key)
            registration_columns = tuple(case((proof_key, runs.c[field.name]), else_=None).label("proof_run_" + field.name)
                                         for field in fields(RunRecord))
        query = select(
            *columns.values(), table.c.body_sha256, table.c.fields_sha256, or_(*invalid).label("invalid_run_reference"),
            *source_columns, *registration_columns,
        ).select_from(table.outerjoin(runs, and_(
            *(() if expected else (table.c.run_fields != 0,)),
            runs.c.store_locator == table.c.store_locator, runs.c.store_key == table.c.store_key,
            runs.c.run_id == table.c.run_id,
        ))).where(*self._run_predicates(table, run_id)).order_by(table.c.id)
        async with self._context.sessions() as session:
            rows = (await self._context.run_statement(session.execute(query))).mappings().all()
        if any(row["invalid_run_reference"] for row in rows):
            raise ValueError("event run reference is missing or corrupt")
        result = []
        matched = 0
        for row in rows:
            event = self.decode_event(self._record_data(row))
            candidate = next((item for item in expected if item.idempotency_key == event.idempotency_key), None)
            if candidate is not None:
                verified = self.decode_started_source(dict(row) | {"run_fields": 0})
                if self._started_body(verified) != self._started_body(candidate):
                    raise ValueError("native lifecycle readback differs from the receipt")
                if candidate is expected[0]:
                    registration = self._record_data({field.name: row["proof_run_" + field.name] for field in fields(RunRecord)})
                    if _RUN_ADAPTER.validate_python(registration) != self.run_from_receipt(started_receipt):
                        raise ValueError("native lifecycle registration differs from the receipt")
                matched += 1
            if isinstance(event, NativeLifecycleEvent) or row["body_sha256"] is not None or row["fields_sha256"] is not None:
                if (_digest(self._started_body(event)) != row["body_sha256"]
                        or _digest(self._started_fields_body(event)) != row["fields_sha256"]):
                    raise ValueError("native event body is corrupt")
            result.append(event)
        if matched != len(expected):
            raise ValueError("native lifecycle readback is missing or ambiguous")
        return result

    def confirmed_run(self, run_id: str) -> RunRecord | None:
        """Return detached committed run proof when retained, without reading SQL."""
        validate_resource_id(run_id)
        source = self._confirmed_runs.get(run_id)
        if source is None or self._context.closed:
            return None
        return self.run_from_receipt(StepWriteReceipt((), (), source))

    def run_from_receipt(self, receipt: StepWriteReceipt) -> RunRecord | None:
        """Detach a source from this store's receipt, independently of cache limits.

        A prepared-write receipt is provisional until its enclosing transaction
        commits. Callers must resolve that outcome before confirming projections.
        """
        source = receipt.run_fields
        if source is None:
            return None
        if source.engine is not self._context.engine or source.store_key != self._store_key:
            raise ValueError("run receipt belongs to another native store")
        data = dict(zip(_EVENT_RUN_FIELDS, source.values))
        data.update(run_id=source.run_id, registration_id=source.registration_id, started_at=source.started_at)
        return _RUN_ADAPTER.validate_python(self._record_data(data))

    @classmethod
    def _started_body(cls, event: StepEvent) -> str:
        if (event.kind != "run_started" and not isinstance(event, NativeLifecycleEvent)) or event.idempotency_key is None:
            raise ValueError("managed source requires a keyed lifecycle event")
        values = cls._event_values(event)
        if isinstance(event, NativeLifecycleEvent):
            cls.decode_event(values)
        return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _started_fields_body(cls, event: StepEvent) -> str:
        if (event.kind != "run_started" and not isinstance(event, NativeLifecycleEvent)) or event.idempotency_key is None:
            raise ValueError("managed source requires a keyed lifecycle event")
        validate_native_replay_key(event.idempotency_key)
        if event.tool_call_id is not None:
            validate_resource_id(event.tool_call_id)
        values = {field.name: getattr(event, field.name) for field in fields(event) if field.name != "metadata"}
        values["timestamp"] = event.timestamp.isoformat()
        return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _with_started(cls, source: _RunFields, event: StepEvent) -> _RunFields:
        body = cls._started_body(event)
        previous = 0 if source.started_body is None else len(source.started_body.encode("utf-8"))
        return replace(source, started_body=body, byte_size=source.byte_size - previous + len(body.encode("utf-8")))

    @classmethod
    def _with_terminal(cls, source: _RunFields, event: NativeLifecycleEvent) -> _RunFields:
        body = cls._started_body(event)
        if event.idempotency_key != NativeLifecycleEvent.source_key():
            raise ValueError("managed terminal requires its fixed lifecycle source key")
        if source.terminal_bodies:
            if source.terminal_bodies != (body,):
                raise ValueError("native lifecycle replay body differs from its source")
            return source
        return replace(source, terminal_bodies=(body,), byte_size=source.byte_size + len(body.encode("utf-8")))

    def _validate_started_replay(
        self, run: RunRecord, started: StepEvent, source: _RunFields,
    ) -> None:
        """Reuse authoritative linked metadata without weakening source identity."""
        receipt = StepWriteReceipt((), (), source)
        actual_run = self.run_from_receipt(receipt)
        actual_start = self.started_from_receipt(receipt)
        if actual_run is None or replace(actual_run, metadata={}) != replace(run, metadata={}):
            raise ValueError("run replay body differs from its registration")
        if (actual_start is None or self._started_body(replace(actual_start, metadata={}))
                != self._started_body(replace(started, metadata={}))):
            raise ValueError("run_started replay body differs from its source")
        if actual_run.metadata != run.metadata or actual_start.metadata != started.metadata:
            if started.metadata != run.metadata or actual_start.metadata != actual_run.metadata:
                raise ValueError("run_started replay body differs from its linked metadata source")

    def started_from_receipt(self, receipt: StepWriteReceipt) -> StepEvent | None:
        """Detach the immutable start source; outer-transaction receipts are provisional."""
        self.run_from_receipt(receipt)
        source = receipt.run_fields
        if source is None or source.started_body is None:
            return None
        return self.decode_event(json.loads(source.started_body))

    @classmethod
    def decode_event(cls, data: Mapping[str, object]) -> StepEvent:
        """Restore explicit lifecycle fields without changing ordinary upstream inputs."""
        cls._validate_record_ids(data)
        return (_LIFECYCLE_ADAPTER if data.get("outcome") is not None else _EVENT_ADAPTER).validate_python(data)

    def events_from_receipt(self, receipt: StepWriteReceipt) -> tuple[StepEvent, ...]:
        """Detach all managed lifecycle sources proven by one provisional write."""
        started = self.started_from_receipt(receipt)
        result = [] if started is None else [started]
        source = receipt.run_fields
        if source is not None:
            for body in source.terminal_bodies:
                event = self.decode_event(json.loads(body))
                if self._started_body(event) != body:
                    raise ValueError("event receipt has a noncanonical source body")
                result.append(event)
        return tuple(result)

    def confirmed_started(self, run_id: str) -> StepEvent | None:
        """Return a retained committed start source without a SQL read."""
        validate_resource_id(run_id)
        source = self._confirmed_runs.get(run_id)
        if source is None or self._context.closed:
            return None
        return self.started_from_receipt(StepWriteReceipt((), (), source))

    def decode_started_source(
        self, event: Mapping[str, object], *, run: Mapping[str, object] | None = None,
    ) -> StepEvent:
        """Verify a joined native source row and resolve its inherited run fields.

        The caller supplies native SQL columns unchanged, including locator and
        body_sha256. This performs no query and never confirms a provisional row.
        """
        return self.decode_started_row(event, expected_store_key=self._store_key, run=run)

    @classmethod
    def decode_started_row(
        cls, event: Mapping[str, object], *, expected_store_key: bytes,
        run: Mapping[str, object] | None = None,
    ) -> StepEvent:
        """Validate joined source columns without constructing or opening a store."""
        data = dict(event)
        data["idempotency_key"] = decode_native_replay_key(data.get("idempotency_key"))
        if "outcome" not in data:
            raise ValueError("native lifecycle source outcome is missing")
        if data.get("store_key") != expected_store_key:
            raise ValueError("native start source is missing or belongs to another store")
        mask = data.get("run_fields")
        if type(mask) is not int or not 0 <= mask < (1 << len(_EVENT_RUN_FIELDS)):
            raise ValueError("native start source run fields are corrupt")
        if mask:
            if run is None or any(run.get(name) != data.get(name) for name in ("store_key", "run_id")):
                raise ValueError("native start source run reference is missing or corrupt")
        for bit, name in enumerate(_EVENT_RUN_FIELDS):
            if mask & (1 << bit):
                if data.get(name) is not None or run.get(name) is None:
                    raise ValueError("native start source run reference is corrupt")
                data[name] = run[name]
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        names = tuple(field.name for field in fields(NativeLifecycleEvent if data.get("outcome") is not None else StepEvent))
        if any(name not in data for name in names):
            raise ValueError("native start source body is incomplete")
        raw_body = json.dumps({name: data[name] for name in names}, ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"))
        result = cls.decode_event(data)
        body = cls._started_body(result)
        if raw_body != body:
            raise ValueError("native start source body has noncanonical field types")
        if (result.idempotency_key is None or data.get("body_sha256") != _digest(body)
                or data.get("fields_sha256") != _digest(cls._started_fields_body(result))):
            raise ValueError("native start source body or locator is corrupt")
        return result

    @classmethod
    def decode_started_fields(
        cls, event: Mapping[str, object], *, expected_store_key: bytes,
    ) -> StepEvent:
        """Verify a metadata-free Trace projection, never a complete source proof.

        Supply SQL-resolved event fields except metadata, plus native scope,
        original run/replay key and fields_sha256. Metadata is intentionally not read or
        authenticated here; full source readback uses decode_started_row instead.
        """
        if "outcome" not in event:
            raise ValueError("native lifecycle projected outcome is missing")
        names = tuple(field.name for field in fields(
            NativeLifecycleEvent if event.get("outcome") is not None else StepEvent,
        ) if field.name != "metadata")
        if any(name not in event for name in names):
            raise ValueError("native start projected fields are incomplete")
        data = {name: event[name] for name in names}
        data["idempotency_key"] = decode_native_replay_key(data["idempotency_key"])
        data["metadata"] = {}
        result = cls.decode_event(data)
        raw_body = json.dumps({name: data[name] for name in names}, ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"))
        body = cls._started_fields_body(result)
        if (event.get("store_key") != expected_store_key
                or result.idempotency_key is None
                or event.get("fields_sha256") != _digest(body) or raw_body != body):
            raise ValueError("native start projected fields or locator are corrupt")
        return result

    @property
    def store_key(self) -> bytes:
        """Physical namespace and tenant partition of these native records."""
        return self._store_key

    @property
    def storage_context(self) -> SqlStorageContext:
        """The borrowed physical context used to verify same-group composition."""
        return self._context

    @property
    def backup_supported(self) -> bool:
        """Whether this owner has no unexported external effect ledger."""
        return self._tool_effect_store is None

    @property
    def backup_source(self) -> NativeStepSource | None:
        """Actual bound message group, never inferred from table presence."""
        if not self.backup_supported or self._message_archive is None:
            return None
        return NativeStepSource(group_id=self._message_archive.scope_id, store_key=self._store_key)

    @classmethod
    def validate_backup(cls, backup: NativeStepBackup) -> None:
        """Validate raw row identities and descriptors without opening SQL."""
        metadata = build_step_sql_metadata()
        runs = {}
        managed_starts = {}
        for name in ("runs", "events", "snapshots"):
            table = metadata.tables[f"ai_step_{name}"]
            columns = {column.name for column in table.columns} - {"id", "updated_at", "created_at"}
            seen = set()
            for row in getattr(backup, name):
                if set(row) != columns or row["store_key"] != backup.store_key:
                    raise ValueError("native backup row fields or scope are invalid")
                run_id = row["run_id"]
                cls._validate_record_ids(row)
                replay = None if name == "runs" else row["idempotency_key"]
                key = run_id if name == "runs" else (run_id, replay)
                if name == "runs" or replay is not None:
                    if key in seen:
                        raise ValueError("native backup contains duplicate owner identities")
                    seen.add(key)
                if name == "runs":
                    _RUN_ADAPTER.validate_python(cls._record_data(row))
                    marker = row["managed_start_key"]
                    if marker is not None and decode_native_replay_key(marker) is None:
                        raise ValueError("managed start identity must be a public replay key")
                    runs[run_id] = row
                    continue
                if name == "snapshots":
                    if type(row["pruned"]) is not bool:
                        raise ValueError("native backup tombstone flag is invalid")
                    data = cls._record_data(dict(row, messages=[]))
                    _SNAPSHOT_ADAPTER.validate_python(data)
                    if row["pruned"]:
                        if row["messages"] != "{}" or row["sequence_root"] is not None:
                            raise ValueError("native backup tombstone retains message content")
                    else:
                        cls.snapshot_message_sequence(row)
                        NativeCheckpoint.validate_descriptor(json.loads(row["messages"])["checkpoint"])
                    continue
                mask = row["run_fields"]
                if type(mask) is not int or not 0 <= mask < (1 << len(_EVENT_RUN_FIELDS)):
                    raise ValueError("native backup event run reference is invalid")
                data = dict(row)
                data["idempotency_key"] = decode_native_replay_key(row["idempotency_key"])
                run = runs.get(run_id)
                for bit, field in enumerate(_EVENT_RUN_FIELDS):
                    if mask & (1 << bit):
                        if run is None or data[field] is not None or run[field] is None:
                            raise ValueError("native backup event run reference is missing or corrupt")
                        data[field] = json.loads(run[field]) if field == "metadata" else run[field]
                event = cls.decode_event(data)
                if (isinstance(event, NativeLifecycleEvent) or row["body_sha256"] is not None
                        or row["fields_sha256"] is not None):
                    cls.decode_started_row(row, expected_store_key=backup.store_key, run=run)
                    if event.kind == "run_started":
                        if run_id in managed_starts or run is None or run["managed_start_key"] != replay:
                            raise ValueError("managed start differs from its registration identity")
                        managed_starts[run_id] = replay
        for run_id, row in runs.items():
            if row["managed_start_key"] != managed_starts.get(run_id):
                raise ValueError("native backup is missing its committed managed start")

    async def capture_backup(
        self, reader: SqlStateRowReader, *, accept: Callable[[object], None] | None = None,
        validate_messages: bool = True,
    ) -> NativeStepBackup:
        """Capture and strictly verify every owner row in the caller's read transaction.

        The caller begins a consistent read and captures the shared message
        closure in that same transaction. No connection or transaction is opened
        here, and no corrupt snapshot is skipped. With validate_messages=False,
        only owner rows are validated: the caller must capture and budget the
        message closure, then call decode_backup_snapshot for every unpruned
        row before publishing the backup. This defers hydration, not integrity.
        """
        from sqlalchemy import select

        if type(validate_messages) is not bool:
            raise TypeError("validate_messages must be a bool")
        if not self.backup_supported:
            raise ValueError("native backup cannot include an external effect ledger")
        batches = []
        for name in ("runs", "events", "snapshots"):
            table = self._table(name)
            columns = tuple(column for column in table.columns
                            if column.name not in {"id", "updated_at", "created_at"})
            rows = []
            cursor = 0
            while True:
                page = await reader.read_sql_rows(self._context, select(table.c.id, *columns).where(
                    table.c.store_locator == self._store_locator, table.c.store_key == self._store_key, table.c.id > cursor,
                ).order_by(table.c.id).limit(_BATCH_SIZE))
                for row in page:
                    value = {column.name: row[column.name] for column in columns}
                    if accept is not None:
                        encoded = dict(value, store_key=encode_native_store_key(value["store_key"]))
                        for key in ("idempotency_key", "managed_start_key"):
                            if key in encoded and encoded[key] is not None:
                                encoded[key] = base64.b64encode(encoded[key]).decode("ascii")
                        accept(encoded)
                    rows.append(value)
                if len(page) < _BATCH_SIZE:
                    break
                cursor = page[-1]["id"]
            batches.append(tuple(rows))
        backup = NativeStepBackup(self._store_key, *batches)
        self.validate_backup(backup)
        if validate_messages:
            await self.validate_backup_messages(reader, backup)
        return backup

    async def validate_backup_messages(self, reader: SqlStateRowReader, backup: NativeStepBackup) -> None:
        """Verify descriptors against messages in the caller's consistent read."""
        async def read(statement: "Select") -> Sequence[Mapping[str, object]]:
            return await reader.read_sql_rows(self._context, statement)
        await self._validate_backup_messages(read, backup)

    async def _validate_backup_messages(
        self, read: Callable[["Select"], Awaitable[Sequence[Mapping[str, object]]]], backup: NativeStepBackup,
    ) -> None:
        if not self.backup_supported or backup.store_key != self._store_key:
            raise ValueError("native backup belongs to another owner or has external effects")
        self.validate_backup(backup)
        from sqlalchemy import select

        sequences = tuple(self.snapshot_message_sequence(row) for row in backup.snapshots if not row["pruned"])
        table = self._table("sequences")
        columns = ("store_key", "sha256", "message_sha256", "tree_count",
                   "left_sha256", "right_sha256", "next_sha256")
        nodes: dict[str, Mapping[str, object]] = {}
        pending = {(span.first, True) for sequence in sequences for span in sequence.slices}
        visited: set[tuple[str, bool]] = set()
        while pending:
            frontier = pending - visited
            if not frontier:
                break
            missing = sorted({identity for identity, _ in frontier} - nodes.keys())
            for offset in range(0, len(missing), _BATCH_SIZE):
                batch = missing[offset:offset + _BATCH_SIZE]
                rows = await read(select(*(table.c[name] for name in columns)).where(
                    table.c.store_locator == self._store_locator, table.c.store_key == self._store_key, table.c.sha256.in_(batch)))
                if len(rows) != len(batch) or {row["sha256"] for row in rows} != set(batch):
                    raise ValueError("native backup sequence closure is incomplete")
                nodes.update((row["sha256"], row) for row in rows)
            pending = set()
            for identity, forest in frontier:
                row = nodes[identity]
                pending.update((row[name], False) for name in ("left_sha256", "right_sha256")
                               if row[name] is not None)
                if forest and row["next_sha256"] is not None:
                    pending.add((row["next_sha256"], True))
            visited.update(frontier)
        closure = NativeMessageClosure(tuple(nodes.values()))
        for sequence in sequences:
            closure.mark(self._store_key, sequence.slices)
        table = self._table("messages")
        identities = sorted(identity for scope, identity in closure.retained_messages)
        payloads = {}
        for offset in range(0, len(identities), _BATCH_SIZE):
            batch = identities[offset:offset + _BATCH_SIZE]
            rows = await read(select(table.c.store_key, table.c.sha256, table.c.message,
                                    table.c.compression, table.c.raw_size, table.c.payload_json).where(
                table.c.store_locator == self._store_locator, table.c.store_key == self._store_key, table.c.sha256.in_(batch)))
            if len(rows) != len(batch) or {row["sha256"] for row in rows} != set(batch):
                raise ValueError("native backup message closure is incomplete")
            for row in rows:
                content = validate_native_message_content(row,
                    group_id=None if self._message_archive is None else self._message_archive.scope_id,
                    verify_inline=False)
                if content.payload is None:
                    encoded = content.body
                elif content.payload.encoding == "json-attachments":
                    if self._message_archive is None:
                        raise ValueError("native message ObjectStore resolver is required")
                    encoded = await read_attachment_payload(content.payload,
                                                            read_object=self._message_archive.read_payload_object)
                else:
                    raise ValueError("native message payload is not an attachment manifest")
                raw = decode_payload_bytes(encoded, codec=content.compression,
                                           raw_digest=content.raw_digest, raw_size=content.raw_size)
                decode_model_message_payloads((raw,))
                payloads[(self._store_key, row["sha256"])] = raw
        self.validate_backup_payloads(backup, closure, payloads)

    @classmethod
    def validate_backup_payloads(
        cls, backup: NativeStepBackup, closure: NativeMessageClosure,
        payloads: Mapping[tuple[bytes, str], bytes],
    ) -> None:
        """Bind checkpoint headers to an authenticated complete message closure.

        Callers authenticate every unique canonical message before this method.
        Only the terminal message participates in the checkpoint retry digest;
        shared historical prefixes need not be hydrated again for each header.
        """
        for row in backup.snapshots:
            if row["pruned"]:
                continue
            sequence = cls.snapshot_message_sequence(row)
            closure.mark(backup.store_key, sequence.slices)
            tail = None
            if sequence.count:
                span = sequence.slices[-1]
                identities = closure.message_identities(backup.store_key,
                    (NativeMessageSlice(span.first, span.count, span.end - 1, span.end),))
                tail = payloads[(backup.store_key, identities[0])]
            cls.validate_backup_snapshot_tail(row, tail)
        if not closure.retained_messages.issubset(payloads):
            raise ValueError("native backup message closure is incomplete")

    async def restore_backup(self, session: "AsyncSession", backup: NativeStepBackup) -> None:
        """Restore native history, reusing equal canonical run registrations."""
        from sqlalchemy import select

        async def read(statement: "Select") -> Sequence[Mapping[str, object]]:
            return (await session.execute(statement)).mappings().all()
        await self._validate_backup_messages(read, backup)
        for name in ("events", "snapshots"):
            table = self._table(name)
            if (await session.execute(select(table.c.id).where(
                    table.c.store_locator == self._store_locator, table.c.store_key == self._store_key).limit(1))).first() is not None:
                raise ValueError("native backup restore requires an empty owner")
        run_table = self._table("runs")
        existing_runs = {}
        identifiers = tuple(row["run_id"] for row in backup.runs)
        for offset in range(0, len(identifiers), _BATCH_SIZE):
            rows = (await session.execute(select(run_table).where(
                run_table.c.store_locator == self._store_locator,
                run_table.c.store_key == self._store_key,
                run_table.c.run_id.in_(identifiers[offset:offset + _BATCH_SIZE]),
            ))).mappings().all()
            existing_runs.update((row["run_id"], row) for row in rows)
        for row in backup.runs:
            existing = existing_runs.get(row["run_id"])
            if existing is not None and (_RUN_ADAPTER.validate_python(self._record_data(existing))
                                         != _RUN_ADAPTER.validate_python(self._record_data(row))):
                raise ValueError("native backup registration differs from its canonical owner")
        for name in ("runs", "events", "snapshots"):
            table = self._table(name)
            # Individual inserts preserve ordering even on dialects that do not
            # guarantee the generated ID order of executemany operations.
            for row in getattr(backup, name):
                if name == "runs" and row["run_id"] in existing_runs:
                    existing = existing_runs[row["run_id"]]
                    if existing["managed_start_key"] is not None:
                        raise ValueError("native restore requires metadata-only registrations")
                    if row["managed_start_key"] is not None:
                        from sqlalchemy import update
                        changed = await session.execute(update(table).where(
                            *self._run_predicates(table, row["run_id"]),
                            table.c.managed_start_key.is_(None),
                        ).values(managed_start_key=row["managed_start_key"]))
                        if changed.rowcount != 1:
                            raise ValueError("native restore registration changed concurrently")
                    continue
                inserted = await session.execute(table.insert().values(**dict(row)))
                if name == "snapshots" and not row["pruned"]:
                    from ._native_content_index import resolve_targets
                    root = self.snapshot_message_sequence(row)
                    targets = await resolve_targets(session, "sequence", self._store_key,
                                                    tuple(span.first for span in root.slices))
        self.invalidate_context(self._context)

    @classmethod
    def for_state_store(cls, store: StateStore, *, namespace: str, tenant_id: str) -> "SqlStepStore | None":
        """Borrow a SQL owner's native message partition without opening a connection."""
        from ._sql import SqlStateStore

        if not isinstance(store, SqlStateStore):
            return None
        return cls(store.context, namespace=namespace, tenant_id=tenant_id)

    async def prepare_archived_messages(
        self, messages: Sequence[ModelMessage], prepared: PreparedArchive,
    ) -> NativeMessageRoot:
        value = await self.prepare_message_content(self._frozen_messages.snapshot(messages), archive=prepared)
        prepared.native_messages[value.root] = value
        prepared.native_publishers[value.root] = self.publish_archived_messages
        return self.message_root(value.root)

    async def prepare_archived_message_payloads(
        self, payloads: Sequence[bytes], prepared: PreparedArchive,
    ) -> NativeMessageRoot:
        """Preserve validated singleton source bytes in the native content owner."""
        return await self.prepare_archived_messages(self._frozen_messages.snapshot_payloads(payloads), prepared)

    async def publish_archived_messages(self, transaction: StateTransaction, prepared: PreparedArchive) -> None:
        """Publish dependencies inside the owner's atomic mutation, then observe commit."""
        if not prepared.native_messages:
            return
        if not isinstance(transaction, SqlMutationObserver):
            raise ValueError("native messages require an observable SQL mutation")
        receipts = await self.publish_message_content(transaction, prepared, owner_written=True)
        transaction.observe_mutation(lambda committed: (
            self.confirm_message_content(receipts) if committed else self._message_content.discard_uncommitted(receipts)
        ))

    def bind_message_archive(self, archive: ContentArchive) -> None:
        """Bind the already opened physical group scope and its object route."""
        from ._sql import SqlStateStorageGroup

        group = archive.store.storage_group
        if not isinstance(group, SqlStateStorageGroup) or group.context.engine is not self._context.engine:
            raise ValueError("native message archive belongs to another SQL group")
        group.bind_native_partition(self._store_key)
        self._message_content.bind_group(archive.scope_id)
        self._message_archive = archive
        archive.bind_message_publisher(self.publish_archived_messages)

    def message_root(self, sequence: NativeMessageSequence) -> NativeMessageRoot:
        return self._message_content.scoped_root(sequence)

    def seed_snapshot_baseline(self, run_id: str, root: NativeMessageRoot, *, raw_digest: str) -> bool:
        """Carry this owner's verified context into a new run without publishing it."""
        try:
            self._message_content.validate_root(root)
        except ValueError:
            return False
        integrity = self._message_integrity.get(root.sequence)
        if integrity is None or integrity[0] != raw_digest:
            return False
        return self._seed_snapshot_baseline(run_id, root.sequence, integrity)

    def inherit_snapshot_baseline(self, run_id: str, *, source_run_id: str) -> bool:
        """Carry a committed or read-back source run's content into its next attempt."""
        previous = self._message_inputs.get(source_run_id)
        if previous is None:
            return False
        digest, size, _count, sequence = previous
        return self._seed_snapshot_baseline(run_id, sequence, (digest, size))

    def _seed_snapshot_baseline(
        self, run_id: str, sequence: NativeMessageSequence, integrity: tuple[str, int],
    ) -> bool:
        if (sequence not in self._message_proofs or self._message_integrity.get(sequence) != integrity
                or len(sequence.slices) > 1 or sequence.slices and (
                    sequence.slices[0].start != 0 or sequence.slices[0].end != sequence.slices[0].count)):
            return False
        self._remember_snapshot_input(run_id, sequence, integrity)
        return True

    def _remember_snapshot_input(
        self, run_id: str, sequence: NativeMessageSequence, integrity: tuple[str, int],
    ) -> None:
        digest, size = integrity
        self._message_inputs[run_id] = (digest, size, sequence.count, sequence)
        self._message_inputs.move_to_end(run_id)
        while len(self._message_inputs) > _RUN_CACHE_SIZE:
            self._message_inputs.popitem(last=False)

    @classmethod
    def invalidate_context(cls, context: SqlStorageContext) -> None:
        """Discard all same-engine content proofs after offline reclamation."""
        for store in tuple(cls._instances):
            if store.storage_context.engine is context.engine:
                store.invalidate_message_content()

    def invalidate_message_content(self) -> None:
        self._confirmed_messages.clear()
        self._confirmed_sequences.clear()
        self._message_content.invalidate()
        self._message_proofs.clear()
        self._message_integrity.clear()
        self._message_snapshots.clear()
        self._prepared_messages.clear()
        self._message_inputs.clear()

    async def _prepare_message_payload(
        self, raw: bytes, existing: ContentRecord | None = None,
    ) -> PreparedMessagePayload:
        if self._message_archive is None:
            compression, encoded = encode_payload_bytes(raw)
            payload = None
        else:
            compression, payload, encoded = (
                await self._message_archive.prepare_payload(raw) if existing is None
                else await self._message_archive.prepare_payload(raw, existing=existing)
            )
        return PreparedMessagePayload(raw, encoded=encoded, compression=compression, payload=payload)

    async def prepare_message_content(
        self, messages: MessageSnapshot, *,
        baseline: tuple[NativeMessageRoot | NativeMessageSequence, str, int] | None = None,
        single_sequence: bool = False,
        archive: PreparedArchive | None = None,
    ) -> PreparedNativeMessages:
        """Use frozen bytes; mutable callers must first encode and check all inputs."""
        base = None
        start = 0
        if baseline is not None:
            root, digest, size = baseline
            self._message_content.validate_root(root)
            sequence = root.sequence if isinstance(root, NativeMessageRoot) else root
            proof = self._message_proofs.get(sequence)
            complete = (not sequence.slices or len(sequence.slices) == 1
                        and sequence.slices[0].start == 0
                        and sequence.slices[0].end == sequence.slices[0].count)
            if proof is not None and complete and sequence.count <= len(messages):
                known = self._message_snapshots.get(sequence)
                shared = (known is not None and self._message_integrity.get(sequence) == (digest, size)
                          and messages.shared_prefix_length(known) == sequence.count)
                if shared or messages[:sequence.count].encoded_digest() == (digest, size):
                    base, start = proof, sequence.count
        prepared = await self._message_content.prepare_append(
            messages[start:], prepare_payload=self._prepare_message_payload, base=base,
            archive=archive,
            reuse_payload=self._prepare_message_payload,
        )
        messages.encoded_digest()
        self._prepared_messages[prepared] = messages
        candidate = ref(prepared)

        async def rebuild() -> None:
            previous = candidate()
            if previous is None:
                return
            cold = await self._message_content.prepare_append(
                messages, prepare_payload=self._prepare_message_payload, use_cache=False,
            )
            self._message_content.replace_prepared(previous, cold)
            self._prepared_messages[cold] = messages

        self._message_content.bind_recovery(prepared, rebuild)
        return prepared

    async def publish_message_content(
        self, transaction: StateTransaction, prepared: PreparedArchive, *, owner_written: bool,
    ) -> tuple[NativeMessageReceipt, ...]:
        return tuple([await self._message_content.publish_retained(
            transaction, value, owner_written=owner_written,
        ) for value in prepared.native_messages.values()])

    def confirm_message_content(self, receipts: Sequence[NativeMessageReceipt]) -> None:
        for receipt in receipts:
            if receipt.root is None:
                continue
            proof = self._message_content.confirm_current(receipt)
            if proof is not None:
                messages = self._prepared_messages.get(receipt.prepared)
                self._remember_message_proof(proof, messages=messages,
                    integrity=None if messages is None else messages.encoded_digest())

    def _remember_message_proof(
        self, proof: CommittedNativeMessages, *, integrity: tuple[str, int] | None = None,
        messages: MessageSnapshot | None = None,
    ) -> None:
        if not self._max_cached_messages:
            return
        self._message_proofs[proof.root] = proof
        self._message_proofs.move_to_end(proof.root)
        if integrity is not None:
            self._message_integrity[proof.root] = integrity
        if messages is not None:
            self._message_snapshots[proof.root] = messages
        while len(self._message_proofs) > _RUN_CACHE_SIZE:
            root, _proof = self._message_proofs.popitem(last=False)
            self._message_integrity.pop(root, None)
            self._message_snapshots.pop(root, None)

    def can_reuse_message_root(self, root: NativeMessageRoot, *, raw_digest: str, raw_size: int) -> bool:
        """Reuse only an actual same-group committed/readback integrity proof."""
        try:
            self._message_content.validate_root(root)
        except ValueError:
            return False
        for source in tuple(self._instances):
            if (source.storage_context.engine is not self.storage_context.engine
                    or source.store_key != self.store_key):
                continue
            try:
                source._message_content.validate_root(root)
            except ValueError:
                continue
            if (root.sequence in source._message_proofs
                    and source._message_integrity.get(root.sequence) == (raw_digest, raw_size)):
                return True
        return False

    async def read_message_roots(
        self, roots: Sequence[NativeMessageRoot],
    ) -> tuple[tuple[bytes, ...], ...]:
        return await (await self.capture_message_roots(roots))()

    async def capture_message_roots(
        self, roots: Sequence[NativeMessageRoot], *, transaction: SqlStateRowReader | None = None,
        reserve: Callable[[int], None] | None = None,
    ) -> Callable[[], Awaitable[tuple[tuple[bytes, ...], ...]]]:
        """Capture SQL rows with their owners; decode objects after the read scope."""
        from .state import active_state_transaction

        roots = tuple(roots)
        query = self._message_content.range_query(roots)
        if transaction is None and self._message_archive is not None:
            active = active_state_transaction(self._message_archive.store)
            if active is not None:
                if not isinstance(active, SqlStateRowReader):
                    raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
                transaction = active
        rows = ()
        if query is not None:
            if transaction is not None:
                rows = await transaction.read_sql_rows(self._context, query.statement)
            else:
                async with self._context.sessions() as session, session.begin():
                    await self._context.run_statement(self._context.dialect.begin_consistent_read(session))
                    rows = tuple(dict(row) for row in (await self._context.run_statement(
                        session.execute(query.statement),
                    )).mappings())

        if reserve is not None:
            # Structural ancestors carry NULL bodies. Only physical message
            # bodies consume decompression memory, once per distinct identity.
            sizes = {}
            for row in rows:
                if row.get("message") is None and row.get("payload_json") is None:
                    continue
                size = row.get("raw_size")
                if type(size) is not int or size < 0:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                sizes[row["message_sha256"]] = size
            reserve(sum(sizes.values()))

        async def hydrate() -> tuple[tuple[bytes, ...], ...]:
            try:
                readbacks = await self._message_content.read_hydrated_ranges(
                    roots, rows, read_object=(None if self._message_archive is None
                                             else self._message_archive.read_payload_object),
                )
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            for result in readbacks:
                raw = combine_model_message_payloads(result.payloads)
                self._remember_message_proof(self._message_content.adopt_read(result),
                                             integrity=(hashlib.sha256(raw).hexdigest(), len(raw)))
            return tuple(result.payloads for result in readbacks)
        return hydrate

    async def prepare_step(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None, managed_start: bool = False,
    ) -> PreparedStep:
        """Capture native records; only an explicitly owned start mints source proof."""
        if type(managed_start) is not bool:
            raise ValueError("managed_start must be a bool")
        if managed_start:
            starts = [event for event in events if event.kind == "run_started"]
            if (run is None or len(starts) != 1 or starts[0].idempotency_key is None
                    or starts[0].run_id != run.run_id):
                raise ValueError("managed start requires one keyed run_started and its registration")
        for record in (() if run is None else (run,)) + tuple(events) + (() if snapshot is None else (snapshot,)):
            validate_native_step_ids(run_id=record.run_id, conversation_id=record.conversation_id,
                                     parent_run_id=record.parent_run_id)
        await self.initialize()
        run, events = deepcopy(run), tuple(deepcopy(event) for event in events)
        if snapshot is not None:
            snapshot = replace(snapshot, messages=self._frozen_messages.snapshot(snapshot.messages))
        values, messages = (None, None) if snapshot is None else await self._prepare_snapshot(snapshot)
        return PreparedStep(run, events, snapshot, values, messages,
                            tuple(encode_native_replay_key(event.idempotency_key) for event in events), managed_start)

    async def write_prepared_step(
        self, transaction: StateTransaction, prepared: PreparedStep, *,
        confirmed: StepWriteReceipt | None = None,
    ) -> StepWriteReceipt:
        """Join the existing same-engine transaction without opening another.

        A caller-held receipt may outlive the bounded cache. Supply it only after
        its enclosing transaction committed; provisional receipts are not proof.
        """
        if not isinstance(transaction, SqlStateTransaction):
            raise TypeError("native SQL records require a SQL state transaction")
        # A reusable preparation is not an append occurrence. Keep each public
        # publication's physical keys fixed for its writer, not for later calls.
        values = prepared.snapshot_values
        if prepared.snapshot is not None and prepared.snapshot.idempotency_key is None and values is not None:
            values = {**values, "idempotency_key": encode_native_replay_key(None)}
        publication = replace(prepared, snapshot_values=values, event_keys=tuple(
            encode_native_replay_key(None) if event.idempotency_key is None else key
            for event, key in zip(prepared.events, prepared.event_keys, strict=True)
        ))
        return await transaction.run_sql(
            self._context, lambda session: self._write_prepared_step(
                session, publication, confirmed=confirmed,
            ),
        )

    async def _write_prepared_step(
        self, session: "AsyncSession", prepared: PreparedStep, *,
        confirmed: StepWriteReceipt | None = None,
    ) -> StepWriteReceipt:
        started = None
        if prepared.managed_start:
            starts = [event for event in prepared.events if event.kind == "run_started"]
            if (prepared.run is None or len(starts) != 1 or starts[0].idempotency_key is None
                    or starts[0].run_id != prepared.run.run_id):
                raise ValueError("managed start requires one keyed run_started and its registration")
            started = starts[0]
        elif confirmed is not None and not any(isinstance(event, NativeLifecycleEvent) for event in prepared.events):
            raise ValueError("a source receipt requires an explicitly managed lifecycle")
        source = None if confirmed is None else confirmed.run_fields
        if confirmed is not None:
            self.events_from_receipt(confirmed)
        if confirmed is not None and started is not None:
            self.run_from_receipt(confirmed)
            if prepared.run is None or confirmed.run_fields is None:
                raise ValueError("confirmed receipt does not prove the supplied run_started source")
            self._validate_started_replay(prepared.run, started, confirmed.run_fields)
        if prepared.run is not None:
            known = (confirmed.run_fields if confirmed is not None
                     else self._confirmed_runs.get(prepared.run.run_id))
            if (started is not None and known is not None and known.started_body is not None
                    and (confirmed is not None or prepared.run.registration_id is not None)):
                self._validate_started_replay(prepared.run, started, known)
                source = known
            else:
                source = await self._write_run(session, prepared.run, replay=True, started=started)
        if source is None and any(isinstance(event, NativeLifecycleEvent) for event in prepared.events):
            source = self._confirmed_runs.get(prepared.events[0].run_id)
        events = prepared.events
        replay_keys = {id(event): key for event, key in zip(events, prepared.event_keys, strict=True)}
        fresh = started
        if started is not None and source is not None and source.started_body is not None:
            events = tuple(event for event in events if event is not started)
            fresh = None
        for managed, group in groupby(events, key=lambda event: isinstance(event, NativeLifecycleEvent)):
            batch = tuple(group)
            if managed:
                for event in batch:
                    source = await self._write_lifecycle_event(session, event, source=source)
            else:
                await self._write_events(session, batch, source=source, started=fresh, replay_keys=replay_keys)
                if fresh is not None and any(event is fresh for event in batch) and source is not None:
                    source = self._with_started(source, fresh)
                    fresh = None
        receipt = StepWriteReceipt((), ())
        if prepared.snapshot is not None and prepared.snapshot_values is not None:
            receipt = await self._write_snapshot(
                session, prepared.snapshot_values, prepared.messages,
            )
        return replace(receipt, run_fields=source, lifecycle_keys=tuple(
            event.idempotency_key for event in prepared.events
            if event is started or isinstance(event, NativeLifecycleEvent)
        ))

    def confirm_prepared_step(self, receipt: StepWriteReceipt) -> None:
        """Advance immutable source knowledge after the enclosing commit is confirmed."""
        if receipt.run_fields is not None:
            self._confirm_run(receipt.run_fields)
        self._confirm_records(self._confirmed_messages, receipt.message_digests)
        self._confirm_records(self._confirmed_sequences, receipt.sequence_digests)
        if receipt.native_messages is not None:
            self.confirm_message_content((receipt.native_messages,))

    def _confirm_run(self, source: _RunFields) -> None:
        if source.engine is not self._context.engine or source.store_key != self._store_key:
            raise ValueError("run receipt belongs to another native store")
        known = self._confirmed_runs.get(source.run_id)
        if known is not None:
            base = replace(known, started_body=None, terminal_bodies=(), byte_size=source.byte_size)
            candidate = replace(source, started_body=None, terminal_bodies=())
            if base == candidate:
                if known.started_body is not None and source.started_body is None:
                    source = self._with_started(source, self.decode_event(json.loads(known.started_body)))
                for body in known.terminal_bodies:
                    source = self._with_terminal(source, self.decode_event(json.loads(body)))
        if source.byte_size > _RUN_CACHE_BYTES:
            return
        self._confirmed_runs[source.run_id] = source
        self._confirmed_runs.move_to_end(source.run_id)
        size = sum(item.byte_size for item in self._confirmed_runs.values())
        while len(self._confirmed_runs) > _RUN_CACHE_SIZE or size > _RUN_CACHE_BYTES:
            _, evicted = self._confirmed_runs.popitem(last=False)
            size -= evicted.byte_size

    async def commit_step(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> None:
        """Commit a standalone native boundary in its own short transaction.

        Runtime same-group writers instead prepare once and join their existing
        execution transaction. A separate caller-owned store uses this explicit
        split-storage operation and cannot claim atomic execution publication.
        """
        if run is None and not events and snapshot is None:
            return
        prepared = await self.prepare_step(run=run, events=events, snapshot=snapshot)
        from ._native_content_index import run_native_mutation
        receipt = await run_native_mutation(self._context,
            lambda session: self._write_prepared_step(session, prepared), domain="steps.boundary",
        )
        self.confirm_prepared_step(receipt)
        self._publish_receipt(receipt)

    async def commit_step_receipt(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> StepWriteReceipt:
        """Observe the public boundary operation without bypassing overrides."""
        with self._capture_receipts() as receipts:
            await self.commit_step(run=run, events=events, snapshot=snapshot)
        receipt = self._matching_receipt(receipts, run_id=None if run is None else run.run_id)
        self._publish_receipt(receipt)
        return receipt

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self.initialize()
        prepared = await self._prepare_snapshot(snapshot)

        async def write(session: "AsyncSession") -> StepWriteReceipt:
            return await self._write_snapshot(session, *prepared)

        from ._native_content_index import run_native_mutation
        receipt = await run_native_mutation(self._context, write, domain="steps.snapshot")
        self.confirm_prepared_step(receipt)

    async def _prepare_snapshot(
        self, snapshot: ContinuableSnapshot,
    ) -> "tuple[dict[str, SqlValue], PreparedNativeMessages]":
        decoded = NativeCheckpoint.decode(snapshot)
        if isinstance(snapshot, NativeCheckpoint):
            # Only the tail carries the retry envelope; the captured history is
            # already independent and must retain its structural prefix proof.
            snapshot = NativeCheckpoint.encode(replace(snapshot, messages=snapshot.messages[-1:]),
                                               output_retries_used=snapshot.output_retries_used)
        values = self._record_values(snapshot)
        checkpoint = None
        if isinstance(decoded, NativeCheckpoint):
            checkpoint = {
                "output_retries_used": decoded.output_retries_used,
                "message_sha256": hashlib.sha256(encode_model_messages(snapshot.messages[-1:])).hexdigest(),
            }
        messages = self._frozen_messages.snapshot(decoded.messages)
        previous = self._message_inputs.get(snapshot.run_id)
        baseline = None if previous is None else (previous[3], previous[0], previous[1])
        prepared = await self.prepare_message_content(messages, baseline=baseline, single_sequence=True)
        sequence = prepared.root
        if (sequence.count != len(messages) or len(sequence.slices) > 1
                or sequence.slices and (sequence.slices[0].start != 0
                                       or sequence.slices[0].end != sequence.slices[0].count)):
            raise ValueError("snapshot requires one complete native message sequence")
        values["messages"] = _snapshot_descriptor_json({"stored_messages": sequence.count, "checkpoint": checkpoint})
        values["pruned"] = False
        values["sequence_root"] = sequence.slices[0].first if sequence.slices else None
        self._remember_snapshot_input(snapshot.run_id, sequence, messages.encoded_digest())
        return values, prepared

    async def _write_snapshot(
        self, session: "AsyncSession", values: "dict[str, SqlValue]", prepared: PreparedNativeMessages,
    ) -> StepWriteReceipt:
        values = values | self._run_values(values["run_id"])
        if decode_native_replay_key(values["idempotency_key"]) is None:
            result = await session.execute(self._table("snapshots").insert().values(values))
            snapshot_id = result.inserted_primary_key[0]
        else:
            result = await self._context.dialect.insert_ignore_conflict(
                session, table=self._table("snapshots"), values=values,
                index_elements=("store_locator", "run_id", "idempotency_key"),
            )
            if not result.inserted:
                from sqlalchemy import select
                table = self._table("snapshots")
                raw = (await session.execute(select(table.c.store_key).where(
                    table.c.store_locator == self._store_locator,
                    table.c.run_id == values["run_id"], table.c.idempotency_key == values["idempotency_key"],
                ))).scalar_one()
                if raw != self._store_key:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                # A conflicting owner key proves no candidate content, even if
                # another owner has already committed that candidate sequence.
                return StepWriteReceipt((), ())
            snapshot_id = result.row_id
        if snapshot_id is None:
            raise RuntimeError("snapshot insertion did not return its physical identity")
        content = await self._message_content.publish_in_session(session, prepared, owner_inserted=True)
        targets = content.publication_targets(
            session, store_key=self._store_key, roots=tuple(span.first for span in prepared.root.slices),
        )
        await self._prune_snapshots(session, values["run_id"])
        return StepWriteReceipt((), (), native_messages=content)

    def _confirm_records(self, cache: "OrderedDict[str, None]", confirmed: Sequence[str]) -> None:
        if self._max_cached_messages:
            for key in confirmed:
                cache[key] = None
                cache.move_to_end(key)
                if len(cache) > self._max_cached_messages:
                    cache.popitem(last=False)

    async def latest_snapshot(
        self, *, run_id: str, include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None:
        snapshots = await self._load_snapshots(run_id, include_interrupted=include_interrupted, latest=True)
        return None if not snapshots else snapshots[0]

    async def list_snapshots(
        self, *, run_id: str, include_interrupted: bool = False,
    ) -> list[ContinuableSnapshot]:
        """Return retained upstream snapshots in insertion order."""
        return await self._load_snapshots(run_id, include_interrupted=include_interrupted, latest=False)

    def with_tool_effect_store(self, store: StepStore) -> "SqlStepStore":
        """Borrow this SQL partition with an attempt's existing effect owner."""
        bound = copy(self)
        bound._tool_effect_store = store
        bound._owns_context = False
        self._instances.add(bound)
        return bound

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        validate_resource_id(record.run_id)
        await self._require_tool_effect_store().record_tool_effect(record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        validate_resource_id(run_id)
        return await self._require_tool_effect_store().get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        validate_resource_id(run_id)
        return await self._require_tool_effect_store().list_unresolved_tool_effects(run_id=run_id)

    def _require_tool_effect_store(self) -> StepStore:
        if self._tool_effect_store is None:
            raise RuntimeError("SqlStepStore requires a bound tool-effect store")
        return self._tool_effect_store

    async def _prune_snapshots(self, session: "AsyncSession", run_id: str) -> None:
        from sqlalchemy import false, or_, select

        if self._max_snapshots_per_run is None:
            return
        table = self._table("snapshots")
        active = (*self._run_predicates(table, run_id), table.c.pruned == false())
        # Return only the retention boundary and the complete recovery point,
        # even when a new adapter reduces an existing run's retention limit.
        boundary = select(table.c.id).where(*active).order_by(table.c.id.desc()).offset(
            self._max_snapshots_per_run,
        ).limit(1).scalar_subquery()
        complete = select(table.c.id).where(*active, table.c.state == "complete").order_by(
            table.c.id.desc(),
        ).limit(1).scalar_subquery()
        cutoff, last_complete = (await session.execute(select(boundary, complete))).one()
        if cutoff is None:
            return
        conditions = (*active, table.c.id <= cutoff)
        # A current-read UPDATE can see a lower-ID complete row that committed
        # after the retention read. Only retire complete points older than the
        # one actually observed; a concurrent new recovery point stays retained.
        complete_safe = table.c.state != "complete"
        if last_complete is not None:
            complete_safe = or_(complete_safe, table.c.id < last_complete)
        conditions += (complete_safe,)
        after = 0
        while True:
            retired = (await session.execute(select(table.c.id).where(
                *conditions, table.c.id > after,
            ).order_by(table.c.id).limit(128))).all()
            if not retired:
                break
            ids = tuple(row.id for row in retired)
            await session.execute(table.update().where(
                *self._run_predicates(table, run_id), table.c.id.in_(ids),
            ).values(pruned=True, messages="{}", sequence_root=None))
            after = retired[-1].id
            if len(retired) < 128:
                break

    @classmethod
    def snapshot_message_sequence(cls, snapshot: Mapping[str, object]) -> NativeMessageSequence:
        """Read the snapshot's sole root column and bounded control descriptor."""
        try:
            descriptor = json.loads(snapshot["messages"])
        except (TypeError, ValueError) as error:
            raise ValueError("snapshot message descriptor is invalid") from error
        if not isinstance(descriptor, dict) or set(descriptor) != {"stored_messages", "checkpoint"}:
            raise ValueError("snapshot message references are incomplete")
        count, root = descriptor["stored_messages"], snapshot["sequence_root"]
        if type(count) is not int or not 0 <= count <= (1 << 63) - 1 or (root is None) != (count == 0):
            raise ValueError("snapshot message references are incomplete")
        return NativeMessageSequence(() if not count else (NativeMessageSlice(root, count, 0, count),))

    async def _load_snapshots(
        self, run_id: str, *, include_interrupted: bool, latest: bool,
    ) -> list[ContinuableSnapshot]:
        from sqlalchemy import false, select

        await self.initialize()
        table = self._table("snapshots")
        query = select(table).where(*self._run_predicates(table, run_id), table.c.pruned == false())
        if not include_interrupted:
            query = query.where(table.c.state == "complete")
        query = query.order_by(table.c.id.desc() if latest else table.c.id)
        if latest:
            query = query.limit(1)
        batches = []
        async with self._context.sessions() as session, session.begin():
            await self._context.run_statement(self._context.dialect.begin_consistent_read(session))
            rows = (await self._context.run_statement(session.execute(query))).mappings().all()
            for offset in range(0, len(rows), _BATCH_SIZE):
                selected = []
                sequences = []
                for row in rows[offset:offset + _BATCH_SIZE]:
                    try:
                        sequence = self.snapshot_message_sequence(row)
                    except ValueError:
                        if latest:
                            raise
                        _logger.warning("Skipping corrupt snapshot descriptor for run %s", run_id)
                        continue
                    selected.append(row)
                    sequences.append(sequence)
                sequences = tuple(sequences)
                if any(sequence.count for sequence in sequences):
                    statement = self._message_content.range_query(sequences).statement
                    found = (await self._context.run_statement(session.execute(statement))).mappings().all()
                else:
                    found = ()
                batches.append((selected, sequences, found))
        results = []
        verified = []
        read_object = None if self._message_archive is None else self._message_archive.read_payload_object
        for selected, sequences, found in batches:
            try:
                readbacks = await self._message_content.read_hydrated_ranges(sequences, found, read_object=read_object)
            except (ValueError, AIError):
                if latest:
                    raise
                isolated = self._message_content.partition_range_rows(sequences, found)
                recovered = []
                for sequence, subset in zip(sequences, isolated, strict=True):
                    try:
                        recovered.append((await self._message_content.read_hydrated_ranges(
                            (sequence,), subset, read_object=read_object))[0])
                    except (ValueError, AIError):
                        recovered.append(None)
                readbacks = recovered
            for row, readback in zip(selected, readbacks, strict=True):
                if readback is None:
                    _logger.warning("Skipping corrupt snapshot for run %s", run_id)
                    continue
                try:
                    # Verify the returned owner and replay key before restoring content.
                    if row["run_id"] != run_id or row["store_key"] != self._store_key:
                        raise ValueError("snapshot owner identity is invalid")
                    snapshot, raw = self._decode_snapshot(row, readback.payloads)
                    integrity = hashlib.sha256(raw).hexdigest(), len(raw)
                except (ValueError, AIError):
                    if latest:
                        raise
                    _logger.warning("Skipping corrupt snapshot for run %s", run_id)
                    continue
                verified.append((readback, integrity))
                results.append(snapshot)
        for readback, integrity in verified:
            self._remember_message_proof(self._message_content.adopt_read(readback),
                                         integrity=integrity)
            self._seed_snapshot_baseline(run_id, readback.root, integrity)
        return results

    def _sequence_query(self, snapshots: "Sequence[RowMapping]") -> "Select | None":
        sequences = tuple(self.snapshot_message_sequence(row) for row in snapshots)
        if not any(sequence.count for sequence in sequences):
            return None
        return self._message_content.range_query(sequences).statement

    @classmethod
    def validate_backup_snapshot_tail(
        cls, data: Mapping[str, object], payload: bytes | None,
    ) -> None:
        """Verify checkpoint header/retry binding using only its last message.

        The caller authenticates every unique message body and the complete
        sequence closure separately. The checkpoint digest binds its terminal
        message plus control header, so repeated prefixes need not be decoded.
        """
        sequence = cls.snapshot_message_sequence(data)
        if (payload is None) != (sequence.count == 0):
            raise ValueError("snapshot terminal message does not match its count")
        descriptor = json.loads(data["messages"])
        _raw, messages = decode_model_message_payloads(() if payload is None else (payload,))
        if len(messages) != (0 if payload is None else 1):
            raise ValueError("snapshot terminal message is invalid")
        header = dict(data, messages=[])
        header.pop("sequence_root")
        snapshot = replace(_SNAPSHOT_ADAPTER.validate_python(cls._record_data(header)), messages=list(messages))
        NativeCheckpoint.restore_descriptor(snapshot, descriptor["checkpoint"])

    @classmethod
    def decode_backup_snapshot(
        cls, data: Mapping[str, object], payloads: tuple[bytes, ...],
    ) -> ContinuableSnapshot:
        """Verify an unpruned snapshot's complete header, messages, and retry binding.

        Payloads are canonical message bytes in occurrence order, already
        authenticated against the shared native message closure by the caller.
        This operation does not open a store or modify the supplied row.
        """
        return cls._decode_snapshot(data, payloads)[0]

    @classmethod
    def _decode_snapshot(
        cls, data: Mapping[str, object], payloads: tuple[bytes, ...],
    ) -> tuple[ContinuableSnapshot, bytes]:
        sequence = cls.snapshot_message_sequence(data)
        descriptor = json.loads(data["messages"])
        raw, messages = decode_model_message_payloads(payloads)
        if len(messages) != sequence.count:
            raise ValueError("snapshot message references are incomplete")
        data = dict(data)
        data.pop("sequence_root")
        # Validate native control fields without sending canonical host/model
        # values back through the SDK's narrower message union.
        data["messages"] = []
        snapshot = replace(_SNAPSHOT_ADAPTER.validate_python(cls._record_data(data)), messages=list(messages))
        return NativeCheckpoint.restore_descriptor(snapshot, descriptor["checkpoint"]), raw

    def _table(self, name: str) -> "Table":
        return self._metadata.tables[f"ai_step_{name}"]

    def _run_values(self, run_id: str) -> "dict[str, SqlValue]":
        validate_resource_id(run_id)
        return {"store_key": self._store_key, "store_locator": self._store_locator, "run_id": run_id}

    def _run_predicates(self, table: "Table", run_id: str) -> "tuple[ColumnElement[bool], ...]":
        validate_resource_id(run_id)
        return (table.c.store_locator == self._store_locator, table.c.store_key == self._store_key, table.c.run_id == run_id)

    @classmethod
    def _event_values(cls, event: StepEvent) -> dict[str, object]:
        validate_native_step_ids(run_id=event.run_id, conversation_id=event.conversation_id,
                                 parent_run_id=event.parent_run_id)
        validate_native_replay_key(event.idempotency_key)
        if event.tool_call_id is not None:
            validate_resource_id(event.tool_call_id)
        values = {field.name: getattr(event, field.name) for field in fields(event)}
        values["timestamp"] = event.timestamp.isoformat()
        values["metadata"] = dict(event.metadata)
        return values

    @classmethod
    def _record_values(cls, record: RunRecord | StepEvent | ContinuableSnapshot) -> "dict[str, SqlValue]":
        validate_native_step_ids(run_id=record.run_id, conversation_id=record.conversation_id,
                                 parent_run_id=record.parent_run_id)
        if isinstance(record, (StepEvent, ContinuableSnapshot)):
            validate_native_replay_key(record.idempotency_key)
        values = {field.name: getattr(record, field.name) for field in fields(record) if field.name != "messages"}
        for name in ("timestamp", "started_at"):
            if name in values:
                values[name] = values[name].isoformat()
        if "metadata" in values:
            values["metadata"] = json.dumps(values["metadata"])
        if "idempotency_key" in values:
            values["idempotency_key"] = encode_native_replay_key(record.idempotency_key)
        return values

    @classmethod
    def _validate_record_ids(cls, row: Mapping[str, object]) -> None:
        try:
            validate_native_step_ids(run_id=row.get("run_id"), conversation_id=row.get("conversation_id"),
                                     parent_run_id=row.get("parent_run_id"))
            if "idempotency_key" in row:
                value = row["idempotency_key"]
                validate_native_replay_key(decode_native_replay_key(value) if isinstance(value, bytes) else value)
            if "store_key" in row:
                validate_native_store_key(row["store_key"])
                if "store_locator" in row and row["store_locator"] != native_store_locator(row["store_key"]):
                    raise ValueError("native store locator differs from raw identity")
        except (AIError, TypeError, ValueError) as error:
            # Persisted identities are integrity evidence, not caller input.
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    @classmethod
    def _record_data(cls, row: "Mapping[str, object] | RowMapping") -> dict[str, object]:
        data = dict(row)
        if "idempotency_key" in data:
            data["idempotency_key"] = decode_native_replay_key(data["idempotency_key"])
        cls._validate_record_ids(data)
        if "metadata" in data:
            data["metadata"] = json.loads(data["metadata"])
        return data


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _message_body(value: object, *, sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


def _snapshot_descriptor_json(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value)
    # MySQL without strict mode may truncate bounded columns instead of failing.
    if len(encoded.encode("utf-8")) > 1024:
        raise ValueError("snapshot message descriptor exceeds 1024 UTF-8 bytes")
    return encoded



__all__ = ["PreparedStep", "SqlStepStore", "StepWriteReceipt", "StepContentCleanupResult"]
