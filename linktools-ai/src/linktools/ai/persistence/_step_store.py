#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy adaptation of the public upstream StepStore protocol."""

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
from itertools import groupby
from typing import TYPE_CHECKING
from uuid import uuid4
from weakref import WeakKeyDictionary, WeakSet

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

from ..errors import AIError
from ..storage import SqlStorageContext, create_sql_storage_context, namespace_digest
from ._archive import ContentArchive, PreparedArchive
from ._message_root import NativeMessageRoot, NativeMessageSequence, NativeMessageSlice
from ._step_message_content import (
    CommittedNativeMessages, NativeMessageContent, NativeMessageReceipt, PreparedMessagePayload,
    PreparedNativeMessages,
)
from ._transcript_staging import MessageArchive, MessageSnapshot
from ._payload import encode_payload_bytes
from ..model.messages import combine_model_message_payloads, decode_model_message_payloads, encode_model_messages
from ..storage import StoredPayload
from ._native_checkpoint import NativeCheckpoint
from ._step_maintenance import StepContentCleanupResult, purge_step_content
from ._step_schema import build_step_sql_metadata
from .state import SqlStateTransaction, SqlMutationObserver, StateStore, StateTransaction
from ..steps import NativeLifecycleEvent

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
    "outcome", "store_digest", "run_digest", "key_digest", "run_fields", "body_sha256", "fields_sha256",
)


@dataclass(frozen=True, slots=True)
class _RunFields:
    """Detached native fields from a successful insert or an existing run read."""

    engine: "AsyncEngine"
    store_digest: str
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
            row["store_digest"], row["run_id"], row["registration_id"], row["started_at"], *values,
        ) if isinstance(value, str))
        return cls(engine, row["store_digest"], row["run_id"], row["registration_id"], row["started_at"], values, byte_size)


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """Native boundary records and encoded messages prepared outside SQL."""

    run: RunRecord | None
    events: tuple[StepEvent, ...]
    snapshot: ContinuableSnapshot | None
    snapshot_values: "dict[str, SqlValue] | None"
    messages: PreparedNativeMessages | None
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
        store_digest = self.scope_digest(namespace=namespace, tenant_id=tenant_id)
        if max_snapshots_per_run is not None and (
            not isinstance(max_snapshots_per_run, int) or max_snapshots_per_run < 1
        ):
            raise ValueError("max_snapshots_per_run must be an int >= 1 or None")
        if not isinstance(max_cached_messages, int) or max_cached_messages < 0:
            raise ValueError("max_cached_messages must be an int >= 0")
        self._context = engine if isinstance(engine, SqlStorageContext) else create_sql_storage_context(engine)
        self._owns_context = not isinstance(engine, SqlStorageContext)
        self._metadata = build_step_sql_metadata()
        self._store_digest = store_digest
        self._tool_effect_store = tool_effect_store
        self._max_snapshots_per_run = max_snapshots_per_run
        self._initialized = False
        self._max_cached_messages = max_cached_messages
        self._confirmed_messages: OrderedDict[str, None] = OrderedDict()
        self._confirmed_sequences: OrderedDict[str, None] = OrderedDict()
        self._message_content = NativeMessageContent(self._context, store_digest=store_digest,
            max_cached_messages=max_cached_messages, max_cached_nodes=max_cached_messages)
        self._message_archive: ContentArchive | None = None
        self._message_proofs: OrderedDict[NativeMessageSequence, CommittedNativeMessages] = OrderedDict()
        self._message_integrity: OrderedDict[NativeMessageSequence, tuple[str, int]] = OrderedDict()
        self._prepared_integrity: WeakKeyDictionary[PreparedNativeMessages, tuple[str, int]] = WeakKeyDictionary()
        self._message_inputs: OrderedDict[str, tuple[str, int, int, NativeMessageSequence]] = OrderedDict()
        self._frozen_messages = MessageArchive()
        self._confirmed_runs: OrderedDict[str, _RunFields] = OrderedDict()
        self._instances.add(self)
        self._receipt_sink: ContextVar[list[StepWriteReceipt] | None] = ContextVar("step_write_receipts", default=None)

    @classmethod
    def scope_digest(cls, *, namespace: str, tenant_id: str) -> str:
        """Derive the native partition identity from its logical scope."""
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        return namespace_digest(json.dumps([namespace, tenant_id], separators=(",", ":")))

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
                result = await purge_step_content(self._context, self._metadata, self._store_digest,
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
        values = self._run_values(record.run_id) | self._record_values(record)
        values["conversation_digest"] = _optional_digest(record.conversation_id)
        values["parent_digest"] = _optional_digest(record.parent_run_id)
        result = await self._context.dialect.insert_ignore_conflict(
            session, table=self._table("runs"), values=values,
            index_elements=("store_digest", "run_digest"),
        )
        if not result.inserted:
            if replay and record.registration_id is not None:
                source_names = terminal_names = _EVENT_SOURCE_COLUMNS
                query = self._run_replay_statement(record, started=started)
                row = (await session.execute(query)).mappings().first()
                if row is not None and row["registration_id"] == record.registration_id:
                    source = _RunFields.from_row(row, engine=self._context.engine)
                    if started is not None:
                        event = self.decode_started_source(
                            {name: row["start_" + name] for name in source_names}, run=row,
                        )
                        source = self._with_started(source, event)
                        self._validate_started_replay(record, started, source)
                        if row["terminal_key_digest"] is not None:
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
        query = select(table.c.store_digest, table.c.run_id, table.c.registration_id, table.c.started_at,
                       *(table.c[name] for name in _EVENT_RUN_FIELDS))
        if started is not None:
            query = query.add_columns(table.c.run_digest,
                *(events.c[name].label("start_" + name) for name in _EVENT_SOURCE_COLUMNS))
            joined = table.outerjoin(events, and_(
                events.c.key_digest == self.event_key_digest(started),
                events.c.store_digest == table.c.store_digest, events.c.run_digest == table.c.run_digest,
                events.c.run_id == table.c.run_id,
            ))
            terminal = events.alias("terminal")
            key = _replay_digest(self._store_digest, _digest(record.run_id), NativeLifecycleEvent.source_key())
            joined = joined.outerjoin(terminal, and_(terminal.c.key_digest == key,
                terminal.c.store_digest == table.c.store_digest,
                terminal.c.run_digest == table.c.run_digest, terminal.c.run_id == table.c.run_id))
            query = query.add_columns(*(terminal.c[name].label("terminal_" + name) for name in _EVENT_SOURCE_COLUMNS))
            query = query.select_from(joined)
        return query.where(*self._run_predicates(table, record.run_id))

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

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None,
    ) -> list[RunRecord]:
        from sqlalchemy import select

        await self.initialize()
        table = self._table("runs")
        query = select(table).where(table.c.store_digest == self._store_digest)
        if parent_run_id is not None:
            query = query.where(table.c.parent_digest == _digest(parent_run_id), table.c.parent_run_id == parent_run_id)
        if conversation_id is not None:
            query = query.where(table.c.conversation_digest == _digest(conversation_id), table.c.conversation_id == conversation_id)
        async with self._context.sessions() as session:
            rows = (await self._context.run_statement(session.execute(query.order_by(table.c.id)))).mappings().all()
        records = [_RUN_ADAPTER.validate_python(self._record_data(row)) for row in rows]
        for row in rows:
            self._confirm_run(_RunFields.from_row(row, engine=self._context.engine))
        return sorted(records, key=lambda r: r.started_at)

    async def append_event(self, event: StepEvent) -> None:
        await self.initialize()

        async def write(session: "AsyncSession") -> _RunFields | None:
            if isinstance(event, NativeLifecycleEvent):
                return await self._write_lifecycle_event(session, event, source=self._confirmed_runs.get(event.run_id))
            await self._write_events(session, (event,))
            return None

        source = await self._context.run_mutation(write, domain="steps.event")
        if source is not None:
            self.confirm_prepared_step(StepWriteReceipt((), (), source))

    async def _write_events(
        self, session: "AsyncSession", events: Sequence[StepEvent], *, source: _RunFields | None = None,
        started: StepEvent | None = None,
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
                row["key_digest"] = _replay_digest(self._store_digest, row["run_digest"], row["idempotency_key"])
                rows.append(row)
            # An unkeyed append must fail rather than silently suppress a
            # colliding physical occurrence identity.
            batch_size = _BATCH_SIZE if keyed and not managed else 1
            for offset in range(0, len(rows), batch_size):
                if not keyed or managed:
                    await session.execute(self._table("events").insert().values(rows[offset]))
                    continue
                await self._context.dialect.insert_ignore_conflict_many(
                    session, table=self._table("events"), rows=rows[offset:offset + batch_size],
                    index_elements=("key_digest",),
                )

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
            table.c.store_digest, table.c.run_digest, table.c.key_digest,
        )
        registration_columns = ()
        if expected:
            proof_key = table.c.key_digest == self.event_key_digest(expected[0])
            registration_columns = tuple(case((proof_key, runs.c[field.name]), else_=None).label("proof_run_" + field.name)
                                         for field in fields(RunRecord))
        query = select(
            *columns.values(), table.c.body_sha256, table.c.fields_sha256, or_(*invalid).label("invalid_run_reference"),
            *source_columns, *registration_columns,
        ).select_from(table.outerjoin(runs, and_(
            *(() if expected else (table.c.run_fields != 0,)),
            runs.c.store_digest == table.c.store_digest,
            runs.c.run_digest == table.c.run_digest,
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
        if source.engine is not self._context.engine or source.store_digest != self._store_digest:
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

    def event_key_digest(self, event: StepEvent) -> str:
        """Exact native locator for a keyed event in this store's partition."""
        if event.idempotency_key is None:
            raise ValueError("an event locator requires an idempotency key")
        return _replay_digest(self._store_digest, _digest(event.run_id), event.idempotency_key)

    def started_from_receipt(self, receipt: StepWriteReceipt) -> StepEvent | None:
        """Detach the immutable start source; outer-transaction receipts are provisional."""
        self.run_from_receipt(receipt)
        source = receipt.run_fields
        if source is None or source.started_body is None:
            return None
        return _EVENT_ADAPTER.validate_python(json.loads(source.started_body))

    @classmethod
    def decode_event(cls, data: Mapping[str, object]) -> StepEvent:
        """Restore explicit lifecycle fields without changing ordinary upstream inputs."""
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
        return self.decode_started_row(event, expected_store_digest=self._store_digest, run=run)

    @classmethod
    def decode_started_row(
        cls, event: Mapping[str, object], *, expected_store_digest: str,
        run: Mapping[str, object] | None = None,
    ) -> StepEvent:
        """Validate joined source columns without constructing or opening a store."""
        data = dict(event)
        if "outcome" not in data:
            raise ValueError("native lifecycle source outcome is missing")
        if data.get("store_digest") != expected_store_digest:
            raise ValueError("native start source is missing or belongs to another store")
        run_id = data.get("run_id")
        if not isinstance(run_id, str) or data.get("run_digest") != _digest(run_id):
            raise ValueError("native start source run identity is corrupt")
        mask = data.get("run_fields")
        if type(mask) is not int or not 0 <= mask < (1 << len(_EVENT_RUN_FIELDS)):
            raise ValueError("native start source run fields are corrupt")
        if mask:
            if run is None or any(run.get(name) != data.get(name) for name in ("store_digest", "run_digest", "run_id")):
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
        locator = _replay_digest(expected_store_digest, _digest(result.run_id), result.idempotency_key)
        if (data.get("key_digest") != locator or data.get("body_sha256") != _digest(body)
                or data.get("fields_sha256") != _digest(cls._started_fields_body(result))):
            raise ValueError("native start source body or locator is corrupt")
        return result

    @classmethod
    def decode_started_fields(
        cls, event: Mapping[str, object], *, expected_store_digest: str,
    ) -> StepEvent:
        """Verify a metadata-free Trace projection, never a complete source proof.

        Supply SQL-resolved event fields except metadata, plus native scope,
        run/key digests and fields_sha256. Metadata is intentionally not read or
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
        data["metadata"] = {}
        result = cls.decode_event(data)
        raw_body = json.dumps({name: event[name] for name in names}, ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"))
        body = cls._started_fields_body(result)
        locator = _replay_digest(expected_store_digest, _digest(result.run_id), result.idempotency_key)
        if (event.get("store_digest") != expected_store_digest
                or event.get("run_digest") != _digest(result.run_id)
                or event.get("key_digest") != locator
                or event.get("fields_sha256") != _digest(body) or raw_body != body):
            raise ValueError("native start projected fields or locator are corrupt")
        return result

    @property
    def store_digest(self) -> str:
        """Physical namespace and tenant partition of these native records."""
        return self._store_digest

    @property
    def storage_context(self) -> SqlStorageContext:
        """The borrowed physical context used to verify same-group composition."""
        return self._context

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
        value = await self.prepare_message_content(self._frozen_messages.snapshot(messages))
        prepared.native_messages[value.root] = value
        prepared.native_publishers[value.root] = self.publish_archived_messages
        return self.message_root(value.root)

    async def publish_archived_messages(self, transaction: StateTransaction, prepared: PreparedArchive) -> None:
        """Publish dependencies inside the owner's atomic mutation, then observe commit."""
        if not prepared.native_messages:
            return
        if not isinstance(transaction, SqlMutationObserver):
            raise ValueError("native messages require an observable SQL mutation")
        receipts = await self.publish_message_content(transaction, prepared, owner_written=True)
        transaction.observe_mutation(lambda committed: (
            self.confirm_message_content(receipts) if committed else self.invalidate_message_content()
        ))

    def bind_message_archive(self, archive: ContentArchive) -> None:
        """Bind the already opened physical group scope and its object route."""
        from ._sql import SqlStateStorageGroup

        group = archive.store.storage_group
        if not isinstance(group, SqlStateStorageGroup) or group.context.engine is not self._context.engine:
            raise ValueError("native message archive belongs to another SQL group")
        self._message_content.bind_group(archive.scope_id)
        self._message_archive = archive
        archive.bind_message_publisher(self.publish_archived_messages)

    def message_root(self, sequence: NativeMessageSequence) -> NativeMessageRoot:
        return self._message_content.scoped_root(sequence)

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
        self._prepared_integrity.clear()
        self._message_inputs.clear()

    async def _prepare_message_payload(self, raw: bytes) -> PreparedMessagePayload:
        if self._message_archive is None:
            compression, _digest, _size, encoded = encode_payload_bytes(raw)
            payload = StoredPayload.inline_bytes(encoded)
        else:
            compression, payload, encoded = await self._message_archive.prepare_payload(raw)
        return PreparedMessagePayload(raw, encoded=encoded, compression=compression, payload=payload)

    async def prepare_message_content(
        self, messages: MessageSnapshot, *,
        baseline: tuple[NativeMessageRoot | NativeMessageSequence, str, int] | None = None,
        single_sequence: bool = False,
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
            if proof is not None and (not single_sequence or complete) and sequence.count <= len(messages):
                prefix = messages[:sequence.count]
                if prefix.encoded_digest() == (digest, size):
                    base, start = proof, sequence.count
        prepared = await self._message_content.prepare_append(
            messages[start:], prepare_payload=self._prepare_message_payload, base=base,
        )
        self._prepared_integrity[prepared] = messages.encoded_digest()
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
                self._remember_message_proof(proof, integrity=self._prepared_integrity.get(receipt.prepared))

    def _remember_message_proof(
        self, proof: CommittedNativeMessages, *, integrity: tuple[str, int] | None = None,
    ) -> None:
        if not self._max_cached_messages:
            return
        self._message_proofs[proof.root] = proof
        self._message_proofs.move_to_end(proof.root)
        if integrity is not None:
            self._message_integrity[proof.root] = integrity
        while len(self._message_proofs) > _RUN_CACHE_SIZE:
            root, _proof = self._message_proofs.popitem(last=False)
            self._message_integrity.pop(root, None)

    def can_reuse_message_root(self, root: NativeMessageRoot, *, raw_digest: str, raw_size: int) -> bool:
        """Reuse only an actual same-group committed/readback integrity proof."""
        try:
            self._message_content.validate_root(root)
        except ValueError:
            return False
        for source in tuple(self._instances):
            if (source.storage_context.engine is not self.storage_context.engine
                    or source.store_digest != self.store_digest):
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
        """Read all selected native bodies in one query, resolving objects after SQL."""
        if not roots:
            return ()
        query = self._message_content.range_query(roots)
        async def read(session: "AsyncSession") -> list[Mapping[str, object]]:
            return list((await session.execute(query)).mappings().all())
        rows = []
        if query is not None:
            query = query.statement
            async with self._context.sessions() as session, session.begin():
                await self._context.run_statement(self._context.dialect.begin_consistent_read(session))
                rows = await self._context.run_statement(read(session))
        readbacks = await self._message_content.read_hydrated_ranges(
            roots, rows, read_object=(None if self._message_archive is None
                                     else self._message_archive.read_payload_object),
        )
        for result in readbacks:
            raw = combine_model_message_payloads(result.payloads)
            self._remember_message_proof(self._message_content.adopt_read(result),
                                         integrity=(hashlib.sha256(raw).hexdigest(), len(raw)))
        return tuple(result.payloads for result in readbacks)

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
        await self.initialize()
        run, events = deepcopy((run, tuple(events)))
        if snapshot is not None:
            snapshot = replace(snapshot, messages=self._frozen_messages.snapshot(snapshot.messages))
        values, messages = (None, None) if snapshot is None else await self._prepare_snapshot(snapshot)
        return PreparedStep(run, events, snapshot, values, messages, managed_start)

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
        return await transaction.run_sql(
            self._context, lambda session: self._write_prepared_step(
                session, prepared, confirmed=confirmed,
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
                await self._write_events(session, batch, source=source, started=fresh)
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
        if source.engine is not self._context.engine or source.store_digest != self._store_digest:
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
        receipt = await self._context.run_mutation(
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

        receipt = await self._context.run_mutation(write, domain="steps.snapshot")
        self.confirm_prepared_step(receipt)

    async def _prepare_snapshot(
        self, snapshot: ContinuableSnapshot,
    ) -> "tuple[dict[str, SqlValue], PreparedNativeMessages]":
        if isinstance(snapshot, NativeCheckpoint):
            snapshot = NativeCheckpoint.encode(snapshot, output_retries_used=snapshot.output_retries_used)
        values = self._record_values(snapshot)
        decoded = NativeCheckpoint.decode(snapshot)
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
        digest, size = messages.encoded_digest()
        self._message_inputs[snapshot.run_id] = (digest, size, len(messages), sequence)
        self._message_inputs.move_to_end(snapshot.run_id)
        while len(self._message_inputs) > _RUN_CACHE_SIZE:
            self._message_inputs.popitem(last=False)
        return values, prepared

    async def _write_snapshot(
        self, session: "AsyncSession", values: "dict[str, SqlValue]", prepared: PreparedNativeMessages,
    ) -> StepWriteReceipt:
        values = values | self._run_values(values["run_id"])
        values["key_digest"] = _replay_digest(self._store_digest, values["run_digest"], values["idempotency_key"])
        if values["idempotency_key"] is None:
            result = await session.execute(self._table("snapshots").insert().values(values))
            snapshot_id = result.inserted_primary_key[0]
        else:
            result = await self._context.dialect.insert_ignore_conflict(
                session, table=self._table("snapshots"), values=values,
                index_elements=("key_digest",),
            )
            if not result.inserted:
                # A conflicting owner key proves no candidate content, even if
                # another owner has already committed that candidate sequence.
                return StepWriteReceipt((), ())
            snapshot_id = result.row_id
        if snapshot_id is None:
            raise RuntimeError("snapshot insertion did not return its physical identity")
        content = await self._message_content.publish_in_session(session, prepared, owner_inserted=True)
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
        return None if not snapshots else await self._snapshot_from_data(*snapshots[0])

    async def list_snapshots(
        self, *, run_id: str, include_interrupted: bool = False,
    ) -> list[ContinuableSnapshot]:
        """Return retained upstream snapshots in insertion order."""
        snapshots = await self._load_snapshots(run_id, include_interrupted=include_interrupted, latest=False)
        result = []
        for snapshot in snapshots:
            try:
                result.append(await self._snapshot_from_data(*snapshot))
            except Exception:
                _logger.warning("Skipping unparsable snapshot for run %s", run_id, exc_info=True)
        return result

    def with_tool_effect_store(self, store: StepStore) -> "SqlStepStore":
        """Borrow this SQL partition with an attempt's existing effect owner."""
        bound = copy(self)
        bound._tool_effect_store = store
        bound._owns_context = False
        self._instances.add(bound)
        return bound

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._require_tool_effect_store().record_tool_effect(record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return await self._require_tool_effect_store().get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return await self._require_tool_effect_store().list_unresolved_tool_effects(run_id=run_id)

    def _require_tool_effect_store(self) -> StepStore:
        if self._tool_effect_store is None:
            raise RuntimeError("SqlStepStore requires a bound tool-effect store")
        return self._tool_effect_store

    async def _prune_snapshots(self, session: "AsyncSession", run_id: str) -> None:
        from sqlalchemy import select

        if self._max_snapshots_per_run is None:
            return
        table = self._table("snapshots")
        rows = (await session.execute(select(table.c.id, table.c.state).where(
            *self._run_predicates(table, run_id), table.c.pruned.is_(False),
        ).order_by(table.c.id))).all()
        retained = {row.id for row in rows[-self._max_snapshots_per_run:]}
        complete = [row.id for row in rows if row.state == "complete"]
        if complete:
            retained.add(complete[-1])
        removed = [row.id for row in rows if row.id not in retained]
        for offset in range(0, len(removed), _BATCH_SIZE):
            ids = removed[offset:offset + _BATCH_SIZE]
            await session.execute(table.update().where(
                table.c.store_digest == self._store_digest, table.c.id.in_(ids),
            ).values(pruned=True, messages="{}", sequence_root=None))

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
    ) -> list[tuple[dict[str, object], tuple[bytes, ...]]]:
        from sqlalchemy import select

        await self.initialize()
        table = self._table("snapshots")
        query = select(table).where(*self._run_predicates(table, run_id), table.c.pruned.is_(False))
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
                    if (row["store_digest"] != self._store_digest or row["run_id"] != run_id
                            or row["run_digest"] != _digest(run_id)
                            or row["idempotency_key"] is not None and row["key_digest"] !=
                               _replay_digest(self._store_digest, _digest(run_id), row["idempotency_key"])):
                        raise ValueError("snapshot owner identity is invalid")
                    data = self._record_data(row)
                    await self._snapshot_from_data(dict(data), readback.payloads)
                except (ValueError, AIError):
                    if latest:
                        raise
                    _logger.warning("Skipping corrupt snapshot for run %s", run_id)
                    continue
                verified.append(readback)
                results.append((data, readback.payloads))
        for readback in verified:
            raw = combine_model_message_payloads(readback.payloads)
            self._remember_message_proof(self._message_content.adopt_read(readback),
                                         integrity=(hashlib.sha256(raw).hexdigest(), len(raw)))
        return results

    def _sequence_query(self, snapshots: "Sequence[RowMapping]") -> "Select | None":
        sequences = tuple(self.snapshot_message_sequence(row) for row in snapshots)
        if not any(sequence.count for sequence in sequences):
            return None
        return self._message_content.range_query(sequences).statement

    async def _snapshot_from_data(
        self, data: dict[str, object], payloads: tuple[bytes, ...],
    ) -> ContinuableSnapshot:
        sequence = self.snapshot_message_sequence(data)
        descriptor = json.loads(data["messages"])
        _raw, messages = decode_model_message_payloads(payloads)
        if len(messages) != sequence.count:
            raise ValueError("snapshot message references are incomplete")
        data.pop("sequence_root")
        # Validate native control fields without sending canonical host/model
        # values back through the SDK's narrower message union.
        data["messages"] = []
        snapshot = replace(_SNAPSHOT_ADAPTER.validate_python(data), messages=list(messages))
        checkpoint = descriptor["checkpoint"]
        if checkpoint is None:
            return snapshot
        if (not isinstance(checkpoint, dict)
                or set(checkpoint) != {"output_retries_used", "message_sha256"}
                or type(checkpoint["output_retries_used"]) is not int
                or checkpoint["output_retries_used"] < 0
                or not isinstance(checkpoint["message_sha256"], str)):
            raise ValueError("snapshot checkpoint state is invalid")
        restored = NativeCheckpoint.encode(snapshot, output_retries_used=checkpoint["output_retries_used"])
        if hashlib.sha256(encode_model_messages(restored.messages[-1:])).hexdigest() != checkpoint["message_sha256"]:
            raise ValueError("snapshot checkpoint state is corrupt or misbound")
        return restored

    def _table(self, name: str) -> "Table":
        return self._metadata.tables[f"ai_step_{name}"]

    def _run_values(self, run_id: str) -> "dict[str, SqlValue]":
        return {"store_digest": self._store_digest, "run_digest": _digest(run_id), "run_id": run_id}

    def _run_predicates(self, table: "Table", run_id: str) -> "tuple[ColumnElement[bool], ...]":
        return (table.c.store_digest == self._store_digest, table.c.run_digest == _digest(run_id), table.c.run_id == run_id)

    @classmethod
    def _event_values(cls, event: StepEvent) -> dict[str, object]:
        values = {field.name: getattr(event, field.name) for field in fields(event)}
        values["timestamp"] = event.timestamp.isoformat()
        values["metadata"] = dict(event.metadata)
        return values

    @classmethod
    def _record_values(cls, record: RunRecord | StepEvent | ContinuableSnapshot) -> "dict[str, SqlValue]":
        values = {field.name: getattr(record, field.name) for field in fields(record) if field.name != "messages"}
        for name in ("timestamp", "started_at"):
            if name in values:
                values[name] = values[name].isoformat()
        if "metadata" in values:
            values["metadata"] = json.dumps(values["metadata"])
        return values

    @classmethod
    def _record_data(cls, row: "Mapping[str, object] | RowMapping") -> dict[str, object]:
        data = dict(row)
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


def _optional_digest(value: str | None) -> str | None:
    return None if value is None else _digest(value)


def _replay_digest(store_digest: str, run_digest: str, value: str | None) -> str:
    identity = [store_digest, run_digest, "occurrence", uuid4().hex] if value is None else [store_digest, run_digest, "key", value]
    return _digest(json.dumps(identity))


__all__ = ["PreparedStep", "SqlStepStore", "StepWriteReceipt", "StepContentCleanupResult"]
