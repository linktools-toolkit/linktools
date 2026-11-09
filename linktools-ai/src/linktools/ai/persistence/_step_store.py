#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy adaptation of the public upstream StepStore protocol."""

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
from itertools import groupby
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai_harness.media import MediaStore, externalize_media, restore_media
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
from ._archive import ImmutableSequenceNode
from ._native_checkpoint import NativeCheckpoint
from ._step_maintenance import StepContentCleanupResult, purge_step_content
from ._step_sequence import build_step_sequence, step_sequence_contents
from ._step_schema import build_step_sql_metadata
from .state import SqlStateTransaction, StateTransaction

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
_SNAPSHOT_ADAPTER = TypeAdapter(ContinuableSnapshot)
_BATCH_SIZE = 128
_RUN_CACHE_SIZE = 128
_RUN_CACHE_BYTES = 1024 * 1024
_EVENT_RUN_FIELDS = ("conversation_id", "parent_run_id", "agent_name", "metadata")


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
    messages: tuple[tuple[str, str], ...]
    sequences: tuple[tuple[str, ImmutableSequenceNode], ...]


@dataclass(frozen=True, slots=True)
class StepWriteReceipt:
    """Provisional body identities and run fields, confirmed by the outer transaction."""

    message_digests: tuple[str, ...]
    sequence_digests: tuple[str, ...]
    run_fields: _RunFields | None = None


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

    def __init__(
        self,
        engine: "AsyncEngine | SqlStorageContext",
        *,
        namespace: str,
        tenant_id: str,
        tool_effect_store: StepStore | None = None,
        media_store: MediaStore | None = None,
        media_threshold_bytes: int = 64 * 1024,
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
        self._media_store = media_store
        self._media_threshold_bytes = media_threshold_bytes
        self._max_snapshots_per_run = max_snapshots_per_run
        self._initialized = False
        self._max_cached_messages = max_cached_messages
        self._confirmed_messages: OrderedDict[str, None] = OrderedDict()
        self._confirmed_sequences: OrderedDict[str, None] = OrderedDict()
        self._confirmed_runs: OrderedDict[str, _RunFields] = OrderedDict()

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
                result = await purge_step_content(self._context, self._metadata, self._store_digest)
            finally:
                # Cancellation or a lost commit acknowledgement must not leave
                # the caller believing an old body still exists after reopening.
                self._confirmed_messages.clear()
                self._confirmed_sequences.clear()
            _logger.info("native content reclaimed: sequences=%s messages=%s",
                         result.deleted_sequences, result.deleted_messages)
            return result

    async def register_run(self, record: RunRecord) -> None:
        await self.initialize()

        async def write(session: "AsyncSession") -> _RunFields | None:
            return await self._write_run(session, record)

        source = await self._context.run_mutation(write, domain="steps.run")
        if source is not None:
            self._confirm_run(source)

    async def _write_run(
        self, session: "AsyncSession", record: RunRecord, *, replay: bool = False,
    ) -> _RunFields | None:
        values = self._run_values(record.run_id) | self._record_values(record)
        values["conversation_digest"] = _optional_digest(record.conversation_id)
        values["parent_digest"] = _optional_digest(record.parent_run_id)
        result = await self._context.dialect.insert_ignore_conflict(
            session, table=self._table("runs"), values=values,
            index_elements=("store_digest", "run_digest"),
        )
        if not result.inserted:
            if replay and record.registration_id is not None:
                from sqlalchemy import select

                table = self._table("runs")
                registration = await session.scalar(select(table.c.registration_id).where(
                    *self._run_predicates(table, record.run_id),
                ))
                if registration == record.registration_id:
                    return None
            raise ValueError(
                f"run_id {record.run_id!r} is already in the store; run_id is single-shot"
            )
        return _RunFields.from_row(values, engine=self._context.engine)

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

        async def write(session: "AsyncSession") -> None:
            await self._write_events(session, (event,))

        await self._context.run_mutation(write, domain="steps.event")

    async def _write_events(
        self, session: "AsyncSession", events: Sequence[StepEvent], *, source: _RunFields | None = None,
    ) -> None:
        for keyed, group in groupby(events, key=lambda event: event.idempotency_key is not None):
            rows = []
            for event in group:
                row = self._run_values(event.run_id) | self._event_values(event)
                known = source if source is not None and source.run_id == event.run_id else self._confirmed_runs.get(event.run_id)
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
            batch_size = _BATCH_SIZE if keyed else 1
            for offset in range(0, len(rows), batch_size):
                if not keyed:
                    await session.execute(self._table("events").insert().values(rows[offset]))
                    continue
                await self._context.dialect.insert_ignore_conflict_many(
                    session, table=self._table("events"), rows=rows[offset:offset + batch_size],
                    index_elements=("key_digest",),
                )

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        from sqlalchemy import and_, case, cast, or_, select, Text

        await self.initialize()
        table, runs = self._table("events"), self._table("runs")
        columns = {field.name: table.c[field.name] for field in fields(StepEvent)}
        columns["metadata"] = cast(table.c.metadata, Text)
        invalid = [table.c.run_fields.bitwise_and((1 << len(_EVENT_RUN_FIELDS)) - 1) != table.c.run_fields,
                   and_(table.c.run_fields != 0, runs.c.id.is_(None))]
        for bit, name in enumerate(_EVENT_RUN_FIELDS):
            referenced = table.c.run_fields.bitwise_and(1 << bit) != 0
            columns[name] = case((referenced, runs.c[name]), else_=columns[name]).label(name)
            invalid.append(and_(referenced, or_(table.c[name].is_not(None), runs.c[name].is_(None))))
        query = select(
            *columns.values(), or_(*invalid).label("invalid_run_reference"),
        ).select_from(table.outerjoin(runs, and_(
            table.c.run_fields != 0,
            runs.c.store_digest == table.c.store_digest,
            runs.c.run_digest == table.c.run_digest,
            runs.c.run_id == table.c.run_id,
        ))).where(*self._run_predicates(table, run_id)).order_by(table.c.id)
        async with self._context.sessions() as session:
            rows = (await self._context.run_statement(session.execute(query))).mappings().all()
        if any(row["invalid_run_reference"] for row in rows):
            raise ValueError("event run reference is missing or corrupt")
        return [_EVENT_ADAPTER.validate_python(self._record_data(row)) for row in rows]

    def confirmed_run(self, run_id: str) -> RunRecord | None:
        """Return detached committed run proof when retained, without reading SQL."""
        source = self._confirmed_runs.get(run_id)
        if source is None or self._context.closed:
            return None
        data = dict(zip(_EVENT_RUN_FIELDS, source.values))
        data.update(run_id=source.run_id, registration_id=source.registration_id, started_at=source.started_at)
        return _RUN_ADAPTER.validate_python(self._record_data(data))

    @property
    def store_digest(self) -> str:
        """Physical namespace and tenant partition of these native records."""
        return self._store_digest

    @property
    def storage_context(self) -> SqlStorageContext:
        """The borrowed physical context used to verify same-group composition."""
        return self._context

    async def prepare_step(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> PreparedStep:
        """Capture native records and perform encoding/media work before SQL."""
        await self.initialize()
        run, events, snapshot = deepcopy((run, tuple(events), snapshot))
        values, messages, sequences = (None, (), ()) if snapshot is None else await self._prepare_snapshot(snapshot)
        return PreparedStep(run, events, snapshot, values, tuple(messages), sequences)

    async def write_prepared_step(
        self, transaction: StateTransaction, prepared: PreparedStep,
    ) -> StepWriteReceipt:
        """Join the existing same-engine transaction without opening another."""
        if not isinstance(transaction, SqlStateTransaction):
            raise TypeError("native SQL records require a SQL state transaction")
        return await transaction.run_sql(
            self._context, lambda session: self._write_prepared_step(session, prepared),
        )

    async def _write_prepared_step(
        self, session: "AsyncSession", prepared: PreparedStep,
    ) -> StepWriteReceipt:
        source = None if prepared.run is None else await self._write_run(session, prepared.run, replay=True)
        await self._write_events(session, prepared.events, source=source)
        receipt = StepWriteReceipt((), ())
        if prepared.snapshot is not None and prepared.snapshot_values is not None:
            receipt = await self._write_snapshot(
                session, prepared.snapshot_values, prepared.messages, prepared.sequences,
            )
        return replace(receipt, run_fields=source)

    def confirm_prepared_step(self, receipt: StepWriteReceipt) -> None:
        """Advance immutable source knowledge after the enclosing commit is confirmed."""
        if receipt.run_fields is not None:
            self._confirm_run(receipt.run_fields)
        self._confirm_records(self._confirmed_messages, receipt.message_digests)
        self._confirm_records(self._confirmed_sequences, receipt.sequence_digests)

    def _confirm_run(self, source: _RunFields) -> None:
        if source.engine is not self._context.engine or source.store_digest != self._store_digest:
            raise ValueError("run receipt belongs to another native store")
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

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self.initialize()
        prepared = await self._prepare_snapshot(snapshot)

        async def write(session: "AsyncSession") -> StepWriteReceipt:
            return await self._write_snapshot(session, *prepared)

        receipt = await self._context.run_mutation(write, domain="steps.snapshot")
        self.confirm_prepared_step(receipt)

    async def _prepare_snapshot(
        self, snapshot: ContinuableSnapshot,
    ) -> "tuple[dict[str, SqlValue], list[tuple[str, str]], tuple[tuple[str, ImmutableSequenceNode], ...]]":
        if isinstance(snapshot, NativeCheckpoint):
            snapshot = NativeCheckpoint.encode(snapshot, output_retries_used=snapshot.output_retries_used)
        values = self._record_values(snapshot)
        decoded = NativeCheckpoint.decode(snapshot)
        checkpoint = None
        if isinstance(decoded, NativeCheckpoint):
            last = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages[-1:]))[0]
            checkpoint = {
                "output_retries_used": decoded.output_retries_used,
                "message_sha256": _digest(_message_body(last, sort_keys=True)),
            }
        messages = json.loads(ModelMessagesTypeAdapter.dump_json(decoded.messages))
        if self._media_store is not None:
            messages = await externalize_media(
                messages, media_store=self._media_store, threshold_bytes=self._media_threshold_bytes,
            )
        if not isinstance(messages, list):
            raise ValueError("snapshot messages must be a list")
        keyed = []
        for message in messages:
            body = _message_body(message)
            keyed.append((_digest(body), body))
        values["messages"] = _snapshot_descriptor_json({"stored_messages": len(keyed), "checkpoint": checkpoint})
        values["pruned"] = False
        values["sequence_root"], sequences = build_step_sequence(tuple(key for key, _ in keyed))

        return values, keyed, sequences

    async def _write_snapshot(
        self, session: "AsyncSession", values: "dict[str, SqlValue]", keyed: Sequence[tuple[str, str]],
        sequences: Sequence[tuple[str, ImmutableSequenceNode]],
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
                return StepWriteReceipt((), ())
            snapshot_id = result.row_id
        if snapshot_id is None:
            raise RuntimeError("snapshot insertion did not return its physical identity")
        message_rows = []
        for key, body in sorted(dict(keyed).items()):
            if key not in self._confirmed_messages:
                message_rows.append({"store_digest": self._store_digest, "sha256": key, "message": body})
        for offset in range(0, len(message_rows), _BATCH_SIZE):
            await self._context.dialect.insert_ignore_conflict_many(
                session, table=self._table("messages"),
                rows=message_rows[offset:offset + _BATCH_SIZE],
                index_elements=("store_digest", "sha256"),
            )
        sequence_rows = [
            {"store_digest": self._store_digest, "sha256": identity,
             "message_sha256": node.content_id, "tree_count": node.tree_count,
             "left_sha256": node.left, "right_sha256": node.right, "next_sha256": node.next_forest}
            for identity, node in sorted(sequences) if identity not in self._confirmed_sequences
        ]
        for offset in range(0, len(sequence_rows), _BATCH_SIZE):
            await self._context.dialect.insert_ignore_conflict_many(
                session, table=self._table("sequences"),
                rows=sequence_rows[offset:offset + _BATCH_SIZE],
                index_elements=("store_digest", "sha256"),
            )
        await self._prune_snapshots(session, values["run_id"])
        return StepWriteReceipt(tuple(dict(keyed)), tuple(identity for identity, _ in sequences))

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

    async def _load_snapshots(
        self, run_id: str, *, include_interrupted: bool, latest: bool,
    ) -> list[tuple[dict[str, object], list[Mapping[str, object]]]]:
        from sqlalchemy import select

        await self.initialize()
        table = self._table("snapshots")
        query = select(table).where(*self._run_predicates(table, run_id), table.c.pruned.is_(False))
        if not include_interrupted:
            query = query.where(table.c.state == "complete")
        query = query.order_by(table.c.id.desc() if latest else table.c.id)
        if latest:
            query = query.limit(1)
        async with self._context.sessions() as session, session.begin():
            await self._context.run_statement(self._context.dialect.begin_consistent_read(session))
            rows = (await self._context.run_statement(session.execute(query))).mappings().all()
            histories: dict[int, list[Mapping[str, object]]] = {}
            for offset in range(0, len(rows), _BATCH_SIZE):
                selected = rows[offset:offset + _BATCH_SIZE]
                query = self._sequence_query(selected)
                if query is None:
                    continue
                found = (await self._context.run_statement(session.execute(query))).mappings().all()
                for item in found:
                    histories.setdefault(item["snapshot_id"], []).append(item)
        return [(self._record_data(row), histories.get(row["id"], [])) for row in rows]

    def _sequence_query(self, snapshots: "Sequence[RowMapping]") -> "Select | None":
        """Expand bounded immutable trees and their bodies in one SQL statement."""
        from sqlalchemy import BigInteger, and_, case, cast, literal, select, union_all

        starts = []
        max_depth = 0
        for snapshot in snapshots:
            try:
                descriptor = json.loads(snapshot["messages"])
            except (TypeError, ValueError):
                continue
            count = descriptor.get("stored_messages") if isinstance(descriptor, dict) else None
            root = snapshot["sequence_root"]
            if type(count) is not int or not 0 <= count <= (1 << 63) - 1 or (root is None) != (count == 0):
                continue
            if root is None:
                continue
            max_depth = max(max_depth, *(index + size.bit_length() - 1
                for index, size in enumerate(ImmutableSequenceNode.forest_weights(count))))
            starts.append(select(
                literal(snapshot["id"], BigInteger).label("snapshot_id"),
                literal(root, self._table("sequences").c.sha256.type).label("sha256"),
                literal(count, BigInteger).label("remaining"),
                literal(0, BigInteger).label("depth"),
                literal(True).label("forest"),
            ))
        if not starts:
            return None
        seeds = (starts[0] if len(starts) == 1 else union_all(*starts)).subquery("snapshot_roots")
        nodes = self._table("sequences")
        edges = union_all(*(select(literal(index).label("edge")) for index in range(3))).subquery("sequence_edges")

        def extend(frontier):
            child = case((edges.c.edge == 0, nodes.c.left_sha256),
                         (edges.c.edge == 1, nodes.c.right_sha256), else_=nodes.c.next_sha256)
            size = case((edges.c.edge == 2, frontier.c.remaining - nodes.c.tree_count),
                        else_=cast((nodes.c.tree_count - 1) / 2, BigInteger))
            return select(
                frontier.c.snapshot_id, child.label("sha256"), size.label("remaining"),
                (frontier.c.depth + 1).label("depth"), (edges.c.edge == 2).label("forest"),
            ).select_from(frontier.join(nodes, and_(
                nodes.c.store_digest == self._store_digest, nodes.c.sha256 == frontier.c.sha256,
            )).join(edges, literal(True))).where(
                child.is_not(None), frontier.c.depth < max_depth,
                nodes.c.tree_count > 0, nodes.c.tree_count <= frontier.c.remaining,
                (edges.c.edge < 2) | frontier.c.forest, size > 0,
            )

        if self._context.engine.dialect.name == "mysql":
            # MySQL 5.7 has no recursive CTEs. Every derived frontier is bounded
            # by the exact tree height; DISTINCT prevents derived-table merging.
            levels = [seeds]
            for depth in range(max_depth):
                levels.append(extend(levels[-1]).distinct().subquery(f"sequence_level_{depth + 1}"))
            closure = (select(levels[0]) if len(levels) == 1 else
                       union_all(*(select(level) for level in levels))).subquery("snapshot_sequences")
        else:
            closure = select(seeds).cte("snapshot_sequences", recursive=True)
            closure = closure.union_all(extend(closure))
        messages = self._table("messages")
        return select(
            closure.c.snapshot_id, nodes.c.sha256, nodes.c.message_sha256,
            nodes.c.tree_count, nodes.c.left_sha256, nodes.c.right_sha256, nodes.c.next_sha256,
            messages.c.message,
        ).select_from(closure.join(nodes, and_(
            nodes.c.store_digest == self._store_digest, nodes.c.sha256 == closure.c.sha256,
        )).outerjoin(messages, and_(
            messages.c.store_digest == self._store_digest, messages.c.sha256 == nodes.c.message_sha256,
        )))

    async def _snapshot_from_data(
        self, data: dict[str, object], references: list[Mapping[str, object]],
    ) -> ContinuableSnapshot:
        descriptor = json.loads(data["messages"])
        if (not isinstance(descriptor, dict) or set(descriptor) != {"stored_messages", "checkpoint"}
                or type(descriptor["stored_messages"]) is not int or descriptor["stored_messages"] < 0):
            raise ValueError("snapshot message references are incomplete")
        nodes: dict[str, ImmutableSequenceNode] = {}
        bodies: dict[str, object] = {}
        for row in references:
            try:
                node = ImmutableSequenceNode(row["message_sha256"], row["tree_count"],
                                             row["left_sha256"], row["right_sha256"], row["next_sha256"])
            except AIError as error:
                raise ValueError("snapshot sequence is missing or corrupt") from error
            body = row["message"]
            if node.identity != row["sha256"] or body is None or _digest(body) != node.content_id:
                raise ValueError("snapshot message references are missing or corrupt")
            nodes[node.identity] = node
            bodies[node.content_id] = json.loads(body)
        contents = step_sequence_contents(data.pop("sequence_root"), descriptor["stored_messages"], nodes)
        messages = [bodies[identity] for identity in contents]
        if self._media_store is not None:
            messages = await restore_media(messages, media_store=self._media_store)
        data["messages"] = ModelMessagesTypeAdapter.validate_python(messages)
        snapshot = _SNAPSHOT_ADAPTER.validate_python(data)
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
        last = json.loads(ModelMessagesTypeAdapter.dump_json(restored.messages[-1:]))[0]
        if _digest(_message_body(last, sort_keys=True)) != checkpoint["message_sha256"]:
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
