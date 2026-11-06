#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy adaptation of the public upstream StepStore protocol."""

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, fields
from itertools import groupby
from typing import TYPE_CHECKING

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

from ..storage import SqlStorageContext, create_sql_storage_context, namespace_digest
from ._step_schema import build_step_sql_metadata
from .state import SqlStateTransaction, StateTransaction

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.engine import RowMapping
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
    from sqlalchemy.sql.elements import ColumnElement

    from ..storage import SqlValue


_logger = environ.get_logger("ai.persistence.steps")
_RUN_ADAPTER = TypeAdapter(RunRecord)
_EVENT_ADAPTER = TypeAdapter(StepEvent)
_SNAPSHOT_ADAPTER = TypeAdapter(ContinuableSnapshot)
_BATCH_SIZE = 128


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """Native boundary records and encoded messages prepared outside SQL."""

    run: RunRecord | None
    events: tuple[StepEvent, ...]
    snapshot: ContinuableSnapshot | None
    snapshot_values: "dict[str, SqlValue] | None"
    messages: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class StepWriteReceipt:
    """Provisional body identities, confirmed only by the outer transaction."""

    message_digests: tuple[str, ...]


class SqlStepStore:
    """Persist upstream records in a fixed namespace and tenant partition.

    The engine is borrowed. Provision production tables through the migration
    boundary; ``initialize(local=True)`` may create a local SQLite schema.
    Tool-effect methods delegate to the caller's existing durable effect owner.
    Stored messages remain after snapshot pruning, as in the upstream Postgres
    backend, because independent concurrent saves may reuse their bodies.
    Confirmed message digests are cached only after commit. Direct deletion or
    replacement of message bodies requires quiescing and closing all adapters
    before maintenance. ``max_cached_messages=0`` disables this optimization.
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
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        if max_snapshots_per_run is not None and (
            not isinstance(max_snapshots_per_run, int) or max_snapshots_per_run < 1
        ):
            raise ValueError("max_snapshots_per_run must be an int >= 1 or None")
        if not isinstance(max_cached_messages, int) or max_cached_messages < 0:
            raise ValueError("max_cached_messages must be an int >= 0")
        self._context = engine if isinstance(engine, SqlStorageContext) else create_sql_storage_context(engine)
        self._owns_context = not isinstance(engine, SqlStorageContext)
        self._metadata = build_step_sql_metadata()
        self._store_digest = namespace_digest(json.dumps([namespace, tenant_id], separators=(",", ":")))
        self._tool_effect_store = tool_effect_store
        self._media_store = media_store
        self._media_threshold_bytes = media_threshold_bytes
        self._max_snapshots_per_run = max_snapshots_per_run
        self._initialized = False
        self._max_cached_messages = max_cached_messages
        self._confirmed_messages: OrderedDict[str, None] = OrderedDict()

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
        if self._owns_context:
            await self._context.close()

    async def register_run(self, record: RunRecord) -> None:
        await self.initialize()

        async def write(session: "AsyncSession") -> None:
            await self._write_run(session, record)

        await self._context.run_mutation(write, domain="steps.run")

    async def _write_run(
        self, session: "AsyncSession", record: RunRecord, *, replay: bool = False,
    ) -> None:
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
                    return
            raise ValueError(
                f"run_id {record.run_id!r} is already in the store; run_id is single-shot"
            )

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        from sqlalchemy import select

        await self.initialize()
        table = self._table("runs")
        async with self._context.sessions() as session:
            row = (await session.execute(select(table).where(*self._run_predicates(table, run_id)))).mappings().first()
        return None if row is None else _RUN_ADAPTER.validate_python(self._record_data(row))

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
            rows = (await session.execute(query.order_by(table.c.id))).mappings().all()
        return sorted((_RUN_ADAPTER.validate_python(self._record_data(row)) for row in rows), key=lambda r: r.started_at)

    async def append_event(self, event: StepEvent) -> None:
        await self.initialize()

        async def write(session: "AsyncSession") -> None:
            await self._write_events(session, (event,))

        await self._context.run_mutation(write, domain="steps.event")

    async def _write_events(self, session: "AsyncSession", events: Sequence[StepEvent]) -> None:
        for keyed, group in groupby(events, key=lambda event: event.idempotency_key is not None):
            rows = [self._run_values(event.run_id) | self._record_values(event)
                    | {"key_digest": _optional_digest(event.idempotency_key)} for event in group]
            # MySQL retries an entire driver batch after a duplicate. Unkeyed
            # events must remain single writes so an earlier prefix cannot repeat.
            batch_size = _BATCH_SIZE if keyed else 1
            for offset in range(0, len(rows), batch_size):
                await self._context.dialect.insert_ignore_conflict_many(
                    session, table=self._table("events"), rows=rows[offset:offset + batch_size],
                    index_elements=("store_digest", "run_digest", "key_digest"),
                )

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        from sqlalchemy import select

        await self.initialize()
        table = self._table("events")
        async with self._context.sessions() as session:
            rows = (await session.execute(select(table).where(
                *self._run_predicates(table, run_id),
            ).order_by(table.c.id))).mappings().all()
        return [_EVENT_ADAPTER.validate_python(self._record_data(row)) for row in rows]

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
        values, messages = (None, ()) if snapshot is None else await self._prepare_snapshot(snapshot)
        return PreparedStep(run, events, snapshot, values, tuple(messages))

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
        if prepared.run is not None:
            await self._write_run(session, prepared.run, replay=True)
        await self._write_events(session, prepared.events)
        confirmed = ()
        if prepared.snapshot is not None and prepared.snapshot_values is not None:
            confirmed = await self._write_snapshot(
                session, prepared.snapshot, prepared.snapshot_values, prepared.messages,
            )
        return StepWriteReceipt(confirmed)

    def confirm_prepared_step(self, receipt: StepWriteReceipt) -> None:
        """Advance body knowledge after the enclosing commit is confirmed."""
        self._confirm_messages(receipt.message_digests)

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

        async def write(session: "AsyncSession") -> tuple[str, ...]:
            return await self._write_snapshot(session, snapshot, *prepared)

        confirmed = await self._context.run_mutation(write, domain="steps.snapshot")
        self._confirm_messages(confirmed)

    async def _prepare_snapshot(
        self, snapshot: ContinuableSnapshot,
    ) -> "tuple[dict[str, SqlValue], list[tuple[str, str]]]":
        messages = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages))
        if self._media_store is not None:
            messages = await externalize_media(
                messages, media_store=self._media_store, threshold_bytes=self._media_threshold_bytes,
            )
        if not isinstance(messages, list):
            raise ValueError("snapshot messages must be a list")
        keyed = []
        for message in messages:
            body = json.dumps(message, separators=(",", ":"))
            keyed.append((_digest(body), body))
        values = self._run_values(snapshot.run_id) | self._record_values(snapshot)
        values["messages"] = json.dumps({"stored_messages": len(keyed)})
        values["key_digest"] = _optional_digest(snapshot.idempotency_key)
        values["pruned"] = False

        return values, keyed

    async def _write_snapshot(
        self, session: "AsyncSession", snapshot: ContinuableSnapshot,
        values: "dict[str, SqlValue]", keyed: Sequence[tuple[str, str]],
    ) -> tuple[str, ...]:
        result = await self._context.dialect.insert_ignore_conflict(
            session, table=self._table("snapshots"), values=values,
            index_elements=("store_digest", "run_digest", "key_digest"),
        )
        if not result.inserted:
            return ()
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
        ref_rows = [{"store_digest": self._store_digest, "snapshot_id": snapshot_id,
                     "position": position, "sha256": key} for position, (key, _) in enumerate(keyed)]
        for offset in range(0, len(ref_rows), _BATCH_SIZE):
            await session.execute(self._table("message_refs").insert(), ref_rows[offset:offset + _BATCH_SIZE])
        await self._prune_snapshots(session, snapshot.run_id)
        return tuple(dict(keyed))

    def _confirm_messages(self, confirmed: Sequence[str]) -> None:
        if self._max_cached_messages:
            for key in confirmed:
                self._confirmed_messages[key] = None
                self._confirmed_messages.move_to_end(key)
                if len(self._confirmed_messages) > self._max_cached_messages:
                    self._confirmed_messages.popitem(last=False)

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
        refs = self._table("message_refs")
        for offset in range(0, len(removed), _BATCH_SIZE):
            ids = removed[offset:offset + _BATCH_SIZE]
            await session.execute(refs.delete().where(refs.c.store_digest == self._store_digest, refs.c.snapshot_id.in_(ids)))
            await session.execute(table.update().where(
                table.c.store_digest == self._store_digest, table.c.id.in_(ids),
            ).values(pruned=True, messages="{}"))

    async def _load_snapshots(
        self, run_id: str, *, include_interrupted: bool, latest: bool,
    ) -> list[tuple[dict[str, object], list[tuple[int, str, str | None]]]]:
        from sqlalchemy import and_, select

        await self.initialize()
        table = self._table("snapshots")
        refs = self._table("message_refs")
        messages = self._table("messages")
        query = select(table).where(*self._run_predicates(table, run_id), table.c.pruned.is_(False))
        if not include_interrupted:
            query = query.where(table.c.state == "complete")
        query = query.order_by(table.c.id.desc() if latest else table.c.id)
        if latest:
            query = query.limit(1)
        async with self._context.sessions() as session, session.begin():
            await self._context.dialect.begin_consistent_read(session)
            rows = (await session.execute(query)).mappings().all()
            histories: dict[int, list[tuple[int, str, str | None]]] = {}
            for offset in range(0, len(rows), _BATCH_SIZE):
                ids = [row["id"] for row in rows[offset:offset + _BATCH_SIZE]]
                found = (await session.execute(select(
                    refs.c.snapshot_id, refs.c.position, refs.c.sha256, messages.c.message,
                ).select_from(refs.outerjoin(messages, and_(
                    messages.c.store_digest == refs.c.store_digest,
                    messages.c.sha256 == refs.c.sha256,
                ))).where(refs.c.store_digest == self._store_digest, refs.c.snapshot_id.in_(ids))
                    .order_by(refs.c.snapshot_id, refs.c.position))).all()
                for ref in found:
                    histories.setdefault(ref.snapshot_id, []).append((ref.position, ref.sha256, ref.message))
        return [(self._record_data(row), histories.get(row["id"], [])) for row in rows]

    async def _snapshot_from_data(
        self, data: dict[str, object], references: list[tuple[int, str, str | None]],
    ) -> ContinuableSnapshot:
        messages = []
        for position, sha256, body in references:
            if position != len(messages) or body is None or _digest(body) != sha256:
                raise ValueError("snapshot message references are missing or corrupt")
            messages.append(json.loads(body))
        if json.loads(data["messages"]) != {"stored_messages": len(messages)}:
            raise ValueError("snapshot message references are incomplete")
        if self._media_store is not None:
            messages = await restore_media(messages, media_store=self._media_store)
        data["messages"] = ModelMessagesTypeAdapter.validate_python(messages)
        return _SNAPSHOT_ADAPTER.validate_python(data)

    def _table(self, name: str) -> "Table":
        return self._metadata.tables[f"ai_step_{name}"]

    def _run_values(self, run_id: str) -> "dict[str, SqlValue]":
        return {"store_digest": self._store_digest, "run_digest": _digest(run_id), "run_id": run_id}

    def _run_predicates(self, table: "Table", run_id: str) -> "tuple[ColumnElement[bool], ...]":
        return (table.c.store_digest == self._store_digest, table.c.run_digest == _digest(run_id), table.c.run_id == run_id)

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


def _optional_digest(value: str | None) -> str | None:
    return None if value is None else _digest(value)


__all__ = ["PreparedStep", "SqlStepStore", "StepWriteReceipt"]
