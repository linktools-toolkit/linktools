#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace start positions backed by the immutable native event owner."""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent as NativeStepEvent

from ..errors import AIError, ErrorCode
from ._plan import PersistenceDomain
from ._schema import build_runtime_sql_metadata
from ._sql import SqlStateStore
from ._step_contracts import AgentRunRecord, StepEvent
from ._official_step_reader import OfficialStepReader
from ._step_schema import build_step_sql_metadata
from ._step_store import PreparedStep, SqlStepStore, StepWriteReceipt
from .codec import _encode_step_envelope, decode_envelope, encode_envelope
from .state import FactQuery, FactScanCursor, SqlStateFactReader, StateTransaction, StoredFact, record_key_digest, stream_digest

if TYPE_CHECKING:
    from sqlalchemy.sql import Select


_TYPE = "native_start_reference"
_RUN_ADAPTER = TypeAdapter(RunRecord)


@dataclass(slots=True)
class _PreparedStart:
    prepared: PreparedStep
    projection: StepEvent
    receipt: StepWriteReceipt | None = None
    trace_confirmed: bool = False
    native_confirmed: bool = False
    source_confirmed: bool = False
    publication_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class NativeEventReferences:
    """Publish the pending start with its original ordered fact and join on reads.

    Pending bodies live only until both first-source publishers are confirmed.
    The native store owns immutable-body proof; the sidecar owns only position.
    """

    def __init__(self, state: SqlStateStore, *, namespace: str, tenant_id: str) -> None:
        self._state, self._namespace, self._tenant_id = state, namespace, tenant_id
        self._store_digest = SqlStepStore.scope_digest(namespace=namespace, tenant_id=tenant_id)
        self._native: SqlStepStore | None = None
        self._pending: dict[str, _PreparedStart] = {}
        metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.TRACE}))
        build_step_sql_metadata(metadata=metadata)
        self._facts, self._events, self._runs = (metadata.tables[name] for name in (
            "ai_state_facts", "ai_step_events", "ai_step_runs",
        ))

    def bind(self, native: SqlStepStore) -> None:
        if (native.storage_context.engine is self._state.context.engine
                and native.store_digest == self._store_digest):
            if self._native is not None and self._native is not native:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._native = native

    def accepts(self, native: object) -> bool:
        return type(native) is SqlStepStore and self._native is native

    @classmethod
    def project(cls, source: NativeStepEvent, *, event_index: int) -> StepEvent:
        if (source.kind != "run_started" or source.idempotency_key is None
                or source.tool_call_id is not None or source.tool_name is not None or source.error is not None
                or type(event_index) is not int or event_index < 0):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return StepEvent(
            agent_run_id=source.run_id, event_type="AGENT_RUN_STARTED", step_index=source.step_index,
            timestamp=source.timestamp, agent_conversation_id=source.conversation_id,
            parent_agent_run_id=source.parent_run_id, agent_id=source.agent_name,
            metadata={}, event_index=event_index,
            idempotency_key=f"{event_index}:{source.step_index}:AGENT_RUN_STARTED:",
        )

    async def stage(self, run: RunRecord, source: NativeStepEvent, event: StepEvent) -> None:
        if self._native is None or run.run_id != source.run_id or self.project(source, event_index=event.event_index) != event:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prepared = await self._native.prepare_step(run=run, events=(source,), managed_start=True)
        previous = self._pending.get(run.run_id)
        if previous is not None:
            if previous.prepared != prepared or previous.projection != event:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        self._pending[run.run_id] = _PreparedStart(prepared, event)

    @asynccontextmanager
    async def publication(self, run_id: str) -> AsyncIterator[None]:
        """Serialize the two first-source publishers through commit confirmation."""
        pending = self._pending.get(run_id)
        if pending is None:
            yield
        else:
            async with pending.publication_lock:
                yield

    def _adopt(self, pending: _PreparedStart, receipt: StepWriteReceipt) -> None:
        source = self._native.started_from_receipt(receipt)
        run = self._native.run_from_receipt(receipt)
        expected = pending.prepared.events[0]
        if (source is None or run is None or source.idempotency_key != expected.idempotency_key
                or self.project(source, event_index=pending.projection.event_index) != pending.projection):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        pending.prepared = replace(pending.prepared, run=run, events=(source,))
        pending.receipt = receipt

    def resolve_run(self, run: AgentRunRecord) -> AgentRunRecord:
        pending = self._pending.get(run.agent_run_id)
        if pending is None or pending.receipt is None:
            return run
        source = self._native.run_from_receipt(pending.receipt)
        if source is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return OfficialStepReader.project_run(source)

    async def publish(self, transaction: StateTransaction, events: Sequence[StepEvent]) -> StepWriteReceipt | None:
        """Return source proof usable in this same transaction, not a global confirmation."""
        receipt = None
        for event in events:
            pending = self._pending.get(event.agent_run_id)
            if pending is not None and event == pending.projection:
                confirmed = pending.receipt if pending.source_confirmed else None
                if confirmed is None:
                    pending.receipt = None
                receipt = await self._native.write_prepared_step(transaction, pending.prepared, confirmed=confirmed)
                self._adopt(pending, receipt)
        return receipt

    def encode(self, event: StepEvent) -> dict[str, object] | None:
        pending = self._pending.get(event.agent_run_id)
        if pending is None or pending.projection != event:
            return None
        if pending.receipt is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source = self._native.started_from_receipt(pending.receipt)
        if source != pending.prepared.events[0]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return encode_envelope({"type": _TYPE, "payload": {
            "key": self._native.event_key_digest(source), "index": event.event_index,
        }})

    def confirm(self, events: Sequence[StepEvent]) -> tuple[StepWriteReceipt, ...]:
        receipts = []
        for event in events:
            pending = self._pending.get(event.agent_run_id)
            if pending is not None and pending.projection == event:
                if pending.receipt is None:
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                self._native.confirm_prepared_step(pending.receipt)
                receipts.append(pending.receipt)
                pending.trace_confirmed = pending.source_confirmed = True
                if pending.native_confirmed:
                    del self._pending[event.agent_run_id]
        return tuple(receipts)

    def confirm_native(self, receipt: StepWriteReceipt) -> None:
        if self._native is None:
            return
        source = self._native.started_from_receipt(receipt)
        pending = None if source is None else self._pending.get(source.run_id)
        if pending is not None:
            self._adopt(pending, receipt)
            pending.native_confirmed = pending.source_confirmed = True
            if pending.trace_confirmed:
                del self._pending[source.run_id]

    def pending_source(self, run_id: str) -> tuple[RunRecord, NativeStepEvent] | None:
        """Exact registration and start to verify before acknowledging an unknown commit."""
        pending = self._pending.get(run_id)
        if pending is None or pending.receipt is None:
            return None
        run = self._native.run_from_receipt(pending.receipt)
        event = self._native.started_from_receipt(pending.receipt)
        if run is None or event is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return run, event

    def has_pending(self, run_id: str) -> bool:
        return run_id in self._pending

    def confirmed_receipt(self, run_id: str) -> StepWriteReceipt | None:
        pending = self._pending.get(run_id)
        return pending.receipt if pending is not None and pending.source_confirmed else None

    def clear(self, run_id: str | None = None) -> None:
        if run_id is None:
            self._pending.clear()
        else:
            self._pending.pop(run_id, None)

    def statement(self, *, query: FactQuery | None = None, sequences: Sequence[int] | None = None,
                  scan: tuple[FactScanCursor | None, int] | None = None,
                  full_source: bool = False) -> "Select":
        from sqlalchemy import JSON, Text, and_, bindparam, case, cast, func, or_, select, type_coerce
        from sqlalchemy.sql import visitors
        from sqlalchemy.sql.elements import BindParameter

        facts, events, runs = self._facts, self._events, self._runs
        encoded = type_coerce(facts.c.payload_json, JSON)[()].as_string()
        payload = cast(encoded, JSON) if self._state.context.dialect.name == "postgresql" else type_coerce(encoded, JSON)
        reference = payload["value"]["payload"]
        columns = {field.name: events.c[field.name] for field in fields(NativeStepEvent)
                   if full_source or field.name != "metadata"}
        if full_source:
            columns["metadata"] = cast(events.c.metadata, Text)
        inherited = events.c.run_fields if full_source else events.c.run_fields.bitwise_and(7)
        invalid = [events.c.run_fields.bitwise_and(15) != events.c.run_fields,
                   and_(inherited != 0, runs.c.id.is_(None))]
        names = ("conversation_id", "parent_run_id", "agent_name", "metadata") if full_source else (
            "conversation_id", "parent_run_id", "agent_name",
        )
        for bit, name in enumerate(names):
            referenced = events.c.run_fields.bitwise_and(1 << bit) != 0
            if not full_source:
                columns[name] = case((referenced, runs.c[name]), else_=columns[name])
            invalid.append(and_(referenced, or_(events.c[name].is_not(None), runs.c[name].is_(None))))
        columns.update((name, events.c[name]) for name in ("store_digest", "run_digest", "key_digest", "fields_sha256"))
        registration_columns = ()
        if full_source:
            columns["body_sha256"] = events.c.body_sha256
            columns["run_fields"] = events.c.run_fields
            registration_columns = tuple(runs.c[name].label("registration_" + name) for name in (
                "store_digest", "run_digest", *(field.name for field in fields(RunRecord)),
            ))
        statement = select(facts, *(column.label("native_" + name) for name, column in columns.items()),
                           *registration_columns, or_(*invalid).label("invalid_native_reference")).select_from(
            facts.outerjoin(events, and_(facts.c.kind == "step_event", events.c.store_digest == self._store_digest,
                                         events.c.key_digest == reference["key"].as_string())).outerjoin(runs, and_(
                *(() if full_source else (inherited != 0,)), runs.c.store_digest == events.c.store_digest,
                runs.c.run_digest == events.c.run_digest, runs.c.run_id == events.c.run_id,
            )))
        statement = visitors.replacement_traverse(statement, {}, lambda node: (
            bindparam(None, node.value, type_=node.type, literal_execute=True)
            if isinstance(node, BindParameter) else None
        )).where(facts.c.store_digest == self._state.store_digest.hex())
        if query is not None:
            conditions = [facts.c.stream_digest == query.stream_digest.hex()]
            if query.after_sequence is not None:
                conditions.append(facts.c.sequence > query.after_sequence)
            if query.subject_digest is not None:
                conditions.append(facts.c.subject_digest == query.subject_digest.hex())
            if sequences is not None:
                conditions.append(facts.c.sequence.in_(tuple(sequences)))
            if query.latest_per_subject:
                latest = select(func.max(facts.c.sequence)).where(
                    facts.c.store_digest == self._state.store_digest.hex(), *conditions,
                ).group_by(facts.c.subject_digest)
                conditions = [facts.c.stream_digest == query.stream_digest.hex(), facts.c.sequence.in_(latest)]
            statement = statement.where(*conditions).order_by(facts.c.sequence.desc() if query.latest else facts.c.sequence)
            if query.limit is not None or query.latest:
                statement = statement.limit(1 if query.latest else query.limit)
        if scan is not None:
            after, limit = scan
            if type(limit) is not int or not 1 <= limit <= 1000:
                raise ValueError("fact page limit must be between 1 and 1000")
            if after is not None:
                statement = statement.where(or_(facts.c.stream_digest > after.stream_digest.hex(), and_(
                    facts.c.stream_digest == after.stream_digest.hex(), facts.c.sequence > after.sequence,
                )))
            statement = statement.order_by(facts.c.stream_digest, facts.c.sequence).limit(limit)
        return statement

    async def read(self, transaction: StateTransaction, *, query: FactQuery | None = None,
                   sequences: Sequence[int] | None = None,
                   scan: tuple[FactScanCursor | None, int] | None = None,
                   full_source: bool = False) -> tuple[StoredFact, ...]:
        if sequences is not None and not sequences:
            return ()
        if not isinstance(transaction, SqlStateFactReader):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        values = []
        for fact, row in await transaction.read_joined_facts(self._state.context, self.statement(
                query=query, sequences=sequences, scan=scan, full_source=full_source)):
            value = fact.data.get("value")
            if isinstance(value, Mapping) and value.get("type") == _TYPE:
                payload = decode_envelope(fact.data).value
                reference = payload.get("payload")
                if (set(payload) != {"type", "payload"} or not isinstance(reference, Mapping)
                        or set(reference) != {"key", "index"} or row["invalid_native_reference"]):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    source_data = {key.removeprefix("native_"): entry for key, entry in row.items()
                                   if key.startswith("native_")}
                    if full_source:
                        registration = {key.removeprefix("registration_"): entry for key, entry in row.items()
                                        if key.startswith("registration_")}
                        source = SqlStepStore.decode_started_row(source_data,
                            expected_store_digest=self._store_digest, run=registration)
                        registration["metadata"] = json.loads(registration["metadata"])
                        native_run = _RUN_ADAPTER.validate_python(registration)
                    else:
                        source = SqlStepStore.decode_started_fields(source_data, expected_store_digest=self._store_digest)
                    event = self.project(source, event_index=reference["index"])
                except (TypeError, ValueError, KeyError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if (reference["key"] != row["native_key_digest"] or fact.kind != "step_event"
                        or fact.state != "AGENT_RUN_STARTED" or fact.subject_digest is not None
                        or fact.owner_key_digest != record_key_digest(self._namespace, self._tenant_id, "trace", "agent_run", source.run_id)
                        or fact.stream_digest != stream_digest(self._namespace, self._tenant_id, "trace", "event", source.run_id)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                pending = self._pending.get(source.run_id)
                if full_source and pending is not None:
                    if pending.prepared.events[0] != source or pending.prepared.run != native_run:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                fact = replace(fact, data=_encode_step_envelope(event))
            values.append(fact)
        return tuple(values)


__all__ = ["NativeEventReferences"]
