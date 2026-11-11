#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace lifecycle positions backed by the immutable native event owner."""

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
from ..steps import NativeLifecycleEvent
from .codec import _encode_step_envelope, decode_envelope, encode_envelope
from .state import FactQuery, FactScanCursor, SqlStateFactReader, StateTransaction, StoredFact, record_key_digest, stream_digest

if TYPE_CHECKING:
    from sqlalchemy.sql import Select


_TYPE = "native_event_reference"
_RUN_ADAPTER = TypeAdapter(RunRecord)


@dataclass(slots=True)
class _PreparedEvent:
    prepared: PreparedStep
    projection: StepEvent
    receipt: StepWriteReceipt | None = None
    trace_confirmed: bool = False
    native_confirmed: bool = False


@dataclass(slots=True)
class _RunPublication:
    pending: dict[str, _PreparedEvent] = field(default_factory=dict)
    confirmed: StepWriteReceipt | None = None
    native_keys: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class NativeEventReferences:
    """Publish pending lifecycle sources with their ordered facts and join on reads.

    Pending bodies live only until both first-source publishers are confirmed.
    The native store owns immutable-body proof; the sidecar owns only position.
    """

    def __init__(self, state: SqlStateStore, *, namespace: str, tenant_id: str) -> None:
        self._state, self._namespace, self._tenant_id = state, namespace, tenant_id
        self._store_digest = SqlStepStore.scope_digest(namespace=namespace, tenant_id=tenant_id)
        self._native: SqlStepStore | None = None
        self._runs_pending: dict[str, _RunPublication] = {}
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
        if (source.idempotency_key is None or source.tool_call_id is not None or source.tool_name is not None
                or type(event_index) is not int or event_index < 0):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(source, NativeLifecycleEvent):
            source.__post_init__()
            event_type = source.event_type
        elif source.kind == "run_started" and source.error is None:
            event_type = "AGENT_RUN_STARTED"
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return StepEvent(
            agent_run_id=source.run_id, event_type=event_type, step_index=source.step_index,
            timestamp=source.timestamp, agent_conversation_id=source.conversation_id,
            parent_agent_run_id=source.parent_run_id, agent_id=source.agent_name,
            error=source.error, metadata={}, event_index=event_index,
            idempotency_key=f"{event_index}:{source.step_index}:{event_type}:",
        )

    @classmethod
    def decode_reference(cls, data: Mapping[str, object]) -> Mapping[str, object]:
        """Validate the physical event locator without querying its owner."""
        value = decode_envelope(data).value
        reference = value.get("payload")
        if (set(value) != {"type", "payload"} or value["type"] != _TYPE
                or not isinstance(reference, Mapping) or set(reference) != {"key", "index"}
                or type(reference["index"]) is not int or reference["index"] < 0
                or not isinstance(reference["key"], str) or len(reference["key"]) != 64
                or any(character not in "0123456789abcdef" for character in reference["key"])):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return reference

    @classmethod
    def project_source(cls, fact: StoredFact, row: Mapping[str, object], *,
                       run: Mapping[str, object] | None, namespace: str, tenant_id: str) -> StoredFact:
        """Verify complete detached source bytes and their physical Trace position."""
        reference = cls.decode_reference(fact.data)
        try:
            source = SqlStepStore.decode_started_row(row,
                expected_store_digest=SqlStepStore.scope_digest(namespace=namespace, tenant_id=tenant_id), run=run)
            event = cls.project(source, event_index=reference["index"])
        except (TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if (reference["key"] != row.get("key_digest") or fact.kind != "step_event"
                or fact.state != event.event_type or fact.subject_digest is not None
                or fact.owner_key_digest != record_key_digest(namespace, tenant_id, "trace", "agent_run", source.run_id)
                or fact.stream_digest != stream_digest(namespace, tenant_id, "trace", "event", source.run_id)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return replace(fact, data=_encode_step_envelope(event))

    async def stage(self, run: RunRecord, source: NativeStepEvent, event: StepEvent) -> None:
        if self._native is None or run.run_id != source.run_id or self.project(source, event_index=event.event_index) != event:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        started = source.kind == "run_started"
        prepared = await self._native.prepare_step(run=run if started else None, events=(source,), managed_start=started)
        publication = self._runs_pending.setdefault(source.run_id, _RunPublication())
        key = source.idempotency_key
        previous = publication.pending.get(key)
        if previous is not None:
            if previous.prepared != prepared or previous.projection != event:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        pending = _PreparedEvent(prepared, event)
        receipt = publication.confirmed
        if receipt is not None and source.idempotency_key in {
            item.idempotency_key for item in self._native.events_from_receipt(receipt)
        }:
            self._adopt(pending, receipt)
            pending.native_confirmed = key in publication.native_keys
        publication.pending[key] = pending

    @asynccontextmanager
    async def publication(self, run_id: str) -> AsyncIterator[None]:
        """Serialize native and Trace publishers through commit confirmation."""
        publication = self._runs_pending.get(run_id)
        if publication is None:
            yield
        else:
            async with publication.lock:
                yield

    def _adopt(self, pending: _PreparedEvent, receipt: StepWriteReceipt) -> None:
        expected = pending.prepared.events[0]
        source = next((item for item in self._native.events_from_receipt(receipt)
                       if (item.run_id, item.idempotency_key) == (expected.run_id, expected.idempotency_key)), None)
        run = self._native.run_from_receipt(receipt)
        if (source is None or (pending.prepared.run is not None and run is None)
                or (isinstance(expected, NativeLifecycleEvent) and source != expected)
                or self.project(source, event_index=pending.projection.event_index) != pending.projection):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        pending.prepared = replace(pending.prepared, run=run if pending.prepared.run is not None else None, events=(source,))
        pending.receipt = receipt

    def resolve_run(self, run: AgentRunRecord) -> AgentRunRecord:
        publication = self._runs_pending.get(run.agent_run_id)
        receipts = () if publication is None else (publication.confirmed, *(item.receipt for item in publication.pending.values()))
        for receipt in receipts:
            if receipt is not None:
                source = self._native.run_from_receipt(receipt)
                if source is not None:
                    return OfficialStepReader.project_run(source)
        return run

    def _for_event(self, event: StepEvent) -> _PreparedEvent | None:
        publication = self._runs_pending.get(event.agent_run_id)
        return None if publication is None else next((pending for pending in publication.pending.values()
                                                      if pending.projection == event), None)

    async def publish(self, transaction: StateTransaction, events: Sequence[StepEvent]) -> StepWriteReceipt | None:
        """Return proof usable in this transaction, not a global confirmation."""
        receipt = None
        for event in events:
            pending = self._for_event(event)
            if pending is not None:
                publication = self._runs_pending[event.agent_run_id]
                confirmed = receipt or publication.confirmed
                if confirmed is None:
                    pending.receipt = None
                receipt = await self._native.write_prepared_step(transaction, pending.prepared, confirmed=confirmed)
                self._adopt(pending, receipt)
        return receipt

    def encode(self, event: StepEvent) -> dict[str, object] | None:
        pending = self._for_event(event)
        if pending is None:
            return None
        if pending.receipt is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source = pending.prepared.events[0]
        if source not in self._native.events_from_receipt(pending.receipt):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return encode_envelope({"type": _TYPE, "payload": {
            "key": self._native.event_key_digest(source), "index": event.event_index,
        }})

    def confirm(self, events: Sequence[StepEvent]) -> tuple[StepWriteReceipt, ...]:
        receipts = []
        for event in events:
            pending = self._for_event(event)
            if pending is not None:
                if pending.receipt is None:
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                self._native.confirm_prepared_step(pending.receipt)
                receipts.append(pending.receipt)
                pending.trace_confirmed = True
                source = pending.prepared.events[0]
                publication = self._runs_pending[source.run_id]
                publication.confirmed = pending.receipt
                if pending.native_confirmed:
                    del publication.pending[source.idempotency_key]
        return tuple(receipts)

    def confirm_native(self, receipt: StepWriteReceipt) -> None:
        if self._native is None:
            return
        sources = {source.idempotency_key: source for source in self._native.events_from_receipt(receipt)}
        if not set(receipt.lifecycle_keys).issubset(sources):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for key in receipt.lifecycle_keys:
            source = sources[key]
            publication = self._runs_pending.setdefault(source.run_id, _RunPublication())
            publication.confirmed = receipt
            key = source.idempotency_key
            publication.native_keys.add(key)
            pending = publication.pending.get(key)
            if pending is not None:
                self._adopt(pending, receipt)
                pending.native_confirmed = True
                if pending.trace_confirmed:
                    del publication.pending[key]

    def pending_source(self, run_id: str) -> tuple[RunRecord, NativeStepEvent | None, NativeStepEvent | None] | None:
        """Exact pending lifecycle sources for the existing owner UNKNOWN readback."""
        publication = self._runs_pending.get(run_id)
        if publication is None:
            return None
        run = None
        started = terminal = None
        for pending in publication.pending.values():
            if pending.receipt is not None:
                run = self._native.run_from_receipt(pending.receipt)
                source = pending.prepared.events[0]
                if source.kind == "run_started":
                    started = source
                else:
                    terminal = source
        if run is None:
            return None
        return run, started, terminal

    def has_pending(self, run_id: str) -> bool:
        publication = self._runs_pending.get(run_id)
        return publication is not None and any(item.prepared.managed_start for item in publication.pending.values())

    def confirmed_receipt(self, run_id: str) -> StepWriteReceipt | None:
        publication = self._runs_pending.get(run_id)
        return None if publication is None else publication.confirmed

    def release_settled(self, run_id: str) -> None:
        """Drop settled source bodies after a quiescent run boundary is confirmed."""
        publication = self._runs_pending.get(run_id)
        if (publication is not None and not publication.lock.locked()
                and all(pending.trace_confirmed for pending in publication.pending.values())):
            # At a confirmed quiescent boundary a failed native publisher cannot
            # echo again. Keep only unresolved Trace publication/UNKNOWN proof.
            del self._runs_pending[run_id]

    def clear(self, run_id: str | None = None) -> None:
        if run_id is None:
            self._runs_pending.clear()
        else:
            self._runs_pending.pop(run_id, None)

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
        columns["outcome"] = events.c.outcome
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
        columns.update((name, events.c[name]) for name in ("store_digest", "key_digest", "fields_sha256"))
        registration_columns = ()
        if full_source:
            columns["body_sha256"] = events.c.body_sha256
            columns["run_fields"] = events.c.run_fields
            registration_columns = tuple(runs.c[name].label("registration_" + name) for name in (
                "store_digest", *(field.name for field in fields(RunRecord)),
            ))
        statement = select(facts, *(column.label("native_" + name) for name, column in columns.items()),
                           *registration_columns, or_(*invalid).label("invalid_native_reference")).select_from(
            facts.outerjoin(events, and_(facts.c.kind == "step_event", events.c.store_digest == self._store_digest,
                                         events.c.key_digest == reference["key"].as_string())).outerjoin(runs, and_(
                *(() if full_source else (inherited != 0,)), runs.c.store_digest == events.c.store_digest,
                runs.c.run_id == events.c.run_id,
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
                reference = self.decode_reference(fact.data)
                if row["invalid_native_reference"]:
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
                        or fact.state != event.event_type or fact.subject_digest is not None
                        or fact.owner_key_digest != record_key_digest(self._namespace, self._tenant_id, "trace", "agent_run", source.run_id)
                        or fact.stream_digest != stream_digest(self._namespace, self._tenant_id, "trace", "event", source.run_id)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                publication = self._runs_pending.get(source.run_id)
                pending = None if publication is None else publication.pending.get(source.idempotency_key)
                if full_source and pending is not None:
                    expected_run = (pending.prepared.run if pending.receipt is None
                                    else self._native.run_from_receipt(pending.receipt))
                    if pending.prepared.events[0] != source or expected_run != native_run:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                fact = replace(fact, data=_encode_step_envelope(event))
            values.append(fact)
        return tuple(values)


__all__ = ["NativeEventReferences"]
