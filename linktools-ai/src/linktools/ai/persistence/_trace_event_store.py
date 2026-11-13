#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace-owned immutable events and query projections."""

import copy
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from ..errors import AIError, ErrorCode
from ._record_key import decode_key_components
from ._run_counters import run_counter_address
from ._schema import build_runtime_sql_metadata
from ._plan import PersistenceDomain
from ._sql import SqlStateStore
from ._step_contracts import StepEvent, TOOL_ERROR_CODE_METADATA_KEY
from .codec import _decode_step_envelope, _encode_step_envelope
from .state import (FactQuery, SqlStateRowReader, SqlStateTransaction, StateTransaction,
                    StoredFact, STREAM_KEY_MAX_BYTES, record_key, stream_key)

if TYPE_CHECKING:
    from sqlalchemy import Table, Select
    from sqlalchemy.ext.asyncio import AsyncSession
    from ._native_event_reference import NativeEventReferences

_STEP_EVENT_ADAPTER = TypeAdapter(StepEvent)

_REQUEST = "linktools.ai.model_request_seq"
_MESSAGE = "linktools.ai.message_seq"
_INLINE_SCALARS = ("agent_conversation_id", "parent_agent_run_id", "agent_id", "tool_name", "error", "idempotency_key")
_PROMOTED = ("agent_run_id", "event_type", "step_index", "timestamp", "tool_call_id", "event_index", *_INLINE_SCALARS)
_REQUEST_KINDS = ("MODEL_REQUEST_STARTED", "MODEL_REQUEST_SUCCEEDED", "MODEL_REQUEST_FAILED", "MODEL_REQUEST_CANCELLED")
_TOOL_KINDS = ("TOOL_CALL_STARTED", "TOOL_CALL_SUCCEEDED", "TOOL_CALL_FAILED")


def _utf8(value: str | None) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.encode("utf-8")


def _identity(value: str | None) -> bytes | None:
    result = _utf8(value)
    if result is not None and len(result) > 128:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _text(value: object) -> str | None:
    return None if value is None else bytes(value).decode("utf-8")


def _request_number(value: str | None) -> bytes | None:
    if value is None:
        return None
    if not value.isdigit():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        number = int(value)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if number < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _identity(str(number))


def _order(timestamp: datetime, sequence: int) -> bytes:
    return f"{timestamp.astimezone(timezone.utc).isoformat(timespec='microseconds')}:{sequence:020d}".encode("ascii")


def history_identities(event: StepEvent) -> tuple[tuple[str, str], ...]:
    """Unique history roles, excluding ambiguous external tool outcomes."""
    result = []
    request, message = event.metadata.get(_REQUEST), event.metadata.get(_MESSAGE)
    if event.event_type.startswith("MODEL_REQUEST_") and request is not None:
        result.append((event.event_type, request))
    if event.event_type == "MODEL_REQUEST_SUCCEEDED" and message is not None:
        result.append(("response", message))
    if (event.event_type in _TOOL_KINDS and event.tool_call_id is not None
            and not (event.event_type == "TOOL_CALL_FAILED"
                     and event.metadata.get(TOOL_ERROR_CODE_METADATA_KEY) == ErrorCode.TOOL_EFFECT_UNKNOWN.value)):
        result.append((event.event_type, event.tool_call_id))
    return tuple(result)


def _history_identity(event: StepEvent) -> bytes | None:
    return next((_identity(identity) for role, identity in history_identities(event) if role != "response"), None)


class SqlTraceEventStore:
    def __init__(self, state: SqlStateStore, *, namespace: str, tenant_id: str) -> None:
        self.state, self.namespace, self.tenant_id = state, namespace, tenant_id
        self._table = build_runtime_sql_metadata(frozenset({PersistenceDomain.TRACE})).tables["ai_trace_events"]

    @property
    def table(self) -> "Table":
        return self._table

    def run_id(self, stream: bytes) -> str:
        parts = decode_key_components(stream, STREAM_KEY_MAX_BYTES)
        if len(parts) != 6 or parts[:5] != ("stream", self.namespace, self.tenant_id, "trace", "event"):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return parts[5]

    def row_values(self, fact: StoredFact, event: StepEvent) -> dict[str, object]:
        if (self.run_id(fact.stream_key) != event.agent_run_id or fact.kind != "step_event"
                or fact.state != event.event_type or fact.subject_id is not None
                or fact.owner_key != record_key(self.namespace, self.tenant_id, "trace", "agent_run", (event.agent_run_id,))):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload = copy.deepcopy(dict(fact.data))
        inline = payload["value"]["type"] == "step_event"
        response = event.event_type == "MODEL_REQUEST_SUCCEEDED"
        if payload["value"]["type"] == "step_event":
            if _decode_step_envelope(fact.data) != event:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            fields = payload["value"]["payload"]["fields"]
            payload = {"type": "step_event", **{key: value for key, value in fields.items()
                                                   if key not in _PROMOTED}, "metadata": {
                key: value for key, value in event.metadata.items()
                if key not in (_REQUEST, TOOL_ERROR_CODE_METADATA_KEY)
                and not (response and key == _MESSAGE)
            }}
        elif payload["value"]["type"] != "native_event_reference":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        message_identity = _identity(event.metadata.get(_MESSAGE))
        return dict(store_key=self.state.store_key, run_id=_identity(event.agent_run_id), sequence=fact.sequence,
                    event_index=event.event_index, kind=event.event_type, step_index=event.step_index,
                    timestamp=event.timestamp.isoformat(), order_key=_order(event.timestamp, fact.sequence),
                    request_identity=_identity(event.metadata.get(_REQUEST)),
                    request_number=_request_number(event.metadata.get(_REQUEST)),
                    message_identity=message_identity if response else None, tool_call_id=_identity(event.tool_call_id),
                    history_identity=_history_identity(event),
                    error_code=_identity(event.metadata.get(TOOL_ERROR_CODE_METADATA_KEY)), payload_json=payload,
                    **{name: _utf8(getattr(event, name)) if inline else None for name in _INLINE_SCALARS})

    def row_to_fact(self, row: Mapping[str, object]) -> StoredFact:
        try:
            if row["store_key"] != self.state.store_key:
                raise ValueError("Trace partition differs")
            run_id = _text(row["run_id"])
            sequence = row["sequence"]
            if type(sequence) is not int or sequence < 1:
                raise ValueError("Trace sequence must be positive")
            for name in ("run_id", "request_identity", "request_number", "message_identity", "tool_call_id", "error_code", "history_identity"):
                if _identity(_text(row[name])) != row[name]:
                    raise ValueError("Trace identity is not canonical binary")
            scalars = {name: _text(row[name]) for name in _INLINE_SCALARS}
            if any(_utf8(value) != row[name] for name, value in scalars.items()):
                raise ValueError("Trace scalar is not canonical UTF-8")
            timestamp = datetime.fromisoformat(row["timestamp"])
            if timestamp.tzinfo is None or timestamp.isoformat() != row["timestamp"] or len(row["timestamp"]) > 64:
                raise ValueError("Trace timestamp is not canonical")
            if row["order_key"] != _order(timestamp, sequence):
                raise ValueError("Trace order differs")
            if row["request_number"] != _request_number(_text(row["request_identity"])):
                raise ValueError("Trace numeric request projection differs")
            payload = copy.deepcopy(row["payload_json"])
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                raise ValueError("Trace payload must be an object")
            if payload.get("type") == "step_event":
                residual = {key: value for key, value in payload.items() if key != "type"}
                if any(key in residual for key in _PROMOTED):
                    raise ValueError("Trace body duplicates promoted fields")
                metadata = residual["metadata"]
                if not isinstance(metadata, dict):
                    raise ValueError("Trace metadata must be an object")
                response = row["kind"] == "MODEL_REQUEST_SUCCEEDED"
                if any(key in metadata for key in (_REQUEST, TOOL_ERROR_CODE_METADATA_KEY)) or response and _MESSAGE in metadata:
                    raise ValueError("Trace metadata duplicates promoted fields")
                if not response and row["message_identity"] is not None:
                    raise ValueError("Trace message identity has no response role")
                metadata.update((key, value) for key, value in (
                    (_REQUEST, _text(row["request_identity"])), (_MESSAGE, _text(row["message_identity"])),
                    (TOOL_ERROR_CODE_METADATA_KEY, _text(row["error_code"]))) if value is not None)
                event = _STEP_EVENT_ADAPTER.validate_python(dict(
                    agent_run_id=run_id, event_type=row["kind"], step_index=row["step_index"],
                    timestamp=timestamp, tool_call_id=_text(row["tool_call_id"]),
                    event_index=row["event_index"], metadata=metadata, **scalars,
                ))
                payload = _encode_step_envelope(event)
                payload["value"]["payload"]["fields"].update(
                    (key, value) for key, value in residual.items() if key != "metadata"
                )
                if row["order_key"] != _order(event.timestamp, sequence) or row["request_number"] != _request_number(event.metadata.get(_REQUEST)):
                    raise ValueError("Trace query projection differs")
                message_identity = _identity(event.metadata.get(_MESSAGE))
                if (row["history_identity"] != _history_identity(event)
                        or row["message_identity"] != (message_identity if response else None)):
                    raise ValueError("Trace history projection differs")
            elif payload.get("value", {}).get("type") == "native_event_reference":
                reference = payload["value"]["payload"]
                if (reference.get("run") != run_id or reference.get("index") != row["event_index"]
                        or any(value is not None for value in scalars.values()) or row["history_identity"] is not None):
                    raise ValueError("Trace native position differs")
            else:
                raise ValueError("Unknown Trace payload")
            return StoredFact(stream_key(self.namespace, self.tenant_id, "trace", "event", (run_id,)), sequence,
                              record_key(self.namespace, self.tenant_id, "trace", "agent_run", (run_id,)),
                              "step_event", None, row["kind"], payload)
        except (KeyError, TypeError, ValueError, UnicodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def _rows(self, transaction: StateTransaction, statement: "Select") -> tuple[Mapping[str, object], ...]:
        if not isinstance(transaction, SqlStateRowReader):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        return await transaction.read_sql_rows(self.state.context, statement)

    async def read(self, transaction: StateTransaction, query: FactQuery, *,
                   sequences: Sequence[int] | None = None) -> tuple[StoredFact, ...]:
        return tuple(self.row_to_fact(row) for row in await self.read_rows(transaction, query, sequences=sequences))

    async def read_rows(self, transaction: StateTransaction, query: FactQuery, *,
                        sequences: Sequence[int] | None = None,
                        native_events: "NativeEventReferences | None" = None,
                        full_source: bool = False) -> tuple[Mapping[str, object], ...]:
        from sqlalchemy import select
        table = self.table
        statement = select(table)
        if native_events is not None:
            statement = native_events.join_trace(statement, table, full_source=full_source)
        statement = statement.where(table.c.store_key == self.state.store_key,
                                    table.c.run_id == _identity(self.run_id(query.stream_key)))
        if query.subject_id is not None or sequences is not None and not sequences:
            return ()
        if query.after_sequence is not None:
            statement = statement.where(table.c.sequence > query.after_sequence)
        if sequences is not None:
            statement = statement.where(table.c.sequence.in_(tuple(sequences)))
        statement = statement.order_by(table.c.sequence.desc() if query.latest else table.c.sequence)
        if query.limit is not None or query.latest:
            statement = statement.limit(1 if query.latest else query.limit)
        return await self._rows(transaction, statement)

    async def select(self, transaction: StateTransaction, run_id: str, *, high_water: int,
                     request_identities: Sequence[str] | None = None, message_identities: Sequence[str] | None = None,
                     tool_call_ids: Sequence[str] | None = None, history: bool = False,
                     after_timestamp: datetime | None = None, after_sequence: int = 0, limit: int | None = None,
                     model_request_seq: int | None = None, step_index: int | None = None,
                     tool_call_id: str | None = None) -> tuple[StoredFact, ...]:
        from sqlalchemy import and_, or_, select, union_all
        t = self.table.c
        statement = select(self.table).where(t.store_key == self.state.store_key, t.run_id == _identity(run_id),
                                             t.sequence <= high_water)
        if history:
            requests = frozenset(map(_identity, request_identities or ()))
            tools = frozenset(map(_identity, tool_call_ids or ()))
            identities = tuple(sorted(requests | tools))
            messages = tuple(dict.fromkeys(map(_identity, message_identities or ())))
            found: dict[int, StoredFact] = {}
            # Each branch seeks one correlation index. Sorting their bounded
            # union avoids a cross-index OR competing with the run sequence scan.
            for offset in range(0, max(len(identities), len(messages)), 32):
                queries = []
                batch = identities[offset:offset + 32]
                if batch:
                    request_batch = tuple(value for value in batch if value in requests)
                    tool_batch = tuple(value for value in batch if value in tools)
                    roles = []
                    if request_batch:
                        roles.append(and_(t.kind.in_(_REQUEST_KINDS), t.history_identity.in_(request_batch)))
                    if tool_batch:
                        roles.append(and_(t.kind.in_(_TOOL_KINDS), t.history_identity.in_(tool_batch),
                            or_(t.kind != "TOOL_CALL_FAILED", t.error_code.is_(None),
                                t.error_code != ErrorCode.TOOL_EFFECT_UNKNOWN.value.encode("utf-8"))))
                    queries.append(statement.where(t.history_identity.in_(batch), or_(*roles)))
                message_batch = messages[offset:offset + 32]
                if message_batch:
                    queries.append(statement.where(t.kind == "MODEL_REQUEST_SUCCEEDED",
                                                   t.message_identity.in_(message_batch)))
                query = queries[0] if len(queries) == 1 else select(union_all(*queries).subquery())
                for row in await self._rows(transaction, query):
                    fact = self.row_to_fact(row)
                    found[fact.sequence] = fact
            ordered = sorted(found)
            return tuple(found[key] for key in (ordered if limit is None else ordered[:limit]))
        statement = statement.where(t.kind.in_((*_REQUEST_KINDS, *_TOOL_KINDS)))
        if after_timestamp is not None:
            statement = statement.where(t.order_key > _order(after_timestamp, after_sequence))
        if model_request_seq is not None:
            statement = statement.where(t.request_number == _identity(str(model_request_seq)))
        if step_index is not None:
            statement = statement.where(t.step_index == step_index)
        if tool_call_id is not None:
            statement = statement.where(t.tool_call_id == _identity(tool_call_id))
        statement = statement.order_by(t.order_key)
        if limit is not None:
            statement = statement.limit(limit)
        return tuple(self.row_to_fact(row) for row in await self._rows(transaction, statement))

    async def history_associations(
        self, transaction: StateTransaction, run_id: str, *, event_stream: bytes,
        high_water: int, message_identities: Sequence[str], tool_call_ids: Sequence[str],
    ) -> tuple[StoredFact, ...]:
        from hashlib import sha256
        from sqlalchemy import and_, literal, or_, select, true, union, union_all

        if self.run_id(event_stream) != run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        table = self.table
        t = table.c
        counters = build_runtime_sql_metadata(frozenset({PersistenceDomain.TRACE})).tables["ai_step_run_admissions"]
        address = run_counter_address(event_stream)
        if address is None or address[1] != "event_high_water":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        counter_owner = address[0]
        messages = tuple(dict.fromkeys(map(_identity, message_identities)))
        tools = tuple(dict.fromkeys(map(_identity, tool_call_ids)))
        found: dict[int, StoredFact] = {}
        for offset in range(0, max(len(messages), len(tools)), 32):
            # Only correlation keys are repeated in the derived branches. Each
            # selected event body is fetched once after sequence deduplication.
            seed = select(t.sequence, t.request_identity).where(
                t.store_key == self.state.store_key, t.run_id == _identity(run_id), t.sequence <= high_water)
            branches = []
            if tools[offset:offset + 32]:
                branches.append(seed.where(t.history_identity.in_(tools[offset:offset + 32]),
                    t.kind.in_(_TOOL_KINDS), or_(t.kind != "TOOL_CALL_FAILED", t.error_code.is_(None),
                        t.error_code != ErrorCode.TOOL_EFFECT_UNKNOWN.value.encode("utf-8"))))
            if messages[offset:offset + 32]:
                branches.append(seed.where(t.message_identity.in_(messages[offset:offset + 32]),
                                           t.kind == "MODEL_REQUEST_SUCCEEDED"))
            selected = (branches[0] if len(branches) == 1 else union_all(*branches)).subquery("selected_events")
            requests = select(selected.c.request_identity).where(
                selected.c.request_identity.is_not(None)).distinct().subquery("selected_requests")
            dependency = table.alias("dependency")
            dependent_sequences = select(dependency.c.sequence).select_from(dependency.join(requests,
                dependency.c.history_identity == requests.c.request_identity)).where(
                    dependency.c.store_key == self.state.store_key, dependency.c.run_id == _identity(run_id),
                    dependency.c.sequence <= high_water, dependency.c.kind.in_(_REQUEST_KINDS))
            sequences = union(select(selected.c.sequence), dependent_sequences).subquery("selected_sequences")
            events = select(table).select_from(table.join(sequences, t.sequence == sequences.c.sequence)).where(
                t.store_key == self.state.store_key, t.run_id == _identity(run_id)).subquery("events")
            anchor = select(literal(1).label("present")).subquery("anchor")
            source = anchor.outerjoin(counters, and_(counters.c.store_key == self.state.store_key,
                counters.c.key_locator == sha256(counter_owner).hexdigest())).outerjoin(events, true())
            statement = select(counters.c.key.label("counter_owner"),
                counters.c.key_locator.label("counter_locator"), counters.c.event_high_water.label("counter_value"), *events.c).select_from(source).order_by(events.c.sequence)
            for row in await self._rows(transaction, statement):
                value = row["counter_value"]
                if row["counter_owner"] is not None and (row["counter_owner"] != counter_owner
                        or row["counter_locator"] != sha256(counter_owner).hexdigest()):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if value is None:
                    value = 0
                elif type(value) is not int or value < 0:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if value < high_water:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if row["sequence"] is not None:
                    fact = self.row_to_fact(row)
                    found[fact.sequence] = fact
        return tuple(found[key] for key in sorted(found))

    async def insert(self, transaction: StateTransaction, facts: Sequence[StoredFact], events: Sequence[StepEvent]) -> None:
        if not facts:
            return
        if (len(facts) != len(events) or any(event.agent_run_id != events[0].agent_run_id for event in events)
                or tuple(fact.sequence for fact in facts) != tuple(range(facts[0].sequence, facts[0].sequence + len(facts)))
                or await transaction.get_sequence(facts[0].stream_key) != facts[-1].sequence):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        # The run CAS serializes role-conflict validation with the publication.
        roles = {}
        for fact, event in zip(facts, events, strict=True):
            for role in history_identities(event):
                if role in roles and roles[role] != fact.sequence:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                roles[role] = fact.sequence
        run_id = events[0].agent_run_id
        existing = await self.select(transaction, run_id, high_water=facts[0].sequence - 1, history=True,
                                     request_identities=tuple(identity for role, identity in roles if role in _REQUEST_KINDS),
                                     message_identities=tuple(identity for role, identity in roles if role == "response"),
                                     tool_call_ids=tuple(identity for role, identity in roles if role in _TOOL_KINDS))
        for fact in existing:
            event = _decode_step_envelope(fact.data)
            if any(role in roles for role in history_identities(event)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self.restore(transaction, tuple(self.row_values(fact, event) for fact, event in zip(facts, events, strict=True)))

    async def restore(self, transaction: StateTransaction, rows: Sequence[Mapping[str, object]]) -> None:
        from sqlalchemy import insert
        if not isinstance(transaction, SqlStateTransaction):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        values = [{key: value for key, value in row.items() if key not in ("id", "updated_at", "created_at")} for row in rows]
        for row in values:
            self.row_to_fact(row)
        async def write(session: "AsyncSession") -> None:
            for offset in range(0, len(values), 32):
                await self.state.context.run_statement(session.execute(insert(self.table), values[offset:offset + 32]))
        await transaction.run_sql(self.state.context, write)

    async def scan(self, transaction: StateTransaction, *, after: tuple[bytes, int] | None = None,
                   limit: int = 256, native_events: "NativeEventReferences | None" = None,
                   full_source: bool = False) -> tuple[Mapping[str, object], ...]:
        from sqlalchemy import and_, or_, select
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Trace page limit must be between 1 and 1000")
        t = self.table.c
        statement = select(self.table)
        if native_events is not None:
            statement = native_events.join_trace(statement, self.table, full_source=full_source)
        statement = statement.where(t.store_key == self.state.store_key)
        if after is not None:
            statement = statement.where(or_(t.run_id > after[0], and_(t.run_id == after[0], t.sequence > after[1])))
        return await self._rows(transaction, statement.order_by(t.run_id, t.sequence).limit(limit))

    async def delete(self, transaction: StateTransaction, run_ids: Sequence[str]) -> None:
        from sqlalchemy import delete
        if not isinstance(transaction, SqlStateTransaction):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        async def write(session: "AsyncSession") -> None:
            conditions = (self.table.c.store_key == self.state.store_key,
                          self.table.c.run_id.in_(tuple(map(_identity, run_ids))))
            await self.state.context.run_statement(session.execute(delete(self.table).where(*conditions)))
        await transaction.run_sql(self.state.context, write)


__all__ = ["SqlTraceEventStore", "history_identities"]
