#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace-owned immutable events and query projections."""

import copy
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ..core import canonical_json_bytes
from ._record_key import decode_key_components
from ._schema import build_runtime_sql_metadata
from ._plan import PersistenceDomain
from ._sql import SqlStateStore
from ._step_contracts import StepEvent, TOOL_ERROR_CODE_METADATA_KEY
from .codec import _decode_step_envelope
from .state import (FactQuery, SqlStateRowReader, SqlStateTransaction, StateTransaction,
                    StoredFact, STREAM_KEY_MAX_BYTES, record_key, stream_key)

if TYPE_CHECKING:
    from sqlalchemy import Table, Select
    from sqlalchemy.ext.asyncio import AsyncSession

_REQUEST = "linktools.ai.model_request_seq"
_MESSAGE = "linktools.ai.message_seq"
_PROMOTED = ("agent_run_id", "event_type", "step_index", "timestamp", "tool_call_id", "event_index")
_REQUEST_KINDS = ("MODEL_REQUEST_STARTED", "MODEL_REQUEST_SUCCEEDED", "MODEL_REQUEST_FAILED", "MODEL_REQUEST_CANCELLED")
_TOOL_KINDS = ("TOOL_CALL_STARTED", "TOOL_CALL_SUCCEEDED", "TOOL_CALL_FAILED")


def _identity(value: str | None) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value.encode("utf-8")) > 128:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.encode("utf-8")


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
        if payload["value"]["type"] == "step_event":
            if _decode_step_envelope(fact.data) != event:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            fields = payload["value"]["payload"]["fields"]
            for name in _PROMOTED:
                fields.pop(name)
            fields["metadata"]["$mapping"] = [pair for pair in fields["metadata"]["$mapping"]
                                                if pair[0] not in (_REQUEST, _MESSAGE, TOOL_ERROR_CODE_METADATA_KEY)]
        elif payload["value"]["type"] != "native_event_reference":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return dict(store_key=self.state.store_key, run_id=_identity(event.agent_run_id), sequence=fact.sequence,
                    event_index=event.event_index, kind=event.event_type, step_index=event.step_index,
                    timestamp=event.timestamp.isoformat(), order_key=_order(event.timestamp, fact.sequence),
                    request_identity=_identity(event.metadata.get(_REQUEST)),
                    request_number=_request_number(event.metadata.get(_REQUEST)),
                    message_identity=_identity(event.metadata.get(_MESSAGE)), tool_call_id=_identity(event.tool_call_id),
                    error_code=_identity(event.metadata.get(TOOL_ERROR_CODE_METADATA_KEY)), payload_json=payload)

    def row_to_fact(self, row: Mapping[str, object]) -> StoredFact:
        try:
            if row["store_key"] != self.state.store_key:
                raise ValueError("Trace partition differs")
            run_id = _text(row["run_id"])
            sequence = row["sequence"]
            if type(sequence) is not int or sequence < 1:
                raise ValueError("Trace sequence must be positive")
            for name in ("run_id", "request_identity", "request_number", "message_identity", "tool_call_id", "error_code"):
                if _identity(_text(row[name])) != row[name]:
                    raise ValueError("Trace identity is not canonical binary")
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
            if payload["value"]["type"] == "step_event":
                fields = payload["value"]["payload"]["fields"]
                if any(name in fields for name in _PROMOTED):
                    raise ValueError("Trace body duplicates promoted fields")
                fields.update(agent_run_id=run_id, event_type=row["kind"], step_index=row["step_index"],
                              timestamp={"$datetime": row["timestamp"]}, tool_call_id=_text(row["tool_call_id"]),
                              event_index=row["event_index"])
                metadata = fields["metadata"]["$mapping"]
                if not isinstance(metadata, list):
                    raise ValueError("Trace metadata must be a mapping sequence")
                previous_key = None
                for pair in metadata:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError("Trace metadata pair is malformed")
                    key = canonical_json_bytes(pair[0])
                    if previous_key is not None and previous_key >= key:
                        raise ValueError("Trace residual metadata is not canonical")
                    previous_key = key
                if any(pair[0] in (_REQUEST, _MESSAGE, TOOL_ERROR_CODE_METADATA_KEY) for pair in metadata):
                    raise ValueError("Trace metadata duplicates promoted fields")
                metadata.extend([key, value] for key, value in (
                    (_REQUEST, _text(row["request_identity"])), (_MESSAGE, _text(row["message_identity"])),
                    (TOOL_ERROR_CODE_METADATA_KEY, _text(row["error_code"]))) if value is not None)
                metadata.sort(key=lambda pair: canonical_json_bytes(pair[0]))
                event = _decode_step_envelope(payload)
                if not isinstance(event, StepEvent):
                    raise ValueError("Trace body is not an event")
                if row["order_key"] != _order(event.timestamp, sequence) or row["request_number"] != _request_number(event.metadata.get(_REQUEST)):
                    raise ValueError("Trace query projection differs")
            elif payload["value"]["type"] == "native_event_reference":
                reference = payload["value"]["payload"]
                if reference.get("run") != run_id or reference.get("index") != row["event_index"]:
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
                        sequences: Sequence[int] | None = None) -> tuple[Mapping[str, object], ...]:
        from sqlalchemy import select
        table = self.table
        statement = select(table).where(table.c.store_key == self.state.store_key,
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
        from sqlalchemy import and_, or_, select
        if history and max(map(len, (request_identities or (), message_identities or (), tool_call_ids or ()))) > 32:
            result = {}
            for offset in range(0, max(map(len, (request_identities or (), message_identities or (), tool_call_ids or ()))), 32):
                found = await self.select(transaction, run_id, high_water=high_water, history=True,
                    request_identities=(request_identities or ())[offset:offset + 32],
                    message_identities=(message_identities or ())[offset:offset + 32],
                    tool_call_ids=(tool_call_ids or ())[offset:offset + 32])
                result.update((fact.sequence, fact) for fact in found)
            return tuple(result[key] for key in sorted(result))
        t = self.table.c
        statement = select(self.table).where(t.store_key == self.state.store_key, t.run_id == _identity(run_id),
                                             t.sequence <= high_water)
        if history:
            roles = []
            if request_identities:
                roles.append(and_(t.kind.in_(_REQUEST_KINDS), t.request_identity.in_(tuple(map(_identity, request_identities)))))
            if message_identities:
                roles.append(and_(t.kind == "MODEL_REQUEST_SUCCEEDED", t.message_identity.in_(tuple(map(_identity, message_identities)))))
            if tool_call_ids:
                roles.append(and_(t.kind.in_(_TOOL_KINDS), t.tool_call_id.in_(tuple(map(_identity, tool_call_ids))),
                                  or_(t.kind != "TOOL_CALL_FAILED", t.error_code.is_(None), t.error_code != ErrorCode.TOOL_EFFECT_UNKNOWN.value.encode("utf-8"))))
            if not roles:
                return ()
            statement = statement.where(or_(*roles)).order_by(t.sequence)
        else:
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
                                     request_identities=tuple(event.metadata[_REQUEST] for event in events if _REQUEST in event.metadata),
                                     message_identities=tuple(event.metadata[_MESSAGE] for event in events if _MESSAGE in event.metadata),
                                     tool_call_ids=tuple(event.tool_call_id for event in events if event.tool_call_id is not None))
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
                   limit: int = 256) -> tuple[Mapping[str, object], ...]:
        from sqlalchemy import and_, or_, select
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Trace page limit must be between 1 and 1000")
        t = self.table.c
        statement = select(self.table).where(t.store_key == self.state.store_key)
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
