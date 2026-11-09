#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace run projections whose common fields belong to a native registration."""

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from datetime import datetime
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from pydantic import TypeAdapter
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent as NativeStepEvent

from ..core import canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ._official_step_reader import OfficialStepReader
from ._plan import PersistenceDomain
from ._schema import build_runtime_sql_metadata
from ._sql import SqlStateStore
from ._step_contracts import AgentRunRecord
from ._step_schema import build_step_sql_metadata
from ._step_store import SqlStepStore, StepWriteReceipt
from .codec import _encode_step_envelope, decode_envelope, encode_envelope
from .state import RecordScanCursor, RecordQuery, SqlStateRecordReader, StateTransaction, StoredRecord

if TYPE_CHECKING:
    from sqlalchemy.sql import Select


_RUN_ADAPTER = TypeAdapter(RunRecord)
_TYPE = "native_agent_run_reference"
_KEYS = frozenset({"run", "run_digest", "registration", "projection_sha256", "metadata", "removed", "started_at"})


def is_native_run_reference(data: Mapping[str, object]) -> bool:
    value = data.get("value")
    return isinstance(value, Mapping) and value.get("type") == _TYPE


class NativeRunReferences:
    """Resolve a native source in the same SELECT as its Trace sidecar.

    Only source-confirmed registrations may create references. The physical
    record keeps distinct Runtime evidence and a byte-integrity digest, never
    another copy of the native lineage, agent or shared metadata.
    """

    def __init__(self, state: SqlStateStore, *, namespace: str, tenant_id: str) -> None:
        self._state = state
        self._store_digest = SqlStepStore.scope_digest(namespace=namespace, tenant_id=tenant_id)
        self._native: SqlStepStore | None = None
        self._confirmed: OrderedDict[str, str] = OrderedDict()
        self._projections: WeakKeyDictionary[StateTransaction, dict[bytes, tuple[Mapping[str, object], Mapping[str, object]]]] = WeakKeyDictionary()
        metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.TRACE}))
        build_step_sql_metadata(metadata=metadata)
        self._records = metadata.tables["ai_state_records"]
        self._runs = metadata.tables["ai_step_runs"]
        self._events = metadata.tables["ai_step_events"]

    def bind(self, native: "SqlStepStore") -> None:
        if (native.storage_context.engine is not self._state.context.engine
                or native.store_digest != self._store_digest):
            return
        if self._native is not None and self._native is not native:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._native = native

    def confirm(self, record: RunRecord) -> None:
        if (self._native is None or not record.registration_id
                or self._native.confirmed_run(record.run_id) != record):
            return
        previous = self._confirmed.get(record.run_id)
        if previous is not None and previous != record.registration_id:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._confirmed[record.run_id] = record.registration_id
        self._confirmed.move_to_end(record.run_id)
        while len(self._confirmed) > 128:
            self._confirmed.popitem(last=False)

    def invalidate(self, run_id: str) -> None:
        self._confirmed.pop(run_id, None)

    def clear(self) -> None:
        self._confirmed.clear()
        self._projections.clear()

    def encode(self, run: AgentRunRecord, *, source_receipt: StepWriteReceipt | None = None) -> dict[str, object] | None:
        if self._native is None:
            return None
        if source_receipt is None:
            source = self._native.confirmed_run(run.agent_run_id)
            if source is None or self._confirmed.get(run.agent_run_id) != source.registration_id:
                return None
        else:
            # The caller just published this source in the enclosing transaction.
            # Do not promote its provisional receipt to committed cache knowledge.
            source = self._native.run_from_receipt(source_receipt)
            if source is None or source.run_id != run.agent_run_id or not source.registration_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        projected = OfficialStepReader.project_run(source)
        if (projected.agent_conversation_id, projected.parent_agent_run_id, projected.agent_id) != (
                run.agent_conversation_id, run.parent_agent_run_id, run.agent_id):
            return None
        encoded_run = _encode_step_envelope(run)
        reference = encode_envelope({"type": _TYPE, "payload": {
            "run": source.run_id,
            "run_digest": hashlib.sha256(source.run_id.encode()).hexdigest(),
            "registration": source.registration_id,
            "projection_sha256": canonical_sha256(encoded_run),
            "metadata": {key: value for key, value in run.metadata.items()
                         if projected.metadata.get(key) != value},
            "removed": sorted(projected.metadata.keys() - run.metadata.keys()),
            "started_at": (None if run.started_at.isoformat() == projected.started_at.isoformat()
                           else run.started_at.isoformat()),
        }})
        return reference if len(canonical_json_bytes(reference)) < len(canonical_json_bytes(encoded_run)) else None

    @classmethod
    def decode_reference(cls, data: Mapping[str, object]) -> Mapping[str, object]:
        envelope = decode_envelope(data)
        value = envelope.value
        payload = value.get("payload")
        if (set(value) != {"type", "payload"} or value["type"] != _TYPE
                or not isinstance(payload, Mapping) or set(payload) != _KEYS):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for key in ("run_digest", "projection_sha256"):
            digest = payload[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if any(not isinstance(payload[key], str) or not payload[key] for key in ("run", "registration")):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if payload["run_digest"] != hashlib.sha256(payload["run"].encode()).hexdigest():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        metadata, removed = payload["metadata"], payload["removed"]
        if (not isinstance(metadata, Mapping) or any(not isinstance(key, str) or not isinstance(value, str)
                                                   for key, value in metadata.items())
                or not isinstance(removed, (list, tuple)) or any(not isinstance(key, str) for key in removed)
                or list(removed) != sorted(set(removed)) or set(removed).intersection(metadata)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        started = payload["started_at"]
        if started is not None:
            try:
                timestamp = datetime.fromisoformat(started)
                if timestamp.tzinfo is None or timestamp.isoformat() != started:
                    raise ValueError("invalid timestamp")
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return payload

    def matches(self, data: Mapping[str, object], run: AgentRunRecord) -> bool:
        reference = self.decode_reference(data)
        return (reference["run"] == run.agent_run_id
                and reference["projection_sha256"] == canonical_sha256(_encode_step_envelope(run)))

    def project(self, data: Mapping[str, object], source: RunRecord, *, store_digest: str) -> AgentRunRecord:
        reference = self.decode_reference(data)
        if (store_digest != self._store_digest or reference["run"] != source.run_id
                or reference["run_digest"] != hashlib.sha256(source.run_id.encode()).hexdigest()
                or reference["registration"] != source.registration_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        run = OfficialStepReader.project_run(source)
        for key in reference["removed"]:
            run.metadata.pop(key, None)
        run.metadata.update(reference["metadata"])
        if reference["started_at"] is not None:
            run.started_at = datetime.fromisoformat(reference["started_at"])
        if not self.matches(data, run):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return run

    def remember_prepared(self, transaction: StateTransaction, record: StoredRecord, run: AgentRunRecord,
                          *, source_receipt: StepWriteReceipt | None = None) -> None:
        """Retain a source-proven projection only for its exact transaction row."""
        if not is_native_run_reference(record.data):
            return
        if self.encode(run, source_receipt=source_receipt) != record.data:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._projections.setdefault(transaction, {})[record.key_digest] = (record.data, _encode_step_envelope(run))

    async def read_physical(self, transaction: StateTransaction, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        """Resolve existing sources in the write prefetch, retaining physical CAS records."""
        await self.read(transaction, keys=keys)
        assert isinstance(transaction, SqlStateRecordReader)
        return {key: record for key, record in transaction.cached_records(keys).items() if record is not None}

    def statement(self, *, keys: Sequence[bytes] | None = None, query: RecordQuery | None = None,
                  scan: tuple[RecordScanCursor | None, int] | None = None,
                  expected_start: NativeStepEvent | None = None) -> "Select":
        from sqlalchemy import JSON, and_, bindparam, cast, select, type_coerce
        from sqlalchemy.sql import visitors
        from sqlalchemy.sql.elements import BindParameter

        records, runs = self._records, self._runs
        encoded = type_coerce(records.c.payload_json, JSON)[()].as_string()
        payload = cast(encoded, JSON) if self._state.context.dialect.name == "postgresql" else type_coerce(encoded, JSON)
        reference = payload["value"]["payload"]
        source_run = reference["run"].as_string()
        # The physical digest narrows the index, while the original id prevents
        # treating a colliding lookup digest as a semantic identity.
        run_join = and_(
            records.c.kind == "agent_run", runs.c.store_digest == self._store_digest,
            runs.c.run_digest == reference["run_digest"].as_string(), runs.c.run_id == source_run,
            runs.c.registration_id == reference["registration"].as_string(),
        )
        source_columns = tuple(column.label("native_" + column.name) for column in runs.c)
        if expected_start is not None:
            if self._native is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            # An inline small Run still has a managed start. Verify its exact
            # source using this existing Run SELECT, never a second read.
            run_join = and_(records.c.kind == "agent_run", runs.c.store_digest == self._store_digest,
                            runs.c.run_digest == bindparam("source_run_digest", hashlib.sha256(expected_start.run_id.encode()).hexdigest()),
                            runs.c.run_id == bindparam("source_run_id", expected_start.run_id))
        joined = records.outerjoin(runs, run_join)
        if expected_start is not None:
            events = self._events
            joined = joined.outerjoin(events, and_(events.c.store_digest == runs.c.store_digest,
                events.c.run_digest == runs.c.run_digest, events.c.run_id == runs.c.run_id,
                events.c.key_digest == bindparam("source_event_key", self._native.event_key_digest(expected_start))))
            names = tuple(field.name for field in fields(NativeStepEvent)) + (
                "store_digest", "run_digest", "key_digest", "run_fields", "body_sha256", "fields_sha256",
            )
            source_columns += tuple(events.c[name].label("start_" + name) for name in names)
        statement = select(records, *source_columns).select_from(joined)
        # Render typed invariant JOIN literals without consuming the record
        # reader's existing bind budget. Keys and query inputs remain bound.
        statement = visitors.replacement_traverse(statement, {}, lambda node: (
            bindparam(None, node.value, type_=node.type, literal_execute=True)
            if isinstance(node, BindParameter) and node.key not in {"source_run_digest", "source_run_id", "source_event_key"}
            else None
        )).where(records.c.store_digest == self._state.store_digest.hex())
        if keys is not None:
            statement = statement.where(records.c.key_digest.in_(tuple(key.hex() for key in keys)))
        if query is not None:
            statement = statement.where(records.c.kind == "agent_run")
            if query.scope_digest is not None:
                statement = statement.where(records.c.scope_digest == query.scope_digest.hex())
            if query.parent_digest is not None:
                statement = statement.where(records.c.parent_digest == query.parent_digest.hex())
            statement = statement.order_by(records.c.sort_key, records.c.key_digest)
        if scan is not None:
            after, limit = scan
            if type(limit) is not int or not 1 <= limit <= 1000:
                raise ValueError("record page limit must be between 1 and 1000")
            if after is not None:
                statement = statement.where(records.c.key_digest > after.key_digest.hex())
            statement = statement.order_by(records.c.key_digest).limit(limit)
        return statement

    async def read(
        self, transaction: StateTransaction, *, keys: Sequence[bytes] | None = None, query: RecordQuery | None = None,
        scan: tuple[RecordScanCursor | None, int] | None = None,
        expected_start: NativeStepEvent | None = None,
        expected_run: RunRecord | None = None,
    ) -> tuple[StoredRecord, ...]:
        if (expected_start is None) != (expected_run is None):
            raise ValueError("source readback requires both registration and start")
        if keys is not None and not keys:
            return ()
        if not isinstance(transaction, SqlStateRecordReader):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        projected = self._projections.setdefault(transaction, {})
        values: dict[bytes, StoredRecord] = {}
        missing = keys
        if keys is not None and expected_start is None:
            cached = transaction.cached_records(keys)
            unresolved: list[bytes] = []
            for key in keys:
                if key not in cached:
                    unresolved.append(key)
                    continue
                record = cached[key]
                if record is None:
                    continue
                if not is_native_run_reference(record.data):
                    values[key] = record
                elif key in projected and projected[key][0] == record.data:
                    values[key] = replace(record, data=projected[key][1])
                else:
                    unresolved.append(key)
            missing = tuple(dict.fromkeys(unresolved))
            if not missing:
                return tuple(values[key] for key in dict.fromkeys(keys) if key in values)
        rows = await transaction.read_joined_records(
            self._state.context, self.statement(keys=missing, query=query, scan=scan, expected_start=expected_start), keys=missing or (),
        )
        ordered: list[StoredRecord] = []
        for record, row in rows:
            if expected_start is not None:
                try:
                    source = {key.removeprefix("start_"): value for key, value in row.items()
                              if key.startswith("start_")}
                    run_source = {column.name: row["native_" + column.name] for column in self._runs.c}
                    actual_start = SqlStepStore.decode_started_row(source,
                        expected_store_digest=self._store_digest, run=run_source)
                    run_source["metadata"] = json.loads(run_source["metadata"])
                    actual_run = _RUN_ADAPTER.validate_python(run_source)
                    if actual_start != expected_start or actual_run != expected_run:
                        raise ValueError("terminal source differs from its provisional registration or start")
                except (TypeError, ValueError, KeyError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if is_native_run_reference(record.data):
                try:
                    source = {column.name: row["native_" + column.name] for column in self._runs.c}
                    source["metadata"] = json.loads(source["metadata"])
                    native = _RUN_ADAPTER.validate_python(source)
                except (TypeError, ValueError, KeyError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                run = self.project(record.data, native, store_digest=row["native_store_digest"])
                data = _encode_step_envelope(run)
                projected[record.key_digest] = (record.data, data)
                record = replace(record, data=data)
            ordered.append(record)
            values[record.key_digest] = record
        return (tuple(ordered) if keys is None else
                tuple(values[key] for key in dict.fromkeys(keys) if key in values))


__all__ = ["NativeRunReferences", "is_native_run_reference"]
