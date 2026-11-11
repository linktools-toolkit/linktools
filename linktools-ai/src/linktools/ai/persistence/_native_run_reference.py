#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace run projections whose common fields belong to a native registration."""

import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from pydantic import TypeAdapter
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent as NativeStepEvent

from ..errors import AIError, ErrorCode
from ..steps import native_store_locator, encode_native_replay_key
from ._run_registration import RunRegistrationReference
from ._plan import PersistenceDomain
from ._schema import build_runtime_sql_metadata
from ._sql import SqlStateStore
from ._step_contracts import AgentRunRecord
from ._step_schema import build_step_sql_metadata
from ._step_store import SqlStepStore, StepWriteReceipt
from .codec import _encode_step_envelope
from .state import RecordScanCursor, RecordQuery, SqlStateRecordReader, StateTransaction, StoredRecord

if TYPE_CHECKING:
    from sqlalchemy.sql import Select


_RUN_ADAPTER = TypeAdapter(RunRecord)
_TYPE = "run_reference"


def is_native_run_reference(data: Mapping[str, object]) -> bool:
    value = data.get("value")
    return isinstance(value, Mapping) and value.get("type") == _TYPE


class NativeRunReferences:
    """Resolve a native source in the same SELECT as its Trace sidecar.

    Only source-confirmed registrations may create references. The physical
    record keeps distinct Runtime projection differences, never
    another copy of the native lineage, agent or shared metadata.
    """

    def __init__(self, state: SqlStateStore, *, namespace: str, tenant_id: str) -> None:
        self._state = state
        self._store_key = SqlStepStore.scope_key(namespace=namespace, tenant_id=tenant_id)
        self._store_locator = native_store_locator(self._store_key)
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
                or native.store_key != self._store_key):
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
        source = (self._native.confirmed_run(run.agent_run_id) if source_receipt is None
                  else self._native.run_from_receipt(source_receipt))
        if source is None:
            return None
        return RunRegistrationReference.capture(source, run).to_data()

    @classmethod
    def decode_reference(cls, data: Mapping[str, object]) -> Mapping[str, object]:
        reference = RunRegistrationReference.from_data(data)
        return {"run": reference.run_id, "registration": reference.registration_id}

    def matches(self, data: Mapping[str, object], run: AgentRunRecord, *, transaction: StateTransaction) -> bool:
        expected = _encode_step_envelope(run)
        return any(physical == data and projection == expected
                   for physical, projection in self._projections.get(transaction, {}).values())

    def project(self, data: Mapping[str, object], source: RunRecord, *, store_key: bytes) -> AgentRunRecord:
        return self.project_source(data, source, store_key=store_key, expected_store_key=self._store_key)

    @classmethod
    def project_source(cls, data: Mapping[str, object], source: RunRecord, *,
                       store_key: bytes, expected_store_key: bytes) -> AgentRunRecord:
        """Project an immutable registration without a second whole-run identity."""
        if store_key != expected_store_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return RunRegistrationReference.from_data(data).project(source)

    def remember_prepared(self, transaction: StateTransaction, record: StoredRecord, run: AgentRunRecord,
                          *, source_receipt: StepWriteReceipt | None = None,
                          registration: RunRecord | None = None) -> None:
        """Retain a projection proven by its receipt or prepared canonical owner."""
        if not is_native_run_reference(record.data):
            return
        expected = (self.encode(run, source_receipt=source_receipt)
                    if source_receipt is not None or registration is None
                    else RunRegistrationReference.capture(registration, run).to_data())
        if expected != record.data:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._projections.setdefault(transaction, {})[record.key] = (record.data, _encode_step_envelope(run))

    async def read_physical(self, transaction: StateTransaction, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        """Resolve existing sources in the write prefetch, retaining physical CAS records."""
        await self.read(transaction, keys=keys)
        assert isinstance(transaction, SqlStateRecordReader)
        return {key: record for key, record in transaction.cached_records(keys).items() if record is not None}

    def statement(self, *, keys: Sequence[bytes] | None = None, query: RecordQuery | None = None,
                  scan: tuple[RecordScanCursor | None, int] | None = None,
                  expected_start: NativeStepEvent | None = None,
                  expected_terminal: NativeStepEvent | None = None) -> "Select":
        from ._sql import _key_matches, _keys_match
        from sqlalchemy import JSON, and_, bindparam, cast, select, type_coerce
        from sqlalchemy import literal_column
        from sqlalchemy.sql import visitors
        from sqlalchemy.sql.elements import BindParameter

        records, runs = self._records, self._runs
        encoded = type_coerce(records.c.payload_json, JSON)[()].as_string()
        payload = cast(encoded, JSON) if self._state.context.dialect.name == "postgresql" else type_coerce(encoded, JSON)
        reference = payload["value"]["payload"]
        source_run = reference["run_id"].as_string()
        run_join = and_(
            records.c.kind == "agent_run", runs.c.store_locator == self._store_locator, runs.c.store_key == self._store_key,
            runs.c.run_id == source_run,
            runs.c.registration_id.is_not_distinct_from(reference["registration_id"].as_string()),
        )
        source_columns = tuple(column.label("native_" + column.name) for column in runs.c)
        expected = expected_start if expected_start is not None else expected_terminal
        if expected is not None:
            if self._native is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            # Inline Run views also require full pending lifecycle proof in
            # this existing SELECT, never a second source read.
            run_join = and_(records.c.kind == "agent_run", runs.c.store_locator == self._store_locator, runs.c.store_key == self._store_key,
                            runs.c.run_id == bindparam("source_run_id", expected.run_id))
        joined = records.outerjoin(runs, run_join)
        for prefix, event in (("start", expected_start), ("terminal", expected_terminal)):
            if event is None:
                continue
            events = self._events if prefix == "start" else self._events.alias("terminal_source")
            joined = joined.outerjoin(events, and_(events.c.store_locator == runs.c.store_locator, events.c.store_key == runs.c.store_key,
                events.c.run_id == runs.c.run_id,
                events.c.idempotency_key == bindparam("source_event_key" if prefix == "start" else "source_terminal_key",
                                                encode_native_replay_key(event.idempotency_key))))
            names = tuple(field.name for field in fields(NativeStepEvent)) + (
                "outcome", "store_key", "run_fields", "body_sha256", "fields_sha256",
            )
            source_columns += tuple(events.c[name].label(prefix + "_" + name) for name in names)
        statement = select(records, *source_columns).select_from(joined)
        # Render typed invariant JOIN literals without consuming the record
        # reader's existing bind budget. Keys and query inputs remain bound.
        def inline_constant(node):
            if not isinstance(node, BindParameter) or node.key in {"source_run_id", "source_event_key", "source_terminal_key"}:
                return None
            if isinstance(node.value, bytes):
                hex_value = node.value.hex()
                literal = ("E'\\\\x" + hex_value + "'::bytea" if self._state.context.dialect.name == "postgresql"
                           else "X'" + hex_value + "'")
                return literal_column(literal, type_=node.type)
            return bindparam(None, node.value, type_=node.type, literal_execute=True)

        statement = visitors.replacement_traverse(statement, {}, inline_constant).where(records.c.store_key == self._state.store_key)
        if keys is not None:
            statement = statement.where(_keys_match(records.c.key, tuple(keys)))
        if query is not None:
            statement = statement.where(records.c.kind == "agent_run")
            if query.scope_key is not None:
                statement = statement.where(_key_matches(records.c.scope_key, query.scope_key))
            if query.parent_key is not None:
                statement = statement.where(_key_matches(records.c.parent_key, query.parent_key))
            statement = statement.order_by(records.c.sort_prefix, records.c.sort_key, records.c.key)
        if scan is not None:
            after, limit = scan
            if type(limit) is not int or not 1 <= limit <= 1000:
                raise ValueError("record page limit must be between 1 and 1000")
            if after is not None:
                statement = statement.where(records.c.key > after.key)
            statement = statement.order_by(records.c.key).limit(limit)
        return statement

    async def read(
        self, transaction: StateTransaction, *, keys: Sequence[bytes] | None = None, query: RecordQuery | None = None,
        scan: tuple[RecordScanCursor | None, int] | None = None,
        expected_start: NativeStepEvent | None = None,
        expected_terminal: NativeStepEvent | None = None,
        expected_run: RunRecord | None = None,
    ) -> tuple[StoredRecord, ...]:
        expected = expected_start if expected_start is not None else expected_terminal
        if (expected is None) != (expected_run is None):
            raise ValueError("source readback requires registration and lifecycle source")
        if expected is not None and any(event is not None and event.run_id != expected_run.run_id
                                        for event in (expected_start, expected_terminal)):
            raise ValueError("source readback lifecycle belongs to another registration")
        if keys is not None and not keys:
            return ()
        if not isinstance(transaction, SqlStateRecordReader):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        projected = self._projections.setdefault(transaction, {})
        values: dict[bytes, StoredRecord] = {}
        missing = keys
        if keys is not None and expected is None:
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
        batches = (missing,)
        if missing:
            # Each key binds both its indexed locator and complete raw identity.
            # Count fixed JOIN/filter binds in the actual dialect before batching.
            probe = self.statement(keys=missing[:1], query=query, scan=scan,
                                   expected_start=expected_start, expected_terminal=expected_terminal)
            compiled = probe.compile(dialect=self._state.context.engine.dialect,
                                     compile_kwargs={"render_postcompile": True})
            count = len(compiled.params) if compiled.positiontup is None else len(compiled.positiontup)
            batch_size = 1 + (999 - count) // 2
            if batch_size < 1:
                raise ValueError("native record read exceeds the SQL parameter budget")
            batches = tuple(missing[offset:offset + batch_size] for offset in range(0, len(missing), batch_size))
        rows = []
        for batch in batches:
            rows.extend(await transaction.read_joined_records(
                self._state.context, self.statement(keys=batch, query=query, scan=scan, expected_start=expected_start,
                    expected_terminal=expected_terminal), keys=batch or (),
            ))
        ordered: list[StoredRecord] = []
        for record, row in rows:
            if expected is not None:
                try:
                    run_source = {column.name: row["native_" + column.name] for column in self._runs.c}
                    for prefix, event in (("start", expected_start), ("terminal", expected_terminal)):
                        if event is None:
                            continue
                        source = {key.removeprefix(prefix + "_"): value for key, value in row.items()
                                  if key.startswith(prefix + "_")}
                        actual = SqlStepStore.decode_started_row(source,
                            expected_store_key=self._store_key, run=run_source)
                        if actual != event:
                            raise ValueError("lifecycle source differs from its provisional body")
                    run_source["metadata"] = json.loads(run_source["metadata"])
                    actual_run = _RUN_ADAPTER.validate_python(run_source)
                    if actual_run != expected_run:
                        raise ValueError("lifecycle source differs from its provisional registration")
                except (TypeError, ValueError, KeyError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if is_native_run_reference(record.data):
                try:
                    source = {column.name: row["native_" + column.name] for column in self._runs.c}
                    source["metadata"] = json.loads(source["metadata"])
                    native = _RUN_ADAPTER.validate_python(source)
                except (TypeError, ValueError, KeyError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                run = self.project(record.data, native, store_key=row["native_store_key"])
                data = _encode_step_envelope(run)
                projected[record.key] = (record.data, data)
                record = replace(record, data=data)
            ordered.append(record)
            values[record.key] = record
        return (tuple(ordered) if keys is None else
                tuple(values[key] for key in dict.fromkeys(keys) if key in values))


__all__ = ["NativeRunReferences", "is_native_run_reference"]
