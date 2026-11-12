#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded portable state snapshots and offline object maintenance."""

import hashlib
import json
from bisect import bisect_left, bisect_right
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from ._tool_operation_content import expand_tool_content, tool_content_size, expand_native_tool_content
from ._native_tool_result import restore_native_tool_result_payload
from ._execution_content import (ExecutionContent, expand_snapshot_execution_content, snapshot_execution_content_size,
    expand_native_execution_content, restore_native_text_payload)
from .codec import (decode_envelope, execution_payload_slots, native_text_reference, binding_reference,
    tool_payload_slots, native_tool_result_reference)

from ..core import (
    JsonValue,
    canonical_json_bytes,
    canonical_sha256,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..steps import native_store_locator, NativeStepSource, rekey_native_step_sources, encode_native_store_key
from ._step_backup import NativeStepBackup
from ._native_backup_projection import NativeBackupProjections
from ..storage import ObjectRef, ObjectStore, read_object
from ..task import TaskGraphAdmission, TaskGraphSubmission
from ._runtime_identity import task_graph_binding_capture_key
from ._run_registration import RunRegistrationStore, RunRegistrationReference, RunRegistrationCleanupResult
from pydantic import TypeAdapter
from pydantic_ai_harness.step_persistence import RunRecord
from ._native_event_reference import NativeEventReferences
from ._sql import SqlStateStore
from ._archive import archive_content_scope, ContentRecord
from ._message_root import NativeMessageRoot
from ..model.messages import decode_model_message_payloads
from ..model.attachments import read_attachment_payload, attachment_payload_objects
from ._archive_closure import (
    ArchiveClosure,
    NativeMessageClosure,
    iter_native_message_roots,
    rekey_native_message_roots,
    validate_native_message_content,
    iter_archive_content_refs,
    iter_archive_roots,
    rekey_archive_roots,
)
from ._payload import decode_payload_bytes
from .captures import (
    CAPTURE_RECORD_KINDS, capture_retention_value, map_capture_references, capture_state_closure,
    expand_snapshot_capture, snapshot_capture_size, restore_capture_value,
)
from ._capability_marker import has_capability_partition_marker
from ._plan import PersistenceDomain
from .state import (
    STREAM_KEY_MAX_BYTES,
    validate_stream_key,
    FactScanCursor,
    OperationScanCursor,
    RecordScanCursor,
    StateStorageGroup,
    StateStore,
    StateTransaction,
    SqlStateRowReader,
    SqlStateTransaction,
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
)
from ._snapshot import (
    SnapshotLimits,
    SnapshotExclusiveGuard,
    validate_snapshot_native_rows,
    encode_native_message_row, decode_native_message_row, encode_native_sequence_row, decode_native_sequence_row,
    snapshot_object_ref_payload as _object_ref_payload,
    snapshot_object_ref_from_payload as _object_ref_from_payload,
)
from ._object_cleanup import (
    ObjectCleanupResult, CaptureCleanupResult,
    purge_unreferenced_captures as _purge_unreferenced_captures,
    purge_unreferenced_objects as _purge_unreferenced_objects,
)
from ._snapshot_validation import (
    canonical_snapshot_indexes,
    validate_snapshot_domain,
    validate_snapshot_references,
)
from .codec import (
    decode_alias,
    _decode_enveloped_domain,
    iter_runtime_object_dependencies,
    decode_fact,
    decode_operation,
    decode_record,
    encode_fact,
    encode_operation,
    encode_record,
    _encode_step_envelope,
    iter_runtime_object_refs,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from ._step_store import SqlStepStore


class _SnapshotReader(Protocol):
    async def validate_trace_prefix(self, records: Sequence[StoredRecord], facts: Sequence[StoredFact]) -> None: ...

    async def scan_trace_events_page(
        self, *, after: tuple[bytes, int] | None, limit: int,
    ) -> tuple[tuple[bytes, int], tuple[StoredFact, ...]] | None: ...

    async def get_operation(self, key: bytes) -> StoredOperation | None: ...

    async def scan_records_page(
        self, *, after: RecordScanCursor | None, limit: int,
    ) -> tuple[StoredRecord, ...]: ...

    async def scan_facts_page(
        self, *, after: FactScanCursor | None, limit: int,
    ) -> tuple[StoredFact, ...]: ...

    async def scan_operations_page(
        self, *, after: OperationScanCursor | None, limit: int,
    ) -> tuple[StoredOperation, ...]: ...


class _StateSnapshotReader:
    def __init__(self, store: StateStore, *, native_run_scope: tuple[str, str] | None = None,
                 trace_scope: tuple[str, str] | None = None) -> None:
        from ._trace_event_store import SqlTraceEventStore
        self._store = store
        scope = trace_scope or native_run_scope
        self._trace_scope = scope
        self._trace_events = (SqlTraceEventStore(store, namespace=scope[0], tenant_id=scope[1])
                              if scope is not None and isinstance(store, SqlStateStore) else None)
        self._native_events = (NativeEventReferences(store, namespace=native_run_scope[0], tenant_id=native_run_scope[1])
                               if native_run_scope is not None and isinstance(store, SqlStateStore) else None)

    async def validate_trace_prefix(self, records: Sequence[StoredRecord], facts: Sequence[StoredFact]) -> None:
        from ._step_contracts import AgentRunRecord
        from .state import stream_key
        if self._trace_scope is None:
            return
        namespace, tenant_id = self._trace_scope
        streams = tuple(stream_key(namespace, tenant_id, "trace", "event", (
            RunRegistrationReference.from_data(record.data).run_id,
        )) for record in records if record.kind == "agent_run")
        counts = {}
        for fact in facts:
            if fact.kind == "step_event":
                counts[fact.stream_key] = counts.get(fact.stream_key, 0) + 1
                if fact.sequence != counts[fact.stream_key]:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if set(counts) - set(streams):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for offset in range(0, len(streams), 128):
            batch = streams[offset:offset + 128]
            high_water = await self._store.read(lambda tx: tx.get_sequences(batch))
            if any(high_water.get(key, 0) != counts.get(key, 0) for key in batch):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        return await self._store.read(lambda tx: tx.get_operation(key))

    async def scan_records_page(
        self, *, after: RecordScanCursor | None, limit: int,
    ) -> tuple[StoredRecord, ...]:
        return await self._store.read(lambda tx: tx.scan_records_page(after=after, limit=limit))

    async def scan_facts_page(
        self, *, after: FactScanCursor | None, limit: int,
    ) -> tuple[StoredFact, ...]:
        return await self._store.read(lambda tx: tx.scan_facts_page(after=after, limit=limit))

    async def scan_trace_events_page(
        self, *, after: tuple[bytes, int] | None, limit: int,
    ) -> tuple[tuple[bytes, int], tuple[StoredFact, ...]] | None:
        if self._trace_events is None:
            return None
        async def read(transaction: StateTransaction):
            rows = await self._trace_events.scan(transaction, after=after, limit=limit,
                                                 native_events=self._native_events, full_source=True)
            if not rows:
                return None
            facts = tuple(self._trace_events.row_to_fact(row) for row in rows)
            if self._native_events is not None:
                facts = self._native_events.project_trace_facts(facts, rows=rows, full_source=True)
            return (rows[-1]["run_id"], rows[-1]["sequence"]), facts
        return await self._store.read(read)

    async def scan_operations_page(
        self, *, after: OperationScanCursor | None, limit: int,
    ) -> tuple[StoredOperation, ...]:
        return await self._store.read(lambda tx: tx.scan_operations_page(after=after, limit=limit))


@dataclass(frozen=True, slots=True)
class _CapturedSnapshotReader:
    records: tuple[StoredRecord, ...]
    facts: tuple[StoredFact, ...]
    operations: tuple[StoredOperation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", tuple(sorted(self.facts, key=lambda value: (value.stream_key, value.sequence))))
        object.__setattr__(self, "records", tuple(sorted(self.records, key=lambda value: (value.kind, value.key))))

    async def validate_trace_prefix(self, records: Sequence[StoredRecord], facts: Sequence[StoredFact]) -> None:
        return None

    async def scan_trace_events_page(
        self, *, after: tuple[bytes, int] | None, limit: int,
    ) -> tuple[tuple[bytes, int], tuple[StoredFact, ...]] | None:
        return None

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        index = bisect_left(self.operations, key, key=lambda value: value.key)
        return (self.operations[index] if index < len(self.operations)
                and self.operations[index].key == key else None)

    async def scan_records_page(self, *, after: RecordScanCursor | None, limit: int) -> tuple[StoredRecord, ...]:
        start = 0 if after is None else bisect_right(self.records, (after.kind, after.key),
                                                    key=lambda value: (value.kind, value.key))
        return self.records[start:start + limit]

    async def scan_facts_page(self, *, after: FactScanCursor | None, limit: int) -> tuple[StoredFact, ...]:
        start = 0 if after is None else bisect_right(self.facts, (after.stream_key, after.sequence),
                                                    key=lambda value: (value.stream_key, value.sequence))
        return self.facts[start:start + limit]

    async def scan_operations_page(self, *, after: OperationScanCursor | None, limit: int) -> tuple[StoredOperation, ...]:
        start = 0 if after is None else bisect_right(self.operations, after.key, key=lambda value: value.key)
        return self.operations[start:start + limit]


@dataclass(slots=True)
class _SnapshotCaptureBudget:
    """Bound the detached source inventory separately from final wire accounting."""

    limits: SnapshotLimits
    entries: int = 0
    byte_size: int = 0

    def page_size(self, requested: int) -> int:
        return min(requested, max(1, self.limits.max_entries - self.entries))

    def check_bytes(self, size: int) -> None:
        if self.byte_size + size > self.limits.max_bytes:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

    def reserve_bytes(self, size: int) -> None:
        self.check_bytes(size)
        self.byte_size += size

    def accept(self, value: object) -> None:
        size = len(canonical_json_bytes(value))
        if self.entries + 1 > self.limits.max_entries or self.byte_size + size > self.limits.max_bytes:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        self.entries += 1
        self.byte_size += size


async def _capture_snapshot_reader(reader: _SnapshotReader, *, budget: _SnapshotCaptureBudget, page_size: int = 128, records_only: bool = False) -> _CapturedSnapshotReader:
    records, facts, operations = [], [], []
    cursor = None
    while True:
        limit = budget.page_size(page_size)
        page = await reader.scan_records_page(after=cursor, limit=limit)
        for value in page:
            if value.kind in {"history_association", "step_run"}:
                continue
            budget.accept(encode_record(value))
            records.append(value)
        if len(page) < limit:
            break
        cursor = RecordScanCursor(page[-1].kind, page[-1].key)
    if records_only:
        return _CapturedSnapshotReader(tuple(records), (), ())
    cursor = None
    while True:
        limit = budget.page_size(page_size)
        page = await reader.scan_facts_page(after=cursor, limit=limit)
        for value in page:
            budget.accept(encode_fact(value))
            facts.append(value)
        if len(page) < limit:
            break
        cursor = FactScanCursor(page[-1].stream_key, page[-1].sequence)
    trace_cursor = None
    while True:
        limit = budget.page_size(page_size)
        page = await reader.scan_trace_events_page(after=trace_cursor, limit=limit)
        if page is None:
            break
        trace_cursor, events = page
        for value in events:
            budget.accept(encode_fact(value))
            facts.append(value)
        if len(events) < limit:
            break
    cursor = None
    while True:
        limit = budget.page_size(page_size)
        page = await reader.scan_operations_page(after=cursor, limit=limit)
        for value in page:
            budget.accept(encode_operation(value))
            operations.append(value)
        if len(page) < limit:
            break
        cursor = OperationScanCursor(page[-1].key)
    await reader.validate_trace_prefix(records, facts)
    return _CapturedSnapshotReader(tuple(records), tuple(facts), tuple(operations))


@dataclass(frozen=True, slots=True)
class _SnapshotArchive:
    scope_id: str
    domain: PersistenceDomain
    reader: _SnapshotReader


@dataclass(frozen=True, slots=True)
class _SnapshotExportSource:
    namespace: str
    tenant_id: str
    readers: Mapping[PersistenceDomain, _SnapshotReader]
    archives: AsyncIterator[_SnapshotArchive]
    object_stores: Mapping[PersistenceDomain, ObjectStore]
    archive_object_stores: Mapping[PersistenceDomain, ObjectStore]
    native_rows: Mapping[str, tuple[PersistenceDomain, list[dict[str, object]], list[dict[str, object]]]] | None = None
    native_checkpoints: Mapping[str, NativeStepBackup] | None = None
    native_projections: Mapping[str, _CapturedSnapshotReader] | None = None
    native_payloads: Mapping[str, Mapping[tuple[bytes, str], bytes]] | None = None
    run_registrations: Mapping[str, RunRecord] | None = None

    def archive_object_store(self, domain: PersistenceDomain) -> ObjectStore:
        return self.archive_object_stores[domain]


@dataclass(frozen=True, slots=True)
class SnapshotResources:
    """Explicit routed resources shared by snapshot and offline cleanup."""

    namespace: str
    tenant_id: str
    durable_domains: frozenset[PersistenceDomain]
    stores: Mapping[PersistenceDomain, StateStore]
    object_stores: Mapping[PersistenceDomain, ObjectStore]
    read_only: bool
    portable_export: bool = True
    archive_object_stores: Mapping[PersistenceDomain, ObjectStore] | None = None
    # Borrow the live mapping so cached maintenance sees later capability bindings.
    capability_stores: Mapping[str, StateStore] | None = None
    checkpoint_backup_source: Callable[[NativeStepSource], Awaitable["SqlStepStore"]] | None = None
    run_registrations: RunRegistrationStore | None = None

    def archive_object_store(self, domain: PersistenceDomain) -> ObjectStore:
        return (self.archive_object_stores if self.archive_object_stores is not None else self.object_stores)[domain]


class RuntimeSnapshotOperations:
    """Operate on all routed state, including currently inactive domains."""

    def __init__(
        self,
        resources: SnapshotResources,
        *,
        require_ready: Callable[[], None],
    ) -> None:
        self._resources = resources
        self._require_ready = require_ready

    async def purge_unreferenced_run_registrations(
        self, candidates: tuple[str, ...], *,
        exclusive: Callable[[tuple[StateStorageGroup, ...]], AbstractAsyncContextManager[None]],
        complete_reference_stores: tuple[StateStore, ...] | None = None, limit: int = 100,
    ) -> RunRegistrationCleanupResult:
        """Explicit offline reclamation, never automatic domain-release GC.

        Supply every live State source borrowing this canonical owner, including
        external archives. The callback receives all distinct source groups and
        must drain and exclude all source users before yielding, including
        readers, snapshot exporters, and writers from preparation through
        publication, and hold that exclusion through scan and deletion. Neither
        this Runtime's routes nor an ordinary no-argument snapshot guard proves
        that authority. Omitted/empty inventories and SQL owners remain blocked.
        """
        self._require_ready()
        if self._resources.read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        owner = self._resources.run_registrations
        known = (*self._resources.stores.values(), *(self._resources.capability_stores or {}).values())
        supplied = {store.storage_group for store in complete_reference_stores or ()}
        if (owner is None or self._resources.capability_stores
                or any(store.storage_group not in supplied for store in known)):
            return RunRegistrationCleanupResult(blocked=tuple(dict.fromkeys(candidates)))
        return await owner.purge_unreferenced(
            candidates, exclusive=exclusive, complete_reference_stores=complete_reference_stores, limit=limit,
        )

    async def purge_unreferenced_captures(
        self, candidates: tuple[str, ...], *, exclusive: SnapshotExclusiveGuard, limit: int = 100,
    ) -> CaptureCleanupResult:
        self._require_ready()
        if self._resources.read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        return await _purge_unreferenced_captures(candidates,
            namespace=self._resources.namespace, tenant_id=self._resources.tenant_id,
            stores=self._resources.stores, capability_stores=self._resources.capability_stores,
            exclusive=exclusive, limit=limit)

    async def capture_objects(self, keys: tuple[str, ...]) -> tuple[tuple[PersistenceDomain, ObjectRef], ...]:
        """Read actual file references before capture metadata becomes cleanup work."""
        from .captures import read_capture_record, capture_record_value, input_capture_expiry_key
        from .codec import decode_domain
        self._require_ready()
        result = []
        store = self._resources.stores[PersistenceDomain.TASK]
        for key in keys:
            record = await read_capture_record(store, namespace=self._resources.namespace,
                tenant_id=self._resources.tenant_id, key=key)
            if record is not None:
                result.extend(iter_runtime_object_refs(capture_retention_value(record), default_domain=PersistenceDomain.TASK))
            marker = await read_capture_record(store, namespace=self._resources.namespace,
                tenant_id=self._resources.tenant_id, key=input_capture_expiry_key(key))
            if marker is not None:
                value = capture_record_value(marker)
                if "cleanup_objects" in value:
                    result.extend(decode_domain(value["cleanup_objects"], tuple[tuple[PersistenceDomain, ObjectRef], ...]))
        return tuple(dict.fromkeys(result))

    async def purge_unreferenced_objects(
        self, candidates: tuple[tuple[PersistenceDomain, ObjectRef], ...], *,
        exclusive: SnapshotExclusiveGuard, limit: int = 100,
    ) -> ObjectCleanupResult:
        """Purge core references only; known opaque capability partitions block cleanup.

        The exclusive guard must quiesce every source user. Historical raw
        custom writes without a durable marker require their original owner;
        an absent marker is not proof that arbitrary custom references are absent.
        """
        self._require_ready()
        if self._resources.read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        return await _purge_unreferenced_objects(
            candidates, namespace=self._resources.namespace, tenant_id=self._resources.tenant_id,
            stores=self._resources.stores,
            object_stores=self._resources.object_stores,
            archive_object_stores=self._resources.archive_object_stores,
            capability_stores=self._resources.capability_stores,
            exclusive=exclusive, limit=limit,
        )

    async def export_snapshot(
        self,
        *,
        object_store: ObjectStore,
        limits: SnapshotLimits,
    ) -> ObjectRef:
        """Export this initialized read-only state as one bounded logical snapshot."""
        self._require_ready()
        if not self._resources.portable_export:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Capability persistence partitions are not supported by portable snapshots")
        if not self._resources.read_only:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        if not isinstance(limits, SnapshotLimits):
            raise TypeError("limits must be SnapshotLimits")
        if not self._resources.durable_domains:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

        capture_budget = _SnapshotCaptureBudget(limits)
        archive_sources = []
        native_rows = {}
        native_checkpoints = {}
        native_projections = {}
        run_registrations = {}
        native_payloads = {}
        readers = {}
        groups = {}
        for domain in sorted(self._resources.durable_domains, key=lambda value: value.value):
            store = self._resources.stores[domain]
            groups.setdefault(store.storage_group, []).append((domain, store))
        for group, members in groups.items():
            if not isinstance(group, StateStorageGroup):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            opened = await group.open_archive()
            native = None
            resolver = self._resources.checkpoint_backup_source
            if (opened is not None and resolver is not None
                    and any(domain is PersistenceDomain.EXECUTION and isinstance(store, SqlStateStore)
                            for domain, store in members)):
                from ._step_store import SqlStepStore
                native = await resolver(NativeStepSource(opened[0], SqlStepStore.scope_key(
                    namespace=self._resources.namespace, tenant_id=self._resources.tenant_id)))

            execution_bindings = None
            if PersistenceDomain.EXECUTION in {domain for domain, _ in members}:
                execution_bindings = ExecutionContent(self._resources.stores[PersistenceDomain.EXECUTION],
                    namespace=self._resources.namespace, tenant_id=self._resources.tenant_id)
                await execution_bindings.initialize(create=False)

            async def capture_group(transaction: object) -> None:
                roots = []
                for domain, store in members:
                    reader = _StateSnapshotReader(store,
                        native_run_scope=((self._resources.namespace, self._resources.tenant_id)
                                          if domain is PersistenceDomain.TRACE else None))
                    captured = await _capture_snapshot_reader(reader, budget=capture_budget)
                    readers[domain] = captured
                    for value in (*captured.records, *captured.facts, *captured.operations):
                        projected = (capture_retention_value(value) if isinstance(value, StoredRecord)
                                     and value.kind in CAPTURE_RECORD_KINDS else value.data)
                        roots.extend(iter_native_message_roots(projected))
                if PersistenceDomain.EXECUTION in {domain for domain, _ in members}:
                    from .contracts import ExecutionRecord
                    from ..agent import AgentBindingContract

                    execution_records = tuple(record for record in readers[PersistenceDomain.EXECUTION].records
                                              if record.kind == "execution")
                    for offset in range(0, len(execution_records), 1000):
                        for semantic_record in await execution_bindings.expand_many(
                                execution_records[offset:offset + 1000], reserve=capture_budget.reserve_bytes,
                                reserve_native=capture_budget.check_bytes):
                            execution = _decode_enveloped_domain(semantic_record.data, ExecutionRecord)
                            if not isinstance(execution.binding, AgentBindingContract) or not execution.restart_checkpoints:
                                continue
                            source = execution.native_step_source
                            resolver = self._resources.checkpoint_backup_source
                            if source is None or resolver is None:
                                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED,
                                              "Native StepStore backup requires an admitted portable producer source")
                            if opened is None or source.group_id != opened[0]:
                                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                            if source.group_id in native_checkpoints:
                                if native_checkpoints[source.group_id].store_key != source.store_key:
                                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                                continue
                            if native is None:
                                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                            execution_store = self._resources.stores[PersistenceDomain.EXECUTION]
                            if (not isinstance(execution_store, SqlStateStore)
                                    or native.storage_context.engine is not execution_store.context.engine
                                    or native.store_key != source.store_key or not native.backup_supported):
                                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                            async def capture_native(tx: StateTransaction) -> NativeStepBackup:
                                if not isinstance(tx, SqlStateRowReader):
                                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                                return await native.capture_backup(tx, accept=capture_budget.accept, validate_messages=False)
                            backup = await execution_store.read(capture_native)
                            native_checkpoints[source.group_id] = backup
                            for row in backup.runs:
                                _retain_run_registration(run_registrations, _run_from_native_row(row))
                            roots.extend(backup.message_roots(source.group_id))
                if (opened is not None and opened[0] in native_checkpoints
                        and PersistenceDomain.TRACE in {domain for domain, _ in members}):
                    trace = self._resources.stores[PersistenceDomain.TRACE]
                    physical = await _capture_snapshot_reader(_StateSnapshotReader(trace, trace_scope=(self._resources.namespace, self._resources.tenant_id)), budget=_SnapshotCaptureBudget(limits))
                    projected = readers[PersistenceDomain.TRACE]
                    records = {value.key: value for value in projected.records}
                    facts = {(value.stream_key, value.sequence): value for value in projected.facts}
                    validator = NativeBackupProjections(native_checkpoints[opened[0]],
                        namespace=self._resources.namespace, tenant_id=self._resources.tenant_id)
                    raw_records, raw_facts = [], []
                    for value in physical.records:
                        visible = records.get(value.key)
                        if visible is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if value.data != visible.data:
                            if validator.validate(visible, value.data) != value:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            capture_budget.accept(encode_record(value))
                            raw_records.append(value)
                    for value in physical.facts:
                        visible = facts.get((value.stream_key, value.sequence))
                        if visible is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if value.data != visible.data:
                            if validator.validate(visible, value.data) != value:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            capture_budget.accept(encode_fact(value))
                            raw_facts.append(value)
                    if raw_records or raw_facts:
                        native_projections[opened[0]] = _CapturedSnapshotReader(tuple(raw_records), tuple(raw_facts), ())
                if opened is not None:
                    scope, archive = opened
                    domain, store = members[0]
                    archive_sources.append(_SnapshotArchive(scope, domain,
                        await _capture_snapshot_reader(_StateSnapshotReader(archive), budget=capture_budget, page_size=1000, records_only=True)))
                    if roots or scope in native_checkpoints:
                        if not isinstance(store, SqlStateStore) or any(root.group_id != scope for root in roots):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        for root in roots:
                            _validate_native_partition(root.store_key, self._resources.namespace, self._resources.tenant_id)
                        sequences, messages = await _read_native_messages(store, roots, budget=capture_budget)
                        native_domain = PersistenceDomain.EXECUTION if scope in native_checkpoints else domain
                        native_rows[scope] = native_domain, sequences, messages
                elif roots:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            # Capture each physical group's roots and sources in the same read
            # view. Independent groups cannot share a database transaction.
            await group.read_many(tuple(store for _, store in members), capture_group)

        # Registrations are immutable and published before their references.
        # Read the detached closure after group snapshots to avoid nested owner sessions.
        references = [RunRegistrationReference.from_data(record.data)
                      for captured in readers.values() for record in captured.records
                      if record.kind == "agent_run"]
        missing = tuple(dict.fromkeys(reference.run_id for reference in references
                                      if reference.run_id not in run_registrations))
        if missing:
            owner = self._resources.run_registrations
            if owner is None:
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            for start in range(0, len(missing), 128):
                registrations = await owner.get_many(missing[start:start + 128])
                for identity in missing[start:start + 128]:
                    registration = registrations.get(identity)
                    if registration is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    _retain_run_registration(run_registrations, registration)
                    capture_budget.accept(_run_json(registration))
        for reference in references:
            reference.project(run_registrations[reference.run_id])

        for scope, (domain, sequences, messages) in native_rows.items():
            native_payloads[scope] = await _capture_native_payloads(
                scope, sequences, messages, budget=capture_budget,
                objects=self._resources.archive_object_store(domain),
                backup=native_checkpoints.get(scope))

        async def archives() -> AsyncIterator[_SnapshotArchive]:
            for archive in archive_sources:
                yield archive

        source = _SnapshotExportSource(
            namespace=self._resources.namespace, tenant_id=self._resources.tenant_id,
            readers=readers, archives=archives(), object_stores=self._resources.object_stores,
            native_rows=native_rows, native_checkpoints=native_checkpoints, native_projections=native_projections, native_payloads=native_payloads,
            run_registrations=run_registrations,
            archive_object_stores=(self._resources.archive_object_stores
                                  if self._resources.archive_object_stores is not None
                                  else self._resources.object_stores),
        )
        return await _export_snapshot(source, object_store=object_store, limits=limits)


async def _export_snapshot(
    source: _SnapshotExportSource, *, object_store: ObjectStore, limits: SnapshotLimits,
) -> ObjectRef:
    if not isinstance(limits, SnapshotLimits):
        raise TypeError("limits must be SnapshotLimits")
    if not source.readers:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    for reader in source.readers.values():
        if await has_capability_partition_marker(reader):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Capability persistence partitions are not supported by portable snapshots")

    archive_closures: dict[str, ArchiveClosure] = {}
    archive_domains: dict[str, PersistenceDomain] = {}
    native_roots: dict[str, list[object]] = {}
    async for archive in source.archives:
        if archive.scope_id in archive_closures:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        records: list[StoredRecord] = []
        cursor = None
        while True:
            page = await archive.reader.scan_records_page(after=cursor, limit=1000)
            records.extend(page)
            if len(page) < 1000:
                break
            cursor = RecordScanCursor(page[-1].kind, page[-1].key)
        archive_closures[archive.scope_id] = ArchiveClosure(archive.scope_id, records)
        archive_domains[archive.scope_id] = archive.domain

    domains: dict[str, dict[str, list[object]]] = {}
    snapshot_records: dict[PersistenceDomain, tuple[StoredRecord, ...]] = {}
    snapshot_facts: dict[PersistenceDomain, tuple[StoredFact, ...]] = {}
    objects: list[dict[str, object]] = []
    binding_budget = _SnapshotCaptureBudget(limits)
    copied_objects: set[tuple[str, str, str, int]] = set()
    entry_count = 0
    object_bytes = 0

    def accept(value: object) -> object:
        nonlocal entry_count
        entry_count += 1
        if entry_count > limits.max_entries:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        return value

    async def copy_reference(
        source_domain: PersistenceDomain,
        reference: ObjectRef,
        *, encoded: bytes | None = None,
    ) -> None:
        nonlocal entry_count, object_bytes
        identity = (
            source_domain.value,
            reference.key,
            reference.digest,
            reference.size,
        )
        if identity in copied_objects:
            return
        entry_count += 1
        object_bytes += reference.size
        if (
            entry_count > limits.max_entries
            or object_bytes > limits.max_bytes
        ):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        source_store = (source.archive_object_store(source_domain) if reference.key.startswith("v1/archive/")
                        else source.object_stores[source_domain])
        key = (
            "v1/runtime-storage-object/"
            f"{source_domain.value}/{reference.digest}"
        )
        await object_store.put(
            key,
            source_store.open(reference.key) if encoded is None else _snapshot_chunk(encoded),
            expected_size=reference.size,
            expected_digest=reference.digest,
        )
        objects.append(
            {
                "domain": source_domain.value,
                "source": _object_ref_payload(reference),
                "content": _object_ref_payload(
                    ObjectRef(
                        object_store.store_id,
                        key,
                        reference.digest,
                        reference.size,
                    )
                ),
            }
        )
        copied_objects.add(identity)
        if reference.key.startswith(
            (
                "v1/asset-snapshot/",
            )
        ):
            payload = await read_object(
                source_store,
                reference.key,
                expected_digest=reference.digest,
                expected_size=reference.size,
            )
            for nested_domain, nested in iter_runtime_object_dependencies(
                reference,
                payload,
                default_domain=source_domain,
            ):
                await copy_reference(nested_domain, nested)

    async def copy_references(encoded: object, domain: PersistenceDomain, *, archive_owner: bool = True) -> None:
        if archive_owner:
            _mark_archive_value(archive_closures, encoded)
            for root in iter_native_message_roots(encoded):
                _validate_native_partition(root.store_key, source.namespace, source.tenant_id)
                native_roots.setdefault(root.group_id, []).append(root)
        for source_domain, reference in iter_runtime_object_refs(
            encoded,
            default_domain=domain,
        ):
            await copy_reference(source_domain, reference)

    for domain in sorted(source.readers, key=lambda item: item.value):
        store = source.readers[domain]
        raw_domain: dict[str, list[object]] = {
            "records": [],
            "aliases": [],
            "facts": [],
            "operations": [],
            "sequences": [],
        }
        domain_records: list[StoredRecord] = []
        binding_contents = {key: content for closure in archive_closures.values()
                            for key, content in closure.binding_contents().items()}
        domain_facts: list[StoredFact] = []
        domain_operations: list[StoredOperation] = []

        record_cursor: RecordScanCursor | None = None
        while True:
            page = await store.scan_records_page(after=record_cursor, limit=128)
            if not page:
                break
            for value in page:
                semantic_value = _expand_snapshot_native_content(value,
                    {group: (rows[1], rows[2]) for group, rows in (source.native_rows or {}).items()},
                    source.native_payloads or {}, namespace=source.namespace, tenant_id=source.tenant_id)
                semantic_value = _expand_run_reference(semantic_value, source.run_registrations or {})
                semantic_value = expand_snapshot_execution_content(semantic_value, binding_contents,
                    namespace=source.namespace, tenant_id=source.tenant_id, reserve=binding_budget.reserve_bytes)
                semantic_value = expand_tool_content(semantic_value, binding_contents,
                    content_scope=archive_content_scope(source.namespace, source.tenant_id), reserve=binding_budget.reserve_bytes)
                semantic_value = expand_snapshot_capture(semantic_value, binding_contents,
                    namespace=source.namespace, tenant_id=source.tenant_id, reserve=binding_budget.reserve_bytes)
                if domain is PersistenceDomain.EXECUTION and value.kind == "execution":
                    from .contracts import ExecutionRecord
                    execution = _decode_enveloped_domain(semantic_value.data, ExecutionRecord)
                    from ..agent import AgentBindingContract
                    if (isinstance(execution.binding, AgentBindingContract) and execution.restart_checkpoints
                            and (execution.native_step_source is None
                                 or execution.native_step_source.group_id not in (source.native_checkpoints or {})
                                 or source.native_checkpoints[execution.native_step_source.group_id].store_key
                                 != execution.native_step_source.store_key)):
                        raise AIError(
                            ErrorCode.SNAPSHOT_UNSUPPORTED,
                            "Native StepPersistence executions require a consistent database and StepStore backup; "
                            "portable snapshots do not include the external StepStore.",
                        )
                domain_records.append(semantic_value)
                encoded = encode_record(value)
                raw_domain["records"].append(accept(encoded))
                if value.kind in CAPTURE_RECORD_KINDS:
                    if value.kind == "input_capture":
                        await restore_capture_value(value, read_object=lambda ref: read_object(
                            source.object_stores[PersistenceDomain.TASK], ref.key,
                            expected_digest=ref.digest, expected_size=ref.size))
                    if value.kind == "task_graph_binding_capture":
                        _mark_archive_value(archive_closures, capture_retention_value(value))
                    await copy_references(capture_retention_value(semantic_value), domain, archive_owner=False)
                elif value.kind != "evaluation_cleanup":
                    await copy_references(encoded, domain)
                    if value.kind in {"execution", "tool_operation"}:
                        await copy_references(encode_record(semantic_value), domain, archive_owner=False)
            last = page[-1]
            record_cursor = RecordScanCursor(last.kind, last.key)
            if len(page) < 128:
                break

        fact_cursor: FactScanCursor | None = None
        while True:
            page = await store.scan_facts_page(after=fact_cursor, limit=128)
            if not page:
                break
            for value in page:
                domain_facts.append(value)
                encoded = encode_fact(value)
                raw_domain["facts"].append(accept(encoded))
                await copy_references(encoded, domain)
            last = page[-1]
            fact_cursor = FactScanCursor(last.stream_key, last.sequence)
            if len(page) < 128:
                break

        operation_cursor: OperationScanCursor | None = None
        while True:
            page = await store.scan_operations_page(after=operation_cursor, limit=128)
            if not page:
                break
            for value in page:
                domain_operations.append(value)
                encoded = encode_operation(value)
                raw_domain["operations"].append(accept(encoded))
                await copy_references(encoded, domain)
            operation_cursor = OperationScanCursor(page[-1].key)
            if len(page) < 128:
                break

        aliases, sequences = canonical_snapshot_indexes(
            namespace=source.namespace,
            tenant_id=source.tenant_id,
            domain=domain,
            records=tuple(domain_records),
            facts=tuple(domain_facts),
            operations=tuple(domain_operations),
        )
        for value in aliases:
            raw_domain["aliases"].append(
                accept(
                    {
                        "alias_key": value.alias_key.hex(),
                        "record_key": value.record_key.hex(),
                    }
                )
            )
        for key in sorted(sequences):
            raw_domain["sequences"].append(
                accept({"stream_key": key.hex(), "value": sequences[key]})
            )

        domains[domain.value] = raw_domain
        snapshot_records[domain] = tuple(domain_records)
        snapshot_facts[domain] = tuple(domain_facts)

    validate_snapshot_references(snapshot_records, snapshot_facts)
    capture_state_closure(tuple(record for records in snapshot_records.values() for record in records),
        values=tuple(dict(fact.data) for facts in snapshot_facts.values() for fact in facts),
        namespace=source.namespace, tenant_id=source.tenant_id, require_bindings=True)
    archives = []
    for scope, closure in archive_closures.items():
        domain = archive_domains[scope]
        retained = closure.retained_records
        if not closure.has_roots:
            continue
        for reference in closure.object_refs:
            await copy_reference(domain, reference)
        for record in retained:
            if record.kind == "archive_content":
                content = ContentRecord.from_record(record)
                encoded = await _read_content_payload(content, lambda ref: read_object(
                    source.archive_object_store(domain), ref.key, expected_digest=ref.digest, expected_size=ref.size))
                _validate_archive_content(content, encoded)
        archives.append({"scope_id": scope, "domain": domain.value,
                         "records": [accept(encode_record(record)) for record in retained]})
    native_checkpoints = []
    for group_id, backup in (source.native_checkpoints or {}).items():
        native_roots.setdefault(group_id, []).extend(backup.message_roots(group_id))
        raw = _encode_checkpoint_registration_refs(backup, source.run_registrations or {})
        for name in ("run_refs", "events", "snapshots"):
            for row in raw[name]:
                accept(row)
        native_checkpoints.append({"group_id": group_id, **raw})
    native_messages = []
    for group_id, roots in native_roots.items():
        owner = (source.native_rows or {}).get(group_id)
        if owner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        domain, sequences, messages = owner
        closure = NativeMessageClosure(sequences)
        for root in roots:
            closure.mark(root.store_key, root.slices)
        retained_sequences = [row for row in sequences if (row["store_key"], row["sha256"]) in closure.retained_sequences]
        retained_messages = [row for row in messages if (row["store_key"], row["sha256"]) in closure.retained_messages]
        if {(row["store_key"], row["sha256"]) for row in retained_messages} != closure.retained_messages:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for row in retained_messages:
            captured = (source.native_payloads or {}).get(group_id)
            if captured is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            encoded = captured[(row["store_key"], row["sha256"])]
            content = ContentRecord.from_native_row(row)
            if content.payload is not None and content.payload.encoding == "json-attachments":
                for reference, data in attachment_payload_objects(encoded, content.payload):
                    await copy_reference(domain, reference, encoded=data)
        native_messages.append({"group_id": group_id, "domain": domain.value,
                                "sequences": [accept(encode_native_sequence_row(row)) for row in retained_sequences],
                                "messages": [accept(encode_native_message_row(row)) for row in retained_messages]})
    manifest = {
        "kind": "runtime-storage-snapshot",
        "format_version": 1,
        "namespace": source.namespace,
        "tenant_id": source.tenant_id,
        "domains": domains,
        "run_registrations": [accept(_run_json(run)) for run in sorted(
            (source.run_registrations or {}).values(), key=lambda item: item.run_id)],
        "archives": archives,
        "native_messages": native_messages,
        "native_checkpoints": native_checkpoints,
        "native_projections": [{"group_id": group_id,
            "records": [accept(encode_record(value)) for value in projected.records],
            "facts": [accept(encode_fact(value)) for value in projected.facts]}
            for group_id, projected in (source.native_projections or {}).items()],
        "objects": objects,
    }
    for optional in ("native_messages", "native_checkpoints", "native_projections"):
        if not manifest[optional]:
            del manifest[optional]
    payload = canonical_json_bytes(manifest)
    budget_entries, budget_bytes = snapshot_manifest_usage(manifest)
    if budget_entries > limits.max_entries or len(payload) + budget_bytes > limits.max_bytes:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    digest = hashlib.sha256(payload).hexdigest()
    key = f"v1/runtime-storage-snapshot/{digest}"
    await _put_snapshot_object(object_store, key, payload)
    return ObjectRef(object_store.store_id, key, digest, len(payload))

async def restore_snapshot(
    ref: ObjectRef,
    *,
    object_store: ObjectStore,
    root: str | Path,
    limits: SnapshotLimits,
    open_target: Callable[[Path, str, str], AbstractAsyncContextManager[SnapshotResources]],
) -> None:
    """Restore bounded logical state through an owned target resource scope."""
    if not isinstance(limits, SnapshotLimits):
        raise TypeError("limits must be SnapshotLimits")
    if ref.store_id != object_store.store_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
    if ref.size > limits.max_bytes:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    payload = await read_object(
        object_store,
        ref.key,
        expected_digest=ref.digest,
        expected_size=ref.size,
    )
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(manifest, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if manifest.get("kind") != "runtime-storage-snapshot":
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    format_version = manifest.get("format_version")
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version < 1
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if format_version != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if not isinstance(manifest.get("domains"), dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    namespace = manifest.get("namespace")
    tenant_id = manifest.get("tenant_id")
    if not isinstance(namespace, str) or not isinstance(tenant_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    namespace = validate_persistence_namespace(namespace)
    tenant_id = validate_tenant_id(tenant_id)

    budget_entries, budget_bytes = snapshot_manifest_usage(manifest)
    if budget_entries > limits.max_entries or len(payload) + budget_bytes > limits.max_bytes:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

    run_registrations = _decode_run_registrations(manifest.get("run_registrations", []))
    raw_domains = manifest["domains"]
    raw_objects = manifest.get("objects", [])
    if not isinstance(raw_objects, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    raw_archives = manifest.get("archives", [])
    if not isinstance(raw_archives, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    archive_closures: dict[str, ArchiveClosure] = {}
    archive_domains: dict[str, PersistenceDomain] = {}
    entry_count = len(raw_objects) + len(run_registrations)
    for item in raw_archives:
        if not isinstance(item, dict) or set(item) != {"scope_id", "domain", "records"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        scope, records = item["scope_id"], item["records"]
        if not isinstance(scope, str) or not scope or scope in archive_closures or not isinstance(records, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            archive_domains[scope] = PersistenceDomain(item["domain"])
        except (ValueError, TypeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        archive_closures[scope] = ArchiveClosure(scope, tuple(decode_record(record) for record in records))
        entry_count += len(records)
    if entry_count > limits.max_entries:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    raw_native = manifest.get("native_messages", [])
    if not isinstance(raw_native, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    native_closures = {}
    native_entries = {}
    for entry in raw_native:
        if (not isinstance(entry, dict) or set(entry) != {"group_id", "domain", "sequences", "messages"}
                or not isinstance(entry["group_id"], str) or not entry["group_id"]
                or entry["group_id"] in native_entries
                or not isinstance(entry["sequences"], list) or not isinstance(entry["messages"], list)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            PersistenceDomain(entry["domain"])
        except (ValueError, TypeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        entry_count += len(entry["sequences"]) + len(entry["messages"])
        if entry_count > limits.max_entries:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        validate_snapshot_native_rows(entry["sequences"], entry["messages"])
        entry = {**entry, "sequences": [decode_native_sequence_row(row) for row in entry["sequences"]],
                 "messages": [decode_native_message_row(row) for row in entry["messages"]]}
        native_closures[entry["group_id"]] = NativeMessageClosure(entry["sequences"])
        native_entries[entry["group_id"]] = entry
    raw_checkpoints = manifest.get("native_checkpoints", [])
    if not isinstance(raw_checkpoints, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    native_checkpoints = {}
    for entry in raw_checkpoints:
        if not isinstance(entry, dict) or not isinstance(entry.get("group_id"), str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        group_id = entry["group_id"]
        if group_id in native_checkpoints or group_id not in native_entries:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if native_entries[group_id]["domain"] != PersistenceDomain.EXECUTION.value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        from ._step_store import SqlStepStore
        try:
            backup = _decode_checkpoint_registration_refs(entry, run_registrations)
            SqlStepStore.validate_backup(backup)
        except (TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        _validate_native_partition(backup.store_key, namespace, tenant_id)
        native_checkpoints[group_id] = backup
        entry_count += len(backup.runs) + len(backup.events) + len(backup.snapshots)
        if entry_count > limits.max_entries:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        for message_root in backup.message_roots(group_id):
            native_closures[group_id].mark(message_root.store_key, message_root.slices)
    object_bytes = 0
    expected_objects: set[tuple[str, str, str, int]] = set()
    required_checkpoint_groups: set[str] = set()
    decoded_domains: dict[
        PersistenceDomain,
        tuple[
            tuple[StoredRecord, ...],
            tuple[StoredAlias, ...],
            tuple[StoredFact, ...],
            tuple[StoredOperation, ...],
            Mapping[bytes, int],
        ],
    ] = {}

    actual_objects: set[tuple[str, str, str, int]] = set()
    decoded_objects: list[tuple[PersistenceDomain, ObjectRef, ObjectRef]] = []
    decoded_by_identity: dict[
        tuple[str, str, str, int],
        tuple[PersistenceDomain, ObjectRef, ObjectRef],
    ] = {}
    for raw_object in raw_objects:
        if not isinstance(raw_object, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            domain = PersistenceDomain(raw_object["domain"])
            source = _object_ref_from_payload(raw_object["source"])
            content_ref = _object_ref_from_payload(raw_object["content"])
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if source.digest != content_ref.digest or source.size != content_ref.size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = (domain.value, source.key, source.digest, source.size)
        if identity in actual_objects:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        actual_objects.add(identity)
        decoded = (domain, source, content_ref)
        decoded_objects.append(decoded)
        decoded_by_identity[identity] = decoded
        object_bytes += content_ref.size
        if len(payload) + object_bytes > limits.max_bytes:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

    native_encoded_payloads = {}
    for group_id, entry in native_entries.items():
        checkpoint_payloads = {}
        native_encoded_payloads[group_id] = {}
        closure = native_closures[group_id]
        for row in entry["messages"]:
            content = validate_native_message_content(row, group_id=group_id,
                content_scope=archive_content_scope(namespace, tenant_id))
            async def resolve(reference: ObjectRef) -> bytes:
                identity = entry["domain"], reference.key, reference.digest, reference.size
                expected_objects.add(identity)
                decoded = decoded_by_identity.get(identity)
                if decoded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                content_ref = decoded[2]
                return await read_object(object_store, content_ref.key,
                                         expected_digest=content_ref.digest, expected_size=content_ref.size)
            encoded = await _read_content_payload(content, resolve)
            _validate_archive_content(content, encoded)
            native_encoded_payloads[group_id][(row["store_key"], row["sha256"])] = encoded
            if group_id in native_checkpoints:
                raw = decode_payload_bytes(encoded, codec=content.compression,
                                           raw_digest=content.raw_digest, raw_size=content.raw_size)
                decode_model_message_payloads((raw,))
                checkpoint_payloads[(row["store_key"], row["sha256"])] = raw
        if group_id in native_checkpoints:
            _validate_native_checkpoint_payloads(native_checkpoints[group_id], closure, checkpoint_payloads)

    semantic_domains = {}
    binding_contents = {key: content for closure in archive_closures.values()
                        for key, content in closure.binding_contents().items()}
    for domain_name, raw_domain in raw_domains.items():
        try:
            domain = PersistenceDomain(domain_name)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        expected_domain_fields = {
            "records",
            "aliases",
            "facts",
            "operations",
            "sequences",
        }
        if (
            not isinstance(raw_domain, Mapping)
            or set(raw_domain) != expected_domain_fields
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        raw_records = raw_domain["records"]
        raw_aliases = raw_domain["aliases"]
        raw_facts = raw_domain["facts"]
        raw_operations = raw_domain["operations"]
        raw_sequences = raw_domain["sequences"]
        if not all(
            isinstance(value, list)
            for value in (
                raw_records,
                raw_aliases,
                raw_facts,
                raw_operations,
                raw_sequences,
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        entry_count += sum(
            len(value)
            for value in (
                raw_records,
                raw_aliases,
                raw_facts,
                raw_operations,
                raw_sequences,
            )
        )
        if entry_count > limits.max_entries:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

        records = tuple(decode_record(value) for value in raw_records)
        semantic_records = tuple(expand_snapshot_capture(expand_snapshot_execution_content(
            _expand_run_reference(_expand_snapshot_native_content(record,
                {group: (entry["sequences"], entry["messages"]) for group, entry in native_entries.items()},
                native_encoded_payloads, namespace=namespace, tenant_id=tenant_id), run_registrations), binding_contents,
            namespace=namespace, tenant_id=tenant_id), binding_contents,
            namespace=namespace, tenant_id=tenant_id) for record in records)
        semantic_records = tuple(expand_tool_content(record, binding_contents,
            content_scope=archive_content_scope(namespace, tenant_id)) for record in semantic_records)
        semantic_domains[domain] = semantic_records
        aliases = _decode_snapshot_aliases(raw_aliases)
        facts = tuple(decode_fact(value) for value in raw_facts)
        operations = tuple(decode_operation(value) for value in raw_operations)
        sequences = _decode_snapshot_sequences(raw_sequences)
        validate_snapshot_domain(
            namespace=namespace,
            tenant_id=tenant_id,
            domain=domain,
            records=semantic_records,
            aliases=aliases,
            facts=facts,
            operations=operations,
            sequences=sequences,
        )
        for value in (*records, *facts, *operations):
            projected = (capture_retention_value(value) if isinstance(value, StoredRecord)
                         and value.kind in CAPTURE_RECORD_KINDS else dict(value.data))
            _mark_archive_value(archive_closures, projected)
            for native_root in iter_native_message_roots(projected):
                _validate_native_partition(native_root.store_key, namespace, tenant_id)
                closure = native_closures.get(native_root.group_id)
                if closure is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                closure.mark(native_root.store_key, native_root.slices)
        decoded_domains[domain] = (
            records,
            aliases,
            facts,
            operations,
            sequences,
        )
        for record in semantic_records:
            if record.kind == "execution":
                from .contracts import ExecutionRecord
                execution = _decode_enveloped_domain(record.data, ExecutionRecord)
                from ..agent import AgentBindingContract
                if isinstance(execution.binding, AgentBindingContract) and execution.restart_checkpoints:
                    source = execution.native_step_source
                    if (source is None or source.group_id not in native_checkpoints
                            or source.store_key != native_checkpoints[source.group_id].store_key):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    required_checkpoint_groups.add(source.group_id)
        live_records = tuple(capture_retention_value(record) if record.kind in CAPTURE_RECORD_KINDS else
                             encode_record(record) if record.kind in {"execution", "tool_operation"} else encoded
                             for encoded, record in zip(raw_records, semantic_records) if record.kind != "evaluation_cleanup")
        for encoded in (*live_records, *raw_facts, *raw_operations):
            for source_domain, reference in iter_runtime_object_refs(
                encoded,
                default_domain=domain,
            ):
                expected_objects.add(
                    (
                        source_domain.value,
                        reference.key,
                        reference.digest,
                        reference.size,
                    )
                )

    validate_snapshot_references(
        semantic_domains,
        {domain: values[2] for domain, values in decoded_domains.items()},
    )
    capture_state_closure(tuple(record for records in semantic_domains.values() for record in records),
        values=tuple(dict(value.data) for domain in decoded_domains.values() for value in (*domain[2], *domain[3])),
        namespace=namespace, tenant_id=tenant_id, require_bindings=True)
    if required_checkpoint_groups != set(native_checkpoints):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    referenced_runs = {RunRegistrationReference.from_data(record.data).run_id
                       for values in decoded_domains.values() for record in values[0]
                       if record.kind == "agent_run"}
    referenced_runs.update(row["run_id"] for backup in native_checkpoints.values() for row in backup.runs)
    if referenced_runs != set(run_registrations):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raw_projections = manifest.get("native_projections", [])
    if not isinstance(raw_projections, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    native_record_projections, native_fact_projections = {}, {}
    seen_projection_groups = set()
    trace_values = decoded_domains.get(PersistenceDomain.TRACE, ((), (), (), (), {}))
    trace_records = {value.key: value for value in trace_values[0]}
    trace_facts = {(value.stream_key, value.sequence): value for value in trace_values[2]}
    for entry in raw_projections:
        if (not isinstance(entry, dict) or set(entry) != {"group_id", "records", "facts"}
                or entry["group_id"] not in native_checkpoints or entry["group_id"] in seen_projection_groups
                or entry["records"] != [] or not isinstance(entry["facts"], list)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        seen_projection_groups.add(entry["group_id"])
        validator = NativeBackupProjections(native_checkpoints[entry["group_id"]], namespace=namespace, tenant_id=tenant_id)
        entry_count += len(entry["records"]) + len(entry["facts"])
        if entry_count > limits.max_entries:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        for encoded in entry["facts"]:
            raw = decode_fact(encoded)
            key = raw.stream_key, raw.sequence
            visible = trace_facts.get(key)
            if (visible is None or key in native_fact_projections
                    or validator.validate(visible, raw.data) != raw):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            native_fact_projections[key] = raw
    pending_objects = list(expected_objects)
    while pending_objects:
        identity = pending_objects.pop()
        decoded = decoded_by_identity.get(identity)
        if decoded is None:
            continue
        domain, source, content_ref = decoded
        if not source.key.startswith(
            (
                "v1/asset-snapshot/",
            )
        ):
            continue
        dependency_payload = await read_object(
            object_store,
            content_ref.key,
            expected_digest=content_ref.digest,
            expected_size=content_ref.size,
        )
        for nested_domain, nested in iter_runtime_object_dependencies(
            source,
            dependency_payload,
            default_domain=domain,
        ):
            nested_identity = (
                nested_domain.value,
                nested.key,
                nested.digest,
                nested.size,
            )
            if nested_identity not in expected_objects:
                expected_objects.add(nested_identity)
                pending_objects.append(nested_identity)

    for scope, closure in archive_closures.items():
        if not closure.has_roots or closure.unreachable_records:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        domain = archive_domains[scope]
        for record in closure.retained_records:
            if record.kind != "archive_content":
                continue
            content = ContentRecord.from_record(record)
            async def resolve(reference: ObjectRef) -> bytes:
                identity = domain.value, reference.key, reference.digest, reference.size
                expected_objects.add(identity)
                decoded = decoded_by_identity.get(identity)
                if decoded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                content_ref = decoded[2]
                return await read_object(object_store, content_ref.key, expected_digest=content_ref.digest, expected_size=content_ref.size)
            encoded = await _read_content_payload(content, resolve)
            _validate_archive_content(content, encoded)

    for group_id, entry in native_entries.items():
        closure = native_closures[group_id]
        sequence_keys = [(row["store_key"], row["sha256"]) for row in entry["sequences"]]
        message_keys = [(row["store_key"], row["sha256"]) for row in entry["messages"]]
        if (len(sequence_keys) != len(set(sequence_keys)) or len(message_keys) != len(set(message_keys))
                or set(sequence_keys) != closure.retained_sequences or set(message_keys) != closure.retained_messages):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def read_capture_attachment(reference: ObjectRef) -> bytes:
        identity = PersistenceDomain.TASK.value, reference.key, reference.digest, reference.size
        decoded = decoded_by_identity.get(identity)
        if decoded is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _domain, _source, content = decoded
        return await read_object(object_store, content.key, expected_digest=content.digest, expected_size=content.size)

    for record in semantic_domains.get(PersistenceDomain.TASK, ()):
        if record.kind == "input_capture":
            await restore_capture_value(record, read_object=read_capture_attachment)

    if actual_objects != expected_objects:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    target = Path(root).expanduser().resolve(strict=False)
    if target.exists() and any(target.iterdir()):
        raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
    async with open_target(target, namespace, tenant_id) as resources:
        if run_registrations:
            if resources.run_registrations is None:
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            for registration in run_registrations.values():
                await resources.run_registrations.publish(registration)
        for domain, source, content_ref in decoded_objects:
            if source.key.startswith("v1/archive/"):
                continue
            destination = resources.object_stores[domain]
            current = await destination.stat(source.key)
            if current is not None:
                if current.digest != source.digest or current.size != source.size:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            else:
                await destination.put(
                    source.key,
                    object_store.open(content_ref.key),
                    expected_size=source.size,
                    expected_digest=source.digest,
                )

        for domain, values in decoded_domains.items():
            records, aliases, facts, operations, sequences = values
            referenced_scopes = set()
            reference_values = tuple(capture_retention_value(value) if value.kind in CAPTURE_RECORD_KINDS else value.data
                                     for value in records) + tuple(value.data for value in (*facts, *operations))
            for value in reference_values:
                referenced_scopes.update(root.scope_id for root in iter_archive_roots(value))
                referenced_scopes.update(scope for scope, _, _ in iter_archive_content_refs(value))
            domain_native_roots = tuple(root for value in reference_values
                                        for root in iter_native_message_roots(value))
            domain_checkpoints = native_checkpoints if domain is PersistenceDomain.EXECUTION else {}
            domain_native_roots += tuple(root for group_id, backup in domain_checkpoints.items()
                                         for root in backup.message_roots(group_id))
            native_groups = {root.group_id for root in domain_native_roots} | set(domain_checkpoints)
            scope_mapping = {}
            archive_records: dict[bytes, StoredRecord] = {}
            archive_store = None
            if referenced_scopes or native_groups:
                group = resources.stores[domain].storage_group
                if not isinstance(group, StateStorageGroup):
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                target_scope, archive_store = await group.open_archive(create=True)
                native_mapping = {source_group: target_scope for source_group in native_groups}
                for source_scope in referenced_scopes:
                    scope_mapping[source_scope] = target_scope
                    for record in archive_closures[source_scope].retained_records:
                        imported = record
                        if record.kind == "archive_content":
                            content = ContentRecord.from_record(record)
                            rewritten = {}
                            for reference in content.object_refs():
                                prefix = f"v1/archive/{source_scope}/"
                                if not reference.key.startswith(prefix):
                                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                                key = f"v1/archive/{target_scope}/" + reference.key[len(prefix):]
                                source_domain = archive_domains[source_scope]
                                content_ref = decoded_by_identity[(source_domain.value, reference.key, reference.digest, reference.size)][2]
                                destination = resources.archive_object_store(domain)
                                current = await destination.stat(key)
                                if current is None:
                                    await destination.put(key, object_store.open(content_ref.key),
                                                          expected_size=reference.size, expected_digest=reference.digest)
                                elif current.digest != reference.digest or current.size != reference.size:
                                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                                rewritten[reference] = ObjectRef(reference.store_id, key, reference.digest, reference.size)
                            if rewritten:
                                content = replace(content, payload=content.payload.map_object_refs(rewritten.__getitem__))
                                imported = replace(record, data=content.to_json())
                        previous = archive_records.get(imported.key)
                        if previous is not None and previous != imported:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        archive_records[imported.key] = imported
            native_rows = []
            for source_group in native_groups:
                entry = native_entries[source_group]
                domain_closure = NativeMessageClosure(entry["sequences"])
                for native_root in domain_native_roots:
                    if native_root.group_id == source_group:
                        domain_closure.mark(native_root.store_key, native_root.slices)
                message_rows = []
                for row in entry["messages"]:
                    if (row["store_key"], row["sha256"]) not in domain_closure.retained_messages:
                        continue
                    content = validate_native_message_content(row, group_id=source_group)
                    rewritten = {}
                    for reference in content.object_refs():
                        prefix = f"v1/archive/{source_group}/"
                        if not reference.key.startswith(prefix):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        key = f"v1/archive/{target_scope}/" + reference.key[len(prefix):]
                        content_ref = decoded_by_identity[(entry["domain"], reference.key, reference.digest, reference.size)][2]
                        destination = resources.archive_object_store(domain)
                        await destination.put(key, object_store.open(content_ref.key),
                                              expected_size=reference.size, expected_digest=reference.digest)
                        rewritten[reference] = ObjectRef(reference.store_id, key, reference.digest, reference.size)
                    if rewritten:
                        content = replace(content, payload=content.payload.map_object_refs(rewritten.__getitem__))
                    message_rows.append({**row, **content.to_native_values()})
                sequence_rows = [row for row in entry["sequences"]
                                 if (row["store_key"], row["sha256"]) in domain_closure.retained_sequences]
                native_rows.append((sequence_rows, message_rows))
            def rekey(value: JsonValue) -> JsonValue:
                return rekey_native_step_sources(
                    rekey_native_message_roots(rekey_archive_roots(dict(value) if isinstance(value, Mapping) else value, scope_mapping),
                                              native_mapping if native_groups else {}),
                    native_mapping if native_groups else {})
            semantic_events = {(value.stream_key, value.sequence): value for value in facts if value.kind == "step_event"}
            if domain is PersistenceDomain.TRACE:
                records = tuple(native_record_projections.get(value.key, value) for value in records)
                facts = tuple(native_fact_projections.get((value.stream_key, value.sequence), value) for value in facts)
            records = tuple(map_capture_references(value, rekey) if value.kind in CAPTURE_RECORD_KINDS
                            else replace(value, data=rekey(value.data)) for value in records)
            facts = tuple(replace(value, data=rekey(value.data)) for value in facts)
            operations = tuple(replace(value, data=rekey(value.data)) for value in operations)

            native_restore = None
            if domain_checkpoints:
                resolver = resources.checkpoint_backup_source
                if resolver is None:
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                backup = next(iter(domain_checkpoints.values()))
                native_restore = await resolver(NativeStepSource(target_scope, backup.store_key))

            async def publish(transaction: StateTransaction) -> None:
                if archive_store is not None:
                    async def retain(archive_tx: StateTransaction) -> None:
                        existing = await archive_tx.get_records(tuple(archive_records))
                        missing = []
                        for key, record in archive_records.items():
                            if key not in existing:
                                missing.append(record)
                            elif existing[key].data != record.data or existing[key].kind != record.kind:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if missing:
                            await archive_tx.insert_records(missing)
                    await archive_store.mutate(retain)
                if native_rows:
                    store = resources.stores[domain]
                    if not isinstance(store, SqlStateStore) or not isinstance(transaction, SqlStateTransaction):
                        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                    await _insert_native_messages(transaction, store, native_rows)
                if domain_checkpoints:
                    from ._step_store import SqlStepStore
                    store = resources.stores[domain]
                    if not isinstance(store, SqlStateStore) or not isinstance(transaction, SqlStateTransaction):
                        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                    for backup in domain_checkpoints.values():
                        await transaction.run_sql(store.context, lambda session: native_restore.restore_backup(session, backup))
                state_facts = facts
                store = resources.stores[domain]
                if domain is PersistenceDomain.TRACE and isinstance(store, SqlStateStore):
                    from ._trace_event_store import SqlTraceEventStore
                    from ._step_contracts import StepEvent
                    trace_events = SqlTraceEventStore(store, namespace=resources.namespace, tenant_id=resources.tenant_id)
                    rows = tuple(trace_events.row_values(value, _decode_enveloped_domain(
                        semantic_events[(value.stream_key, value.sequence)].data, StepEvent))
                        for value in facts if value.kind == "step_event")
                    await trace_events.restore(transaction, rows)
                    state_facts = tuple(value for value in facts if value.kind != "step_event")
                await _insert_snapshot_values(transaction, records, aliases, state_facts, operations, sequences)
                if domain is PersistenceDomain.TRACE and not isinstance(store, SqlStateStore):
                    from ._model_interaction_store import ModelInteractionStateStepArchive
                    from ._step_contracts import AgentRunRecord, StepEvent
                    trace = ModelInteractionStateStepArchive(
                        store, object_store=resources.object_stores[domain],
                        namespace=resources.namespace, tenant_id=resources.tenant_id,
                        runtime_domain=PersistenceDomain.TRACE,
                    )
                    by_run = {}
                    for fact in facts:
                        if fact.kind == "step_event":
                            event = _decode_enveloped_domain(fact.data, StepEvent)
                            by_run.setdefault(event.agent_run_id, []).append((fact.sequence, event))
                    for run_id, events in by_run.items():
                        await trace.index_history_events_in_transaction(transaction, AgentRunRecord(run_id), events)

            try:
                await resources.stores[domain].mutate(publish)
            finally:
                if native_rows and isinstance(resources.stores[domain], SqlStateStore):
                    from ._step_store import SqlStepStore
                    SqlStepStore.invalidate_context(resources.stores[domain].context)


def snapshot_manifest_usage(
    manifest: Mapping[str, object],
) -> tuple[int, int]:
    """Return entry count and bytes outside the serialized storage manifest.

    Native and per-execution binding decoded bytes supplement encoded bytes so
    exporters and restorers enforce the same hydration allowance.
    """
    domains = manifest.get("domains")
    objects = manifest.get("objects")
    if not isinstance(domains, Mapping) or not isinstance(objects, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    registrations = _decode_run_registrations(manifest.get("run_registrations", []))
    entries = len(objects) + len(registrations)
    object_bytes = 0
    for raw_domain in domains.values():
        if not isinstance(raw_domain, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for name in ("records", "aliases", "facts", "operations", "sequences"):
            values = raw_domain.get(name)
            if not isinstance(values, list):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            entries += len(values)
    for name, fields in (("archives", ("records",)), ("native_messages", ("sequences", "messages")),
                         ("native_checkpoints", ("run_refs", "events", "snapshots")),
                         ("native_projections", ("records", "facts"))):
        groups = manifest.get(name, [])
        if not isinstance(groups, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for group in groups:
            if not isinstance(group, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for field in fields:
                rows = group.get(field)
                if not isinstance(rows, list):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                entries += len(rows)
    for raw_object in objects:
        if not isinstance(raw_object, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source = _object_ref_from_payload(raw_object.get("source"))
        content = _object_ref_from_payload(raw_object.get("content"))
        if source.digest != content.digest or source.size != content.size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        object_bytes += content.size
    for group in manifest.get("native_messages", []):
        for row in group["messages"]:
            try:
                content = ContentRecord.from_native_row(decode_native_message_row(row))
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            object_bytes += content.raw_size
    binding_contents = {}
    for group in manifest.get("archives", []):
        records = tuple(decode_record(row) for row in group["records"]
                        if isinstance(row, Mapping) and row.get("kind") == "archive_content"
                        and isinstance(row.get("data"), Mapping)
                        and row["data"].get("codec") in {"execution-binding", "task-graph-binding", "execution-payload", "tool-operation-payload"})
        if records:
            scope = group.get("scope_id")
            if not isinstance(scope, str) or not scope:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            binding_contents.update(ArchiveClosure(scope, records).binding_contents())
    for raw_domain in domains.values():
        for row in raw_domain["records"]:
            if isinstance(row, Mapping) and row.get("kind") in {"execution", "task_graph_binding_capture", "tool_operation"}:
                record = decode_record(row)
                object_bytes += snapshot_execution_content_size(record, binding_contents) + snapshot_capture_size(record, binding_contents) + tool_content_size(record, binding_contents)
    return entries, object_bytes


def snapshot_manifest_identity(manifest: Mapping[str, object]) -> str:
    """Compare portable state independently of relocated group/object addresses."""
    archives = manifest.get("archives", [])
    native = manifest.get("native_messages", [])
    domains = manifest.get("domains")
    objects = manifest.get("objects")
    if (not isinstance(archives, list) or not isinstance(native, list)
            or not isinstance(domains, Mapping) or not isinstance(objects, list)):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    neutral = "0" * 32
    groups = {}
    for entry in archives:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("scope_id"), str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        groups[entry["scope_id"]] = neutral
    for entry in native:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("group_id"), str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        groups[entry["group_id"]] = neutral

    def reference(value: object) -> dict[str, object]:
        ref = _object_ref_from_payload(value)
        key = ref.key
        if key.startswith("v1/archive/"):
            parts = key[len("v1/archive/"):].rsplit("/", 2)
            if len(parts) != 3 or parts[0] not in groups:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            key = f"v1/archive/{neutral}/{parts[1]}/{parts[2]}"
        return dict(_object_ref_payload(replace(ref, key=key)))

    def content(parsed: ContentRecord) -> dict[str, object]:
        if parsed.payload is not None:
            parsed = replace(parsed, payload=parsed.payload.map_object_refs(
                lambda ref: _object_ref_from_payload(reference(_object_ref_payload(ref)))))
        return parsed.to_json()

    def ordered_unique(values: list[object]) -> list[object]:
        unique = {canonical_json_bytes(value): value for value in values}
        return [unique[key] for key in sorted(unique)]

    archive_records = []
    nonnative_objects = set()
    for domain_name, values in domains.items():
        try:
            domain = PersistenceDomain(domain_name)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        reference_values = {**values, "records": [capture_retention_value(decode_record(row))
                            if row.get("kind") in CAPTURE_RECORD_KINDS else row for row in values["records"]]}
        for _, ref in iter_runtime_object_refs(reference_values, default_domain=domain):
            nonnative_objects.add((ref.key, ref.digest, ref.size))
    for entry in archives:
        rows = entry.get("records")
        if not isinstance(rows, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for row in rows:
            record = decode_record(row)
            if record.kind == "archive_content":
                parsed = ContentRecord.from_record(record)
                for ref in parsed.object_refs():
                    nonnative_objects.add((ref.key, ref.digest, ref.size))
                record = replace(record, data=content(parsed), payload_bytes=None)
            archive_records.append(encode_record(record))
    native_sequences = []
    native_messages = {}
    native_objects = set()
    for entry in native:
        validate_snapshot_native_rows(entry.get("sequences"), entry.get("messages"))
        rows = [decode_native_message_row(row) for row in entry["messages"]]
        native_sequences.extend(entry["sequences"])
        for row in rows:
            parsed = validate_native_message_content(row, group_id=entry["group_id"])
            # Raw bytes, not compression or transport location, define message
            # identity. Export/restore verify object bytes before this comparison;
            # the shared validator also verifies inline bytes here.
            body = {"codec": parsed.codec, "raw_digest": parsed.raw_digest, "raw_size": parsed.raw_size}
            key = row["store_key"], row["sha256"]
            message = {"store_key": encode_native_store_key(key[0]), "sha256": key[1], "message": body}
            if key in native_messages and native_messages[key] != message:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            native_messages[key] = message
            for ref in parsed.object_refs():
                native_objects.add((ref.key, ref.digest, ref.size))
    normalized_objects = []
    for raw in objects:
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source = reference(raw.get("source"))
        target = _object_ref_from_payload(raw.get("content"))
        if (source["digest"], source["size"]) != (target.digest, target.size):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        original = _object_ref_from_payload(raw.get("source"))
        identity = original.key, original.digest, original.size
        if identity in native_objects and identity not in nonnative_objects:
            continue
        normalized_objects.append({
            "domain": "archive" if source["key"].startswith("v1/archive/") else raw.get("domain"),
            "source": source,
        })
    def normalize(value: JsonValue) -> JsonValue:
        return rekey_native_step_sources(rekey_native_message_roots(rekey_archive_roots(value, groups), groups), groups)

    normalized_domains = {name: {**normalize({key: value for key, value in values.items() if key != "records"}),
        "records": [encode_record(map_capture_references(decode_record(row), normalize))
                    if row.get("kind") in CAPTURE_RECORD_KINDS else normalize(row) for row in values["records"]]}
        for name, values in domains.items()}
    run_registrations = _decode_run_registrations(manifest.get("run_registrations", []))
    checkpoint_values = []
    for entry in manifest.get("native_checkpoints", []):
        if not isinstance(entry, Mapping) or entry.get("group_id") not in groups:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        backup = _decode_checkpoint_registration_refs(entry, run_registrations)
        checkpoint_values.append(backup.to_json())
    return canonical_sha256({
        "kind": manifest.get("kind"), "format_version": manifest.get("format_version"),
        "namespace": manifest.get("namespace"), "tenant_id": manifest.get("tenant_id"),
        "domains": normalized_domains,
        "run_registrations": ordered_unique([_run_json(run) for run in run_registrations.values()]),
        "archives": ordered_unique(archive_records),
        "native_checkpoints": ordered_unique(checkpoint_values),
        "native_projections": ordered_unique([
            {"records": entry["records"], "facts": entry["facts"]}
            for entry in manifest.get("native_projections", [])]),
        "native_sequences": ordered_unique(native_sequences),
        "native_messages": ordered_unique(list(native_messages.values())),
        "objects": ordered_unique(normalized_objects),
    })



_RUN_REGISTRATION_ADAPTER = TypeAdapter(RunRecord)


def _run_json(run: RunRecord) -> dict[str, object]:
    return json.loads(_RUN_REGISTRATION_ADAPTER.dump_json(run))


def _retain_run_registration(values: dict[str, RunRecord], run: RunRecord) -> None:
    previous = values.get(run.run_id)
    if previous is not None and previous != run:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values[run.run_id] = run


def _decode_run_registrations(raw: object) -> dict[str, RunRecord]:
    if not isinstance(raw, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result = {}
    try:
        for value in raw:
            run = _RUN_REGISTRATION_ADAPTER.validate_python(value)
            if run.run_id in result or _run_json(run) != value:
                raise ValueError("duplicate or noncanonical run registration")
            result[run.run_id] = run
    except (ValueError, TypeError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return result


def _run_from_native_row(row: Mapping[str, object]) -> RunRecord:
    try:
        return _RUN_REGISTRATION_ADAPTER.validate_python(dict(row, metadata=json.loads(row["metadata"])))
    except (ValueError, TypeError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _expand_run_reference(record: StoredRecord, runs: Mapping[str, RunRecord]) -> StoredRecord:
    if record.kind != "agent_run":
        return record
    reference = RunRegistrationReference.from_data(record.data)
    source = runs.get(reference.run_id)
    if source is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return replace(record, data=_encode_step_envelope(reference.project(source)))


def _encode_checkpoint_registration_refs(backup: NativeStepBackup, runs: Mapping[str, RunRecord]) -> dict[str, object]:
    result = backup.to_json()
    for row in backup.runs:
        if runs.get(row["run_id"]) != _run_from_native_row(row):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result["run_refs"] = [{"run_id": row["run_id"], "managed_start_key": row["managed_start_key"]}
                          for row in result["runs"]]
    del result["runs"]
    return result


def _decode_checkpoint_registration_refs(entry: Mapping[str, object], runs: Mapping[str, RunRecord]) -> NativeStepBackup:
    from ..steps import decode_native_store_key
    value = {key: item for key, item in entry.items() if key != "group_id"}
    identifiers = value.pop("run_refs", None)
    if ("runs" in value or not isinstance(identifiers, list)
            or any(not isinstance(item, Mapping) or set(item) != {"run_id", "managed_start_key"}
                   or not isinstance(item["run_id"], str) for item in identifiers)
            or len({item["run_id"] for item in identifiers}) != len(identifiers)):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        store_key = decode_native_store_key(value["store_key"])
        rows = []
        for reference in identifiers:
            identity = reference["run_id"]
            row = _run_json(runs[identity])
            row["managed_start_key"] = reference["managed_start_key"]
            row.update(store_key=value["store_key"], store_locator=native_store_locator(store_key),
                       started_at=runs[identity].started_at.isoformat(), metadata=json.dumps(row["metadata"]))
            rows.append(row)
        value["runs"] = rows
        return NativeStepBackup.from_json(value)
    except (ValueError, TypeError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _validate_native_partition(store_key: bytes, namespace: str, tenant_id: str) -> None:
    from ._step_store import SqlStepStore

    if store_key != SqlStepStore.scope_key(namespace=namespace, tenant_id=tenant_id):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_native_checkpoint_payloads(
    backup: NativeStepBackup, closure: NativeMessageClosure,
    payloads: Mapping[tuple[bytes, str], bytes],
) -> None:
    from ._step_store import SqlStepStore

    try:
        SqlStepStore.validate_backup_payloads(backup, closure, payloads)
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _expand_snapshot_native_content(
    record: StoredRecord,
    rows: Mapping[str, tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]],
    payloads: Mapping[str, Mapping[tuple[bytes, str], bytes]], *, namespace: str, tenant_id: str,
) -> StoredRecord:
    if record.kind not in {"execution", "tool_operation"}:
        return record
    value = decode_envelope(record.data).value["payload"]
    is_execution = record.kind == "execution"
    slots = execution_payload_slots(value) if is_execution else tool_payload_slots(value)
    restored = {}
    for _path, payload in slots:
        reference = native_text_reference(payload) if is_execution else native_tool_result_reference(payload)
        if reference is None or reference in restored:
            continue
        root = reference.root
        if is_execution:
            binding = binding_reference(value["fields"].get("binding"))
            if binding is None or binding[0] != root.group_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _validate_native_partition(root.store_key, namespace, tenant_id)
        owner = rows.get(root.group_id)
        if owner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identities = NativeMessageClosure(owner[0]).message_identities(root.store_key, root.slices)
        if len(identities) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = root.store_key, identities[0]
        bodies = {(row["store_key"], row["sha256"]): row for row in owner[1]}
        row = bodies.get(key)
        encoded = payloads.get(root.group_id, {}).get(key)
        if row is None or encoded is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        content = validate_native_message_content(row, group_id=root.group_id,
            content_scope=archive_content_scope(namespace, tenant_id))
        _validate_archive_content(content, encoded)
        raw = decode_payload_bytes(encoded, codec=content.compression,
            raw_digest=content.raw_digest, raw_size=content.raw_size)
        restored[reference] = (restore_native_text_payload(reference, raw) if is_execution
                               else restore_native_tool_result_payload(reference, raw))
    return (expand_native_execution_content(record, restored) if is_execution
            else expand_native_tool_content(record, restored))


async def _capture_native_payloads(
    group_id: str, sequences: list[dict[str, object]], messages: list[dict[str, object]],
    *, budget: _SnapshotCaptureBudget, objects: ObjectStore, backup: NativeStepBackup | None,
) -> dict[tuple[bytes, str], bytes]:
    # Reserve the entire reachable encoded and decoded inventory before opening
    # an object or decompressing an inline body. Row metadata was already charged.
    contents = []
    required_bytes = required_entries = 0
    for row in messages:
        content = ContentRecord.from_native_row(row)
        required_bytes += content.raw_size
        for ref in content.object_refs():
            required_bytes += ref.size
            required_entries += 1
        contents.append(content)
    if (budget.entries + required_entries > budget.limits.max_entries
            or budget.byte_size + required_bytes > budget.limits.max_bytes):
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    budget.entries += required_entries
    budget.byte_size += required_bytes
    encoded_payloads, raw_payloads = {}, {}
    for row, content in zip(messages, contents, strict=True):
        validate_native_message_content(row, group_id=group_id, verify_inline=False)
        encoded = await _read_content_payload(content, lambda ref: read_object(
            objects, ref.key, expected_digest=ref.digest, expected_size=ref.size))
        raw = decode_payload_bytes(encoded, codec=content.compression,
                                   raw_digest=content.raw_digest, raw_size=content.raw_size)
        decode_model_message_payloads((raw,))
        identity = row["store_key"], row["sha256"]
        encoded_payloads[identity] = encoded
        raw_payloads[identity] = raw
    if backup is not None:
        _validate_native_checkpoint_payloads(backup, NativeMessageClosure(sequences), raw_payloads)
    return encoded_payloads


async def _read_native_messages(
    store: SqlStateStore, roots: list[NativeMessageRoot], *, budget: _SnapshotCaptureBudget,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from sqlalchemy import and_, or_, select
    from ._step_schema import build_step_sql_metadata

    metadata = build_step_sql_metadata()
    async def read(transaction: StateTransaction) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if not isinstance(transaction, SqlStateRowReader):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        nodes = metadata.tables["ai_step_sequences"]
        columns = ("store_key", "sha256", "message_sha256", "tree_count",
                   "left_sha256", "right_sha256", "next_sha256")
        sequences = []
        after = None
        partitions = sorted({root.store_key for root in roots})
        while True:
            limit = budget.page_size(128)
            statement = select(*(nodes.c[name] for name in columns)).where(
                nodes.c.store_locator.in_(tuple(native_store_locator(partition) for partition in partitions)),
                nodes.c.store_key.in_(partitions))
            if after is not None:
                statement = statement.where(or_(nodes.c.store_key > after[0],
                    and_(nodes.c.store_key == after[0], nodes.c.sha256 > after[1])))
            rows = await transaction.read_sql_rows(store.context,
                statement.order_by(nodes.c.store_key, nodes.c.sha256).limit(limit))
            for row in rows:
                budget.accept(encode_native_sequence_row(row))
                sequences.append(dict(row))
            if len(rows) < limit:
                break
            after = rows[-1]["store_key"], rows[-1]["sha256"]
        closure = NativeMessageClosure(sequences)
        for root in roots:
            closure.mark(root.store_key, root.slices)
        sequences = [row for row in sequences if (row["store_key"], row["sha256"]) in closure.retained_sequences]
        messages = []
        table = metadata.tables["ai_step_messages"]
        for partition in partitions:
            identities = sorted(identity for scope, identity in closure.retained_messages if scope == partition)
            offset = 0
            while offset < len(identities):
                batch = identities[offset:offset + budget.page_size(128)]
                rows = await transaction.read_sql_rows(store.context, select(
                    table.c.store_key, table.c.sha256, table.c.message,
                    table.c.compression, table.c.raw_size, table.c.payload_json).where(
                    table.c.store_locator == native_store_locator(partition),
                    table.c.store_key == partition, table.c.sha256.in_(batch)).order_by(table.c.sha256))
                if len(rows) != len(batch):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for row in rows:
                    budget.accept(encode_native_message_row(row))
                    messages.append(dict(row))
                offset += len(batch)
        return sequences, messages
    return await store.read(read)


async def _insert_native_messages(
    transaction: SqlStateTransaction, store: SqlStateStore,
    entries: list[tuple[list[dict[str, object]], list[dict[str, object]]]],
) -> None:
    from ._step_schema import build_step_sql_metadata

    metadata = build_step_sql_metadata()
    async def insert(session: "AsyncSession") -> None:
        for sequences, messages in entries:
            for name, rows in (("ai_step_messages", messages), ("ai_step_sequences", sequences)):
                table = metadata.tables[name]
                if not rows:
                    continue
                # Include the derived locator in the per-row portable bind budget.
                batch_size = min(128, 999 // (len(rows[0]) + 1))
                for start in range(0, len(rows), batch_size):
                    await store.context.run_statement(store.context.dialect.insert_immutable_many(
                        session, table=table,
                        rows=[dict(row, store_locator=native_store_locator(row["store_key"]))
                              for row in rows[start:start + batch_size]],
                        index_elements=("store_locator", "sha256"), conflict_column="id"))
            from ._native_content_index import index_nodes
            partitions = sorted({row["store_key"] for row in sequences})
            for partition in partitions:
                await index_nodes(store.context, session, partition,
                                  tuple(row for row in sequences if row["store_key"] == partition), partial_restore=True)
    await transaction.run_sql(store.context, insert)


def _mark_archive_value(closures: Mapping[str, ArchiveClosure], value: object) -> None:
    for root in iter_archive_roots(value):
        closure = closures.get(root.scope_id)
        if closure is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        closure.mark(root)
    for scope, content_scope, identity in iter_archive_content_refs(value):
        closure = closures.get(scope)
        if closure is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        closure.mark_content(scope, content_scope, identity)


async def _read_content_payload(
    content: ContentRecord, reader: Callable[[ObjectRef], Awaitable[bytes]],
) -> bytes:
    if content.payload is None:
        return cast(bytes, content.body)
    return await read_attachment_payload(content.payload, read_object=reader)


def _validate_archive_content(content: ContentRecord, encoded: object) -> None:
    if not isinstance(encoded, bytes):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    decode_payload_bytes(encoded, codec=content.compression,
                         raw_digest=content.raw_digest, raw_size=content.raw_size)


def _decode_snapshot_stream_key(value: object) -> bytes:
    if not isinstance(value, str) or not 2 <= len(value) <= 2 * STREAM_KEY_MAX_BYTES:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if decoded.hex() != value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return validate_stream_key(decoded)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _decode_snapshot_aliases(value: object) -> tuple[StoredAlias, ...]:
    if not isinstance(value, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    aliases: list[StoredAlias] = []
    seen: set[bytes] = set()
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {
            "alias_key",
            "record_key",
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        alias = decode_alias({
            "alias": raw["alias_key"],
            "record": raw["record_key"],
        })
        if alias.alias_key in seen:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        seen.add(alias.alias_key)
        aliases.append(alias)
    return tuple(sorted(aliases, key=lambda item: item.alias_key))


def _decode_snapshot_sequences(value: object) -> Mapping[bytes, int]:
    if not isinstance(value, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    sequences: dict[bytes, int] = {}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {"stream_key", "value"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = _decode_snapshot_stream_key(raw["stream_key"])
        sequence = raw["value"]
        if (
            key in sequences
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sequences[key] = sequence
    return sequences


async def _insert_snapshot_values(
    transaction: StateTransaction,
    records: tuple[StoredRecord, ...],
    aliases: tuple[StoredAlias, ...],
    facts: tuple[StoredFact, ...],
    operations: tuple[StoredOperation, ...],
    sequences: Mapping[bytes, int],
) -> None:
    await transaction.insert_records(records)
    await transaction.insert_aliases(aliases)
    await transaction.insert_facts(facts)
    for operation in operations:
        await transaction.insert_operation(operation)
    if sequences:
        restored = await transaction.reserve_sequences(sequences)
        if dict(restored) != dict(sequences):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _snapshot_chunk(value: bytes) -> AsyncIterator[bytes]:
    yield value


async def _put_snapshot_object(
    object_store: ObjectStore,
    key: str,
    value: bytes,
) -> None:
    digest = hashlib.sha256(value).hexdigest()
    current = await object_store.stat(key)
    if current is not None:
        if current.digest != digest or current.size != len(value):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await read_object(
            object_store,
            key,
            expected_digest=digest,
            expected_size=len(value),
        )
        return
    await object_store.put(
        key,
        _snapshot_chunk(value),
        expected_size=len(value),
        expected_digest=digest,
    )


from ._capability_marker import ensure_capability_partition_marker
from ._snapshot import snapshot_object_ref_from_payload, snapshot_object_ref_payload

__all__ = ['RunRegistrationCleanupResult', 'snapshot_manifest_usage', 'snapshot_manifest_identity', 'RuntimeSnapshotOperations', 'SnapshotExclusiveGuard', 'SnapshotLimits', 'SnapshotResources', 'ensure_capability_partition_marker', 'has_capability_partition_marker', 'restore_snapshot', 'snapshot_object_ref_from_payload', 'snapshot_object_ref_payload']
