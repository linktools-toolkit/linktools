#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded portable state snapshots and offline object maintenance."""

import hashlib
import json
from bisect import bisect_left, bisect_right
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..core import (
    canonical_json_bytes,
    canonical_sha256,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..steps import NativeStepSource, rekey_native_step_sources
from ._step_backup import NativeStepBackup
from ._native_backup_projection import NativeBackupProjections
from ..storage import ObjectRef, ObjectStore, ObjectStoreInspection, StoredPayload, read_object
from ..task import TaskGraphAdmission, TaskGraphSubmission
from ._runtime_identity import task_graph_binding_capture_key
from ._native_run_reference import NativeRunReferences
from ._native_event_reference import NativeEventReferences
from ._sql import SqlStateStore
from ._archive import ContentRecord
from ._message_root import NativeMessageRoot
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
    input_capture_expiry_key,
    input_capture_key,
    iter_input_capture_dependencies,
    validate_input_capture_payload,
)
from ._capability_marker import has_capability_partition_marker
from ._plan import PersistenceDomain
from .state import (
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
    snapshot_object_ref_payload as _object_ref_payload,
    snapshot_object_ref_from_payload as _object_ref_from_payload,
)
from ._object_cleanup import (
    ObjectCleanupResult,
    purge_unreferenced_objects as _purge_unreferenced_objects,
)
from ._snapshot_validation import (
    canonical_snapshot_indexes,
    validate_snapshot_domain,
    validate_snapshot_references,
)
from .codec import (
    _decode_enveloped_domain,
    iter_runtime_object_dependencies,
    decode_fact,
    decode_operation,
    decode_record,
    encode_fact,
    encode_operation,
    encode_record,
    iter_runtime_object_refs,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from ._step_store import SqlStepStore


class _SnapshotReader(Protocol):
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
    def __init__(self, store: StateStore, *, native_run_scope: tuple[str, str] | None = None) -> None:
        self._store = store
        self._native_runs = (NativeRunReferences(store, namespace=native_run_scope[0], tenant_id=native_run_scope[1])
                             if native_run_scope is not None and isinstance(store, SqlStateStore) else None)
        self._native_events = (NativeEventReferences(store, namespace=native_run_scope[0], tenant_id=native_run_scope[1])
                               if native_run_scope is not None and isinstance(store, SqlStateStore) else None)

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        return await self._store.read(lambda tx: tx.get_operation(key))

    async def scan_records_page(
        self, *, after: RecordScanCursor | None, limit: int,
    ) -> tuple[StoredRecord, ...]:
        if self._native_runs is not None:
            return await self._store.read(lambda tx: self._native_runs.read(tx, scan=(after, limit)))
        return await self._store.read(lambda tx: tx.scan_records_page(after=after, limit=limit))

    async def scan_facts_page(
        self, *, after: FactScanCursor | None, limit: int,
    ) -> tuple[StoredFact, ...]:
        if self._native_events is not None:
            return await self._store.read(lambda tx: self._native_events.read(tx, scan=(after, limit)))
        return await self._store.read(lambda tx: tx.scan_facts_page(after=after, limit=limit))

    async def scan_operations_page(
        self, *, after: OperationScanCursor | None, limit: int,
    ) -> tuple[StoredOperation, ...]:
        return await self._store.read(lambda tx: tx.scan_operations_page(after=after, limit=limit))


@dataclass(frozen=True, slots=True)
class _CapturedSnapshotReader:
    records: tuple[StoredRecord, ...]
    facts: tuple[StoredFact, ...]
    operations: tuple[StoredOperation, ...]

    async def get_operation(self, key: bytes) -> StoredOperation | None:
        index = bisect_left(self.operations, key, key=lambda value: value.key_digest)
        return (self.operations[index] if index < len(self.operations)
                and self.operations[index].key_digest == key else None)

    async def scan_records_page(self, *, after: RecordScanCursor | None, limit: int) -> tuple[StoredRecord, ...]:
        start = 0 if after is None else bisect_right(self.records, (after.kind, after.key_digest),
                                                    key=lambda value: (value.kind, value.key_digest))
        return self.records[start:start + limit]

    async def scan_facts_page(self, *, after: FactScanCursor | None, limit: int) -> tuple[StoredFact, ...]:
        start = 0 if after is None else bisect_right(self.facts, (after.stream_digest, after.sequence),
                                                    key=lambda value: (value.stream_digest, value.sequence))
        return self.facts[start:start + limit]

    async def scan_operations_page(self, *, after: OperationScanCursor | None, limit: int) -> tuple[StoredOperation, ...]:
        start = 0 if after is None else bisect_right(self.operations, after.key_digest, key=lambda value: value.key_digest)
        return self.operations[start:start + limit]


@dataclass(slots=True)
class _SnapshotCaptureBudget:
    """Bound the detached source inventory separately from final wire accounting."""

    limits: SnapshotLimits
    entries: int = 0
    byte_size: int = 0

    def page_size(self, requested: int) -> int:
        return min(requested, max(1, self.limits.max_entries - self.entries))

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
            budget.accept(encode_record(value))
            records.append(value)
        if len(page) < limit:
            break
        cursor = RecordScanCursor(page[-1].kind, page[-1].key_digest)
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
        cursor = FactScanCursor(page[-1].stream_digest, page[-1].sequence)
    cursor = None
    while True:
        limit = budget.page_size(page_size)
        page = await reader.scan_operations_page(after=cursor, limit=limit)
        for value in page:
            budget.accept(encode_operation(value))
            operations.append(value)
        if len(page) < limit:
            break
        cursor = OperationScanCursor(page[-1].key_digest)
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
                native = await resolver(NativeStepSource(opened[0], SqlStepStore.scope_digest(
                    namespace=self._resources.namespace, tenant_id=self._resources.tenant_id)))

            async def capture_group(transaction: object) -> None:
                roots = []
                for domain, store in members:
                    reader = _StateSnapshotReader(store,
                        native_run_scope=((self._resources.namespace, self._resources.tenant_id)
                                          if domain is PersistenceDomain.TRACE else None))
                    captured = await _capture_snapshot_reader(reader, budget=capture_budget)
                    readers[domain] = captured
                    for value in (*captured.records, *captured.facts, *captured.operations):
                        roots.extend(iter_native_message_roots(value.data))
                if PersistenceDomain.EXECUTION in {domain for domain, _ in members}:
                    from .contracts import ExecutionRecord
                    from ..agent import AgentBindingContract

                    for record in readers[PersistenceDomain.EXECUTION].records:
                        if record.kind != "execution":
                            continue
                        execution = _decode_enveloped_domain(record.data, ExecutionRecord)
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
                            if native_checkpoints[source.group_id].store_digest != source.store_digest:
                                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                            continue
                        if native is None:
                            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                        execution_store = self._resources.stores[PersistenceDomain.EXECUTION]
                        if (not isinstance(execution_store, SqlStateStore)
                                or native.storage_context.engine is not execution_store.context.engine
                                or native.store_digest != source.store_digest or not native.backup_supported):
                            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                        async def capture_native(tx: StateTransaction) -> NativeStepBackup:
                            if not isinstance(tx, SqlStateRowReader):
                                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                            return await native.capture_backup(tx, accept=capture_budget.accept)
                        backup = await execution_store.read(capture_native)
                        native_checkpoints[source.group_id] = backup
                        roots.extend(backup.message_roots(source.group_id))
                if (opened is not None and opened[0] in native_checkpoints
                        and PersistenceDomain.TRACE in {domain for domain, _ in members}):
                    trace = self._resources.stores[PersistenceDomain.TRACE]
                    physical = await _capture_snapshot_reader(_StateSnapshotReader(trace), budget=_SnapshotCaptureBudget(limits))
                    projected = readers[PersistenceDomain.TRACE]
                    records = {value.key_digest: value for value in projected.records}
                    facts = {(value.stream_digest, value.sequence): value for value in projected.facts}
                    validator = NativeBackupProjections(native_checkpoints[opened[0]],
                        namespace=self._resources.namespace, tenant_id=self._resources.tenant_id)
                    raw_records, raw_facts = [], []
                    for value in physical.records:
                        visible = records.get(value.key_digest)
                        if visible is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if value.data != visible.data:
                            if validator.validate(visible, value.data) != value:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            capture_budget.accept(encode_record(value))
                            raw_records.append(value)
                    for value in physical.facts:
                        visible = facts.get((value.stream_digest, value.sequence))
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
                            _validate_native_partition(root.store_digest, self._resources.namespace, self._resources.tenant_id)
                        sequences, messages = await _read_native_messages(store, roots, budget=capture_budget)
                        native_rows[scope] = (PersistenceDomain.EXECUTION if scope in native_checkpoints else domain), sequences, messages
                elif roots:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            # Capture each physical group's roots and sources in the same read
            # view. Independent groups cannot share a database transaction.
            await group.read_many(tuple(store for _, store in members), capture_group)

        async def archives() -> AsyncIterator[_SnapshotArchive]:
            for archive in archive_sources:
                yield archive

        source = _SnapshotExportSource(
            namespace=self._resources.namespace, tenant_id=self._resources.tenant_id,
            readers=readers, archives=archives(), object_stores=self._resources.object_stores,
            native_rows=native_rows, native_checkpoints=native_checkpoints, native_projections=native_projections,
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
            cursor = RecordScanCursor(page[-1].kind, page[-1].key_digest)
        archive_closures[archive.scope_id] = ArchiveClosure(archive.scope_id, records)
        archive_domains[archive.scope_id] = archive.domain

    domains: dict[str, dict[str, list[object]]] = {}
    snapshot_records: dict[PersistenceDomain, tuple[StoredRecord, ...]] = {}
    snapshot_facts: dict[PersistenceDomain, tuple[StoredFact, ...]] = {}
    objects: list[dict[str, object]] = []
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
            source_store.open(reference.key),
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
        if reference.key.startswith("v1/input-capture/"):
            payload = await read_object(source_store, reference.key, expected_digest=reference.digest, expected_size=reference.size)
            manifest = json.loads(payload)
            validate_input_capture_payload(reference.key, manifest, namespace=source.namespace, tenant_id=source.tenant_id)
            if _capture_contains_archived_context(manifest):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if not reference.key.startswith("v1/input-capture/result/"):
                await copy_references(manifest, PersistenceDomain.TASK, archive_owner=False)
        if reference.key.startswith(
            (
                "v1/asset-snapshot/",
                "v1/task-graph-binding-capture/",
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
                _validate_native_partition(root.store_digest, source.namespace, source.tenant_id)
                native_roots.setdefault(root.group_id, []).append(root)
        for key, digest in iter_input_capture_dependencies(encoded, namespace=source.namespace, tenant_id=source.tenant_id):
            capture_source = source.object_stores[PersistenceDomain.TASK]
            marker = await capture_source.stat(input_capture_expiry_key(key))
            if marker is not None:
                await copy_reference(PersistenceDomain.TASK, ObjectRef(capture_source.store_id, marker.key, marker.digest, marker.size))
                continue
            stat = await capture_source.stat(key)
            if stat is None or stat.digest != digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await copy_reference(PersistenceDomain.TASK, ObjectRef(capture_source.store_id, key, digest, stat.size))
        for source_domain, reference in iter_runtime_object_refs(
            encoded,
            default_domain=domain,
        ):
            await copy_reference(source_domain, reference)

    if PersistenceDomain.TASK in source.readers:
        capture_store = source.object_stores[PersistenceDomain.TASK]
        if not isinstance(capture_store, ObjectStoreInspection):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        async for stat in capture_store.list_objects():
            if not stat.key.startswith(tuple("v1/input-capture/" + kind + "/" for kind in ("agent", "task", "graph", "template", "expired"))):
                continue
            payload = await read_object(capture_store, stat.key, expected_digest=stat.digest, expected_size=stat.size)
            value = json.loads(payload)
            if not isinstance(value, Mapping) or value.get("namespace") != source.namespace or value.get("tenant_id") != source.tenant_id:
                continue
            if not stat.key.startswith("v1/input-capture/expired/") and await capture_store.stat(input_capture_expiry_key(stat.key)) is not None:
                continue
            await copy_reference(PersistenceDomain.TASK, ObjectRef(capture_store.store_id, stat.key, stat.digest, stat.size))

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
        domain_facts: list[StoredFact] = []
        domain_operations: list[StoredOperation] = []

        record_cursor: RecordScanCursor | None = None
        while True:
            page = await store.scan_records_page(after=record_cursor, limit=128)
            if not page:
                break
            for value in page:
                if domain is PersistenceDomain.EXECUTION and value.kind == "execution":
                    from .contracts import ExecutionRecord
                    execution = _decode_enveloped_domain(value.data, ExecutionRecord)
                    from ..agent import AgentBindingContract
                    if (isinstance(execution.binding, AgentBindingContract) and execution.restart_checkpoints
                            and (execution.native_step_source is None
                                 or execution.native_step_source.group_id not in (source.native_checkpoints or {})
                                 or source.native_checkpoints[execution.native_step_source.group_id].store_digest
                                 != execution.native_step_source.store_digest)):
                        raise AIError(
                            ErrorCode.SNAPSHOT_UNSUPPORTED,
                            "Native StepPersistence executions require a consistent database and StepStore backup; "
                            "portable snapshots do not include the external StepStore.",
                        )
                domain_records.append(value)
                encoded = encode_record(value)
                raw_domain["records"].append(accept(encoded))
                if value.kind != "evaluation_cleanup":
                    await copy_references(encoded, domain)
                if value.kind in {"execution", "task_admission"}:
                    if value.kind == "execution":
                        from .contracts import ExecutionRecord
                        identity = _decode_enveloped_domain(value.data, ExecutionRecord).execution_id
                        capture_kind = "invocation"
                    else:
                        identity = _decode_enveloped_domain(value.data, TaskGraphAdmission).graph_id
                        capture_kind = "declaration"
                    key = input_capture_key(source.namespace, source.tenant_id, capture_kind, identity)
                    stat = await source.object_stores[PersistenceDomain.TASK].stat(key)
                    if stat is not None:
                        await copy_reference(PersistenceDomain.TASK, ObjectRef(source.object_stores[PersistenceDomain.TASK].store_id, key, stat.digest, stat.size))
                if (
                    domain is PersistenceDomain.TASK
                    and value.kind in {"task_admission", "task_submission_payload"}
                ):
                    admission = (
                        _decode_enveloped_domain(value.data, TaskGraphAdmission)
                        if value.kind == "task_admission"
                        else _decode_enveloped_domain(value.data, TaskGraphSubmission).admission
                    )
                    key = task_graph_binding_capture_key(
                        source.namespace,
                        admission.principal.tenant_id,
                        admission.graph_id,
                        admission.initial_request_digest,
                    )
                    stat = await source.object_stores[PersistenceDomain.TASK].stat(key)
                    if stat is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    await copy_reference(
                        PersistenceDomain.TASK,
                        ObjectRef(
                            source.object_stores[PersistenceDomain.TASK].store_id,
                            stat.key,
                            stat.digest,
                            stat.size,
                        ),
                    )
            last = page[-1]
            record_cursor = RecordScanCursor(last.kind, last.key_digest)
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
            fact_cursor = FactScanCursor(last.stream_digest, last.sequence)
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
            operation_cursor = OperationScanCursor(page[-1].key_digest)
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
                        "alias_digest": value.alias_digest.hex(),
                        "record_key_digest": value.record_key_digest.hex(),
                    }
                )
            )
        for key in sorted(sequences):
            raw_domain["sequences"].append(
                accept({"key_digest": key.hex(), "value": sequences[key]})
            )

        domains[domain.value] = raw_domain
        snapshot_records[domain] = tuple(domain_records)
        snapshot_facts[domain] = tuple(domain_facts)

    validate_snapshot_references(snapshot_records, snapshot_facts)
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
                content = ContentRecord.from_json(record.data)
                encoded = (content.payload.decode() if content.payload.ref is None else
                           await read_object(source.archive_object_store(domain), content.payload.ref.key,
                                             expected_digest=content.payload.digest, expected_size=content.payload.size))
                _validate_archive_content(content, encoded)
        archives.append({"scope_id": scope, "domain": domain.value,
                         "records": [accept(encode_record(record)) for record in retained]})
    native_checkpoints = []
    for group_id, backup in (source.native_checkpoints or {}).items():
        native_roots.setdefault(group_id, []).extend(backup.message_roots(group_id))
        raw = backup.to_json()
        for name in ("runs", "events", "snapshots"):
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
            closure.mark(root.store_digest, root.slices)
        retained_sequences = [row for row in sequences if (row["store_digest"], row["sha256"]) in closure.retained_sequences]
        retained_messages = [row for row in messages if (row["store_digest"], row["sha256"]) in closure.retained_messages]
        if {(row["store_digest"], row["sha256"]) for row in retained_messages} != closure.retained_messages:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for row in retained_messages:
            content = validate_native_message_content(row, group_id=group_id)
            if content.payload.ref is not None:
                await copy_reference(domain, content.payload.ref)
                encoded = await read_object(source.archive_object_store(domain), content.payload.ref.key,
                                            expected_digest=content.payload.digest, expected_size=content.payload.size)
                _validate_archive_content(content, encoded)
        native_messages.append({"group_id": group_id, "domain": domain.value,
                                "sequences": [accept(row) for row in retained_sequences],
                                "messages": [accept(row) for row in retained_messages]})
    manifest = {
        "kind": "runtime-storage-snapshot",
        "format_version": 1,
        "namespace": source.namespace,
        "tenant_id": source.tenant_id,
        "domains": domains,
        "archives": archives,
        "native_messages": native_messages,
        "native_checkpoints": native_checkpoints,
        "native_projections": [{"group_id": group_id,
            "records": [accept(encode_record(value)) for value in projected.records],
            "facts": [accept(encode_fact(value)) for value in projected.facts]}
            for group_id, projected in (source.native_projections or {}).items()],
        "objects": objects,
    }
    payload = canonical_json_bytes(manifest)
    if len(payload) + object_bytes > limits.max_bytes:
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

    raw_domains = manifest["domains"]
    raw_objects = manifest.get("objects", [])
    if not isinstance(raw_objects, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    raw_archives = manifest.get("archives", [])
    if not isinstance(raw_archives, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    archive_closures: dict[str, ArchiveClosure] = {}
    archive_domains: dict[str, PersistenceDomain] = {}
    entry_count = len(raw_objects)
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
            backup = NativeStepBackup.from_json({key: value for key, value in entry.items() if key != "group_id"})
            SqlStepStore.validate_backup(backup)
        except (TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        _validate_native_partition(backup.store_digest, namespace, tenant_id)
        native_checkpoints[group_id] = backup
        entry_count += len(backup.runs) + len(backup.events) + len(backup.snapshots)
        if entry_count > limits.max_entries:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        for message_root in backup.message_roots(group_id):
            native_closures[group_id].mark(message_root.store_digest, message_root.slices)
    object_bytes = 0
    expected_objects: set[tuple[str, str, str, int]] = set()
    expected_task_object_keys: set[str] = set()
    capture_dependencies: set[tuple[str, str]] = set()
    optional_capture_keys: set[str] = set()
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
        aliases = _decode_snapshot_aliases(raw_aliases)
        facts = tuple(decode_fact(value) for value in raw_facts)
        operations = tuple(decode_operation(value) for value in raw_operations)
        sequences = _decode_snapshot_sequences(raw_sequences)
        validate_snapshot_domain(
            namespace=namespace,
            tenant_id=tenant_id,
            domain=domain,
            records=records,
            aliases=aliases,
            facts=facts,
            operations=operations,
            sequences=sequences,
        )
        for value in (*records, *facts, *operations):
            _mark_archive_value(archive_closures, dict(value.data))
            for native_root in iter_native_message_roots(value.data):
                _validate_native_partition(native_root.store_digest, namespace, tenant_id)
                closure = native_closures.get(native_root.group_id)
                if closure is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                closure.mark(native_root.store_digest, native_root.slices)
        decoded_domains[domain] = (
            records,
            aliases,
            facts,
            operations,
            sequences,
        )
        if domain is PersistenceDomain.TASK:
            for record in records:
                if record.kind not in {"task_admission", "task_submission_payload"}:
                    continue
                admission = (
                    _decode_enveloped_domain(record.data, TaskGraphAdmission)
                    if record.kind == "task_admission"
                    else _decode_enveloped_domain(record.data, TaskGraphSubmission).admission
                )
                expected_task_object_keys.add(
                    task_graph_binding_capture_key(
                        namespace,
                        admission.principal.tenant_id,
                        admission.graph_id,
                        admission.initial_request_digest,
                    )
                )
        for record in records:
            if record.kind == "execution":
                from .contracts import ExecutionRecord
                execution = _decode_enveloped_domain(record.data, ExecutionRecord)
                from ..agent import AgentBindingContract
                if isinstance(execution.binding, AgentBindingContract) and execution.restart_checkpoints:
                    source = execution.native_step_source
                    if (source is None or source.group_id not in native_checkpoints
                            or source.store_digest != native_checkpoints[source.group_id].store_digest):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    required_checkpoint_groups.add(source.group_id)
                identity = execution.execution_id
                optional_capture_keys.add(input_capture_key(namespace, tenant_id, "invocation", identity))
            elif record.kind == "task_admission":
                identity = _decode_enveloped_domain(record.data, TaskGraphAdmission).graph_id
                optional_capture_keys.add(input_capture_key(namespace, tenant_id, "declaration", identity))
        live_records = tuple(encoded for encoded, record in zip(raw_records, records) if record.kind != "evaluation_cleanup")
        for encoded in (*live_records, *raw_facts, *raw_operations):
            capture_dependencies.update(iter_input_capture_dependencies(encoded, namespace=namespace, tenant_id=tenant_id))
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
        {domain: values[0] for domain, values in decoded_domains.items()},
        {domain: values[2] for domain, values in decoded_domains.items()},
    )
    if required_checkpoint_groups != set(native_checkpoints):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raw_projections = manifest.get("native_projections", [])
    if not isinstance(raw_projections, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    native_record_projections, native_fact_projections = {}, {}
    seen_projection_groups = set()
    trace_values = decoded_domains.get(PersistenceDomain.TRACE, ((), (), (), (), {}))
    trace_records = {value.key_digest: value for value in trace_values[0]}
    trace_facts = {(value.stream_digest, value.sequence): value for value in trace_values[2]}
    for entry in raw_projections:
        if (not isinstance(entry, dict) or set(entry) != {"group_id", "records", "facts"}
                or entry["group_id"] not in native_checkpoints or entry["group_id"] in seen_projection_groups
                or not isinstance(entry["records"], list) or not isinstance(entry["facts"], list)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        seen_projection_groups.add(entry["group_id"])
        validator = NativeBackupProjections(native_checkpoints[entry["group_id"]], namespace=namespace, tenant_id=tenant_id)
        entry_count += len(entry["records"]) + len(entry["facts"])
        if entry_count > limits.max_entries:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        for encoded in entry["records"]:
            raw = decode_record(encoded)
            visible = trace_records.get(raw.key_digest)
            if (visible is None or raw.key_digest in native_record_projections
                    or validator.validate(visible, raw.data) != raw):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            native_record_projections[raw.key_digest] = raw
        for encoded in entry["facts"]:
            raw = decode_fact(encoded)
            key = raw.stream_digest, raw.sequence
            visible = trace_facts.get(key)
            if (visible is None or key in native_fact_projections
                    or validator.validate(visible, raw.data) != raw):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            native_fact_projections[key] = raw
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

    for key in expected_task_object_keys:
        matches = tuple(
            identity
            for identity in actual_objects
            if identity[0] == PersistenceDomain.TASK.value
            and identity[1] == key
        )
        if len(matches) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_objects.add(matches[0])

    def capture_identity(key: str, digest: str) -> tuple[str, str, str, int]:
        matches = [identity for identity in actual_objects if identity[0] == PersistenceDomain.TASK.value and identity[1] == key and identity[2] == digest]
        if not matches:
            marker_key = input_capture_expiry_key(key)
            matches = [identity for identity in actual_objects if identity[0] == PersistenceDomain.TASK.value and identity[1] == marker_key]
        if len(matches) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return matches[0]

    for key, digest in capture_dependencies:
        expected_objects.add(capture_identity(key, digest))
    expected_objects.update(identity for identity in actual_objects if identity[0] == PersistenceDomain.TASK.value and identity[1] in optional_capture_keys)
    for identity in actual_objects:
        if identity[0] != PersistenceDomain.TASK.value or not identity[1].startswith(tuple("v1/input-capture/" + kind + "/" for kind in ("agent", "task", "graph", "template", "expired"))):
            continue
        _, source, content_ref = decoded_by_identity[identity]
        raw_capture = await read_object(object_store, content_ref.key, expected_digest=content_ref.digest, expected_size=content_ref.size)
        capture = json.loads(raw_capture)
        validate_input_capture_payload(source.key, capture, namespace=namespace, tenant_id=tenant_id)
        expected_objects.add(identity)
    pending_objects = list(expected_objects)
    while pending_objects:
        identity = pending_objects.pop()
        decoded = decoded_by_identity.get(identity)
        if decoded is None:
            continue
        domain, source, content_ref = decoded
        if source.key.startswith("v1/input-capture/"):
            raw_capture = await read_object(object_store, content_ref.key, expected_digest=content_ref.digest, expected_size=content_ref.size)
            manifest_capture = json.loads(raw_capture)
            validate_input_capture_payload(source.key, manifest_capture, namespace=namespace, tenant_id=tenant_id)
            if _capture_contains_archived_context(manifest_capture):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if not source.key.startswith("v1/input-capture/result/"):
                for key, digest in iter_input_capture_dependencies(manifest_capture, namespace=namespace, tenant_id=tenant_id):
                    nested_identity = capture_identity(key, digest)
                    if nested_identity not in expected_objects:
                        expected_objects.add(nested_identity)
                        pending_objects.append(nested_identity)
                for nested_domain, nested in iter_runtime_object_refs(manifest_capture, default_domain=PersistenceDomain.TASK):
                    nested_identity = (nested_domain.value, nested.key, nested.digest, nested.size)
                    if nested_identity not in expected_objects:
                        expected_objects.add(nested_identity)
                        pending_objects.append(nested_identity)
        if not source.key.startswith(
            (
                "v1/asset-snapshot/",
                "v1/task-graph-binding-capture/",
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
            content = ContentRecord.from_json(record.data)
            reference = content.payload.ref
            if reference is None:
                encoded = content.payload.decode()
            else:
                identity = domain.value, reference.key, reference.digest, reference.size
                expected_objects.add(identity)
                decoded = decoded_by_identity.get(identity)
                if decoded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                content_ref = decoded[2]
                encoded = await read_object(object_store, content_ref.key, expected_digest=content_ref.digest, expected_size=content_ref.size)
            _validate_archive_content(content, encoded)

    for group_id, entry in native_entries.items():
        checkpoint_payloads = {}
        closure = native_closures[group_id]
        sequence_keys = [(row["store_digest"], row["sha256"]) for row in entry["sequences"]]
        message_keys = []
        for row in entry["messages"]:
            content = validate_native_message_content(row, group_id=group_id)
            message_keys.append((row["store_digest"], row["sha256"]))
            reference = content.payload.ref
            if reference is not None:
                identity = entry["domain"], reference.key, reference.digest, reference.size
                expected_objects.add(identity)
                decoded = decoded_by_identity.get(identity)
                if decoded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                content_ref = decoded[2]
                encoded = await read_object(object_store, content_ref.key,
                                            expected_digest=content_ref.digest, expected_size=content_ref.size)
                _validate_archive_content(content, encoded)
            else:
                encoded = content.payload.decode()
            if group_id in native_checkpoints:
                checkpoint_payloads[(row["store_digest"], row["sha256"])] = decode_payload_bytes(
                    encoded, codec=content.compression, raw_digest=content.raw_digest, raw_size=content.raw_size)
        if (len(sequence_keys) != len(set(sequence_keys)) or len(message_keys) != len(set(message_keys))
                or set(sequence_keys) != closure.retained_sequences or set(message_keys) != closure.retained_messages):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if group_id in native_checkpoints:
            from ._step_store import SqlStepStore
            backup = native_checkpoints[group_id]
            try:
                for row in backup.snapshots:
                    if row["pruned"]:
                        continue
                    sequence = SqlStepStore.snapshot_message_sequence(row)
                    identities = closure.message_identities(backup.store_digest, sequence.slices)
                    payloads = tuple(checkpoint_payloads[(backup.store_digest, identity)] for identity in identities)
                    SqlStepStore.decode_backup_snapshot(dict(row), payloads)
            except (TypeError, ValueError, KeyError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    if actual_objects != expected_objects:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    target = Path(root).expanduser().resolve(strict=False)
    if target.exists() and any(target.iterdir()):
        raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
    async with open_target(target, namespace, tenant_id) as resources:
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
            for value in (*records, *facts, *operations):
                referenced_scopes.update(root.scope_id for root in iter_archive_roots(value.data))
                referenced_scopes.update(scope for scope, _, _ in iter_archive_content_refs(value.data))
            domain_native_roots = tuple(root for value in (*records, *facts, *operations)
                                        for root in iter_native_message_roots(value.data))
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
                            content = ContentRecord.from_json(record.data)
                            reference = content.payload.ref
                            if reference is not None:
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
                                content = replace(content, payload=StoredPayload.object(
                                    ObjectRef(reference.store_id, key, reference.digest, reference.size)))
                                imported = replace(record, data=content.to_json())
                        previous = archive_records.get(imported.key_digest)
                        if previous is not None and previous != imported:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        archive_records[imported.key_digest] = imported
            native_rows = []
            for source_group in native_groups:
                entry = native_entries[source_group]
                domain_closure = NativeMessageClosure(entry["sequences"])
                for native_root in domain_native_roots:
                    if native_root.group_id == source_group:
                        domain_closure.mark(native_root.store_digest, native_root.slices)
                message_rows = []
                for row in entry["messages"]:
                    if (row["store_digest"], row["sha256"]) not in domain_closure.retained_messages:
                        continue
                    content = validate_native_message_content(row, group_id=source_group)
                    reference = content.payload.ref
                    if reference is not None and reference.key.startswith("v1/archive/"):
                        prefix = f"v1/archive/{source_group}/"
                        if not reference.key.startswith(prefix):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        key = f"v1/archive/{target_scope}/" + reference.key[len(prefix):]
                        content_ref = decoded_by_identity[(entry["domain"], reference.key, reference.digest, reference.size)][2]
                        destination = resources.archive_object_store(domain)
                        await destination.put(key, object_store.open(content_ref.key),
                                              expected_size=reference.size, expected_digest=reference.digest)
                        content = replace(content, payload=StoredPayload.object(
                            ObjectRef(reference.store_id, key, reference.digest, reference.size)))
                    message_rows.append({**row, "message": canonical_json_bytes(content.to_json()).decode("utf-8")})
                sequence_rows = [row for row in entry["sequences"]
                                 if (row["store_digest"], row["sha256"]) in domain_closure.retained_sequences]
                native_rows.append((sequence_rows, message_rows))
            def rekey(value: Mapping) -> dict:
                return rekey_native_step_sources(
                    rekey_native_message_roots(rekey_archive_roots(dict(value), scope_mapping),
                                              native_mapping if native_groups else {}),
                    native_mapping if native_groups else {})
            if domain is PersistenceDomain.TRACE:
                records = tuple(native_record_projections.get(value.key_digest, value) for value in records)
                facts = tuple(native_fact_projections.get((value.stream_digest, value.sequence), value) for value in facts)
            records = tuple(replace(value, data=rekey(value.data)) for value in records)
            facts = tuple(replace(value, data=rekey(value.data)) for value in facts)
            operations = tuple(replace(value, data=rekey(value.data)) for value in operations)

            native_restore = None
            if domain_checkpoints:
                resolver = resources.checkpoint_backup_source
                if resolver is None:
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                backup = next(iter(domain_checkpoints.values()))
                native_restore = await resolver(NativeStepSource(target_scope, backup.store_digest))

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
                await _insert_snapshot_values(transaction, records, aliases, facts, operations, sequences)

            try:
                await resources.stores[domain].mutate(publish)
            finally:
                if native_rows and isinstance(resources.stores[domain], SqlStateStore):
                    from ._step_store import SqlStepStore
                    SqlStepStore.invalidate_context(resources.stores[domain].context)


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
            parts = key.split("/", 3)
            if len(parts) != 4 or parts[2] not in groups:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            key = f"v1/archive/{neutral}/{parts[3]}"
        return dict(_object_ref_payload(replace(ref, key=key)))

    def content(value: object) -> dict[str, object]:
        parsed = ContentRecord.from_json(value)
        if parsed.payload.ref is not None:
            parsed = replace(parsed, payload=StoredPayload.object(_object_ref_from_payload(
                reference(_object_ref_payload(parsed.payload.ref)))))
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
        for _, ref in iter_runtime_object_refs(values, default_domain=domain):
            nonnative_objects.add((ref.key, ref.digest, ref.size))
    for entry in archives:
        rows = entry.get("records")
        if not isinstance(rows, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for row in rows:
            record = decode_record(row)
            if record.kind == "archive_content":
                parsed = ContentRecord.from_json(record.data)
                if parsed.payload.ref is not None:
                    ref = parsed.payload.ref
                    nonnative_objects.add((ref.key, ref.digest, ref.size))
                record = replace(record, data=content(record.data))
            archive_records.append(encode_record(record))
    native_sequences = []
    native_messages = {}
    native_objects = set()
    for entry in native:
        validate_snapshot_native_rows(entry.get("sequences"), entry.get("messages"))
        native_sequences.extend(entry["sequences"])
        for row in entry["messages"]:
            parsed = validate_native_message_content(row, group_id=entry["group_id"])
            # Raw bytes, not compression or transport location, define message
            # identity. Export/restore verify object bytes before this comparison;
            # the shared validator also verifies inline bytes here.
            body = {"codec": parsed.codec, "raw_digest": parsed.raw_digest, "raw_size": parsed.raw_size}
            key = row["store_digest"], row["sha256"]
            message = {"store_digest": key[0], "sha256": key[1], "message": body}
            if key in native_messages and native_messages[key] != message:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            native_messages[key] = message
            if parsed.payload.ref is not None:
                ref = parsed.payload.ref
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
    normalized_domains = rekey_native_step_sources(
        rekey_native_message_roots(rekey_archive_roots(dict(domains), groups), groups), groups)
    checkpoint_values = []
    for entry in manifest.get("native_checkpoints", []):
        if not isinstance(entry, Mapping) or entry.get("group_id") not in groups:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        backup = NativeStepBackup.from_json({key: value for key, value in entry.items() if key != "group_id"})
        checkpoint_values.append(backup.to_json())
    return canonical_sha256({
        "kind": manifest.get("kind"), "format_version": manifest.get("format_version"),
        "namespace": manifest.get("namespace"), "tenant_id": manifest.get("tenant_id"),
        "domains": normalized_domains,
        "archives": ordered_unique(archive_records),
        "native_checkpoints": ordered_unique(checkpoint_values),
        "native_projections": ordered_unique([
            {"records": entry["records"], "facts": entry["facts"]}
            for entry in manifest.get("native_projections", [])]),
        "native_sequences": ordered_unique(native_sequences),
        "native_messages": ordered_unique(list(native_messages.values())),
        "objects": ordered_unique(normalized_objects),
    })


def _validate_native_partition(store_digest: str, namespace: str, tenant_id: str) -> None:
    from ._step_store import SqlStepStore

    if store_digest != SqlStepStore.scope_digest(namespace=namespace, tenant_id=tenant_id):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


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
        columns = ("store_digest", "sha256", "message_sha256", "tree_count",
                   "left_sha256", "right_sha256", "next_sha256")
        sequences = []
        after = None
        partitions = sorted({root.store_digest for root in roots})
        while True:
            limit = budget.page_size(128)
            statement = select(*(nodes.c[name] for name in columns)).where(nodes.c.store_digest.in_(partitions))
            if after is not None:
                statement = statement.where(or_(nodes.c.store_digest > after[0],
                    and_(nodes.c.store_digest == after[0], nodes.c.sha256 > after[1])))
            rows = await transaction.read_sql_rows(store.context,
                statement.order_by(nodes.c.store_digest, nodes.c.sha256).limit(limit))
            for row in rows:
                budget.accept(row)
                sequences.append(dict(row))
            if len(rows) < limit:
                break
            after = rows[-1]["store_digest"], rows[-1]["sha256"]
        closure = NativeMessageClosure(sequences)
        for root in roots:
            closure.mark(root.store_digest, root.slices)
        sequences = [row for row in sequences if (row["store_digest"], row["sha256"]) in closure.retained_sequences]
        messages = []
        table = metadata.tables["ai_step_messages"]
        for partition in partitions:
            identities = sorted(identity for scope, identity in closure.retained_messages if scope == partition)
            offset = 0
            while offset < len(identities):
                batch = identities[offset:offset + budget.page_size(128)]
                rows = await transaction.read_sql_rows(store.context, select(
                    table.c.store_digest, table.c.sha256, table.c.message).where(
                    table.c.store_digest == partition, table.c.sha256.in_(batch)).order_by(table.c.sha256))
                if len(rows) != len(batch):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for row in rows:
                    budget.accept(row)
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
                for start in range(0, len(rows), 128):
                    await store.context.run_statement(store.context.dialect.insert_ignore_conflict_many(
                        session, table=table, rows=rows[start:start + 128], index_elements=("store_digest", "sha256")))
    await transaction.run_sql(store.context, insert)


def _capture_contains_archived_context(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("kind") != "agent":
        return False
    context = value.get("input_context")
    return isinstance(context, Mapping) and context.get("kind") == "execution-context"


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


def _validate_archive_content(content: ContentRecord, encoded: object) -> None:
    if not isinstance(encoded, bytes):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    decode_payload_bytes(encoded, codec=content.compression,
                         raw_digest=content.raw_digest, raw_size=content.raw_size)


def _decode_snapshot_digest(value: object) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(decoded) != 32:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return decoded


def _decode_snapshot_aliases(value: object) -> tuple[StoredAlias, ...]:
    if not isinstance(value, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    aliases: list[StoredAlias] = []
    seen: set[bytes] = set()
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {
            "alias_digest",
            "record_key_digest",
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        alias = StoredAlias(
            _decode_snapshot_digest(raw["alias_digest"]),
            _decode_snapshot_digest(raw["record_key_digest"]),
        )
        if alias.alias_digest in seen:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        seen.add(alias.alias_digest)
        aliases.append(alias)
    return tuple(sorted(aliases, key=lambda item: item.alias_digest))


def _decode_snapshot_sequences(value: object) -> Mapping[bytes, int]:
    if not isinstance(value, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    sequences: dict[bytes, int] = {}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {"key_digest", "value"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = _decode_snapshot_digest(raw["key_digest"])
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

__all__ = ['snapshot_manifest_identity', 'RuntimeSnapshotOperations', 'SnapshotExclusiveGuard', 'SnapshotLimits', 'SnapshotResources', 'ensure_capability_partition_marker', 'has_capability_partition_marker', 'restore_snapshot', 'snapshot_object_ref_from_payload', 'snapshot_object_ref_payload']
