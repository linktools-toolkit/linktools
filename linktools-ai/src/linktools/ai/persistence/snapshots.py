#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded portable state snapshots and offline object maintenance."""

import hashlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from pathlib import Path

from ..core import (
    canonical_json_bytes,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, ObjectStoreInspection, StoredPayload, read_object
from ..task import TaskGraphAdmission, TaskGraphSubmission
from ._runtime_identity import task_graph_binding_capture_key
from ._archive import ContentRecord
from ._archive_closure import (
    ArchiveClosure,
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
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
)
from ._snapshot import (
    SnapshotLimits,
    SnapshotExclusiveGuard,
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
        self._require_ready()
        if self._resources.read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        return await _purge_unreferenced_objects(
            candidates, namespace=self._resources.namespace, tenant_id=self._resources.tenant_id,
            stores=self._resources.stores,
            object_stores=self._resources.object_stores,
            archive_object_stores=self._resources.archive_object_stores,
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

        for domain in self._resources.durable_domains:
            if await self._resources.stores[domain].read(has_capability_partition_marker):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Capability persistence partitions are not supported by portable snapshots")

        archive_closures: dict[str, ArchiveClosure] = {}
        archive_domains: dict[str, PersistenceDomain] = {}
        groups = {}
        for domain in sorted(self._resources.durable_domains, key=lambda value: value.value):
            groups.setdefault(self._resources.stores[domain].storage_group, domain)
        for group, domain in groups.items():
            if not isinstance(group, StateStorageGroup):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            opened = await group.open_archive()
            if opened is None:
                continue
            scope, archive = opened
            records = []
            cursor = None
            while True:
                page = await archive.read(lambda tx: tx.scan_records_page(after=cursor, limit=1000))
                records.extend(page)
                if len(page) < 1000:
                    break
                cursor = RecordScanCursor(page[-1].kind, page[-1].key_digest)
            if scope in archive_closures:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            archive_closures[scope] = ArchiveClosure(scope, records)
            archive_domains[scope] = domain

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
            source_store = (self._resources.archive_object_store(source_domain) if reference.key.startswith("v1/archive/")
                            else self._resources.object_stores[source_domain])
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
                validate_input_capture_payload(reference.key, manifest, namespace=self._resources.namespace, tenant_id=self._resources.tenant_id)
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
            for key, digest in iter_input_capture_dependencies(encoded, namespace=self._resources.namespace, tenant_id=self._resources.tenant_id):
                source = self._resources.object_stores[PersistenceDomain.TASK]
                marker = await source.stat(input_capture_expiry_key(key))
                if marker is not None:
                    await copy_reference(PersistenceDomain.TASK, ObjectRef(source.store_id, marker.key, marker.digest, marker.size))
                    continue
                stat = await source.stat(key)
                if stat is None or stat.digest != digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await copy_reference(PersistenceDomain.TASK, ObjectRef(source.store_id, key, digest, stat.size))
            for source_domain, reference in iter_runtime_object_refs(
                encoded,
                default_domain=domain,
            ):
                await copy_reference(source_domain, reference)

        if PersistenceDomain.TASK in self._resources.durable_domains:
            capture_store = self._resources.object_stores[PersistenceDomain.TASK]
            if not isinstance(capture_store, ObjectStoreInspection):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            async for stat in capture_store.list_objects():
                if not stat.key.startswith(tuple("v1/input-capture/" + kind + "/" for kind in ("agent", "task", "graph", "template", "expired"))):
                    continue
                payload = await read_object(capture_store, stat.key, expected_digest=stat.digest, expected_size=stat.size)
                value = json.loads(payload)
                if not isinstance(value, Mapping) or value.get("namespace") != self._resources.namespace or value.get("tenant_id") != self._resources.tenant_id:
                    continue
                if not stat.key.startswith("v1/input-capture/expired/") and await capture_store.stat(input_capture_expiry_key(stat.key)) is not None:
                    continue
                await copy_reference(PersistenceDomain.TASK, ObjectRef(capture_store.store_id, stat.key, stat.digest, stat.size))

        for domain in sorted(self._resources.durable_domains, key=lambda item: item.value):
            store = self._resources.stores[domain]
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
                page = await store.read(
                    lambda transaction, cursor=record_cursor: transaction.scan_records_page(
                        after=cursor,
                        limit=128,
                    )
                )
                if not page:
                    break
                for value in page:
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
                        key = input_capture_key(self._resources.namespace, self._resources.tenant_id, capture_kind, identity)
                        stat = await self._resources.object_stores[PersistenceDomain.TASK].stat(key)
                        if stat is not None:
                            await copy_reference(PersistenceDomain.TASK, ObjectRef(self._resources.object_stores[PersistenceDomain.TASK].store_id, key, stat.digest, stat.size))
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
                            self._resources.namespace,
                            admission.principal.tenant_id,
                            admission.graph_id,
                            admission.initial_request_digest,
                        )
                        stat = await self._resources.object_stores[PersistenceDomain.TASK].stat(key)
                        if stat is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await copy_reference(
                            PersistenceDomain.TASK,
                            ObjectRef(
                                self._resources.object_stores[PersistenceDomain.TASK].store_id,
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
                page = await store.read(
                    lambda transaction, cursor=fact_cursor: transaction.scan_facts_page(
                        after=cursor,
                        limit=128,
                    )
                )
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
                page = await store.read(
                    lambda transaction, cursor=operation_cursor: transaction.scan_operations_page(
                        after=cursor,
                        limit=128,
                    )
                )
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
                namespace=self._resources.namespace,
                tenant_id=self._resources.tenant_id,
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
                               await read_object(self._resources.archive_object_store(domain), content.payload.ref.key,
                                                 expected_digest=content.payload.digest, expected_size=content.payload.size))
                    _validate_archive_content(content, encoded)
            archives.append({"scope_id": scope, "domain": domain.value,
                             "records": [accept(encode_record(record)) for record in retained]})
        manifest = {
            "kind": "runtime-storage-snapshot",
            "format_version": 1,
            "namespace": self._resources.namespace,
            "tenant_id": self._resources.tenant_id,
            "domains": domains,
            "archives": archives,
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
    object_bytes = 0
    expected_objects: set[tuple[str, str, str, int]] = set()
    expected_task_object_keys: set[str] = set()
    capture_dependencies: set[tuple[str, str]] = set()
    optional_capture_keys: set[str] = set()
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
                identity = _decode_enveloped_domain(record.data, ExecutionRecord).execution_id
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
            scope_mapping = {}
            archive_records: dict[bytes, StoredRecord] = {}
            archive_store = None
            if referenced_scopes:
                group = resources.stores[domain].storage_group
                if not isinstance(group, StateStorageGroup):
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                target_scope, archive_store = await group.open_archive(create=True)
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
            records = tuple(replace(value, data=rekey_archive_roots(dict(value.data), scope_mapping)) for value in records)
            facts = tuple(replace(value, data=rekey_archive_roots(dict(value.data), scope_mapping)) for value in facts)
            operations = tuple(replace(value, data=rekey_archive_roots(dict(value.data), scope_mapping)) for value in operations)

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
                await _insert_snapshot_values(transaction, records, aliases, facts, operations, sequences)

            await resources.stores[domain].mutate(publish)


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
from ._resource_scope import reject_filesystem_capability_partitions
from ._snapshot import snapshot_object_ref_from_payload, snapshot_object_ref_payload

__all__ = ['RuntimeSnapshotOperations', 'SnapshotExclusiveGuard', 'SnapshotLimits', 'SnapshotResources', 'ensure_capability_partition_marker', 'has_capability_partition_marker', 'reject_filesystem_capability_partitions', 'restore_snapshot', 'snapshot_object_ref_from_payload', 'snapshot_object_ref_payload']
