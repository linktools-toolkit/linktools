#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded portable state snapshots and offline object maintenance."""

import hashlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path

from ...core import (
    canonical_json_bytes,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, ObjectStore, ObjectStoreInspection, read_object
from ...task import TaskGraphAdmission, TaskGraphSubmission
from .._runtime_identity import task_graph_binding_capture_key
from ._input_capture import input_capture_expiry_key, input_capture_key, iter_input_capture_dependencies, validate_input_capture_payload
from ._capability_marker import has_capability_partition_marker
from ._plan import RuntimeDomain
from ._store import (
    FactScanCursor,
    OperationScanCursor,
    RecordScanCursor,
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
from ._object_cleanup import ObjectCleanupResult, purge_unreferenced_objects as _purge_unreferenced_objects
from ._snapshot_validation import (
    canonical_snapshot_indexes,
    validate_snapshot_domain,
    validate_snapshot_references,
)
from ._codec import (
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
    durable_domains: frozenset[RuntimeDomain]
    stores: Mapping[RuntimeDomain, StateStore]
    object_stores: Mapping[RuntimeDomain, ObjectStore]
    read_only: bool
    portable_export: bool = True


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
        self, candidates: tuple[tuple[RuntimeDomain, ObjectRef], ...], *,
        exclusive: SnapshotExclusiveGuard, limit: int = 100,
    ) -> ObjectCleanupResult:
        self._require_ready()
        if self._resources.read_only:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        return await _purge_unreferenced_objects(
            candidates, namespace=self._resources.namespace, tenant_id=self._resources.tenant_id,
            stores=self._resources.stores,
            object_stores=self._resources.object_stores,
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

        domains: dict[str, dict[str, list[object]]] = {}
        snapshot_records: dict[RuntimeDomain, tuple[StoredRecord, ...]] = {}
        snapshot_facts: dict[RuntimeDomain, tuple[StoredFact, ...]] = {}
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
            source_domain: RuntimeDomain,
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
            source_store = self._resources.object_stores[source_domain]
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
                if not reference.key.startswith("v1/input-capture/result/"):
                    await copy_references(manifest, RuntimeDomain.TASK)
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

        async def copy_references(encoded: object, domain: RuntimeDomain) -> None:
            for key, digest in iter_input_capture_dependencies(encoded, namespace=self._resources.namespace, tenant_id=self._resources.tenant_id):
                source = self._resources.object_stores[RuntimeDomain.TASK]
                marker = await source.stat(input_capture_expiry_key(key))
                if marker is not None:
                    await copy_reference(RuntimeDomain.TASK, ObjectRef(source.store_id, marker.key, marker.digest, marker.size))
                    continue
                stat = await source.stat(key)
                if stat is None or stat.digest != digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await copy_reference(RuntimeDomain.TASK, ObjectRef(source.store_id, key, digest, stat.size))
            for source_domain, reference in iter_runtime_object_refs(
                encoded,
                default_domain=domain,
            ):
                await copy_reference(source_domain, reference)

        if RuntimeDomain.TASK in self._resources.durable_domains:
            capture_store = self._resources.object_stores[RuntimeDomain.TASK]
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
                await copy_reference(RuntimeDomain.TASK, ObjectRef(capture_store.store_id, stat.key, stat.digest, stat.size))

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
                            from ._contracts import ExecutionRecord
                            identity = _decode_enveloped_domain(value.data, ExecutionRecord).execution_id
                            capture_kind = "invocation"
                        else:
                            identity = _decode_enveloped_domain(value.data, TaskGraphAdmission).graph_id
                            capture_kind = "declaration"
                        key = input_capture_key(self._resources.namespace, self._resources.tenant_id, capture_kind, identity)
                        stat = await self._resources.object_stores[RuntimeDomain.TASK].stat(key)
                        if stat is not None:
                            await copy_reference(RuntimeDomain.TASK, ObjectRef(self._resources.object_stores[RuntimeDomain.TASK].store_id, key, stat.digest, stat.size))
                    if (
                        domain is RuntimeDomain.TASK
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
                        stat = await self._resources.object_stores[RuntimeDomain.TASK].stat(key)
                        if stat is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await copy_reference(
                            RuntimeDomain.TASK,
                            ObjectRef(
                                self._resources.object_stores[RuntimeDomain.TASK].store_id,
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
        manifest = {
            "kind": "runtime-storage-snapshot",
            "format_version": 1,
            "namespace": self._resources.namespace,
            "tenant_id": self._resources.tenant_id,
            "domains": domains,
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

    entry_count = len(raw_objects)
    object_bytes = 0
    expected_objects: set[tuple[str, str, str, int]] = set()
    expected_task_object_keys: set[str] = set()
    capture_dependencies: set[tuple[str, str]] = set()
    optional_capture_keys: set[str] = set()
    decoded_domains: dict[
        RuntimeDomain,
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
            domain = RuntimeDomain(domain_name)
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
        decoded_domains[domain] = (
            records,
            aliases,
            facts,
            operations,
            sequences,
        )
        if domain is RuntimeDomain.TASK:
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
                from ._contracts import ExecutionRecord
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
    decoded_objects: list[tuple[RuntimeDomain, ObjectRef, ObjectRef]] = []
    decoded_by_identity: dict[
        tuple[str, str, str, int],
        tuple[RuntimeDomain, ObjectRef, ObjectRef],
    ] = {}
    for raw_object in raw_objects:
        if not isinstance(raw_object, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            domain = RuntimeDomain(raw_object["domain"])
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
            if identity[0] == RuntimeDomain.TASK.value
            and identity[1] == key
        )
        if len(matches) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_objects.add(matches[0])

    def capture_identity(key: str, digest: str) -> tuple[str, str, str, int]:
        matches = [identity for identity in actual_objects if identity[0] == RuntimeDomain.TASK.value and identity[1] == key and identity[2] == digest]
        if not matches:
            marker_key = input_capture_expiry_key(key)
            matches = [identity for identity in actual_objects if identity[0] == RuntimeDomain.TASK.value and identity[1] == marker_key]
        if len(matches) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return matches[0]

    for key, digest in capture_dependencies:
        expected_objects.add(capture_identity(key, digest))
    expected_objects.update(identity for identity in actual_objects if identity[0] == RuntimeDomain.TASK.value and identity[1] in optional_capture_keys)
    for identity in actual_objects:
        if identity[0] != RuntimeDomain.TASK.value or not identity[1].startswith(tuple("v1/input-capture/" + kind + "/" for kind in ("agent", "task", "graph", "template", "expired"))):
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
            if not source.key.startswith("v1/input-capture/result/"):
                for key, digest in iter_input_capture_dependencies(manifest_capture, namespace=namespace, tenant_id=tenant_id):
                    nested_identity = capture_identity(key, digest)
                    if nested_identity not in expected_objects:
                        expected_objects.add(nested_identity)
                        pending_objects.append(nested_identity)
                for nested_domain, nested in iter_runtime_object_refs(manifest_capture, default_domain=RuntimeDomain.TASK):
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

    if actual_objects != expected_objects:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    target = Path(root).expanduser().resolve(strict=False)
    if target.exists() and any(target.iterdir()):
        raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
    async with open_target(target, namespace, tenant_id) as resources:
        for domain, source, content_ref in decoded_objects:
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
            await resources.stores[domain].mutate(
                lambda transaction, records=records, aliases=aliases, facts=facts, operations=operations, sequences=sequences: _insert_snapshot_values(
                    transaction,
                    records,
                    aliases,
                    facts,
                    operations,
                    sequences,
                )
            )


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


__all__ = ["SnapshotResources", "RuntimeSnapshotOperations", "restore_snapshot"]
