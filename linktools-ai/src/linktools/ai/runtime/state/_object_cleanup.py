#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline deletion of unreferenced Runtime-owned object candidates."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from ...core import JsonValue, validate_page_limit
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, ObjectStat, ObjectStore, ObjectStoreInspection, ObjectStoreMaintenance, read_object
from ._archive_closure import ArchiveClosure, iter_archive_content_refs, iter_archive_roots
from ._codec import iter_runtime_object_dependencies, iter_runtime_object_refs
from ._input_capture import input_capture_expiry_key, iter_input_capture_dependencies, validate_input_capture_payload
from ._plan import RuntimeDomain
from ._snapshot import SnapshotExclusiveGuard
from ._store import FactScanCursor, OperationScanCursor, RecordScanCursor, StateStorageGroup, StateStore

ObjectCandidate = tuple[RuntimeDomain, ObjectRef]
_MANIFEST_PREFIXES = ("v1/input-capture/", "v1/asset-snapshot/", "v1/task-graph-binding-capture/")


@dataclass(frozen=True, slots=True)
class ObjectCleanupResult:
    deleted: tuple[ObjectCandidate, ...] = ()
    missing: tuple[ObjectCandidate, ...] = ()
    retained: tuple[ObjectCandidate, ...] = ()
    blocked: tuple[ObjectCandidate, ...] = ()


async def purge_unreferenced_objects(
    candidates: tuple[ObjectCandidate, ...], *, namespace: str, tenant_id: str,
    stores: Mapping[RuntimeDomain, StateStore],
    object_stores: Mapping[RuntimeDomain, ObjectStore],
    exclusive: SnapshotExclusiveGuard, limit: int = 100,
    archive_object_stores: Mapping[RuntimeDomain, ObjectStore] | None = None,
) -> ObjectCleanupResult:
    """Delete only after the caller quiesces all users of the routed stores.

    Object keys can be shared across domain routes. A reference from any state
    record or retained object manifest therefore protects the physical key.
    Cleanup receipts are deletion work, rather than content retention roots.
    The limit bounds deletions, not scans or references retained by other owners.
    """
    archive_objects = object_stores if archive_object_stores is None else archive_object_stores

    def source_for(domain: RuntimeDomain, reference: ObjectRef) -> ObjectStore:
        return (archive_objects if reference.key.startswith("v1/archive/") else object_stores)[domain]

    validate_page_limit(limit)
    selected = tuple(dict.fromkeys(candidates))
    if not selected:
        return ObjectCleanupResult()
    if any(not isinstance(store, ObjectStoreInspection) for store in (*object_stores.values(), *archive_objects.values())):
        return ObjectCleanupResult(blocked=selected)
    groups = {}
    for domain, store in stores.items():
        groups.setdefault(store.storage_group, domain)
    if any(not isinstance(group, StateStorageGroup) for group in groups):
        return ObjectCleanupResult(blocked=selected)
    async with exclusive.offline_exclusivity():
        archive_closures = {}
        archive_stores = {}
        partitions = {}
        for group, domain in groups.items():
            for partition in await group.scan_stores():
                partitions.setdefault(partition, domain)
            opened = await group.open_archive()
            if opened is not None:
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
                archive_stores[scope] = archive
        candidate_locations = {(id(source_for(domain, reference)), reference.key) for domain, reference in candidates}
        protected: set[str] = set()
        visited: set[tuple[int, str, str, int]] = set()
        metadata: dict[tuple[int, str], ObjectStat | None] = {}

        async def read_metadata(source: ObjectStore, key: str) -> ObjectStat | None:
            location = id(source), key
            if location not in metadata:
                metadata[location] = await source.stat(key)
            return metadata[location]

        async def visit(reference: ObjectRef, domain: RuntimeDomain, *, root: bool = True) -> None:
            source = source_for(domain, reference)
            if reference.key.startswith("v1/input-capture/") and await read_metadata(source, input_capture_expiry_key(reference.key)) is not None:
                return
            if root:
                protected.add(reference.key)
            identity = id(source), reference.key, reference.digest, reference.size
            if identity in visited or not reference.key.startswith(_MANIFEST_PREFIXES):
                return
            visited.add(identity)
            payload = await read_object(source, reference.key,
                                        expected_digest=reference.digest, expected_size=reference.size)
            if reference.key.startswith("v1/input-capture/"):
                try:
                    value = json.loads(payload)
                except (ValueError, UnicodeDecodeError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if (isinstance(value, Mapping) and value.get("kind") == "agent"
                        and isinstance(value.get("input_context"), Mapping)
                        and value["input_context"].get("kind") == "execution-context"):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                capture_namespace = value.get("namespace", namespace) if isinstance(value, Mapping) else namespace
                capture_tenant = value.get("tenant_id", tenant_id) if isinstance(value, Mapping) else tenant_id
                validate_input_capture_payload(reference.key, value,
                    namespace=capture_namespace, tenant_id=capture_tenant)
                if not reference.key.startswith(("v1/input-capture/result/", "v1/input-capture/expired/")):
                    await collect(value, RuntimeDomain.TASK,
                                  capture_namespace=capture_namespace, capture_tenant=capture_tenant, validate_scope=False, archive_owner=False)
            else:
                for nested_domain, nested in iter_runtime_object_dependencies(reference, payload, default_domain=domain):
                    await visit(nested, nested_domain)

        async def collect(value: JsonValue, domain: RuntimeDomain, *,
                          capture_namespace: str = namespace, capture_tenant: str = tenant_id,
                          validate_scope: bool = True, archive_owner: bool = True) -> None:
            for root in iter_archive_roots(value) if archive_owner else ():
                closure = archive_closures.get(root.scope_id)
                if closure is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                closure.mark(root)
            for scope, content_scope, identity in iter_archive_content_refs(value) if archive_owner else ():
                closure = archive_closures.get(scope)
                if closure is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                closure.mark_content(scope, content_scope, identity)
            for source_domain, reference in iter_runtime_object_refs(value, default_domain=domain):
                await visit(reference, source_domain)
            for key, digest in iter_input_capture_dependencies(value,
                    namespace=capture_namespace, tenant_id=capture_tenant, validate_scope=validate_scope):
                source = object_stores[RuntimeDomain.TASK]
                if await read_metadata(source, input_capture_expiry_key(key)) is not None:
                    continue
                stat = await read_metadata(source, key)
                if stat is None or stat.digest != digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await visit(ObjectRef(source.store_id, key, digest, stat.size), RuntimeDomain.TASK)

        for store, domain in partitions.items():
            record_cursor = None
            while True:
                records = await store.read(lambda tx: tx.scan_records_page(after=record_cursor, limit=1000))
                for record in records:
                    if record.kind not in {"evaluation_cleanup", "archive_content", "archive_sequence"}:
                        await collect(dict(record.data), domain)
                if len(records) < 1000:
                    break
                record_cursor = RecordScanCursor(records[-1].kind, records[-1].key_digest)
            fact_cursor = None
            while True:
                facts = await store.read(lambda tx: tx.scan_facts_page(after=fact_cursor, limit=1000))
                for fact in facts:
                    await collect(dict(fact.data), domain)
                if len(facts) < 1000:
                    break
                fact_cursor = FactScanCursor(facts[-1].stream_digest, facts[-1].sequence)
            operation_cursor = None
            while True:
                operations = await store.read(lambda tx: tx.scan_operations_page(after=operation_cursor, limit=1000))
                for operation in operations:
                    await collect(dict(operation.data), domain)
                if len(operations) < 1000:
                    break
                operation_cursor = OperationScanCursor(operations[-1].key_digest)

        # Captures are public durable handles and need not occur in a state row.
        inspected: set[int] = set()
        for domain, source in object_stores.items():
            if id(source) in inspected:
                continue
            inspected.add(id(source))
            async for stat in cast(ObjectStoreInspection, source).list_objects():
                if (id(source), stat.key) not in candidate_locations and stat.key.startswith(_MANIFEST_PREFIXES):
                    await visit(ObjectRef(source.store_id, stat.key, stat.digest, stat.size), domain, root=False)

        for scope, closure in archive_closures.items():
            protected.update(reference.key for reference in closure.object_refs)
            unreachable = closure.unreachable_records
            archive = archive_stores[scope]
            for offset in range(0, len(unreachable), 1000):
                keys = tuple(record.key_digest for record in unreachable[offset:offset + 1000])
                await archive.mutate(lambda tx, keys=keys: tx.delete_records(keys))

        deleted, missing, retained, blocked = [], [], [], []
        for candidate in selected:
            domain, reference = candidate
            source = source_for(domain, reference)
            if reference.key in protected:
                retained.append(candidate)
            elif not isinstance(source, ObjectStoreMaintenance) or len(deleted) + len(missing) >= limit:
                blocked.append(candidate)
            else:
                removed = await source.delete_object(reference.key, expected_digest=reference.digest)
                (deleted if removed else missing).append(candidate)
        return ObjectCleanupResult(tuple(deleted), tuple(missing), tuple(retained), tuple(blocked))


__all__ = ["ObjectCleanupResult", "purge_unreferenced_objects"]
