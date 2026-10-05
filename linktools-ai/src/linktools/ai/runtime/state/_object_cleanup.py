#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline deletion of unreferenced Runtime-owned object candidates."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from ...core import JsonValue, validate_page_limit
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, ObjectStore, ObjectStoreInspection, ObjectStoreMaintenance, read_object
from ._codec import iter_runtime_object_dependencies, iter_runtime_object_refs
from ._input_capture import input_capture_expiry_key, iter_input_capture_dependencies, validate_input_capture_payload
from ._plan import RuntimeDomain
from ._snapshot import SnapshotExclusiveGuard
from ._store import FactScanCursor, OperationScanCursor, RecordScanCursor, StateStore

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
) -> ObjectCleanupResult:
    """Delete only after the caller quiesces all users of the routed stores.

    Object keys can be shared across domain routes. A reference from any state
    record or retained object manifest therefore protects the physical key.
    Cleanup receipts are deletion work, rather than content retention roots.
    The limit bounds deletions, not scans or references retained by other owners.
    """
    validate_page_limit(limit)
    selected = tuple(dict.fromkeys(candidates))
    if not selected:
        return ObjectCleanupResult()
    if any(not isinstance(store, ObjectStoreInspection) for store in object_stores.values()):
        return ObjectCleanupResult(blocked=selected)
    async with exclusive.offline_exclusivity():
        candidate_locations = {(id(object_stores[domain]), reference.key) for domain, reference in candidates}
        protected: set[str] = set()
        visited: set[tuple[RuntimeDomain, str, str]] = set()

        async def visit(reference: ObjectRef, domain: RuntimeDomain, *, root: bool = True) -> None:
            source = object_stores[domain]
            if reference.key.startswith("v1/input-capture/") and await source.stat(input_capture_expiry_key(reference.key)) is not None:
                return
            if root:
                protected.add(reference.key)
            identity = domain, reference.key, reference.digest
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
                capture_namespace = value.get("namespace", namespace) if isinstance(value, Mapping) else namespace
                capture_tenant = value.get("tenant_id", tenant_id) if isinstance(value, Mapping) else tenant_id
                validate_input_capture_payload(reference.key, value,
                    namespace=capture_namespace, tenant_id=capture_tenant)
                if not reference.key.startswith(("v1/input-capture/result/", "v1/input-capture/expired/")):
                    await collect(value, RuntimeDomain.TASK,
                                  capture_namespace=capture_namespace, capture_tenant=capture_tenant, validate_scope=False)
            else:
                for nested_domain, nested in iter_runtime_object_dependencies(reference, payload, default_domain=domain):
                    await visit(nested, nested_domain)

        async def collect(value: JsonValue, domain: RuntimeDomain, *,
                          capture_namespace: str = namespace, capture_tenant: str = tenant_id,
                          validate_scope: bool = True) -> None:
            for source_domain, reference in iter_runtime_object_refs(value, default_domain=domain):
                await visit(reference, source_domain)
            for key, digest in iter_input_capture_dependencies(value,
                    namespace=capture_namespace, tenant_id=capture_tenant, validate_scope=validate_scope):
                source = object_stores[RuntimeDomain.TASK]
                if await source.stat(input_capture_expiry_key(key)) is not None:
                    continue
                stat = await source.stat(key)
                if stat is None or stat.digest != digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await visit(ObjectRef(source.store_id, key, digest, stat.size), RuntimeDomain.TASK)

        for domain, store in stores.items():
            record_cursor = None
            while True:
                records = await store.read(lambda tx: tx.scan_records_page(after=record_cursor, limit=1000))
                for record in records:
                    if record.kind != "evaluation_cleanup":
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
        for domain, source in object_stores.items():
            async for stat in cast(ObjectStoreInspection, source).list_objects():
                if (id(source), stat.key) not in candidate_locations and stat.key.startswith(_MANIFEST_PREFIXES):
                    await visit(ObjectRef(source.store_id, stat.key, stat.digest, stat.size), domain, root=False)

        deleted, missing, retained, blocked = [], [], [], []
        for candidate in selected:
            domain, reference = candidate
            source = object_stores[domain]
            if reference.key in protected:
                retained.append(candidate)
            elif not isinstance(source, ObjectStoreMaintenance) or len(deleted) + len(missing) >= limit:
                blocked.append(candidate)
            else:
                removed = await source.delete_object(reference.key, expected_digest=reference.digest)
                (deleted if removed else missing).append(candidate)
        return ObjectCleanupResult(tuple(deleted), tuple(missing), tuple(retained), tuple(blocked))


__all__ = ["ObjectCleanupResult", "purge_unreferenced_objects"]
