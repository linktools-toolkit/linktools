#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Runtime storage validation and object mark-and-sweep."""

from collections.abc import Mapping
from typing import Protocol

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import ObjectStoreMaintenance
from ._codec import iter_runtime_object_refs
from ._plan import RuntimeDomain
from ._store import StateStore, StoredFact, StoredOperation, StoredRecord

_logger = environ.get_logger("ai.runtime.state.maintenance")
_OBJECT_DOMAINS = frozenset(
    {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.RECOVERY,
    }
)


class ObjectRouter(Protocol):
    def object_store(self, domain: RuntimeDomain) -> object: ...


class RuntimeStorageMaintenance:
    """Run scans and object deletion only when the caller owns the workspace."""

    def __init__(
        self,
        stores: Mapping[RuntimeDomain, StateStore],
        objects: ObjectRouter,
        *,
        durable_domains: frozenset[RuntimeDomain],
    ) -> None:
        self._stores = dict(stores)
        self._objects = objects
        self._durable_domains = durable_domains

    async def compact_objects(self) -> int:
        references: dict[int, set[str]] = {}
        object_stores: dict[int, ObjectStoreMaintenance] = {}
        for domain in self._durable_domains:
            store = self._stores[domain]
            await store.validate_integrity()
            records, facts, operations = await store.read(_scan_state)
            self._collect_references(
                domain,
                records,
                facts,
                operations,
                references,
            )
            if domain not in _OBJECT_DOMAINS:
                continue
            object_store = self._objects.object_store(domain)
            if not isinstance(object_store, ObjectStoreMaintenance):
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            object_stores[id(object_store)] = object_store
        for object_store in object_stores.values():
            await object_store.validate_integrity()
        deleted = 0
        for object_store_id, object_store in object_stores.items():
            async for value in object_store.list_objects():
                if value.key in references.get(object_store_id, set()):
                    continue
                if await object_store.delete_object(
                    value.key,
                    expected_digest=value.digest,
                ):
                    deleted += 1
        for object_store in object_stores.values():
            await object_store.validate_integrity()
        _logger.info(
            "runtime object compaction completed: stores=%s deleted=%s",
            len(object_stores),
            deleted,
        )
        return deleted

    def _collect_references(
        self,
        domain: RuntimeDomain,
        records: tuple[StoredRecord, ...],
        facts: tuple[StoredFact, ...],
        operations: tuple[StoredOperation, ...],
        references: dict[int, set[str]],
    ) -> None:
        values = (
            *(record.data for record in records),
            *(fact.data for fact in facts),
            *(operation.data for operation in operations),
        )
        for value in values:
            for source_domain, reference in iter_runtime_object_refs(
                value,
                default_domain=domain,
            ):
                object_store = self._objects.object_store(source_domain)
                references.setdefault(id(object_store), set()).add(reference.key)


async def _scan_state(
    transaction,
) -> tuple[tuple[StoredRecord, ...], tuple[StoredFact, ...], tuple[StoredOperation, ...]]:
    return (
        await transaction.scan_records(),
        await transaction.scan_facts(),
        await transaction.scan_operations(),
    )


__all__ = ["RuntimeStorageMaintenance"]
