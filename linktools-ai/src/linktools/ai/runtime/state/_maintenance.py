#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Runtime storage validation and object mark-and-sweep."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Protocol

from linktools.core import environ

from ...core import ExecutionEventType
from ...errors import AIError, ErrorCode
from ...storage import ObjectStoreInspection, ObjectStoreMaintenance
from ._codec import (
    _VERSION_CODECS,
    _decode_domain,
    _iter_enveloped_runtime_object_refs,
    decode_envelope,
)
from ._plan import RuntimeDomain
from ._store import (
    FactScanCursor,
    OperationScanCursor,
    RecordScanCursor,
    StateStore,
    StoredFact,
    StoredOperation,
    StoredRecord,
)

_logger = environ.get_logger("ai.runtime.state.maintenance")
_OBJECT_DOMAINS = frozenset(
    {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.RECOVERY,
        RuntimeDomain.TASK,
    }
)
_MAINTENANCE_PAGE_SIZE = 128
_ENVELOPED_FACT_KINDS = frozenset(
    {
        "step_effect",
        "step_event",
        "step_snapshot",
        "transcript_chunk",
    }
)
_EXECUTION_EVENT_FACT_KINDS = frozenset(value.value for value in ExecutionEventType)
_READ_MODEL_RECORD_KIND = "execution_read_model"
_READ_MODEL_FACT_KINDS = frozenset(
    {
        "execution_read_history",
        "execution_read_trace",
        "execution_read_transcript",
    }
)
_READ_MODEL_FIELDS = frozenset(
    {
        "execution_id",
        "tenant_id",
        "source_digest",
        "model_version",
        "status",
        "trace_count",
        "history_count",
        "transcript_count",
        "revision",
    }
)
_LEASE_PROJECTED_WIRE_IDS = frozenset({"task_node_view", "tool_operation"})
_LEASE_FIELDS = frozenset({"owner", "fence", "lease_expires_at"})


class ObjectRouter(Protocol):
    def object_store(self, domain: RuntimeDomain) -> object: ...


class OfflineExclusiveStorage(Protocol):
    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]: ...


class RuntimeStorageInspection:
    """Inspect durable state and calculate object reachability."""

    def __init__(
        self,
        stores: Mapping[RuntimeDomain, StateStore],
        objects: ObjectRouter,
        *,
        durable_domains: frozenset[RuntimeDomain],
        state_validators: Sequence[Callable[[], Awaitable[None]]] = (),
    ) -> None:
        self._stores = dict(stores)
        self._objects = objects
        self._durable_domains = durable_domains
        self._state_validators = tuple(state_validators)

    async def inspect_objects(self) -> Mapping[int, frozenset[str]]:
        references: dict[int, set[str]] = {}
        await self.validate_state_stores()
        for domain in self._durable_domains:
            store = self._stores[domain]
            record_cursor: RecordScanCursor | None = None
            while True:
                records = await store.read(
                    lambda transaction, cursor=record_cursor: transaction.scan_records_page(
                        after=cursor,
                        limit=_MAINTENANCE_PAGE_SIZE,
                    )
                )
                if not records:
                    break
                self._collect_references(domain, records, (), (), references)
                last = records[-1]
                record_cursor = RecordScanCursor(last.kind, last.key_digest)
            fact_cursor: FactScanCursor | None = None
            while True:
                facts = await store.read(
                    lambda transaction, cursor=fact_cursor: transaction.scan_facts_page(
                        after=cursor,
                        limit=_MAINTENANCE_PAGE_SIZE,
                    )
                )
                if not facts:
                    break
                self._collect_references(domain, (), facts, (), references)
                last = facts[-1]
                fact_cursor = FactScanCursor(last.stream_digest, last.sequence)
            operation_cursor: OperationScanCursor | None = None
            while True:
                operations = await store.read(
                    lambda transaction, cursor=operation_cursor: transaction.scan_operations_page(
                        after=cursor,
                        limit=_MAINTENANCE_PAGE_SIZE,
                    )
                )
                if not operations:
                    break
                self._collect_references(domain, (), (), operations, references)
                operation_cursor = OperationScanCursor(operations[-1].key_digest)
        return {key: frozenset(value) for key, value in references.items()}

    async def validate_state_stores(self) -> None:
        for domain in self._durable_domains:
            await self._stores[domain].validate_integrity()
        for validator in self._state_validators:
            await validator()

    async def estimate_orphans(self) -> int:
        references = await self.inspect_objects()
        total = 0
        for object_store_id, object_store in (
            (id(value), value) for value in self.object_inspection_stores()
        ):
            total += sum(
                value.key not in references.get(object_store_id, frozenset())
                async for value in object_store.list_objects()
            )
        return total

    def object_inspection_stores(self) -> tuple[ObjectStoreInspection, ...]:
        stores: dict[int, ObjectStoreInspection] = {}
        for domain in self._durable_domains:
            if domain not in _OBJECT_DOMAINS:
                continue
            object_store = self._objects.object_store(domain)
            if not isinstance(object_store, ObjectStoreInspection):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stores[id(object_store)] = object_store
        return tuple(stores.values())

    def object_maintenance_stores(self) -> tuple[ObjectStoreMaintenance, ...]:
        stores: dict[int, ObjectStoreMaintenance] = {}
        for domain in self._durable_domains:
            if domain not in _OBJECT_DOMAINS:
                continue
            object_store = self._objects.object_store(domain)
            if not isinstance(object_store, ObjectStoreMaintenance):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stores[id(object_store)] = object_store
        return tuple(stores.values())

    async def _compact_objects(self) -> int:
        references = dict(await self.inspect_objects())
        object_stores = {
            id(value): value for value in self.object_maintenance_stores()
        }
        candidates: dict[int, tuple[object, ...]] = {}
        for object_store_id, object_store in object_stores.items():
            candidates[object_store_id] = tuple(
                [value async for value in object_store.list_objects()]
            )
        await self.validate_state_stores()
        for object_store in object_stores.values():
            await object_store.validate_integrity()
        deleted = 0
        for object_store_id, values in candidates.items():
            object_store = object_stores[object_store_id]
            for value in values:
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
        for record in records:
            if record.kind == _READ_MODEL_RECORD_KIND:
                _validate_read_model_record(record.data)
            else:
                self._collect_enveloped_references(domain, record.data, references)
        for fact in facts:
            if fact.kind in _ENVELOPED_FACT_KINDS:
                self._collect_enveloped_references(domain, fact.data, references)
            elif fact.kind in _EXECUTION_EVENT_FACT_KINDS:
                continue
            elif fact.kind in _READ_MODEL_FACT_KINDS:
                _validate_read_model_fact(fact.data)
            else:
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        for operation in operations:
            self._collect_enveloped_references(domain, operation.data, references)

    def _collect_enveloped_references(
        self,
        domain: RuntimeDomain,
        value: Mapping[str, object],
        references: dict[int, set[str]],
    ) -> None:
        _validate_enveloped_value(value)
        self._record_references(
            _iter_enveloped_runtime_object_refs(
                value,
                default_domain=domain,
            ),
            references,
        )

    def _record_references(
        self,
        values: object,
        references: dict[int, set[str]],
    ) -> None:
        for source_domain, reference in values:
            object_store = self._objects.object_store(source_domain)
            references.setdefault(id(object_store), set()).add(reference.key)


class OfflineRuntimeStorageMaintenance:
    """Run destructive object collection under an explicit exclusive guard."""

    def __init__(
        self,
        inspection: RuntimeStorageInspection,
        exclusive_guard: OfflineExclusiveStorage | None = None,
    ) -> None:
        self._inspection = inspection
        self._exclusive_guard = exclusive_guard

    async def compact_objects(self) -> int:
        if self._exclusive_guard is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async with self._exclusive_guard.offline_exclusivity():
            await self._inspection.validate_state_stores()
            async with AsyncExitStack() as stack:
                for object_store in self._inspection.object_maintenance_stores():
                    await stack.enter_async_context(
                        object_store.offline_exclusivity()
                    )
                return await self._inspection._compact_objects()


def _validate_enveloped_value(value: Mapping[str, object]) -> None:
    envelope = decode_envelope(value)
    if set(envelope.value) != {"type", "payload"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    codec = _VERSION_CODECS.get(envelope.version)
    if codec is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    wire_id = envelope.value.get("type")
    if not isinstance(wire_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    target = codec.domain_types.get(wire_id)
    if target is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    payload = envelope.value.get("payload")
    if payload is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    partial_projected_lease = False
    if wire_id in _LEASE_PROJECTED_WIRE_IDS:
        if not isinstance(payload, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        fields_value = payload.get("fields")
        if not isinstance(fields_value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        present = _LEASE_FIELDS.intersection(fields_value)
        partial_projected_lease = bool(present and present != _LEASE_FIELDS)
        payload = _restore_projected_lease_fields(payload)
    try:
        _decode_domain(payload, target, codec, persisted=True)
    except AIError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if partial_projected_lease:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _restore_projected_lease_fields(value: object) -> object:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    restored = dict(value)
    restored["fields"] = {
        "owner": None,
        "fence": 0,
        "lease_expires_at": None,
        **fields,
    }
    return restored


def _validate_read_model_record(value: Mapping[str, object]) -> None:
    if not _READ_MODEL_FIELDS.issubset(value):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = value.get("model_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _validate_read_model_fact(value: Mapping[str, object]) -> None:
    if "items" not in value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    items = value.get("items")
    if not isinstance(items, list) or any(
        not isinstance(item, Mapping) for item in items
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


RuntimeStorageMaintenance = RuntimeStorageInspection


__all__ = [
    "OfflineRuntimeStorageMaintenance",
    "RuntimeStorageInspection",
    "RuntimeStorageMaintenance",
]
