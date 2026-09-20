#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime storage validation and object reachability inspection."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol, cast

from linktools.core import environ

from ...core import ExecutionEventType, validate_persistence_namespace
from ...errors import AIError, ErrorCode
from ...storage import (
    ObjectRef,
    ObjectStore,
    ObjectStoreInspection,
    ObjectStoreMaintenance,
    read_object,
)
from ...task import TaskEventType, TaskGraphAdmission
from .._runtime_identity import task_capability_snapshot_key
from ._codec import (
    _VERSION_CODECS,
    _decode_domain,
    _decode_enveloped_domain,
    _iter_enveloped_runtime_object_refs,
    decode_envelope,
    iter_runtime_object_dependencies,
)
from ._offline_maintenance import OfflineRuntimeStorageMaintenance
from ._plan import RuntimeDomain, runtime_domain_uses_object_store
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
_MAINTENANCE_PAGE_SIZE = 128
_ENVELOPED_FACT_KINDS = frozenset(
    {
        "step_effect",
        "step_event",
        "step_snapshot",
        "transcript_chunk",
    }
)
_REFERENCE_FREE_RECORD_VERSIONS = {
    "agent_plan": 1,
    "session_turn_commit": 1,
}
_REFERENCE_FREE_FACT_VERSIONS = {
    "session_turn": 1,
    **{value.value: 1 for value in TaskEventType},
}
_REFERENCE_FREE_UNVERSIONED_FACT_KINDS = frozenset(
    value.value for value in ExecutionEventType
)
_LEASE_PROJECTED_WIRE_IDS = frozenset({"task_node_view", "tool_operation"})
_LEASE_FIELDS = frozenset({"owner", "fence", "lease_expires_at"})


class ObjectRouter(Protocol):
    def object_store(self, domain: RuntimeDomain) -> object: ...


class RuntimeStorageInspection:
    """Inspect durable state and calculate object reachability."""

    def __init__(
        self,
        stores: Mapping[RuntimeDomain, StateStore],
        objects: ObjectRouter,
        *,
        namespace: str,
        durable_domains: frozenset[RuntimeDomain],
        state_validators: Sequence[Callable[[], Awaitable[None]]] = (),
    ) -> None:
        self._stores = dict(stores)
        self._objects = objects
        self._namespace = validate_persistence_namespace(namespace)
        self._durable_domains = durable_domains
        self._state_validators = tuple(state_validators)

    async def inspect_objects(self) -> Mapping[int, frozenset[str]]:
        await self.validate_state_stores()
        return await self._scan_object_references()

    async def _scan_object_references(self) -> Mapping[int, frozenset[str]]:
        references: dict[int, set[str]] = {}
        pending: list[tuple[RuntimeDomain, ObjectRef]] = []
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
                self._collect_references(
                    domain,
                    records,
                    (),
                    (),
                    references,
                    pending,
                )
                if domain is RuntimeDomain.TASK:
                    for record in records:
                        if record.kind == "task_admission":
                            await self._collect_task_capability_reference(
                                record,
                                references,
                                pending,
                            )
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
                self._collect_references(
                    domain,
                    (),
                    facts,
                    (),
                    references,
                    pending,
                )
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
                self._collect_references(
                    domain,
                    (),
                    (),
                    operations,
                    references,
                    pending,
                )
                operation_cursor = OperationScanCursor(operations[-1].key_digest)
        await self._expand_object_dependencies(references, pending)
        return {key: frozenset(value) for key, value in references.items()}

    async def validate_state_stores(self) -> None:
        representatives: dict[int, StateStore] = {}
        for domain in self._durable_domains:
            store = self._stores[domain]
            representatives.setdefault(id(store.storage_group), store)
        _logger.debug(
            "runtime state physical validation: domains=%s groups=%s",
            len(self._durable_domains),
            len(representatives),
        )
        for store in representatives.values():
            await store.validate_integrity()
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
            if not runtime_domain_uses_object_store(domain):
                continue
            object_store = self._objects.object_store(domain)
            if not isinstance(object_store, ObjectStoreInspection):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stores[id(object_store)] = object_store
        return tuple(stores.values())

    def object_maintenance_stores(self) -> tuple[ObjectStoreMaintenance, ...]:
        stores: dict[int, ObjectStoreMaintenance] = {}
        for domain in self._durable_domains:
            if not runtime_domain_uses_object_store(domain):
                continue
            object_store = self._objects.object_store(domain)
            if not isinstance(object_store, ObjectStoreMaintenance):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stores[id(object_store)] = object_store
        return tuple(stores.values())

    async def _compact_objects(self) -> int:
        await self.validate_state_stores()
        object_stores = {
            id(value): value for value in self.object_maintenance_stores()
        }
        for object_store in object_stores.values():
            await object_store.validate_integrity()
        references = dict(await self._scan_object_references())
        candidates: dict[int, tuple[object, ...]] = {}
        for object_store_id, object_store in object_stores.items():
            candidates[object_store_id] = tuple(
                [value async for value in object_store.list_objects()]
            )
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
        pending: "list[tuple[RuntimeDomain, ObjectRef]] | None" = None,
    ) -> None:
        selected_pending = [] if pending is None else pending
        for record in records:
            expected_version = _REFERENCE_FREE_RECORD_VERSIONS.get(record.kind)
            if expected_version is not None:
                _validate_reference_free_version(
                    record.data,
                    expected_version=expected_version,
                )
                continue
            self._collect_enveloped_references(
                domain,
                record.data,
                references,
                selected_pending,
            )
        for fact in facts:
            if fact.kind in _ENVELOPED_FACT_KINDS:
                self._collect_enveloped_references(
                    domain,
                    fact.data,
                    references,
                    selected_pending,
                )
                continue
            expected_version = _REFERENCE_FREE_FACT_VERSIONS.get(fact.kind)
            if expected_version is not None:
                _validate_reference_free_version(
                    fact.data,
                    expected_version=expected_version,
                )
                continue
            if fact.kind in _REFERENCE_FREE_UNVERSIONED_FACT_KINDS:
                continue
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        for operation in operations:
            self._collect_enveloped_references(
                domain,
                operation.data,
                references,
                selected_pending,
            )

    def _collect_enveloped_references(
        self,
        domain: RuntimeDomain,
        value: Mapping[str, object],
        references: dict[int, set[str]],
        pending: list[tuple[RuntimeDomain, ObjectRef]],
    ) -> None:
        _validate_enveloped_value(value)
        self._record_references(
            _iter_enveloped_runtime_object_refs(
                value,
                default_domain=domain,
            ),
            references,
            pending,
        )

    def _record_references(
        self,
        values: object,
        references: dict[int, set[str]],
        pending: list[tuple[RuntimeDomain, ObjectRef]],
    ) -> None:
        for source_domain, reference in values:
            self._remember_reference(
                source_domain,
                reference,
                references,
                pending,
            )

    def _remember_reference(
        self,
        source_domain: RuntimeDomain,
        reference: ObjectRef,
        references: dict[int, set[str]],
        pending: list[tuple[RuntimeDomain, ObjectRef]],
    ) -> None:
        object_store = self._objects.object_store(source_domain)
        keys = references.setdefault(id(object_store), set())
        if reference.key in keys:
            return
        keys.add(reference.key)
        pending.append((source_domain, reference))

    async def _collect_task_capability_reference(
        self,
        record: StoredRecord,
        references: dict[int, set[str]],
        pending: list[tuple[RuntimeDomain, ObjectRef]],
    ) -> None:
        admission = _decode_enveloped_domain(record.data, TaskGraphAdmission)
        key = task_capability_snapshot_key(
            self._namespace,
            admission.principal.tenant_id,
            admission.graph_id,
            admission.initial_request_digest,
        )
        object_store = cast(
            ObjectStore,
            self._objects.object_store(RuntimeDomain.TASK),
        )
        stat = await object_store.stat(key)
        if stat is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._remember_reference(
            RuntimeDomain.TASK,
            ObjectRef(
                object_store.store_id,
                stat.key,
                stat.digest,
                stat.size,
            ),
            references,
            pending,
        )

    async def _expand_object_dependencies(
        self,
        references: dict[int, set[str]],
        pending: list[tuple[RuntimeDomain, ObjectRef]],
    ) -> None:
        while pending:
            source_domain, reference = pending.pop()
            if not reference.key.startswith(
                ("v1/skill-source-snapshot/", "v1/task-capability-snapshot/")
            ):
                continue
            object_store = cast(
                ObjectStore,
                self._objects.object_store(source_domain),
            )
            payload = await read_object(
                object_store,
                reference.key,
                expected_digest=reference.digest,
                expected_size=reference.size,
            )
            self._record_references(
                iter_runtime_object_dependencies(
                    reference,
                    payload,
                    default_domain=source_domain,
                ),
                references,
                pending,
            )


def _validate_reference_free_version(
    value: Mapping[str, object],
    *,
    expected_version: int,
) -> None:
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != expected_version:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


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


__all__ = ["OfflineRuntimeStorageMaintenance", "RuntimeStorageInspection"]
