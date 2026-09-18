#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeState lifecycle owner."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from ...core import (
    canonical_json_bytes,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ...errors import AIError, ErrorCode
from ...storage import FilesystemObjectStore, ObjectRef, ObjectStore, read_object
from ._contracts import (
    ArtifactState,
    ConversationState,
    EvaluationState,
    ExecutionState,
    MemoryState,
    RecoveryState,
    TaskState,
)
from ._plan import (
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeStatePlan,
    RuntimeStateRoute,
    runtime_domain_uses_object_store,
)
from ._store import (
    FactScanCursor,
    OperationScanCursor,
    RecordScanCursor,
    StateStore,
    StateTransaction,
    StoredAlias,
)
from ._snapshot import SnapshotLimits
from ._codec import (
    decode_fact,
    decode_operation,
    decode_record,
    encode_fact,
    encode_operation,
    encode_record,
    iter_runtime_object_refs,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ._maintenance import RuntimeStorageInspection
    from ._materializer import _MaterializedRuntimeState
    from ._object_router import _RuntimeObjectRouter
    from ._retention import RuntimeRetentionController
    from ._steps import RuntimeStepStore


class _RuntimeStateLifecycle(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"


class RuntimeState:
    """Own materialized domain states and every resource acquired for them."""

    def __init__(
        self,
        plan: RuntimeStatePlan,
        *,
        object_store: "ObjectStore | None" = None,
    ) -> None:
        _validate_state_configuration(plan, object_store)
        self._plan = plan
        self._external_object_store = object_store
        self._lifecycle = _RuntimeStateLifecycle.NEW
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._close_cursor = 0
        self._close_actions: tuple[Callable[[], Awaitable[None]], ...] = ()
        self._namespace: str | None = None
        self._tenant_id: str | None = None
        self._conversation: ConversationState | None = None
        self._execution: ExecutionState | None = None
        self._memory: MemoryState | None = None
        self._artifact: ArtifactState | None = None
        self._task: TaskState | None = None
        self._evaluation: EvaluationState | None = None
        self._recovery: RecoveryState | None = None
        self._objects: _RuntimeObjectRouter | None = None
        self._steps: RuntimeStepStore | None = None
        self._retention: RuntimeRetentionController | None = None
        self._maintenance: RuntimeStorageInspection | None = None
        self._stores: dict[RuntimeDomain, StateStore] = {}
        self._read_only = False

    @classmethod
    def in_memory(cls) -> "RuntimeState":
        return cls(RuntimeStatePlan())

    @classmethod
    def filesystem(
        cls,
        path: "str | Path",
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeState":
        base = _normalize_path(path)
        return cls(
            RuntimeStatePlan(
                **{
                    domain.value: RuntimeStateRoute.filesystem(
                        base / domain.value,
                        transaction_root=base,
                    )
                    for domain in RuntimeDomain
                }
            ),
            object_store=object_store,
        )

    @classmethod
    def sqlite(
        cls,
        path: "str | Path",
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeState":
        route = RuntimeStateRoute.sqlite(path)
        return cls(
            RuntimeStatePlan(
                **{domain.value: route for domain in RuntimeDomain}
            ),
            object_store=object_store,
        )

    @classmethod
    def from_root(cls, root: "str | Path") -> "RuntimeState":
        base = _normalize_path(root)
        return cls.sqlite(
            base / "runtime.db",
            object_store=FilesystemObjectStore(base / "objects"),
        )

    @classmethod
    def sql(
        cls,
        engine: "AsyncEngine",
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeState":
        route = RuntimeStateRoute.sql(engine)
        return cls(
            RuntimeStatePlan(
                **{domain.value: route for domain in RuntimeDomain}
            ),
            object_store=object_store,
        )

    @classmethod
    def from_plan(
        cls,
        plan: RuntimeStatePlan,
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeState":
        return cls(plan, object_store=object_store)

    @property
    def plan(self) -> RuntimeStatePlan:
        return self._plan

    @property
    def ready(self) -> bool:
        return self._lifecycle is _RuntimeStateLifecycle.READY

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def namespace(self) -> str:
        self._require_ready()
        if self._namespace is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._namespace

    @property
    def tenant_id(self) -> str:
        self._require_ready()
        if self._tenant_id is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._tenant_id

    @property
    def conversation(self) -> ConversationState:
        return self._require_state(self._conversation)

    @property
    def execution(self) -> ExecutionState:
        return self._require_state(self._execution)

    @property
    def memory(self) -> MemoryState:
        return self._require_state(self._memory)

    @property
    def artifact(self) -> ArtifactState:
        return self._require_state(self._artifact)

    @property
    def task(self) -> TaskState:
        return self._require_state(self._task)

    @property
    def evaluation(self) -> EvaluationState:
        return self._require_state(self._evaluation)

    @property
    def recovery(self) -> RecoveryState:
        return self._require_state(self._recovery)

    @property
    def steps(self) -> "RuntimeStepStore":
        return self._require_state(self._steps)

    @property
    def retention(self) -> "RuntimeRetentionController":
        return self._require_state(self._retention)

    async def initialize(
        self,
        *,
        namespace: str,
        tenant_id: str,
        read_only: bool = False,
    ) -> None:
        async with self._lock:
            if self._lifecycle is not _RuntimeStateLifecycle.NEW:
                raise AIError(
                    ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                    "RuntimeState must be NEW",
                )
            validate_persistence_namespace(namespace)
            if not tenant_id.strip():
                raise ValueError("tenant_id is required")
            if not isinstance(read_only, bool):
                raise TypeError("read_only must be bool")
            self._lifecycle = _RuntimeStateLifecycle.INITIALIZING
            try:
                from ._materializer import materialize_runtime_state

                materialized = await materialize_runtime_state(
                    self._plan,
                    namespace=namespace,
                    tenant_id=tenant_id,
                    object_store=self._external_object_store,
                    read_only=read_only,
                )
                self._assign_materialized(
                    materialized,
                    namespace,
                    tenant_id,
                )
                self._read_only = read_only
                self._lifecycle = _RuntimeStateLifecycle.READY
            except BaseException:
                self._lifecycle = _RuntimeStateLifecycle.CLOSED
                raise

    async def close(self) -> None:
        async with self._lock:
            if self._lifecycle is _RuntimeStateLifecycle.NEW:
                self._lifecycle = _RuntimeStateLifecycle.CLOSED
                return
            if self._lifecycle is _RuntimeStateLifecycle.CLOSED:
                return
            self._lifecycle = _RuntimeStateLifecycle.CLOSING
            task = self._close_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._run_close_actions(),
                    name="linktools-runtime-state-close",
                )
                task.add_done_callback(self._consume_close_result)
                self._close_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                self._consume_close_result(task)
            raise

    def _consume_close_result(self, task: "asyncio.Task[None]") -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            pass

    async def _run_close_actions(self) -> None:
        while self._close_cursor < len(self._close_actions):
            action = self._close_actions[self._close_cursor]
            await action()
            self._close_cursor += 1
        self._lifecycle = _RuntimeStateLifecycle.CLOSED

    def _assign_materialized(
        self,
        value: "_MaterializedRuntimeState",
        namespace: str,
        tenant_id: str,
    ) -> None:
        self._conversation = value.conversation
        self._execution = value.execution
        self._memory = value.memory
        self._artifact = value.artifact
        self._task = value.task
        self._evaluation = value.evaluation
        self._recovery = value.recovery
        self._objects = value.objects
        self._steps = value.steps
        self._retention = value.retention
        self._maintenance = value.maintenance
        self._stores = dict(value.stores)
        self._close_actions = value.close_actions
        self._namespace = namespace
        self._tenant_id = tenant_id

    def _require_ready(self) -> None:
        if self._lifecycle is not _RuntimeStateLifecycle.READY:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "RuntimeState is not ready",
            )

    def _require_state(self, value: object) -> object:
        self._require_ready()
        if value is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "RuntimeState is not ready",
            )
        return value

    def object_store(self, domain: RuntimeDomain) -> ObjectStore:
        self._require_ready()
        if self._objects is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            )
        return self._objects.object_store(domain)

    def working_object_store(
        self,
        domain: RuntimeDomain,
        *,
        owner_scope: str,
    ) -> ObjectStore:
        self._require_ready()
        if self._objects is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            )
        return self._objects.working_object_store(
            domain,
            owner_scope=owner_scope,
        )

    def local_paths(self) -> tuple[Path, ...]:
        """Return local physical paths owned by this state instance."""
        self._require_ready()
        paths: list[Path] = []
        for domain in RuntimeDomain:
            route = self._plan.route(domain)
            if route.path is not None:
                paths.append(route.path.resolve())
            if route.transaction_root is not None:
                paths.append(route.transaction_root.resolve())
        if self._objects is not None:
            paths.extend(self._objects.local_paths())
        return tuple(dict.fromkeys(paths))

    async def export_snapshot(
        self,
        *,
        object_store: ObjectStore,
        limits: SnapshotLimits,
    ) -> ObjectRef:
        """Export this initialized read-only state as one bounded logical snapshot."""
        self._require_ready()
        if not self._read_only:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        if not isinstance(limits, SnapshotLimits):
            raise TypeError("limits must be SnapshotLimits")
        if not self._plan.durable_domains:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

        domains: dict[str, dict[str, list[object]]] = {}
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

        async def copy_references(encoded: object, domain: RuntimeDomain) -> None:
            nonlocal entry_count, object_bytes
            for source_domain, reference in iter_runtime_object_refs(
                encoded,
                default_domain=domain,
            ):
                identity = (
                    source_domain.value,
                    reference.key,
                    reference.digest,
                    reference.size,
                )
                if identity in copied_objects:
                    continue
                entry_count += 1
                object_bytes += reference.size
                if (
                    entry_count > limits.max_entries
                    or object_bytes > limits.max_bytes
                ):
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                key = (
                    "v1/runtime-state-object/"
                    f"{source_domain.value}/{reference.digest}"
                )
                await object_store.put(
                    key,
                    self.object_store(source_domain).open(reference.key),
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

        for domain in sorted(self._plan.durable_domains, key=lambda item: item.value):
            store = self._stores[domain]
            raw_domain: dict[str, list[object]] = {
                "records": [],
                "aliases": [],
                "facts": [],
                "operations": [],
                "sequences": [],
            }

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
                    encoded = encode_record(value)
                    raw_domain["records"].append(accept(encoded))
                    await copy_references(encoded, domain)
                last = page[-1]
                record_cursor = RecordScanCursor(last.kind, last.key_digest)
                if len(page) < 128:
                    break

            alias_cursor: bytes | None = None
            while True:
                page = await store.read(
                    lambda transaction, cursor=alias_cursor: transaction.scan_aliases_page(
                        after=cursor,
                        limit=128,
                    )
                )
                if not page:
                    break
                for value in page:
                    raw_domain["aliases"].append(
                        accept(
                            {
                                "alias_digest": value.alias_digest.hex(),
                                "record_key_digest": value.record_key_digest.hex(),
                            }
                        )
                    )
                alias_cursor = page[-1].alias_digest
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
                    encoded = encode_operation(value)
                    raw_domain["operations"].append(accept(encoded))
                    await copy_references(encoded, domain)
                operation_cursor = OperationScanCursor(page[-1].key_digest)
                if len(page) < 128:
                    break

            sequence_cursor: bytes | None = None
            while True:
                page = await store.read(
                    lambda transaction, cursor=sequence_cursor: transaction.scan_sequences_page(
                        after=cursor,
                        limit=128,
                    )
                )
                if not page:
                    break
                for key in sorted(page):
                    raw_domain["sequences"].append(
                        accept({"key_digest": key.hex(), "value": page[key]})
                    )
                sequence_cursor = max(page)
                if len(page) < 128:
                    break

            domains[domain.value] = raw_domain

        manifest = {
            "kind": "runtime-state-snapshot",
            "format_version": 1,
            "namespace": self.namespace,
            "tenant_id": self.tenant_id,
            "domains": domains,
            "objects": objects,
        }
        payload = canonical_json_bytes(manifest)
        if len(payload) + object_bytes > limits.max_bytes:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        digest = hashlib.sha256(payload).hexdigest()
        key = f"v1/runtime-state-snapshot/{digest}"
        await _put_snapshot_object(object_store, key, payload)
        return ObjectRef(object_store.store_id, key, digest, len(payload))

    @classmethod
    async def restore_snapshot(
        cls,
        ref: ObjectRef,
        *,
        object_store: ObjectStore,
        root: str | Path,
        limits: SnapshotLimits,
    ) -> None:
        """Restore a bounded logical state snapshot into a new local RuntimeState."""
        if not isinstance(limits, SnapshotLimits):
            raise TypeError("limits must be SnapshotLimits")
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
        if (
            not isinstance(manifest, dict)
            or manifest.get("kind") != "runtime-state-snapshot"
            or manifest.get("format_version") != 1
            or not isinstance(manifest.get("domains"), dict)
        ):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
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
        decoded_domains: dict[
            RuntimeDomain,
            tuple[
                tuple[object, ...],
                tuple[StoredAlias, ...],
                tuple[object, ...],
                tuple[object, ...],
                Mapping[bytes, int],
            ],
        ] = {}

        for domain_name, raw_domain in raw_domains.items():
            try:
                domain = RuntimeDomain(domain_name)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if not isinstance(raw_domain, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raw_records = raw_domain.get("records", [])
            raw_aliases = raw_domain.get("aliases", [])
            raw_facts = raw_domain.get("facts", [])
            raw_operations = raw_domain.get("operations", [])
            raw_sequences = raw_domain.get("sequences", [])
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
            decoded_domains[domain] = (
                records,
                aliases,
                facts,
                operations,
                sequences,
            )
            for encoded in (*raw_records, *raw_facts, *raw_operations):
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

        actual_objects: set[tuple[str, str, str, int]] = set()
        decoded_objects: list[tuple[RuntimeDomain, ObjectRef, ObjectRef]] = []
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
            decoded_objects.append((domain, source, content_ref))
            object_bytes += content_ref.size
            if len(payload) + object_bytes > limits.max_bytes:
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        if actual_objects != expected_objects:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        target = Path(root).expanduser().resolve(strict=False)
        if target.exists() and any(target.iterdir()):
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        state = cls.from_root(root)
        await state.initialize(namespace=namespace, tenant_id=tenant_id)
        try:
            for domain, source, content_ref in decoded_objects:
                destination = state.object_store(domain)
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
                await state._stores[domain].mutate(
                    lambda transaction, records=records, aliases=aliases, facts=facts, operations=operations, sequences=sequences: _insert_snapshot_values(
                        transaction,
                        records,
                        aliases,
                        facts,
                        operations,
                        sequences,
                    )
                )
        finally:
            await state.close()


def _validate_state_configuration(
    plan: RuntimeStatePlan,
    object_store: "ObjectStore | None",
) -> None:
    if not isinstance(plan, RuntimeStatePlan):
        raise TypeError("plan must be a RuntimeStatePlan")
    if object_store is not None and not any(
        runtime_domain_uses_object_store(domain)
        and plan.route(domain).retention is RuntimeRetentionMode.DURABLE
        for domain in RuntimeDomain
    ):
        raise ValueError(
            "object_store requires at least one durable object-capable RuntimeDomain"
        )
    if (
        plan.route(RuntimeDomain.CONVERSATION).retention
        is RuntimeRetentionMode.DURABLE
    ):
        if (
            plan.route(RuntimeDomain.EXECUTION).retention
            is not RuntimeRetentionMode.DURABLE
        ):
            raise ValueError("durable conversation requires durable execution")
        if (
            plan.route(RuntimeDomain.RECOVERY).retention
            is not RuntimeRetentionMode.DURABLE
        ):
            raise ValueError("durable conversation requires durable recovery")


def _normalize_path(value: "str | Path") -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("RuntimeState path is required")
    return Path(value).expanduser().resolve(strict=False)


def _object_ref_payload(ref: ObjectRef) -> dict[str, object]:
    return {
        "store_id": ref.store_id,
        "key": ref.key,
        "digest": ref.digest,
        "size": ref.size,
    }


def _object_ref_from_payload(value: object) -> ObjectRef:
    if not isinstance(value, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return ObjectRef(
            str(value["store_id"]),
            str(value["key"]),
            str(value["digest"]),
            int(value["size"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


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
    records: tuple[object, ...],
    aliases: tuple[StoredAlias, ...],
    facts: tuple[object, ...],
    operations: tuple[object, ...],
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


async def _snapshot_chunk(value: bytes):
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


__all__ = ["RuntimeState"]
