#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed Runtime composition state and owned resource lifecycle."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from ...core import validate_persistence_namespace
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, ObjectStore, StorageRoute
from ._root import RuntimeStorage, _validate_storage_configuration
from ._plan import RuntimeDomain, RuntimeStoragePlan, runtime_domain_uses_object_store
from ._contracts import (ArtifactRepositories, ConversationRepositories, EvaluationRepositories,
                         ExecutionRepositories, MemoryRepositories, RecoveryRepositories, TaskRepositories)
from ._store import StateStore
from ._snapshot import SnapshotLimits, SnapshotExclusiveGuard
from ._snapshot_operations import SnapshotResources, RuntimeSnapshotOperations, restore_snapshot
from ._object_cleanup import ObjectCleanupResult

if TYPE_CHECKING:
    from ._materializer import _MaterializedRuntimeStorage
    from ._object_router import _RuntimeObjectRouter
    from ._retention import RuntimeRetentionController
    from ._steps import RuntimeAgentRunStore


class _RuntimeStateLifecycle(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"


class RuntimeState:
    """Runtime composition state and lifecycle for typed module resources."""

    def __init__(
        self,
        storage: RuntimeStorage,
    ) -> None:
        if not isinstance(storage, RuntimeStorage):
            raise TypeError("storage must be RuntimeStorage")
        self._plan = storage.plan
        self._external_object_store = storage.object_store
        self._lifecycle = _RuntimeStateLifecycle.NEW
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._close_cursor = 0
        self._close_actions: tuple[Callable[[], Awaitable[None]], ...] = ()
        self._namespace: str | None = None
        self._tenant_id: str | None = None
        self._conversation: ConversationRepositories | None = None
        self._execution: ExecutionRepositories | None = None
        self._memory: MemoryRepositories | None = None
        self._artifact: ArtifactRepositories | None = None
        self._task: TaskRepositories | None = None
        self._evaluation: EvaluationRepositories | None = None
        self._recovery: RecoveryRepositories | None = None
        self._objects: _RuntimeObjectRouter | None = None
        self._run_store: RuntimeAgentRunStore | None = None
        self._retention: RuntimeRetentionController | None = None
        self._stores: dict[RuntimeDomain, StateStore] = {}
        self.capability_stores: dict[str, StateStore] = {}
        self.durable_capability_ids: frozenset[str] = frozenset()
        self._read_only = False

    @property
    def plan(self) -> RuntimeStoragePlan:
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
    def conversation(self) -> ConversationRepositories:
        return self._require_materialized(self._conversation)

    @property
    def execution(self) -> ExecutionRepositories:
        return self._require_materialized(self._execution)

    @property
    def memory(self) -> MemoryRepositories:
        return self._require_materialized(self._memory)

    @property
    def artifact(self) -> ArtifactRepositories:
        return self._require_materialized(self._artifact)

    @property
    def task(self) -> TaskRepositories:
        return self._require_materialized(self._task)

    @property
    def evaluation(self) -> EvaluationRepositories:
        return self._require_materialized(self._evaluation)

    @property
    def recovery(self) -> RecoveryRepositories:
        return self._require_materialized(self._recovery)

    @property
    def run_store(self) -> "RuntimeAgentRunStore":
        return self._require_materialized(self._run_store)

    @property
    def retention(self) -> "RuntimeRetentionController":
        return self._require_materialized(self._retention)

    async def initialize(
        self,
        *,
        namespace: str,
        tenant_id: str,
        read_only: bool = False,
        module_routes: Mapping[RuntimeDomain, StorageRoute] | None = None,
        capability_routes: Mapping[str, StorageRoute] | None = None,
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
            routes = dict(module_routes or {})
            if any(not isinstance(route, StorageRoute) for route in routes.values()):
                raise TypeError("module storage must use StorageRoute")
            if not routes.keys() <= {RuntimeDomain.TRACE, RuntimeDomain.CHECKPOINT}:
                raise ValueError("module storage only configures Trace and Checkpoint resources")
            self._plan = self._plan.with_module_routes(routes)
            _validate_storage_configuration(self._plan, self._external_object_store)
            self._lifecycle = _RuntimeStateLifecycle.INITIALIZING
            try:
                from ._materializer import materialize_runtime_storage

                materialized = await materialize_runtime_storage(
                    self._plan,
                    namespace=namespace,
                    tenant_id=tenant_id,
                    object_store=self._external_object_store,
                    read_only=read_only,
                    capability_routes=capability_routes,
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
                    name="linktools-runtime-storage-close",
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
        value: "_MaterializedRuntimeStorage",
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
        self._run_store = value.run_store
        self._retention = value.retention
        self._stores = dict(value.stores)
        self.capability_stores = dict(value.capability_stores)
        self.durable_capability_ids = value.durable_capability_ids
        self._close_actions = value.close_actions
        self._namespace = namespace
        self._tenant_id = tenant_id

    def _require_ready(self) -> None:
        if self._lifecycle is not _RuntimeStateLifecycle.READY:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "RuntimeState is not ready",
            )

    def _require_materialized(self, value: object) -> object:
        self._require_ready()
        if value is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "RuntimeState is not ready",
            )
        return value

    async def purge_unreferenced_objects(
        self, candidates: tuple[tuple[RuntimeDomain, ObjectRef], ...], *,
        exclusive: SnapshotExclusiveGuard, limit: int = 100,
    ) -> ObjectCleanupResult:
        return await self.snapshot_operations().purge_unreferenced_objects(
            candidates, exclusive=exclusive, limit=limit,
        )

    def object_store(self, domain: RuntimeDomain) -> ObjectStore:
        self._require_ready()
        if self._objects is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            )
        return self._objects.object_store(domain)

    async def read_object(self, domain: RuntimeDomain, reference: ObjectRef) -> bytes:
        """Read a referenced object through this storage's content cache."""
        self._require_ready()
        if self._objects is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._objects.read_object(domain, reference)

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

    def capability_marker_store(self) -> StateStore | None:
        """Choose an existing durable partition visible to portable snapshots."""
        self._require_ready()
        domains = self.plan.durable_domains
        if RuntimeDomain.EXECUTION in domains:
            return self._stores[RuntimeDomain.EXECUTION]
        return next((self._stores[domain] for domain in RuntimeDomain if domain in domains), None)

    def snapshot_resources(self) -> SnapshotResources:
        self._require_ready()
        return SnapshotResources(
            namespace=self.namespace, tenant_id=self.tenant_id,
            durable_domains=self.plan.durable_domains, stores=self._stores,
            object_stores={domain: self.object_store(domain) for domain in RuntimeDomain
                           if runtime_domain_uses_object_store(domain)},
            read_only=self.read_only,
            portable_export=not self.durable_capability_ids,
        )

    def snapshot_operations(self) -> RuntimeSnapshotOperations:
        return RuntimeSnapshotOperations(self.snapshot_resources(), require_ready=self._require_ready)

    async def export_snapshot(self, *, object_store: ObjectStore, limits: SnapshotLimits) -> ObjectRef:
        return await self.snapshot_operations().export_snapshot(object_store=object_store, limits=limits)

    @classmethod
    async def restore_snapshot(
        cls, ref: ObjectRef, *, object_store: ObjectStore, root: str | Path, limits: SnapshotLimits,
    ) -> None:
        @asynccontextmanager
        async def open_target(path: Path, namespace: str, tenant_id: str) -> AsyncIterator[SnapshotResources]:
            state = cls(RuntimeStorage.from_root(path))
            try:
                await state.initialize(namespace=namespace, tenant_id=tenant_id)
                yield state.snapshot_resources()
            finally:
                await state.close()

        await restore_snapshot(ref, object_store=object_store, root=root, limits=limits, open_target=open_target)


__all__ = ["RuntimeState"]
