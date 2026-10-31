#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared persistence scope, repositories and owned resource lifecycle."""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, TypeVar

from linktools.core import environ
from ..observe.stream import await_stream_cleanup

from ..core import validate_persistence_namespace, validate_tenant_id, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import (
    CacheBackend,
    TurnCoordinationBackend,
    StreamBackend,
    ObjectRef,
    ObjectStore,
    StorageRetentionMode,
)
from ._root import PersistenceStorage, _validate_storage_configuration
from ._plan import PersistenceDomain, PersistencePlan, runtime_domain_uses_object_store
from .contracts import (ArtifactRepositories, ConversationRepositories, EvaluationRepositories,
                         ExecutionRepositories, MemoryRepositories, RecoveryRepositories, TaskRepositories)
from .state import StateStore
from ._snapshot import SnapshotLimits, SnapshotExclusiveGuard
from .snapshots import SnapshotResources, RuntimeSnapshotOperations, restore_snapshot
from ._object_cleanup import ObjectCleanupResult

if TYPE_CHECKING:
    from ._materializer import _MaterializedRuntimeStorage
    from ._object_router import _RuntimeObjectRouter
    from ._retention import RuntimeRetentionController
    from .runs import RuntimeAgentRunStore


_ValueT = TypeVar("_ValueT")
_logger = environ.get_logger(__name__)


class PersistenceStatus(TypedDict):
    domain: PersistenceDomain
    backend: str
    retention: StorageRetentionMode
    path: str | None
    object_store_id: str
    object_store_paths: tuple[str, ...]


class _RuntimeStateLifecycle(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"


class Persistence:
    """Runtime composition state and lifecycle for typed module resources."""

    def __init__(
        self,
        storage: PersistenceStorage,
    ) -> None:
        if not isinstance(storage, PersistenceStorage):
            raise TypeError("storage must be PersistenceStorage")
        self._coordination_backend = storage.coordination
        self._local_sqlite_path = storage.local_sqlite_path
        self._plan = storage.plan
        self._external_object_store = storage.object_store
        self._lifecycle = _RuntimeStateLifecycle.NEW
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._close_cursor = 0
        self._close_actions: tuple[Callable[[], Awaitable[None]], ...] = ()
        self._runtime_owners: dict[object, bool] = {}
        self._task_owners: set[object] = set()
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
        self._stores: dict[PersistenceDomain, StateStore] = {}
        self.capability_stores: dict[str, StateStore] = {}
        self.durable_capability_ids: frozenset[str] = frozenset()
        self._read_only = False

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        namespace: str,
        *,
        storage: PersistenceStorage,
        tenant_id: str = "default",
        read_only: bool = False,
    ) -> AsyncIterator["Persistence"]:
        """Own the storage lifetime shared by independently borrowed producers."""
        state = cls(storage)
        await state.initialize(
            namespace=namespace,
            tenant_id=validate_tenant_id(tenant_id),
            read_only=read_only,
        )
        try:
            yield state
        except BaseException as primary_error:
            try:
                await await_stream_cleanup(state.close(), None)
            except BaseException as error:
                _logger.error(
                    "secondary cleanup failed: phase=persistence.close code=%s exception_type=%s",
                    error.code.value if isinstance(error, AIError) else None,
                    type(error).__name__,
                )
                raise primary_error from error
            raise
        else:
            await await_stream_cleanup(state.close(), None)

    @property
    def coordination_backend(self) -> CacheBackend | TurnCoordinationBackend | StreamBackend | None:
        """Read the borrowed physical coordination resource, not a live producer."""
        self._require_ready()
        return self._coordination_backend

    @property
    def plan(self) -> PersistencePlan:
        return self._plan

    def status(self) -> tuple[PersistenceStatus, ...]:
        """Describe bound routes and object resources without probing their health."""
        self._require_ready()
        result: list[PersistenceStatus] = []
        for domain in PersistenceDomain:
            route = self._plan.route(domain)
            path = self._plan.filesystem_path(domain) if route.kind == "filesystem" else route.path
            objects = self.object_store(domain)
            result.append({
                "domain": domain,
                "backend": route.kind,
                "retention": route.retention,
                "path": str(path) if path is not None else None,
                "object_store_id": objects.store_id,
                "object_store_paths": tuple(str(path) for path in objects.local_paths()),
            })
        return tuple(result)

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

    async def capability_store(self, identity: str) -> StateStore:
        """Borrow an isolated capability partition, using the execution route by default."""
        if not isinstance(identity, str) or not identity:
            raise ValueError("capability identity must be nonempty")
        async with self._lock:
            self._require_ready()
            if self._read_only:
                raise AIError(ErrorCode.STORAGE_READ_ONLY)
            store = self.capability_stores.get(identity)
            if store is None:
                key = hashlib.sha256(canonical_json_bytes([
                    "capability-store", self.namespace, self.tenant_id, identity,
                ])).digest()
                store = await self.execution.executions.state_store.storage_group.open_partition(key, create=True)
                if store is None:
                    raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                self.capability_stores[identity] = store
                if self._plan.route(PersistenceDomain.EXECUTION).retention is StorageRetentionMode.DURABLE:
                    self.durable_capability_ids = self.durable_capability_ids | {identity}
            return store

    async def create_agent_run_store(self) -> "RuntimeAgentRunStore":
        """Create a producer-owned capture lifetime over these borrowed repositories."""
        self._require_ready()
        from ._step_materializer import build_runtime_agent_run_store

        if self._objects is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        store = build_runtime_agent_run_store(
            self._plan, self._stores, self._objects,
            history_repository=self.conversation.histories,
            execution_repository=self.execution.executions,
            namespace=self.namespace, tenant_id=self.tenant_id,
        )
        try:
            await store.initialize()
        except BaseException:
            await store.preflight_close()
            await store.close()
            raise
        return store

    @property
    def exclusive_execution_writer(self) -> bool:
        """Whether this live resource owns the physical Execution writer permission."""
        self._require_ready()
        if self._read_only:
            return False
        from ._sql import SqlStateStorageGroup

        group = self.execution.executions.state_store.storage_group
        if isinstance(group, SqlStateStorageGroup):
            return group.context.exclusive_writer
        return self._plan.route(PersistenceDomain.EXECUTION).kind == "filesystem"

    async def bind_runtime(
        self, *, execution_producer: bool, owns_persistence: bool,
    ) -> Callable[[], Awaitable[None]]:
        """Admit a Runtime and return its retryable lifetime cleanup.

        The caller releases only after its producers have fully stopped. An
        owning Runtime also closes the physical resources before releasing.
        This admission does not replace a backend's cross-process ownership.
        """
        owner = object()
        async with self._lock:
            self._require_ready()
            if self._read_only:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "Persistence is read-only")
            if (
                execution_producer
                and self.exclusive_execution_writer
                and any(self._runtime_owners.values())
            ):
                raise AIError(
                    ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                    "Persistence already has an exclusive execution producer",
                )
            self._runtime_owners[owner] = execution_producer

        async def cleanup() -> None:
            if owns_persistence:
                await self._close(runtime_owner=owner)
            async with self._lock:
                self._runtime_owners.pop(owner, None)

        return cleanup

    def bind_task_engine(self) -> Callable[[], None]:
        """Retain a TaskEngine lifetime until its synchronous close notifications finish."""
        self._require_ready()
        owner = object()
        self._task_owners.add(owner)

        def release() -> None:
            self._task_owners.discard(owner)

        return release

    @property
    def retention(self) -> "RuntimeRetentionController":
        return self._require_materialized(self._retention)

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
                    "Persistence must be NEW",
                )
            validate_persistence_namespace(namespace)
            if not tenant_id.strip():
                raise ValueError("tenant_id is required")
            if not isinstance(read_only, bool):
                raise TypeError("read_only must be bool")
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
                    local_sqlite_path=self._local_sqlite_path,
                    capability_routes=self._plan.capability_routes,
                    register_cleanup=self._register_initialization_cleanup,
                )
                self._assign_materialized(
                    materialized,
                    namespace,
                    tenant_id,
                )
                self._read_only = read_only
                self._lifecycle = _RuntimeStateLifecycle.READY
            except BaseException as error:
                self._lifecycle = _RuntimeStateLifecycle.CLOSING
                primary_error = error
            else:
                return
        try:
            await await_stream_cleanup(self.close(), None)
        except BaseException as cleanup_error:
            raise primary_error from cleanup_error
        raise primary_error

    def _register_initialization_cleanup(
        self, actions: tuple[Callable[[], Awaitable[None]], ...],
    ) -> None:
        self._close_actions = (*actions, *self._close_actions)

    async def close(self) -> None:
        await self._close()

    async def _close(self, *, runtime_owner: object | None = None) -> None:
        async with self._lock:
            if self._lifecycle is _RuntimeStateLifecycle.NEW:
                self._lifecycle = _RuntimeStateLifecycle.CLOSED
                return
            if self._lifecycle is _RuntimeStateLifecycle.CLOSED:
                return
            if self._task_owners or any(owner is not runtime_owner for owner in self._runtime_owners):
                raise AIError(
                    ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                    "Persistence still has active Runtime or TaskEngine owners",
                )
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
                "Persistence is not ready",
            )

    def _require_materialized(self, value: _ValueT | None) -> _ValueT:
        self._require_ready()
        if value is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "Persistence is not ready",
            )
        return value

    async def purge_unreferenced_objects(
        self, candidates: tuple[tuple[PersistenceDomain, ObjectRef], ...], *,
        exclusive: SnapshotExclusiveGuard, limit: int = 100,
    ) -> ObjectCleanupResult:
        return await self.snapshot_operations().purge_unreferenced_objects(
            candidates, exclusive=exclusive, limit=limit,
        )

    def object_store(self, domain: PersistenceDomain) -> ObjectStore:
        self._require_ready()
        if self._objects is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            )
        return self._objects.object_store(domain)

    def archive_object_store(self, domain: PersistenceDomain) -> ObjectStore:
        self._require_ready()
        if self._objects is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._objects.archive_object_store(domain)

    async def read_object(self, domain: PersistenceDomain, reference: ObjectRef) -> bytes:
        """Read a referenced object through this storage's content cache."""
        self._require_ready()
        if self._objects is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._objects.read_object(domain, reference)

    async def read_archive_object(self, domain: PersistenceDomain, reference: ObjectRef) -> bytes:
        """Read a referenced object through this storage's content cache."""
        self._require_ready()
        if self._objects is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._objects.read_archive_object(domain, reference)

    def working_object_store(
        self,
        domain: PersistenceDomain,
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
        for domain in PersistenceDomain:
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
        if PersistenceDomain.EXECUTION in domains:
            return self._stores[PersistenceDomain.EXECUTION]
        return next((self._stores[domain] for domain in PersistenceDomain if domain in domains), None)

    def snapshot_resources(self) -> SnapshotResources:
        self._require_ready()
        return SnapshotResources(
            namespace=self.namespace, tenant_id=self.tenant_id,
            durable_domains=self.plan.durable_domains, stores=self._stores,
            object_stores={domain: self.object_store(domain) for domain in PersistenceDomain
                           if runtime_domain_uses_object_store(domain)},
            archive_object_stores={domain: self.archive_object_store(domain) for domain in PersistenceDomain},
            capability_stores=self.capability_stores,
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
        exclusive_writer: bool = False,
    ) -> None:
        @asynccontextmanager
        async def open_target(path: Path, namespace: str, tenant_id: str) -> AsyncIterator[SnapshotResources]:
            state = cls(PersistenceStorage.from_root(path, exclusive_writer=exclusive_writer))
            try:
                await state.initialize(namespace=namespace, tenant_id=tenant_id)
                yield state.snapshot_resources()
            finally:
                await state.close()

        await restore_snapshot(ref, object_store=object_store, root=root, limits=limits, open_target=open_target)


__all__ = ["Persistence", "PersistenceStatus"]
