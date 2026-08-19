#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize Runtime repositories and their owned StateStore resources."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from linktools.core import environ

from ...core import canonical_sha256
from ...errors import AIError, ErrorCode
from ...storage import (
    FilesystemObjectStore,
    InMemoryObjectStore,
    ObjectRef,
    ObjectStore,
    SqlObjectStore,
    SqlStorageContext,
    StorageMetrics,
    TransientObjectStore,
    build_object_sql_metadata,
    create_sql_storage_context,
    namespace_digest,
)
from ._contracts import (
    ArtifactState,
    ConversationState,
    EvaluationState,
    ExecutionState,
    MemoryState,
    RecoveryState,
    TaskState,
)
from ._filesystem import FilesystemStateStorageGroup, FilesystemStateStore
from ._memory import MemoryStateStorageGroup, MemoryStateStore
from ._maintenance import RuntimeStorageMaintenance
from ._plan import (
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeStatePlan,
    RuntimeStateRoute,
)
from ._repositories import build_repository_bundle
from ._retention import RuntimeRetentionController
from ._sql import SqlStateStorageGroup, SqlStateStore
from ._steps import (
    InMemoryStepArchive,
    RuntimeStepStore,
    StagingStepStore,
    StateStepArchive,
)

_logger = environ.get_logger("ai.runtime.state.materializer")
_OBJECT_DOMAINS = frozenset(
    {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.RECOVERY,
    }
)
_STEP_DOMAINS = (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY)


@dataclass(frozen=True, slots=True)
class _MaterializedRuntimeState:
    conversation: ConversationState
    execution: ExecutionState
    memory: MemoryState
    artifact: ArtifactState
    task: TaskState
    evaluation: EvaluationState
    recovery: RecoveryState
    objects: "_RuntimeObjectRouter"
    steps: RuntimeStepStore
    retention: RuntimeRetentionController
    maintenance: RuntimeStorageMaintenance
    metrics: StorageMetrics
    close_actions: tuple[Callable[[], Awaitable[None]], ...]


class _RuntimeObjectRouter:
    def __init__(self, stores: Mapping[RuntimeDomain, ObjectStore]) -> None:
        self._stores = dict(stores)

    def object_store(self, domain: RuntimeDomain) -> ObjectStore:
        try:
            return self._stores[domain]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY) from error

    def resolve_object(self, domain: RuntimeDomain, reference: ObjectRef) -> ObjectStore:
        """Resolve an object by its durable runtime domain, never by store id."""
        store = self.object_store(domain)
        if reference.store_id != store.store_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return store

    def working_object_store(self, domain: RuntimeDomain, *, owner_scope: str) -> ObjectStore:
        store = self.object_store(domain)
        if isinstance(store, TransientObjectStore):
            return store.scoped(f"runtime:{domain.value}:{owner_scope}")
        return store

    async def release_object_scope(self, domain: RuntimeDomain, *, owner_scope: str) -> None:
        store = self.object_store(domain)
        if isinstance(store, TransientObjectStore):
            await store.release_scope(f"runtime:{domain.value}:{owner_scope}")

    async def clear_transient(self) -> None:
        seen: set[int] = set()
        for store in self._stores.values():
            if isinstance(store, TransientObjectStore) and id(store) not in seen:
                store.clear()
                seen.add(id(store))


async def materialize_runtime_state(
    plan: RuntimeStatePlan,
    *,
    namespace: str,
    tenant_id: str,
    object_store: ObjectStore | None,
) -> _MaterializedRuntimeState:
    stores: dict[RuntimeDomain, object] = {}
    sql_contexts: dict[RuntimeDomain, SqlStorageContext] = {}
    cleanups: list[Callable[[], Awaitable[None]]] = []
    try:
        sql_groups: dict[tuple[str, object], list[RuntimeDomain]] = {}
        sql_routes: dict[tuple[str, object], RuntimeStateRoute] = {}
        filesystem_domains: dict[Path, list[RuntimeDomain]] = {}
        filesystem_routes: dict[Path, RuntimeStateRoute] = {}
        memory_group = MemoryStateStorageGroup()
        for domain in RuntimeDomain:
            route = plan.route(domain)
            if route.kind in {"sqlite", "sql"}:
                if route.kind == "sqlite":
                    if route.path is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    key = ("sqlite", route.path)
                else:
                    if route.engine is None:
                        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                    key = ("sql", route.engine)
                sql_groups.setdefault(key, []).append(domain)
                sql_routes[key] = route
                continue
            if route.kind == "memory":
                stores[domain] = MemoryStateStore(memory_group)
            elif route.kind == "filesystem":
                if route.path is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                group_root = route.transaction_root or route.path
                filesystem_domains.setdefault(group_root, []).append(domain)
                filesystem_routes[group_root] = route
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        for group_root, domains in filesystem_domains.items():
            route = filesystem_routes[group_root]
            member_roots = {
                domain: _route_domain_path(plan.route(domain), namespace, tenant_id)
                for domain in domains
            }
            standalone = route.transaction_root is None
            scope = _filesystem_group_scope(namespace, tenant_id, group_root, member_roots)
            group = FilesystemStateStorageGroup(
                group_root if not standalone else member_roots[domains[0]],
                namespace=namespace,
                tenant_id=tenant_id,
                scope_digest=scope,
                standalone=standalone,
            )
            for domain in domains:
                store = FilesystemStateStore(
                    member_roots[domain],
                    namespace=namespace,
                    tenant_id=tenant_id,
                    runtime_domain=domain.value,
                    group=group,
                )
                stores[domain] = store
                cleanups.append(store.close)
            cleanups.append(group.close)

        for store in stores.values():
            if isinstance(store, MemoryStateStore) or isinstance(store, FilesystemStateStore):
                await store.initialize()

        for key, domains in sql_groups.items():
            route = sql_routes[key]
            if key[0] == "sqlite":
                if route.path is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await asyncio.to_thread(route.path.parent.mkdir, parents=True, exist_ok=True)
                from sqlalchemy.ext.asyncio import create_async_engine

                engine = create_async_engine(f"sqlite+aiosqlite:///{route.path}")
                context = create_sql_storage_context(engine, owns_engine=True)
            else:
                if route.engine is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                context = create_sql_storage_context(route.engine)
            group_stores: list[SqlStateStore] = []
            group: SqlStateStorageGroup | None = None
            try:
                from sqlalchemy import MetaData

                metadata = MetaData()
                from ._schema import build_runtime_sql_metadata

                build_runtime_sql_metadata(frozenset(domains), metadata=metadata)
                if object_store is None and set(domains) & _OBJECT_DOMAINS:
                    build_object_sql_metadata(metadata=metadata)
                group = SqlStateStorageGroup(
                    context,
                    metadata,
                    owns_context=key[0] == "sqlite",
                )
                for domain in domains:
                    store = SqlStateStore(
                        context.engine,
                        metadata=metadata,
                        context=context,
                        runtime_domain=domain,
                        group=group,
                    )
                    await store.initialize()
                    group_stores.append(store)
                    stores[domain] = store
                    sql_contexts[domain] = context
            except BaseException:
                for store in reversed(group_stores):
                    await store.close()
                if group is not None:
                    await group.close()
                raise
            cleanups.extend(store.close for store in group_stores)
            if group is not None:
                cleanups.append(group.close)

        bundles = {
            domain: build_repository_bundle(stores[domain], namespace=namespace, tenant_id=tenant_id, domain=domain)
            for domain in RuntimeDomain
        }
        components = tuple(
            value
            for bundle in bundles.values()
            for value in bundle.values()
            if hasattr(value, "initialize") and hasattr(value, "close")
        )
        for component in _unique(components):
            await component.initialize()

        states = _states(bundles)
        objects = _build_object_router(plan, object_store, stores, sql_contexts)
        steps = _build_steps(
            plan,
            stores,
            objects,
            history_repository=bundles[RuntimeDomain.CONVERSATION]["histories"],
            namespace=namespace,
            tenant_id=tenant_id,
        )
        await steps.initialize()
        retention = RuntimeRetentionController(
            conversation=states.conversation,
            execution=states.execution,
            memory=states.memory,
            artifact=states.artifact,
            task=states.task,
            evaluation=states.evaluation,
            recovery=states.recovery,
            objects=objects,
            steps=steps,
            plan=plan,
            namespace=namespace,
        )
        maintenance = RuntimeStorageMaintenance(
            {domain: stores[domain] for domain in RuntimeDomain},
            objects,
            durable_domains=plan.durable_domains,
        )
        actions: list[Callable[[], Awaitable[None]]] = [steps.preflight_close, retention.close, steps.close]
        actions.extend(cleanups)
        _logger.info(
            "runtime state materialized: namespace=%s domains=%s",
            namespace,
            ",".join(domain.value for domain in RuntimeDomain),
        )
        return _MaterializedRuntimeState(
            conversation=states.conversation,
            execution=states.execution,
            memory=states.memory,
            artifact=states.artifact,
            task=states.task,
            evaluation=states.evaluation,
            recovery=states.recovery,
            objects=objects,
            steps=steps,
            retention=retention,
            maintenance=maintenance,
            metrics=StorageMetrics(),
            close_actions=tuple(actions),
        )
    except BaseException as primary:
        for cleanup in reversed(cleanups):
            try:
                await cleanup()
            except BaseException:
                _logger.error("runtime materialization cleanup failed", exc_info=True)
        raise primary


def _states(bundles: Mapping[RuntimeDomain, Mapping[str, object]]) -> object:
    try:
        return type(
            "RuntimeStates",
            (),
            {
                "conversation": ConversationState(
                    bundles[RuntimeDomain.CONVERSATION]["sessions"],
                    bundles[RuntimeDomain.CONVERSATION]["histories"],
                    bundles[RuntimeDomain.CONVERSATION]["operations"],
                ),
                "execution": ExecutionState(
                    bundles[RuntimeDomain.EXECUTION]["executions"],
                    bundles[RuntimeDomain.EXECUTION]["events"],
                    bundles[RuntimeDomain.EXECUTION]["idempotency"],
                    bundles[RuntimeDomain.EXECUTION]["operations"],
                ),
                "memory": MemoryState(
                    bundles[RuntimeDomain.MEMORY]["records"], bundles[RuntimeDomain.MEMORY]["operations"]
                ),
                "artifact": ArtifactState(
                    bundles[RuntimeDomain.ARTIFACT]["records"], bundles[RuntimeDomain.ARTIFACT]["operations"]
                ),
                "task": TaskState(bundles[RuntimeDomain.TASK]["tasks"], bundles[RuntimeDomain.TASK]["operations"]),
                "evaluation": EvaluationState(
                    bundles[RuntimeDomain.EVALUATION]["records"],
                    bundles[RuntimeDomain.EVALUATION]["idempotency"],
                    bundles[RuntimeDomain.EVALUATION]["operations"],
                ),
                "recovery": RecoveryState(
                    bundles[RuntimeDomain.RECOVERY]["approvals"],
                    bundles[RuntimeDomain.RECOVERY]["external_calls"],
                    bundles[RuntimeDomain.RECOVERY]["checkpoints"],
                    bundles[RuntimeDomain.RECOVERY]["operations"],
                    bundles[RuntimeDomain.RECOVERY]["tools"],
                ),
            },
        )()
    except KeyError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _build_object_router(
    plan: RuntimeStatePlan,
    external: ObjectStore | None,
    stores: Mapping[RuntimeDomain, object],
    contexts: Mapping[RuntimeDomain, SqlStorageContext],
) -> _RuntimeObjectRouter:
    values: dict[RuntimeDomain, ObjectStore] = {}
    for domain in _OBJECT_DOMAINS:
        route = plan.route(domain)
        if route.retention is RuntimeRetentionMode.DURABLE and external is not None:
            values[domain] = external
        elif route.retention is RuntimeRetentionMode.VOLATILE:
            values[domain] = InMemoryObjectStore()
        elif route.retention is RuntimeRetentionMode.TRANSIENT:
            values[domain] = TransientObjectStore()
        elif route.kind == "filesystem" and route.path is not None:
            values[domain] = FilesystemObjectStore(route.path / "objects")
        elif route.kind in {"sqlite", "sql"} and domain in contexts:
            values[domain] = SqlObjectStore.from_context(contexts[domain])
        else:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    return _RuntimeObjectRouter(values)


def _build_steps(
    plan: RuntimeStatePlan,
    stores: Mapping[RuntimeDomain, object],
    objects: _RuntimeObjectRouter,
    history_repository: object,
    *,
    namespace: str,
    tenant_id: str,
) -> RuntimeStepStore:
    archives: dict[RuntimeDomain, object] = {}
    for domain in _STEP_DOMAINS:
        route = plan.route(domain)
        if route.retention is RuntimeRetentionMode.TRANSIENT and domain is not RuntimeDomain.CONVERSATION:
            continue
        if route.retention is RuntimeRetentionMode.DURABLE:
            context_sources = None
            conversation_archive = archives.get(RuntimeDomain.CONVERSATION)
            if isinstance(conversation_archive, StateStepArchive):
                context_sources = {
                    RuntimeDomain.CONVERSATION: conversation_archive.transcript_repository,
                }
            archives[domain] = StateStepArchive(
                stores[domain],
                object_store=objects.object_store(domain),
                namespace=namespace,
                tenant_id=tenant_id,
                runtime_domain=domain,
                context_sources=context_sources,
                history_repository=(
                    history_repository
                    if domain is RuntimeDomain.CONVERSATION
                    else None
                ),
            )
        else:
            archives[domain] = InMemoryStepArchive(domain)
    return RuntimeStepStore(
        StagingStepStore(),
        conversation_archive=archives[RuntimeDomain.CONVERSATION],
        execution_archive=archives.get(RuntimeDomain.EXECUTION),
        recovery_archive=archives.get(RuntimeDomain.RECOVERY),
        conversation_retention=plan.route(RuntimeDomain.CONVERSATION).retention,
        execution_retention=plan.route(RuntimeDomain.EXECUTION).retention,
        recovery_retention=plan.route(RuntimeDomain.RECOVERY).retention,
    )


def _unique(values: tuple[object, ...]) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[int] = set()
    for value in values:
        if id(value) not in seen:
            result.append(value)
            seen.add(id(value))
    return tuple(result)


def _tenant_scope_digest(tenant_id: str) -> str:
    return hashlib.sha256(("tenant:" + tenant_id).encode("utf-8")).hexdigest()


def _route_domain_path(route: RuntimeStateRoute, namespace: str, tenant_id: str) -> Path:
    if route.path is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return route.path / namespace_digest(namespace) / _tenant_scope_digest(tenant_id)


def _filesystem_group_scope(
    namespace: str,
    tenant_id: str,
    transaction_root: Path,
    member_roots: Mapping[RuntimeDomain, Path],
) -> str:
    members = tuple(
        (
            domain.value,
            member_roots[domain].relative_to(transaction_root).as_posix(),
        )
        for domain in sorted(member_roots, key=lambda value: value.value)
    )
    return canonical_sha256(
        {
            "namespace": namespace,
            "tenant_id": tenant_id,
            "members": members,
        }
    )[:32]


__all__ = ["materialize_runtime_state"]
