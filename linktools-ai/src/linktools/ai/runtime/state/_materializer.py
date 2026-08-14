#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize RuntimeState domain contracts and owned resources."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import (
    InMemoryObjectStore,
    ObjectStore,
    SqlObjectStore,
    SqlStorageContext,
    StorageMetrics,
    TransientObjectStore,
    create_sql_storage_context,
    provision_sql,
    validate_sql,
)
from ._contracts import (
    ArtifactState,
    ConversationState,
    EvaluationState,
    ExecutionState,
    MemoryState,
    RecoveryState,
    RuntimeRepository,
    TaskState,
)
from ._memory import _build_in_memory_domains
from ._filesystem import _FilesystemDomainBackend, _build_filesystem_domain
from ._plan import RuntimeDomain, RuntimeRetentionMode, RuntimeStatePlan
from ._retention import RuntimeRetentionController
from ._steps import InMemoryStepArchive, RuntimeStepStore, StagingStepStore
from ._transaction import TransactionHub

_logger = environ.get_logger("ai.runtime.state.materializer")
_OBJECT_DOMAINS = frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY})
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
    metrics: StorageMetrics
    close_actions: tuple[Callable[[], Awaitable[None]], ...]


class _RuntimeObjectRouter:
    def __init__(self, stores: Mapping[RuntimeDomain, ObjectStore]) -> None:
        self._stores = dict(stores)

    def object_store(self, domain: RuntimeDomain) -> ObjectStore:
        try:
            return self._stores[domain]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def working_object_store(self, domain: RuntimeDomain, *, owner_scope: str) -> ObjectStore:
        store = self.object_store(domain)
        if isinstance(store, TransientObjectStore):
            return store.scoped(_transient_scope(domain, owner_scope))
        return store

    async def release_object_scope(self, domain: RuntimeDomain, *, owner_scope: str) -> None:
        store = self.object_store(domain)
        if isinstance(store, TransientObjectStore):
            await store.release_scope(_transient_scope(domain, owner_scope))

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
    object_store: "ObjectStore | None",
) -> _MaterializedRuntimeState:
    """Acquire every selected component and return independent close actions."""
    cleanup: list[Callable[[], Awaitable[None]]] = []
    hub = TransactionHub()
    memory_domains = frozenset(domain for domain in RuntimeDomain if plan.route(domain).kind == "memory")
    states: dict[RuntimeDomain, object] = {}
    filesystem_backends: dict[RuntimeDomain, _FilesystemDomainBackend] = {}
    sql_contexts: dict[RuntimeDomain, SqlStorageContext] = {}
    components: tuple[RuntimeRepository, ...] = ()
    try:
        if memory_domains:
            from ._memory import RuntimeTransactionBinding

            binding = RuntimeTransactionBinding()
            parts = _build_in_memory_domains(
                namespace=namespace,
                domains=memory_domains,
                transaction_hub=hub,
                transaction_binding=binding,
            )
            states.update(parts.states)
            components = parts.components
            await _initialize_components(components, cleanup)
        if len(states) != len(memory_domains):
            missing = sorted(domain.value for domain in memory_domains if domain not in states)
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, safe_details={"domains": missing})

        for domain in RuntimeDomain:
            route = plan.route(domain)
            if route.kind != "filesystem":
                continue
            if route.path is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            backend = await _build_filesystem_domain(
                route.path,
                namespace=namespace,
                tenant_id=tenant_id,
                domain=domain,
            )
            await backend.prepare()
            filesystem_backends[domain] = backend
            states[domain] = backend.state
            cleanup.append(backend.release)
            await _initialize_components(backend.components, cleanup)

        for route, domains in _sql_route_groups(plan):
            owns_engine = route.kind == "sqlite"
            if owns_engine:
                if route.path is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    route.path.parent.mkdir(parents=True, exist_ok=True)
                except OSError as error:
                    raise AIError(ErrorCode.STORAGE_UNAVAILABLE, "failed to prepare SQLite runtime directory") from error
                from sqlalchemy.ext.asyncio import create_async_engine

                engine = create_async_engine(f"sqlite+aiosqlite:///{route.path}")
            else:
                if route.engine is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                engine = route.engine
            context = create_sql_storage_context(engine, owns_engine=owns_engine)
            cleanup.append(context.close)
            group_plan = RuntimeStatePlan(**{
                item.value: plan.route(item) if item in domains else RuntimeStatePlan().route(item)
                for item in RuntimeDomain
            })
            from ..schema_api import build_runtime_sql_metadata

            metadata = build_runtime_sql_metadata(
                group_plan,
                include_object_tables=object_store is None,
            )
            if owns_engine:
                await provision_sql(engine, metadata)
                await validate_sql(engine, metadata)
            else:
                await validate_sql(engine, metadata)
            await context.initialize()
            sql_contexts.update({domain: context for domain in domains})
            from ._sql import _build_sql_domains

            parts = _build_sql_domains(
                context,
                namespace=namespace,
                tenant_id=tenant_id,
                domains=domains,
                metadata=metadata,
                transaction_hub=hub,
            )
            states.update(parts.states)
            components = (*components, *parts.components)
            await _initialize_components(parts.components, cleanup)

        objects = _build_object_router(plan, object_store, filesystem_backends, sql_contexts)
        steps = _build_steps(plan, objects, filesystem_backends, sql_contexts, namespace=namespace, tenant_id=tenant_id)
        try:
            await steps.initialize()
        except BaseException:
            raise
        cleanup.append(steps.close)
        retention = RuntimeRetentionController(
            conversation=_state(states, RuntimeDomain.CONVERSATION),
            execution=_state(states, RuntimeDomain.EXECUTION),
            memory=_state(states, RuntimeDomain.MEMORY),
            artifact=_state(states, RuntimeDomain.ARTIFACT),
            task=_state(states, RuntimeDomain.TASK),
            evaluation=_state(states, RuntimeDomain.EVALUATION),
            recovery=_state(states, RuntimeDomain.RECOVERY),
            objects=objects,
            steps=steps,
            plan=plan,
            namespace=namespace,
        )
        normal_actions: list[Callable[[], Awaitable[None]]] = [steps.preflight_close, retention.close, steps.close]
        normal_actions.extend(_close_once(component.close) for component in _unique_reversed(components))
        normal_actions.extend(backend.release for backend in _unique_reversed(tuple(filesystem_backends.values())))
        normal_actions.extend(_close_once(context.close) for context in _unique_reversed(tuple(sql_contexts.values())))
        _logger.info("runtime state materialized: namespace=%s domains=%s", namespace, sorted(domain.value for domain in RuntimeDomain))
        return _MaterializedRuntimeState(
            conversation=_state(states, RuntimeDomain.CONVERSATION),
            execution=_state(states, RuntimeDomain.EXECUTION),
            memory=_state(states, RuntimeDomain.MEMORY),
            artifact=_state(states, RuntimeDomain.ARTIFACT),
            task=_state(states, RuntimeDomain.TASK),
            evaluation=_state(states, RuntimeDomain.EVALUATION),
            recovery=_state(states, RuntimeDomain.RECOVERY),
            objects=objects,
            steps=steps,
            retention=retention,
            metrics=StorageMetrics(),
            close_actions=tuple(normal_actions),
        )
    except BaseException as primary:
        await _cleanup_reverse(cleanup, primary)
        raise


async def _initialize_components(components: tuple[RuntimeRepository, ...], cleanup: list[Callable[[], Awaitable[None]]]) -> None:
    initialized: set[int] = set()
    for component in components:
        if id(component) in initialized:
            continue
        await component.initialize()
        initialized.add(id(component))
        cleanup.append(component.close)


def _build_object_router(
    plan: RuntimeStatePlan,
    external: "ObjectStore | None",
    filesystem_backends: Mapping[RuntimeDomain, _FilesystemDomainBackend],
    sql_contexts: Mapping[RuntimeDomain, SqlStorageContext],
) -> _RuntimeObjectRouter:
    stores: dict[RuntimeDomain, ObjectStore] = {}
    for domain in _OBJECT_DOMAINS:
        route = plan.route(domain)
        if route.retention is RuntimeRetentionMode.DURABLE and external is not None:
            stores[domain] = external
        elif route.retention is RuntimeRetentionMode.VOLATILE:
            stores[domain] = InMemoryObjectStore()
        elif route.retention is RuntimeRetentionMode.TRANSIENT:
            stores[domain] = TransientObjectStore()
        elif route.kind == "filesystem" and domain in filesystem_backends:
            stores[domain] = filesystem_backends[domain].object_store
        elif route.kind in {"sqlite", "sql"} and domain in sql_contexts:
            stores[domain] = SqlObjectStore.from_context(sql_contexts[domain])
        else:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    return _RuntimeObjectRouter(stores)


def _build_steps(
    plan: RuntimeStatePlan,
    objects: _RuntimeObjectRouter,
    filesystem_backends: Mapping[RuntimeDomain, _FilesystemDomainBackend],
    sql_contexts: Mapping[RuntimeDomain, SqlStorageContext],
    *,
    namespace: str,
    tenant_id: str,
) -> RuntimeStepStore:
    archives: dict[RuntimeDomain, object] = {}
    for domain in _STEP_DOMAINS:
        route = plan.route(domain)
        if route.retention is RuntimeRetentionMode.TRANSIENT and domain is not RuntimeDomain.CONVERSATION:
            continue
        if route.kind == "filesystem":
            from ._steps import FilesystemStepArchive

            archives[domain] = FilesystemStepArchive.from_runtime(
                route.path,
                namespace=namespace,
                tenant_id=tenant_id,
                runtime_domain=domain,
                object_store=objects.object_store(domain),
                writer_lock=filesystem_backends[domain].writer_lock,
            )
        elif route.kind in {"sqlite", "sql"}:
            from ._steps import SqlStepArchive

            archives[domain] = SqlStepArchive.from_runtime(
                sql_contexts[domain].engine,
                namespace=namespace,
                tenant_id=tenant_id,
                runtime_domain=domain,
                object_store=objects.object_store(domain),
                context=sql_contexts[domain],
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


def _state(states: Mapping[RuntimeDomain, object], domain: RuntimeDomain) -> object:
    try:
        return states[domain]
    except KeyError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, safe_details={"domain": domain.value}) from error


def _sql_route_groups(plan: RuntimeStatePlan) -> tuple[tuple[object, frozenset[RuntimeDomain]], ...]:
    groups: list[tuple[object, frozenset[RuntimeDomain]]] = []
    for domain in RuntimeDomain:
        route = plan.route(domain)
        if route.kind not in {"sqlite", "sql"}:
            continue
        for index, (existing, domains) in enumerate(groups):
            same = route.kind == existing.kind and (
                route.kind == "sqlite" and route.path == existing.path
                or route.kind == "sql" and route.engine is existing.engine
            )
            if same:
                groups[index] = (existing, domains | {domain})
                break
        else:
            groups.append((route, frozenset({domain})))
    return tuple(groups)


def _unique_reversed(values: tuple[object, ...]) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[int] = set()
    for value in reversed(values):
        if id(value) not in seen:
            result.append(value)
            seen.add(id(value))
    return tuple(result)


async def _cleanup_reverse(actions: list[Callable[[], Awaitable[None]]], primary: BaseException) -> None:
    del primary
    for action in reversed(actions):
        task = asyncio.create_task(action(), name="linktools-runtime-state-cleanup")
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(task)
            except BaseException:
                _logger.error("runtime state cleanup failed after cancellation", exc_info=True)
            continue
        except BaseException:
            _logger.error("runtime state cleanup failed", exc_info=True)


def _close_once(close_method: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
    async def close() -> None:
        await close_method()

    return close


def _transient_scope(domain: RuntimeDomain, owner_scope: str) -> str:
    if not owner_scope:
        raise ValueError("owner_scope must not be empty")
    return f"runtime:{domain.value}:{owner_scope}"


__all__ = ["materialize_runtime_state"]
