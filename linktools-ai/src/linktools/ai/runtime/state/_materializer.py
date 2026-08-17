#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize Runtime repositories and their owned StateStore resources."""

import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import (
    FilesystemObjectStore,
    InMemoryObjectStore,
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
from ._filesystem import FilesystemStateStore
from ._memory import MemoryStateStore
from ._plan import RuntimeDomain, RuntimeRetentionMode, RuntimeStatePlan
from ._repositories import build_repository_bundle
from ._retention import RuntimeRetentionController
from ._sql import SqlStateStore
from ._steps import InMemoryStepArchive, RuntimeStepStore, StagingStepStore, StateStepArchive

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
        for domain in RuntimeDomain:
            route = plan.route(domain)
            if route.kind == "memory":
                store: object = MemoryStateStore()
                await store.initialize()
                cleanups.append(store.close)
            elif route.kind == "filesystem":
                if route.path is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                store = FilesystemStateStore(
                    route.path / namespace_digest(namespace) / _tenant_scope_digest(tenant_id),
                    namespace=namespace,
                    tenant_id=tenant_id,
                    runtime_domain=domain.value,
                )
                await store.initialize()
                cleanups.append(store.close)
            elif route.kind in {"sqlite", "sql"}:
                if route.kind == "sqlite":
                    if route.path is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    route.path.parent.mkdir(parents=True, exist_ok=True)
                    from sqlalchemy.ext.asyncio import create_async_engine

                    engine = create_async_engine(f"sqlite+aiosqlite:///{route.path}")
                    context = create_sql_storage_context(engine, owns_engine=True)
                else:
                    if route.engine is None:
                        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                    context = create_sql_storage_context(route.engine)
                from sqlalchemy import MetaData

                from ._schema import build_runtime_sql_metadata

                metadata = MetaData()
                build_runtime_sql_metadata(frozenset({domain}), metadata=metadata)
                if object_store is None and domain in _OBJECT_DOMAINS:
                    build_object_sql_metadata(metadata=metadata)
                await context.initialize(metadata=metadata)
                store = SqlStateStore(context.engine, metadata=metadata, context=context)
                await store.initialize()
                sql_contexts[domain] = context
                cleanups.append(store.close)
                cleanups.append(context.close)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stores[domain] = store

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
        steps = _build_steps(plan, stores, namespace=namespace, tenant_id=tenant_id)
        await steps.initialize()
        cleanups.append(steps.close)
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
                    bundles[RuntimeDomain.CONVERSATION]["sessions"], bundles[RuntimeDomain.CONVERSATION]["operations"]
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
    plan: RuntimeStatePlan, stores: Mapping[RuntimeDomain, object], *, namespace: str, tenant_id: str
) -> RuntimeStepStore:
    archives: dict[RuntimeDomain, object] = {}
    for domain in _STEP_DOMAINS:
        route = plan.route(domain)
        if route.retention is RuntimeRetentionMode.TRANSIENT and domain is not RuntimeDomain.CONVERSATION:
            continue
        if route.retention is RuntimeRetentionMode.DURABLE:
            archives[domain] = StateStepArchive(
                stores[domain], namespace=namespace, tenant_id=tenant_id, runtime_domain=domain
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


__all__ = ["materialize_runtime_state"]
