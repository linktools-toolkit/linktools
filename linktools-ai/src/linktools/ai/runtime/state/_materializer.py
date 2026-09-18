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
    ObjectStore,
    SqlStorageContext,
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
from ._maintenance import RuntimeStorageInspection
from ._memory import MemoryStateStorageGroup, MemoryStateStore
from ._object_router import _RuntimeObjectRouter, build_runtime_object_router
from ._plan import (
    RuntimeDomain,
    RuntimeStatePlan,
    RuntimeStateRoute,
    runtime_domain_uses_object_store,
)
from ._recovery_repositories import build_recovery_repository_bundle
from ._repositories import OperationLedgerRepository, build_repository_bundle
from ._retention import RuntimeRetentionController
from ._sql import SqlStateStorageGroup, SqlStateStore
from ._step_materializer import build_runtime_steps
from ._steps import RuntimeStepStore
from ._store import StateStore
from ._task_admission_repository import TaskAdmissionRepositoryImpl
from ._task_repository import TaskRepositoryImpl

_logger = environ.get_logger("ai.runtime.state.materializer")


@dataclass(frozen=True, slots=True)
class _MaterializedRuntimeState:
    conversation: ConversationState
    execution: ExecutionState
    memory: MemoryState
    artifact: ArtifactState
    task: TaskState
    evaluation: EvaluationState
    recovery: RecoveryState
    objects: _RuntimeObjectRouter
    steps: RuntimeStepStore
    retention: RuntimeRetentionController
    maintenance: RuntimeStorageInspection
    close_actions: tuple[Callable[[], Awaitable[None]], ...]


@dataclass(frozen=True, slots=True)
class _RuntimeStates:
    conversation: ConversationState
    execution: ExecutionState
    memory: MemoryState
    artifact: ArtifactState
    task: TaskState
    evaluation: EvaluationState
    recovery: RecoveryState


async def materialize_runtime_state(
    plan: RuntimeStatePlan,
    *,
    namespace: str,
    tenant_id: str,
    object_store: ObjectStore | None,
) -> _MaterializedRuntimeState:
    stores: dict[RuntimeDomain, StateStore] = {}
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
            scope = _filesystem_group_scope(
                namespace, tenant_id, group_root, member_roots
            )
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
                    _range_index=domain is RuntimeDomain.MEMORY,
                    group=group,
                )
                stores[domain] = store
                cleanups.append(store.close)
            cleanups.append(group.close)

        for store in stores.values():
            if isinstance(store, (MemoryStateStore, FilesystemStateStore)):
                await store.initialize()

        for key, domains in sql_groups.items():
            route = sql_routes[key]
            bootstrap_local_schema = False
            if key[0] == "sqlite":
                if route.path is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                bootstrap_local_schema = not route.path.exists()
                await asyncio.to_thread(
                    route.path.parent.mkdir, parents=True, exist_ok=True
                )
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
                if key[0] in {"sqlite", "sql"} and object_store is None and any(
                    runtime_domain_uses_object_store(domain)
                    for domain in domains
                ):
                    build_object_sql_metadata(metadata=metadata)
                if bootstrap_local_schema:
                    await context.initialize()
                    async with context.engine.begin() as connection:
                        await connection.run_sync(metadata.create_all)
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
                elif key[0] == "sqlite":
                    await context.close()
                raise
            cleanups.extend(store.close for store in group_stores)
            if group is not None:
                cleanups.append(group.close)

        execution_store = stores[RuntimeDomain.EXECUTION]
        recovery_store = stores[RuntimeDomain.RECOVERY]
        if execution_store.storage_group is not recovery_store.storage_group:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "execution and recovery must share a StateStorageGroup",
            )

        bundles = {
            domain: build_repository_bundle(
                stores[domain], namespace=namespace, tenant_id=tenant_id, domain=domain
            )
            for domain in RuntimeDomain
            if domain not in {RuntimeDomain.RECOVERY, RuntimeDomain.TASK}
        }
        bundles[RuntimeDomain.RECOVERY] = build_recovery_repository_bundle(
            recovery_store,
            namespace=namespace,
            tenant_id=tenant_id,
        )
        task_store = stores[RuntimeDomain.TASK]
        bundles[RuntimeDomain.TASK] = {
            "operations": OperationLedgerRepository(
                task_store,
                namespace=namespace,
                tenant_id=tenant_id,
                domain=RuntimeDomain.TASK,
            ),
            "tasks": TaskRepositoryImpl(
                task_store,
                namespace=namespace,
                tenant_id=tenant_id,
            ),
            "admissions": TaskAdmissionRepositoryImpl(
                task_store,
                namespace=namespace,
                tenant_id=tenant_id,
            ),
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
        objects = build_runtime_object_router(plan, object_store, stores, sql_contexts)
        steps = build_runtime_steps(
            plan,
            stores,
            objects,
            history_repository=bundles[RuntimeDomain.CONVERSATION]["histories"],
            execution_repository=bundles[RuntimeDomain.EXECUTION]["executions"],
            namespace=namespace,
            tenant_id=tenant_id,
        )
        await steps.initialize()
        retention = RuntimeRetentionController(
            conversation=states.conversation,
            execution=states.execution,
            memory=states.memory,
            artifact=states.artifact,
            evaluation=states.evaluation,
            recovery=states.recovery,
            objects=objects,
            steps=steps,
            plan=plan,
            namespace=namespace,
        )
        maintenance = RuntimeStorageInspection(
            {domain: stores[domain] for domain in RuntimeDomain},
            objects,
            durable_domains=plan.durable_domains,
            state_validators=(steps.validate_integrity,),
        )
        actions: list[Callable[[], Awaitable[None]]] = [
            steps.preflight_close,
            objects.preflight_close,
            retention.close,
            steps.close,
        ]
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
            close_actions=tuple(actions),
        )
    except BaseException:
        for cleanup in reversed(cleanups):
            try:
                await cleanup()
            except BaseException as error:
                code = error.code.value if isinstance(error, AIError) else None
                _logger.error(
                    "runtime materialization cleanup failed: phase=%s code=%s "
                    "exception_type=%s",
                    "runtime.state.materialize",
                    code,
                    type(error).__name__,
                )
        raise


def _states(bundles: Mapping[RuntimeDomain, Mapping[str, object]]) -> _RuntimeStates:
    try:
        return _RuntimeStates(
            conversation=ConversationState(
                bundles[RuntimeDomain.CONVERSATION]["sessions"],
                bundles[RuntimeDomain.CONVERSATION]["histories"],
                bundles[RuntimeDomain.CONVERSATION]["operations"],
            ),
            execution=ExecutionState(
                bundles[RuntimeDomain.EXECUTION]["executions"],
                bundles[RuntimeDomain.EXECUTION]["events"],
                bundles[RuntimeDomain.EXECUTION]["idempotency"],
                bundles[RuntimeDomain.EXECUTION]["operations"],
            ),
            memory=MemoryState(
                bundles[RuntimeDomain.MEMORY]["records"],
                bundles[RuntimeDomain.MEMORY]["operations"],
            ),
            artifact=ArtifactState(
                bundles[RuntimeDomain.ARTIFACT]["records"],
                bundles[RuntimeDomain.ARTIFACT]["operations"],
            ),
            task=TaskState(
                bundles[RuntimeDomain.TASK]["tasks"],
                bundles[RuntimeDomain.TASK]["operations"],
                bundles[RuntimeDomain.TASK]["admissions"],
            ),
            evaluation=EvaluationState(
                bundles[RuntimeDomain.EVALUATION]["records"],
                bundles[RuntimeDomain.EVALUATION]["idempotency"],
                bundles[RuntimeDomain.EVALUATION]["operations"],
            ),
            recovery=RecoveryState(
                bundles[RuntimeDomain.RECOVERY]["approvals"],
                bundles[RuntimeDomain.RECOVERY]["external_calls"],
                bundles[RuntimeDomain.RECOVERY]["checkpoints"],
                bundles[RuntimeDomain.RECOVERY]["operations"],
                bundles[RuntimeDomain.RECOVERY]["tools"],
            ),
        )
    except KeyError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


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


def _route_domain_path(
    route: RuntimeStateRoute, namespace: str, tenant_id: str
) -> Path:
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
