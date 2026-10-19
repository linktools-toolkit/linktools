#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize Runtime repositories and their owned StateStore resources."""

import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import (
    CacheBackend,
    StorageRoute,
    StorageRetentionMode,
    ObjectStore,
    SqlStorageContext,
    build_object_sql_metadata,
    namespace_digest,
)
from ._contracts import (
    ArtifactRepositories,
    ConversationRepositories,
    EvaluationRepositories,
    ExecutionRepositories,
    MemoryRepositories,
    RecoveryRepositories,
    TaskRepositories,
)
from ._filesystem import FilesystemStateStore
from ._memory import InMemoryStateStore
from ._object_router import _RuntimeObjectRouter, build_runtime_object_router
from ._plan import RuntimeDomain, RuntimeStoragePlan, runtime_domain_uses_object_store
from ._recovery_repositories import build_recovery_repository_bundle
from ._repositories import OperationLedgerRepositoryImpl, RepositoryBase, build_repository_bundle
from ._resource_scope import (
    FilesystemResource, MemoryResource, StorageResourceRequest, StorageResourceScope,
)
from ._retention import RuntimeRetentionController
from ._sql import SqlStateStore
from ._step_materializer import build_runtime_agent_run_store
from ._steps import RuntimeAgentRunStore
from ._store import StateStore, state_store_digest
from ._task_admission_repository import TaskAdmissionRepositoryImpl
from ._task_repository import TaskRepositoryImpl

if TYPE_CHECKING:
    from sqlalchemy import MetaData

    from .._tool import ToolStateRepository
    from ._contracts import (
        ApprovalRepository,
        ArtifactRepository,
        BudgetRepository,
        ConversationHistoryRepository,
        EvaluationRepository,
        EventRepository,
        ExecutionRepository,
        ExternalCallRepository,
        IdempotencyRepository,
        MemoryRepository,
        OperationLedgerRepository,
        RecoveryCheckpointRepository,
        SessionRepository,
        TaskAdmissionRepository,
        TaskRepository,
    )

_logger = environ.get_logger("ai.runtime.state.materializer")


@dataclass(frozen=True, slots=True)
class _MaterializedRuntimeStorage:
    conversation: ConversationRepositories
    execution: ExecutionRepositories
    memory: MemoryRepositories
    artifact: ArtifactRepositories
    task: TaskRepositories
    evaluation: EvaluationRepositories
    recovery: RecoveryRepositories
    objects: _RuntimeObjectRouter
    run_store: RuntimeAgentRunStore
    retention: RuntimeRetentionController
    stores: Mapping[RuntimeDomain, StateStore]
    capability_stores: Mapping[str, StateStore]
    durable_capability_ids: frozenset[str]
    close_actions: tuple[Callable[[], Awaitable[None]], ...]


@dataclass(frozen=True, slots=True)
class _RuntimeRepositories:
    conversation: ConversationRepositories
    execution: ExecutionRepositories
    memory: MemoryRepositories
    artifact: ArtifactRepositories
    task: TaskRepositories
    evaluation: EvaluationRepositories
    recovery: RecoveryRepositories


async def materialize_runtime_storage(
    plan: RuntimeStoragePlan,
    *,
    namespace: str,
    tenant_id: str,
    object_store: ObjectStore | None,
    read_only: bool = False,
    capability_routes: Mapping[str, StorageRoute] | None = None,
) -> _MaterializedRuntimeStorage:
    from ._schema import build_runtime_sql_metadata

    def configure_schema(domain: RuntimeDomain, metadata: "MetaData") -> None:
        build_runtime_sql_metadata(frozenset({domain}), metadata=metadata)
        if object_store is None and runtime_domain_uses_object_store(domain):
            build_object_sql_metadata(metadata=metadata)

    requests = tuple(
        StorageResourceRequest(
            name=domain.value,
            route=plan.route(domain),
            member_path=(
                _route_domain_path(plan, domain, namespace, tenant_id)
                if plan.route(domain).kind == "filesystem" else None
            ),
            configure_schema=partial(configure_schema, domain),
        )
        for domain in RuntimeDomain
    )
    participant_routes = dict(capability_routes or {})
    requests += tuple(
        StorageResourceRequest(
            name=f"capability:{identity}", route=route,
            member_path=(route.path / "capabilities" / hashlib.sha256(identity.encode()).hexdigest()
                         / namespace_digest(namespace) / _tenant_scope_digest(tenant_id)
                         if route.kind == "filesystem" and route.path is not None else None),
            configure_schema=lambda metadata: build_runtime_sql_metadata(frozenset({RuntimeDomain.EXECUTION}), metadata=metadata),
        )
        for identity, route in participant_routes.items()
    )
    capability_stores: dict[str, StateStore] = {}
    stores: dict[RuntimeDomain, StateStore] = {}
    sql_contexts: dict[RuntimeDomain, SqlStorageContext] = {}
    cleanups: list[Callable[[], Awaitable[None]]] = []
    try:
        resources = await StorageResourceScope.open(
            requests, namespace=namespace, tenant_id=tenant_id, read_only=read_only,
        )
        cleanups.append(resources.close)
        for domain, resource in zip(RuntimeDomain, resources.resources):
            selected = plan.route(domain).cache
            cache = selected if isinstance(selected, CacheBackend) else None
            store: StateStore
            if isinstance(resource, MemoryResource):
                store = InMemoryStateStore(resource.group, cache=cache)
            elif isinstance(resource, FilesystemResource):
                store = FilesystemStateStore(
                    resource.member_path,
                    namespace=namespace,
                    tenant_id=tenant_id,
                    runtime_domain=domain.value,
                    _range_index=domain is RuntimeDomain.MEMORY,
                    group=resource.group,
                    cache=cache,
                )
            else:
                store = SqlStateStore(
                    resource.context.engine,
                    metadata=resource.metadata,
                    context=resource.context,
                    runtime_domain=domain,
                    group=resource.group,
                    cache=cache,
                    store_digest=state_store_digest(namespace, tenant_id, domain.value),
                )
                sql_contexts[domain] = resource.context
            stores[domain] = store
            cleanups.append(store.close)

        for (identity, route), resource in zip(participant_routes.items(), resources.resources[len(RuntimeDomain):]):
            partition = f"capability:{identity}"
            if isinstance(resource, MemoryResource):
                store = InMemoryStateStore(resource.group, cache=route.cache)
            elif isinstance(resource, FilesystemResource):
                store = FilesystemStateStore(
                    resource.member_path, namespace=namespace, tenant_id=tenant_id,
                    runtime_domain=partition, group=resource.group, cache=route.cache,
                )
            else:
                store = SqlStateStore(
                    resource.context.engine, metadata=resource.metadata, context=resource.context,
                    runtime_domain=partition, group=resource.group, cache=route.cache,
                    store_digest=state_store_digest(namespace, tenant_id, partition),
                )
            capability_stores[identity] = store
            cleanups.append(store.close)

        # Filesystem groups seal their membership when initialization starts.
        for store in (*stores.values(), *capability_stores.values()):
            await store.initialize()

        execution_store = stores[RuntimeDomain.EXECUTION]
        recovery_store = stores[RuntimeDomain.RECOVERY]
        if execution_store.storage_group is not recovery_store.storage_group:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "execution and recovery must share a StateStorageGroup",
            )

        atomic_domains = {RuntimeDomain.TRACE, RuntimeDomain.CHECKPOINT}
        for domain in atomic_domains:
            if stores[domain].storage_group is not execution_store.storage_group:
                raise AIError(
                    ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                    f"{domain.value} and execution must share a StateStorageGroup",
                )

        bundles: dict[RuntimeDomain, dict[str, RepositoryBase]] = {
            domain: build_repository_bundle(
                stores[domain], namespace=namespace, tenant_id=tenant_id, domain=domain
            )
            for domain in RuntimeDomain
            if domain not in {RuntimeDomain.RECOVERY, RuntimeDomain.TASK, RuntimeDomain.TRACE, RuntimeDomain.CHECKPOINT}
        }
        bundles[RuntimeDomain.RECOVERY] = build_recovery_repository_bundle(
            recovery_store,
            namespace=namespace,
            tenant_id=tenant_id,
        )
        task_store = stores[RuntimeDomain.TASK]
        bundles[RuntimeDomain.TASK] = {
            "operations": OperationLedgerRepositoryImpl(
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
        )
        for component in _unique(components):
            await component.initialize()

        repositories = _repositories(bundles)
        objects = build_runtime_object_router(
            plan, object_store, stores, sql_contexts,
            namespace=namespace, tenant_id=tenant_id,
        )
        run_store = build_runtime_agent_run_store(
            plan,
            stores,
            objects,
            history_repository=repositories.conversation.histories,
            execution_repository=repositories.execution.executions,
            namespace=namespace,
            tenant_id=tenant_id,
        )
        await run_store.initialize()
        retention = RuntimeRetentionController(
            conversation=repositories.conversation,
            execution=repositories.execution,
            objects=objects,
            run_store=run_store,
            plan=plan,
            namespace=namespace,
        )
        actions: list[Callable[[], Awaitable[None]]] = [
            run_store.preflight_close,
            objects.drain_reads,
            objects.preflight_close,
            retention.close,
            run_store.close,
            repositories.execution.budgets.close,
        ]
        actions.extend(cleanups[1:])
        actions.append(resources.close)
        _logger.info(
            "runtime storage materialized: namespace=%s domains=%s",
            namespace,
            ",".join(domain.value for domain in RuntimeDomain),
        )
        return _MaterializedRuntimeStorage(
            conversation=repositories.conversation,
            execution=repositories.execution,
            memory=repositories.memory,
            artifact=repositories.artifact,
            task=repositories.task,
            evaluation=repositories.evaluation,
            recovery=repositories.recovery,
            objects=objects,
            run_store=run_store,
            retention=retention,
            stores=dict(stores),
            capability_stores=capability_stores,
            durable_capability_ids=frozenset(identity for identity, route in participant_routes.items() if route.retention is StorageRetentionMode.DURABLE),
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



def _repositories(
    bundles: Mapping[RuntimeDomain, Mapping[str, RepositoryBase]],
) -> _RuntimeRepositories:
    try:
        return _RuntimeRepositories(
            conversation=ConversationRepositories(
                cast("SessionRepository", bundles[RuntimeDomain.CONVERSATION]["sessions"]),
                cast("ConversationHistoryRepository", bundles[RuntimeDomain.CONVERSATION]["histories"]),
                cast("OperationLedgerRepository", bundles[RuntimeDomain.CONVERSATION]["operations"]),
            ),
            execution=ExecutionRepositories(
                cast("ExecutionRepository", bundles[RuntimeDomain.EXECUTION]["executions"]),
                cast("EventRepository", bundles[RuntimeDomain.EXECUTION]["events"]),
                cast("IdempotencyRepository", bundles[RuntimeDomain.EXECUTION]["idempotency"]),
                cast("OperationLedgerRepository", bundles[RuntimeDomain.EXECUTION]["operations"]),
                cast("BudgetRepository", bundles[RuntimeDomain.EXECUTION]["budgets"]),
            ),
            memory=MemoryRepositories(
                cast("MemoryRepository", bundles[RuntimeDomain.MEMORY]["records"]),
                cast("OperationLedgerRepository", bundles[RuntimeDomain.MEMORY]["operations"]),
            ),
            artifact=ArtifactRepositories(
                cast("ArtifactRepository", bundles[RuntimeDomain.ARTIFACT]["records"]),
                cast("OperationLedgerRepository", bundles[RuntimeDomain.ARTIFACT]["operations"]),
            ),
            task=TaskRepositories(
                cast("TaskRepository", bundles[RuntimeDomain.TASK]["tasks"]),
                cast("OperationLedgerRepository", bundles[RuntimeDomain.TASK]["operations"]),
                cast("TaskAdmissionRepository", bundles[RuntimeDomain.TASK]["admissions"]),
            ),
            evaluation=EvaluationRepositories(
                cast("EvaluationRepository", bundles[RuntimeDomain.EVALUATION]["records"]),
                cast("IdempotencyRepository", bundles[RuntimeDomain.EVALUATION]["idempotency"]),
                cast("OperationLedgerRepository", bundles[RuntimeDomain.EVALUATION]["operations"]),
            ),
            recovery=RecoveryRepositories(
                cast("ApprovalRepository", bundles[RuntimeDomain.RECOVERY]["approvals"]),
                cast("ExternalCallRepository", bundles[RuntimeDomain.RECOVERY]["external_calls"]),
                cast("RecoveryCheckpointRepository", bundles[RuntimeDomain.RECOVERY]["checkpoints"]),
                cast("OperationLedgerRepository", bundles[RuntimeDomain.RECOVERY]["operations"]),
                cast("ToolStateRepository", bundles[RuntimeDomain.RECOVERY]["tools"]),
            ),
        )
    except KeyError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _unique(values: tuple[RepositoryBase, ...]) -> tuple[RepositoryBase, ...]:
    result: list[RepositoryBase] = []
    seen: set[int] = set()
    for value in values:
        if id(value) not in seen:
            result.append(value)
            seen.add(id(value))
    return tuple(result)


def _tenant_scope_digest(tenant_id: str) -> str:
    return hashlib.sha256(("tenant:" + tenant_id).encode("utf-8")).hexdigest()


def _route_domain_path(
    plan: RuntimeStoragePlan,
    domain: RuntimeDomain,
    namespace: str,
    tenant_id: str,
) -> Path:
    path = plan.filesystem_path(domain)
    return path / namespace_digest(namespace) / _tenant_scope_digest(tenant_id)


__all__ = ["materialize_runtime_storage"]
