#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize domain repositories, objects, steps, and retention."""

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import (
    SqlStorageContext,
    create_sql_storage_context,
    provision_sql,
    validate_sql,
)
from .._persistence import RuntimeDomainStates
from ._contracts import RuntimeDomain, RuntimeRetentionMode
from ._filesystem import build_filesystem_runtime
from ._memory import RuntimeObjectRouter, build_in_memory_runtime
from ._plan import RuntimeStatePlan, RuntimeStateRoute
from ._retention import RuntimeRetentionController
from ._schema import build_runtime_sql_metadata
from ._sql import build_sql_runtime
from ._steps import (
    FilesystemStepArchive,
    InMemoryStepArchive,
    RuntimeStepStore,
    SqlStepArchive,
    StagingStepStore,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ...storage import ObjectStore

_logger = environ.get_logger("ai.runtime.state")
_OBJECT_DOMAINS = frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY})
_STEP_DOMAINS = (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY)


async def materialize_runtime_state(
    plan: RuntimeStatePlan,
    *,
    namespace: str,
    tenant_id: str,
    object_store: "ObjectStore | None",
) -> "tuple[RuntimeDomainStates, RuntimeStepStore, RuntimeRetentionController, tuple[Callable[[], Awaitable[None]], ...]]":
    runtimes: "list[object]" = []
    contexts: "list[SqlStorageContext]" = []
    route_resources: "dict[RuntimeDomain, object]" = {}
    try:
        memory_domains = frozenset(domain for domain in RuntimeDomain if plan.route(domain).kind == "memory")
        if memory_domains:
            runtime = build_in_memory_runtime(namespace=namespace)
            await runtime.initialize()
            runtimes.append(runtime)
            for domain in memory_domains:
                route_resources[domain] = runtime

        for domain in RuntimeDomain:
            route = plan.route(domain)
            if route.kind != "filesystem":
                continue
            runtime = build_filesystem_runtime(str(route.path), namespace=namespace, tenant_id=tenant_id, persist=frozenset({domain}))
            await runtime.initialize()
            runtimes.append(runtime)
            route_resources[domain] = runtime

        sql_groups: "list[tuple[RuntimeStateRoute, frozenset[RuntimeDomain]]]" = []
        for domain in RuntimeDomain:
            route = plan.route(domain)
            if route.kind not in {"sqlite", "sql"}:
                continue
            for index, (group_route, domains) in enumerate(sql_groups):
                same = route.kind == group_route.kind and (
                    route.kind == "sqlite" and route.path == group_route.path or route.kind == "sql" and route.engine is group_route.engine
                )
                if same:
                    sql_groups[index] = (group_route, domains | {domain})
                    break
            else:
                sql_groups.append((route, frozenset({domain})))

        for route, domains in sql_groups:
            engine: "AsyncEngine"
            owns_engine = route.kind == "sqlite"
            if owns_engine:
                from sqlalchemy.ext.asyncio import create_async_engine

                engine = create_async_engine(f"sqlite+aiosqlite:///{route.path}")
            else:
                if route.engine is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                engine = route.engine
            context = create_sql_storage_context(engine, owns_engine=owns_engine)
            contexts.append(context)
            group_plan = RuntimeStatePlan(**{item.value: plan.route(item) if item in domains else RuntimeStateRoute.memory() for item in RuntimeDomain})
            metadata = build_runtime_sql_metadata(
                group_plan,
                include_object_tables=object_store is None and any(
                    item in _OBJECT_DOMAINS and item in domains for item in RuntimeDomain
                ),
            )
            if owns_engine:
                await provision_sql(engine, metadata)
            else:
                await validate_sql(engine, metadata)
            await context.initialize()
            runtime = build_sql_runtime(
                context,
                namespace=namespace,
                tenant_id=tenant_id,
                plan=group_plan,
                object_store=object_store,
            )
            await runtime.initialize()
            runtimes.append(runtime)
            for domain in domains:
                route_resources[domain] = runtime

        domain = _combine_domain_states(namespace, route_resources)
        object_stores = {
            runtime_domain: object_store
            if object_store is not None and runtime_domain in _OBJECT_DOMAINS and plan.route(runtime_domain).retention is RuntimeRetentionMode.DURABLE
            else _runtime_object_store(route_resources[runtime_domain], runtime_domain)
            for runtime_domain in RuntimeDomain
        }
        domain = replace(domain, object_router=RuntimeObjectRouter(object_stores))
        archives = await _build_step_archives(plan, route_resources, object_stores, namespace=namespace, tenant_id=tenant_id)
        steps = RuntimeStepStore(
            StagingStepStore(),
            conversation_archive=archives[RuntimeDomain.CONVERSATION],
            execution_archive=archives.get(RuntimeDomain.EXECUTION),
            recovery_archive=archives.get(RuntimeDomain.RECOVERY),
            conversation_retention=plan.route(RuntimeDomain.CONVERSATION).retention,
            execution_retention=plan.route(RuntimeDomain.EXECUTION).retention,
            recovery_retention=plan.route(RuntimeDomain.RECOVERY).retention,
        )
        await steps.initialize()
        retention = RuntimeRetentionController(domain, steps, plan, namespace=namespace)
        actions: "list[Callable[[], Awaitable[None]]]" = [steps.preflight_close, retention.close, steps.close]
        actions.extend(_close_once(resource) for resource in (*reversed(runtimes), *reversed(contexts)))
        _logger.info("runtime state materialized: namespace=%s domains=%s", namespace, sorted(domain.value for domain in RuntimeDomain))
        return domain, steps, retention, tuple(actions)
    except BaseException:
        for resource in reversed(runtimes):
            try:
                await resource.close()
            except BaseException:
                _logger.exception("runtime state resource cleanup failed")
        for context in reversed(contexts):
            try:
                await context.close()
            except BaseException:
                _logger.exception("runtime SQL context cleanup failed")
        raise


def _combine_domain_states(namespace: str, resources: "dict[RuntimeDomain, object]") -> RuntimeDomainStates:
    values = {domain: resource.persistence for domain, resource in resources.items()}
    conversation = values[RuntimeDomain.CONVERSATION].conversation
    execution = values[RuntimeDomain.EXECUTION].execution
    memory = values[RuntimeDomain.MEMORY].memory
    artifact = values[RuntimeDomain.ARTIFACT].artifact
    task = values[RuntimeDomain.TASK].task
    evaluation = values[RuntimeDomain.EVALUATION].evaluation
    recovery = values[RuntimeDomain.RECOVERY].recovery
    return RuntimeDomainStates(namespace, conversation, execution, memory, artifact, task, evaluation, recovery)


def _runtime_object_store(runtime: object, domain: RuntimeDomain) -> "ObjectStore":
    return runtime.persistence.object_store(domain)


async def _build_step_archives(
    plan: RuntimeStatePlan,
    resources: "dict[RuntimeDomain, object]",
    object_stores: "dict[RuntimeDomain, ObjectStore]",
    *,
    namespace: str,
    tenant_id: str,
) -> "dict[RuntimeDomain, object]":
    archives: "dict[RuntimeDomain, object]" = {}
    for domain in _STEP_DOMAINS:
        route = plan.route(domain)
        if route.retention is RuntimeRetentionMode.TRANSIENT and domain is not RuntimeDomain.CONVERSATION:
            continue
        runtime = resources[domain]
        if route.kind == "filesystem":
            archives[domain] = FilesystemStepArchive.from_runtime(str(route.path), namespace=namespace, tenant_id=tenant_id, runtime_domain=domain, object_store=object_stores[domain], writer_lock=runtime.writer_lock)
        elif route.kind in {"sqlite", "sql"}:
            context = _find_context(runtime)
            archives[domain] = SqlStepArchive.from_runtime(context.engine, namespace=namespace, tenant_id=tenant_id, runtime_domain=domain, object_store=object_stores[domain], context=context)
        else:
            archives[domain] = InMemoryStepArchive(domain)
    return archives


def _find_context(runtime: object) -> SqlStorageContext:
    return runtime.context


def _close_once(resource: object) -> "Callable[[], Awaitable[None]]":
    async def close() -> None:
        await resource.close()

    return close


__all__ = ["materialize_runtime_state"]
