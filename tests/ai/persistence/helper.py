#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test-owned SQL dependency composition."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from linktools.ai import RuntimeStorage, RuntimeStoragePlan
from linktools.ai.adapter import open_runtime_persistence
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import RuntimeStores
from linktools.ai.workspace import Workspace, open_workspace_runtime
from pydantic_ai_harness.step_persistence import StepStore
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    backend: str
    namespace: str
    domain: RuntimeStores
    steps: StepStore


@asynccontextmanager
async def open_sql_resources(storage: RuntimeStorage, *, connection_url: str | None = None, namespace: str = "test") -> "AsyncIterator[RuntimeResources]":
    if storage.target_kind == "sqlite":
        async with open_runtime_persistence(storage, namespace=namespace, tenant_id=namespace) as persistence:
            yield RuntimeResources("sqlite", namespace, persistence.domain, persistence.steps)
        return
    if connection_url is None:
        raise ValueError("SQL test resources require a connection URL")
    engine: AsyncEngine = create_async_engine(connection_url)
    plan = storage.plan if storage.target_kind == "sql" else RuntimeStoragePlan.all()
    try:
        await provision_runtime_database(engine, plan=plan)
        sql_storage = RuntimeStorage.sql(engine, plan=plan)
        async with open_runtime_persistence(sql_storage, namespace=namespace, tenant_id=namespace) as persistence:
            yield RuntimeResources("sql", namespace, persistence.domain, persistence.steps)
    finally:
        await engine.dispose()

@asynccontextmanager
async def _open_sql_workspace(
    workspace: Workspace,
    storage: RuntimeStorage,
    *,
    runner: object,
) -> AsyncIterator[object]:
    if storage.target_kind == "sqlite":
        if storage.target_path is None:
            raise ValueError("SQLite workspace storage requires a path")
        url = f"sqlite+aiosqlite:///{storage.target_path}"
    elif storage.target_engine is not None:
        url = str(storage.target_engine.url)
    else:
        raise ValueError("SQL workspace storage requires an engine")
    engine = create_async_engine(url)
    try:
        async with open_workspace_runtime(
            workspace,
            runtime_storage=RuntimeStorage.sql(engine, plan=storage.plan),
            task_node_runner=runner,
        ) as runtime:
            yield runtime
    finally:
        await engine.dispose()
