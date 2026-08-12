#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test-owned SQL dependency composition."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from linktools.ai import RuntimeStorage
from linktools.ai.adapter import SqlStepStore, open_sql_runtime
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeStores
from linktools.ai.storage import create_sql_storage_context
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
    location = None if storage.location is None else str(storage.location)
    url = f"sqlite+aiosqlite:///{location}" if storage.target_kind == "sqlite" else connection_url
    if url is None:
        raise ValueError("SQL test resources require a connection URL")
    engine: AsyncEngine = create_async_engine(url)
    try:
        await provision_database(engine)
        context = create_sql_storage_context(engine, namespace)
        await context.initialize()
        domain = await open_sql_runtime(engine, namespace=namespace, persist=storage.persist)
        await domain.initialize()
        step_store = SqlStepStore(engine, namespace=namespace)
        await step_store.initialize()
        try:
            yield RuntimeResources(storage.target_kind, namespace, domain, step_store)
        finally:
            await step_store.close()
            await domain.close()
            await context.close()
    finally:
        await engine.dispose()

@asynccontextmanager
async def _open_sql_workspace(
    workspace: Workspace,
    storage: RuntimeStorage,
    *,
    runner: object,
) -> AsyncIterator[object]:
    url = f"sqlite+aiosqlite:///{storage.location}" if storage.target_kind == "sqlite" else str(storage.location)
    engine = create_async_engine(url)
    try:
        await provision_database(engine)
        async with open_workspace_runtime(
            workspace,
            storage=RuntimeStorage.sql(engine, persist=storage.persist),
            task_node_runner=runner,
        ) as runtime:
            yield runtime
    finally:
        await engine.dispose()
