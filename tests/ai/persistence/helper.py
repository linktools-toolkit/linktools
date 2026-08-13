#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test-owned SQL dependency composition."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from linktools.ai import RuntimeStorage
from linktools.ai.adapter import SqlRuntimeSchema, SqlStepStore
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeStores
from linktools.ai.storage import build_sql_schema_metadata, create_sql_context
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
        context = create_sql_context(engine, namespace)
        metadata, digest = build_sql_schema_metadata()
        await context.initialize(metadata=metadata, schema_manifest_digest=digest)
        domain = await SqlRuntimeSchema._open_sql_runtime(context, persist=storage.persist)
        await domain.initialize()
        step_store = SqlStepStore._from_context(context)
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
