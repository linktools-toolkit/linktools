#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test-owned SQL dependency composition."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path

from linktools.ai import RuntimePersistenceConfig
from linktools.ai.adapter import (
    SqlRuntimeSchema,
    SqlStepStore,
    open_sql_runtime,
)
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeBackend, RuntimePersistence
from linktools.ai.storage import (
    SqlSchemaRegistry,
    build_sqlite_storage,
    build_storage,
    initialize_storage,
)
from linktools.ai.workspace import Workspace, open_workspace_runtime
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepStore
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    backend: RuntimeBackend
    namespace: str
    domain: RuntimePersistence
    steps: StepStore


@asynccontextmanager
async def open_sql_resources(config: RuntimePersistenceConfig, *, connection_url: str | None = None) -> "AsyncIterator[RuntimeResources]":
    location = str(config.location)
    url = f"sqlite+aiosqlite:///{location}" if config.backend is RuntimeBackend.SQLITE else connection_url
    if url is None:
        raise ValueError("SQL test resources require a connection URL")
    engine: AsyncEngine = create_async_engine(url)
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await _provision_sql_schema(engine, config)
        registry = SqlSchemaRegistry()
        tables = SqlRuntimeSchema.register_schema(registry)
        manifest = registry.freeze()
        database = await (
            build_sqlite_storage(session_factory=session_factory, metadata=registry.metadata, schema_manifest_digest=manifest.digest)
            if config.backend is RuntimeBackend.SQLITE
            else build_storage(session_factory=session_factory, metadata=registry.metadata, schema_manifest_digest=manifest.digest)
        )
        await initialize_storage(database)
        domain = await open_sql_runtime(
            database,
            session_factory=session_factory,
            backend=config.backend,
            namespace=config.namespace,
            deployment_id=config.deployment_id,
            tables=tables,
        )
        if config.backend is RuntimeBackend.SQLITE:
            step_store = SqliteStepStore(database=_step_db_path(str(config.location), config.namespace))
        else:
            step_store = SqlStepStore(database, session_factory, config.namespace)
            await step_store.initialize()
        try:
            yield RuntimeResources(config.backend, config.namespace, domain, step_store)
        finally:
            if isinstance(step_store, SqlStepStore):
                await step_store.close()
    finally:
        await engine.dispose()


@asynccontextmanager
async def _open_sql_workspace(
    workspace: Workspace,
    config: RuntimePersistenceConfig,
    *,
    runner: object,
) -> AsyncIterator[object]:
    url = f"sqlite+aiosqlite:///{config.location}" if config.backend is RuntimeBackend.SQLITE else str(config.location)
    engine = create_async_engine(url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await _provision_sql_schema(engine, config)
        async with open_workspace_runtime(
            workspace,
            config=config,
            task_node_runner=runner,
            session_factory=session_factory,
        ) as runtime:
            yield runtime
    finally:
        await engine.dispose()


async def _provision_sql_schema(engine: AsyncEngine, config: RuntimePersistenceConfig) -> None:
    if config.backend is RuntimeBackend.SQLITE:
        registry = SqlSchemaRegistry()
        SqlRuntimeSchema.register_schema(registry)
        async with engine.begin() as connection:
            await connection.run_sync(registry.metadata.create_all)
    else:
        await provision_database(engine)


def _step_db_path(runtime_path: str, namespace: str) -> Path:
    path = Path(runtime_path).expanduser().resolve()
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    return path.with_name(f"{path.name}.steps.{digest}.db")
