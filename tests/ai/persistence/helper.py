#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test-owned SQL dependency composition."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from linktools.ai import (
    RuntimePersistenceConfig,
    RuntimeResources,
    open_runtime_resources,
)
from linktools.ai.app import open_workspace_runtime
from linktools.ai.runtime import RuntimeBackend
from linktools.ai.workspace import Workspace
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@asynccontextmanager
async def open_sql_resources(config: RuntimePersistenceConfig, *, connection_url: str | None = None) -> "AsyncIterator[RuntimeResources]":
    location = str(config.location)
    url = f"sqlite+aiosqlite:///{location}" if config.backend is RuntimeBackend.SQLITE else connection_url
    if url is None:
        raise ValueError("SQL test resources require a connection URL")
    engine: AsyncEngine = create_async_engine(url)
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with open_runtime_resources(config, engine=engine, session_factory=session_factory) as resources:
            yield resources
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
        async with open_workspace_runtime(
            workspace,
            config=config,
            runner=runner,
            engine=engine,
            session_factory=session_factory,
        ) as runtime:
            yield runtime
    finally:
        await engine.dispose()
