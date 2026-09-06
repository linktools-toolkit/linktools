#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Close-free SQLite MetricStore using operation-scoped engines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..core import Page
from ..errors import AIError, ErrorCode
from ..storage import validate_sql
from ._model import MetricDefinition, Observation
from ._sql import SqlMetricStore, build_metrics_sql_metadata

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class SQLiteMetricStore:
    """Path-backed MetricStore without an owned long-lived engine."""

    def __init__(self, path: str | Path) -> None:
        if (
            not isinstance(path, (str, Path))
            or not str(path).strip()
            or str(path) == ":memory:"
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        self._path = Path(path).expanduser().resolve(strict=False)
        self._schema_validated = False
        self._schema_validation_lock = asyncio.Lock()

    async def _ensure_schema(self, engine: "AsyncEngine") -> None:
        if self._schema_validated:
            return
        async with self._schema_validation_lock:
            if self._schema_validated:
                return
            await validate_sql(engine, build_metrics_sql_metadata())
            self._schema_validated = True

    @asynccontextmanager
    async def _store(self) -> AsyncIterator[SqlMetricStore]:
        try:
            from sqlalchemy.engine import URL
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy.pool import NullPool

            engine = create_async_engine(
                URL.create("sqlite+aiosqlite", database=str(self._path)),
                poolclass=NullPool,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise AIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING) from error
        try:
            await self._ensure_schema(engine)
            yield SqlMetricStore(engine, validate_schema=False)
        finally:
            await engine.dispose()

    async def put_definition(
        self,
        namespace: str,
        definition: MetricDefinition,
    ) -> MetricDefinition:
        async with self._store() as store:
            return await store.put_definition(namespace, definition)

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int,
    ) -> MetricDefinition | None:
        async with self._store() as store:
            return await store.get_definition(namespace, name, revision)

    async def latest_definition(
        self,
        namespace: str,
        name: str,
    ) -> MetricDefinition | None:
        async with self._store() as store:
            return await store.latest_definition(namespace, name)

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        async with self._store() as store:
            await store.put_observations(namespace, observations)

    async def get_observation(
        self,
        namespace: str,
        observation_id: str,
    ) -> Observation | None:
        async with self._store() as store:
            return await store.get_observation(namespace, observation_id)

    async def scan_observations(
        self,
        namespace: str,
        kind: str,
        start: datetime,
        end: datetime,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[Observation]:
        async with self._store() as store:
            return await store.scan_observations(
                namespace,
                kind,
                start,
                end,
                cursor=cursor,
                limit=limit,
            )

    async def prune_observations(
        self,
        namespace: str,
        *,
        before: datetime,
    ) -> int:
        async with self._store() as store:
            return await store.prune_observations(namespace, before=before)


__all__ = ["SQLiteMetricStore"]
