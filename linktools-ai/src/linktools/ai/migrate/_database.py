#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit Runtime, Asset, and Metrics schema provisioning."""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.core import environ

from ..asset import build_asset_sql_metadata
from ..errors import AIError, ErrorCode
from ..observe import build_metrics_sql_metadata
from ..runtime.state import (
    RuntimeDomain,
    build_runtime_sql_metadata,
)
from ..storage import build_object_sql_metadata, provision_sql

_logger = environ.get_logger("ai.migrate.database")

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ..storage import ObjectStore


def build_sql_schema_metadata() -> "MetaData":
    """Build the complete Runtime, Object, Asset, and Metrics schema."""
    from sqlalchemy import MetaData

    metadata = MetaData()
    build_runtime_sql_metadata(frozenset(RuntimeDomain), metadata=metadata)
    build_object_sql_metadata(metadata=metadata)
    build_asset_sql_metadata(metadata=metadata)
    build_metrics_sql_metadata(metadata=metadata)
    if len(metadata.tables) != 12:
        raise RuntimeError("complete SQL schema must contain exactly 12 tables")
    return metadata


async def provision_database(engine: "AsyncEngine") -> None:
    """Provision the complete schema from the explicit migration boundary."""
    metadata = build_sql_schema_metadata()
    await provision_sql(engine, metadata)
    _logger.info("complete SQL schema provisioned: tables=%s", len(metadata.tables))


async def provision_runtime_database(
    engine: "AsyncEngine",
    *,
    domains: Iterable[RuntimeDomain] = tuple(RuntimeDomain),
    object_store: "ObjectStore | None" = None,
) -> None:
    selected = frozenset(domains)
    if not selected:
        raise ValueError("at least one RuntimeDomain is required")
    if not selected.issubset(frozenset(RuntimeDomain)):
        raise ValueError("domains must contain RuntimeDomain values")
    from sqlalchemy import MetaData

    metadata = MetaData()
    build_runtime_sql_metadata(selected, metadata=metadata)
    if object_store is None and selected & frozenset(
        {
            RuntimeDomain.CONVERSATION,
            RuntimeDomain.EXECUTION,
            RuntimeDomain.MEMORY,
            RuntimeDomain.ARTIFACT,
            RuntimeDomain.RECOVERY,
            RuntimeDomain.TASK,
        }
    ):
        build_object_sql_metadata(metadata=metadata)
    await provision_sql(engine, metadata)


async def provision_asset_database(engine: "AsyncEngine") -> None:
    from sqlalchemy import MetaData

    metadata = MetaData()
    build_asset_sql_metadata(metadata=metadata)
    build_object_sql_metadata(metadata=metadata)
    await provision_sql(engine, metadata)


async def provision_metrics_database(engine: "AsyncEngine") -> None:
    await provision_sql(engine, build_metrics_sql_metadata())


async def provision_metrics_sqlite(path: str | Path) -> None:
    """Provision a path-backed SQLite Metrics database without exposing an engine."""
    if (
        not isinstance(path, (str, Path))
        or not str(path).strip()
        or str(path) == ":memory:"
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    database = str(Path(path).expanduser().resolve(strict=False))
    try:
        from sqlalchemy.engine import URL
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool
    except (ImportError, ModuleNotFoundError) as error:
        raise AIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING) from error
    engine = create_async_engine(
        URL.create("sqlite+aiosqlite", database=database),
        poolclass=NullPool,
    )
    try:
        await provision_metrics_database(engine)
    finally:
        await engine.dispose()


__all__ = [
    "build_sql_schema_metadata",
    "provision_asset_database",
    "provision_database",
    "provision_metrics_database",
    "provision_metrics_sqlite",
    "provision_runtime_database",
]
