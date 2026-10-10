#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit Metrics schema provisioning and validation."""

from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ..observe import build_metrics_sql_metadata
from ..storage import provision_sql, validate_sql

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


async def provision_metrics_database(engine: "AsyncEngine") -> None:
    await provision_sql(engine, build_metrics_sql_metadata())


async def validate_metrics_database(engine: "AsyncEngine") -> None:
    await validate_sql(engine, build_metrics_sql_metadata())


def _metrics_sqlite_engine(path: str | Path) -> "AsyncEngine":
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
    return create_async_engine(
        URL.create("sqlite+aiosqlite", database=database),
        poolclass=NullPool,
    )


async def provision_metrics_sqlite(path: str | Path) -> None:
    """Provision a path-backed SQLite Metrics database without exposing an engine."""
    engine = _metrics_sqlite_engine(path)
    try:
        await provision_metrics_database(engine)
    finally:
        await engine.dispose()


async def validate_metrics_sqlite(path: str | Path) -> None:
    """Validate a path-backed SQLite Metrics database without mutating its schema."""
    engine = _metrics_sqlite_engine(path)
    try:
        await validate_metrics_database(engine)
    finally:
        await engine.dispose()



__all__ = [
    "provision_metrics_database",
    "provision_metrics_sqlite",
    "validate_metrics_database",
    "validate_metrics_sqlite",
]
