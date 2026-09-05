#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path-backed SQLite MetricStore using the shared SQL boundary."""

from __future__ import annotations

from pathlib import Path

from ..errors import AIError, ErrorCode
from ._sql import SqlMetricStore


class SQLiteMetricStore(SqlMetricStore):
    """SQLite metrics store with operation-scoped connections and no close API."""

    def __init__(self, path: str | Path) -> None:
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

            engine = create_async_engine(
                URL.create("sqlite+aiosqlite", database=database),
                poolclass=NullPool,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise AIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING) from error
        super().__init__(engine)


__all__ = ["SQLiteMetricStore"]
