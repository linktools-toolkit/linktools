#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit SQL profile composition and schema registration."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from linktools.core import environ

from ..adapter.schema import SqlRuntimeSchema, SqlRuntimeTables
from ..adapter.tool import SqlToolState
from ..asset.sql import SqlAlchemyAssetBackend, SqlAssetTables
from ..storage import SqlSchemaManifest, SqlSchemaRegistry

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Table

_logger = environ.get_logger("ai.entry.sql")


@dataclass(frozen=True, slots=True)
class SqlSchemaAssembly:
    asset: SqlAssetTables
    runtime: SqlRuntimeTables
    tool: "Table"
    metadata: "MetaData"
    manifest: SqlSchemaManifest


def register_sql_schema() -> SqlSchemaAssembly:
    registry = SqlSchemaRegistry()
    asset = SqlAlchemyAssetBackend.register_schema(registry)
    runtime = SqlRuntimeSchema.register_schema(registry)
    tool = SqlToolState.register_schema(registry)
    manifest = registry.freeze()
    _logger.info("SQL schema frozen: tables=%s digest=%s", len(manifest.tables), manifest.digest)
    return SqlSchemaAssembly(asset, runtime, tool, registry.metadata, manifest)


__all__ = ["SqlSchemaAssembly", "register_sql_schema"]
