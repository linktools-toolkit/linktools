#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit SQL profile composition and schema registration."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from linktools.core import environ


if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from ..adapter.schema import SqlRuntimeTables
    from ..asset.sql import SqlAssetTables
    from ..storage import SqlSchemaManifest

_logger = environ.get_logger("ai.entry.sql")


@dataclass(frozen=True, slots=True)
class SqlSchemaAssembly:
    asset: "SqlAssetTables"
    runtime: "SqlRuntimeTables"
    metadata: "MetaData"
    manifest: "SqlSchemaManifest"


def register_sql_schema() -> SqlSchemaAssembly:
    from ..adapter.schema import SqlRuntimeSchema
    from ..asset.sql import SqlAlchemyAssetBackend
    from ..storage import SqlSchemaRegistry

    registry = SqlSchemaRegistry()
    asset = SqlAlchemyAssetBackend.register_schema(registry)
    runtime = SqlRuntimeSchema.register_schema(registry)
    manifest = registry.freeze()
    _logger.info("SQL schema frozen: tables=%s digest=%s", len(manifest.tables), manifest.digest)
    return SqlSchemaAssembly(asset, runtime, registry.metadata, manifest)


__all__ = ["SqlSchemaAssembly", "register_sql_schema"]
