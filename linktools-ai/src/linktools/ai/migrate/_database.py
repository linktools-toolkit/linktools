#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and provision the database schema owned by linktools-ai."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from linktools.core import environ

from ..adapter import SqlRuntimeSchema, build_step_schema
from ..asset import SqlAssetBackend
from ..storage import SqlSchemaRegistry

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine


_logger = environ.get_logger("ai.migrate")


@dataclass(frozen=True, slots=True)
class _SchemaGroup:
    owner: str
    metadata: "MetaData"


async def provision_database(engine: "AsyncEngine") -> None:
    """Create all current database tables for an explicit deployment step."""
    for group in _schema_groups():
        await _provision_schema(engine, group.metadata)
        _logger.info(
            "database schema provisioned: owner=%s table_count=%s",
            group.owner,
            len(group.metadata.tables),
        )


def _schema_groups() -> "tuple[_SchemaGroup, ...]":
    runtime_registry = SqlSchemaRegistry()
    SqlRuntimeSchema.register_schema(runtime_registry)
    asset_registry = SqlSchemaRegistry()
    SqlAssetBackend.register_schema(asset_registry)
    return (
        _SchemaGroup("adapter.sql", runtime_registry.metadata),
        _SchemaGroup("adapter._step", build_step_schema()),
        _SchemaGroup("asset.sql", asset_registry.metadata),
    )


async def _provision_schema(engine: "AsyncEngine", metadata: "MetaData") -> None:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)


__all__ = ["provision_database"]
