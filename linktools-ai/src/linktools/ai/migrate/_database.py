#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit Runtime, Asset, and Metrics schema provisioning."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from linktools.core import environ

from ..asset import build_asset_sql_metadata
from ..observe import build_metrics_sql_metadata
from ..persistence import PersistenceDomain
from ..persistence.objects import runtime_domain_uses_object_store
from ..persistence.schema import build_runtime_sql_metadata, build_step_sql_metadata, build_step_domain_sql_metadata
from ..storage import SqlStorageContext, build_object_sql_metadata, provision_sql, validate_sql

_logger = environ.get_logger("ai.migrate.database")

# Bump when the complete standard SQL schema changes. This is not a data-wire
# version and is used only by an explicitly acquired local SQLite writer.
_SQLITE_SCHEMA_VERSION = 1

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from ..storage import ObjectStore


def build_sql_schema_metadata() -> "MetaData":
    """Build the complete Runtime, Object, Asset, and Metrics schema."""
    from sqlalchemy import MetaData

    metadata = MetaData()
    build_runtime_sql_metadata(frozenset(PersistenceDomain), metadata=metadata)
    build_step_sql_metadata(metadata=metadata)
    build_object_sql_metadata(metadata=metadata)
    build_asset_sql_metadata(metadata=metadata)
    build_metrics_sql_metadata(metadata=metadata)
    return metadata


async def provision_database(engine: "AsyncEngine | SqlStorageContext") -> None:
    """Provision the complete schema; an owned SQLite context may cache validation.

    Only a context holding the local SQLite write permission owns user_version.
    Bare engines, shared contexts and partial-schema APIs never read or write it.
    The same version trusts the deployed schema; use validate_sql explicitly to
    diagnose external edits made without changing the schema version.
    """
    metadata = build_sql_schema_metadata()
    if isinstance(engine, SqlStorageContext):
        context = engine
        await context.initialize()
        await context.require_write_access()
        if context.exclusive_writer:
            async def provision_owned(session: "AsyncSession") -> None:
                connection = await session.connection()
                version = (await connection.exec_driver_sql("PRAGMA main.user_version")).scalar_one()
                if version == _SQLITE_SCHEMA_VERSION:
                    return
                await connection.run_sync(metadata.create_all)
                await validate_sql(connection, metadata)
                await connection.exec_driver_sql(f"PRAGMA main.user_version={_SQLITE_SCHEMA_VERSION}")

            await context.run_mutation(provision_owned, domain="schema")
            return
        engine = context.engine
    await provision_sql(engine, metadata)
    _logger.info("complete SQL schema provisioned: tables=%s", len(metadata.tables))


async def provision_runtime_database(
    engine: "AsyncEngine",
    *,
    domains: Iterable[PersistenceDomain] = tuple(PersistenceDomain),
    object_store: "ObjectStore | None" = None,
) -> None:
    selected = frozenset(domains)
    if not selected:
        raise ValueError("at least one PersistenceDomain is required")
    if not selected.issubset(frozenset(PersistenceDomain)):
        raise ValueError("domains must contain PersistenceDomain values")
    from sqlalchemy import MetaData

    metadata = MetaData()
    build_step_domain_sql_metadata(selected, metadata=metadata)
    build_runtime_sql_metadata(selected, metadata=metadata)
    if object_store is None and any(
        runtime_domain_uses_object_store(domain) for domain in selected
    ):
        build_object_sql_metadata(metadata=metadata)
    await provision_sql(engine, metadata)


async def provision_asset_database(engine: "AsyncEngine") -> None:
    from sqlalchemy import MetaData

    metadata = MetaData()
    build_asset_sql_metadata(metadata=metadata)
    build_object_sql_metadata(metadata=metadata)
    await provision_sql(engine, metadata)


__all__ = [
    "build_sql_schema_metadata",
    "provision_asset_database",
    "provision_database",
    "provision_runtime_database",
]
