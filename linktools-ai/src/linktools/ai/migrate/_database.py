#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit Runtime and Asset schema provisioning."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from linktools.core import environ

from ..asset import build_asset_sql_metadata
from ..runtime.state import RuntimeDomain, build_runtime_sql_metadata
from ..storage import provision_sql, sql_integer_id

_logger = environ.get_logger("ai.migrate.database")

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ..storage import ObjectStore


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
    metadata = build_runtime_sql_metadata(
        selected,
        include_object_tables=object_store is None and bool(selected & frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY})),
    )
    if RuntimeDomain.TASK in selected:
        await _upgrade_runtime_task_nodes(engine)
    await provision_sql(engine, metadata)


async def _upgrade_runtime_task_nodes(engine: "AsyncEngine") -> None:
    """Add generic TaskNode payload columns when an existing table lacks them."""
    from sqlalchemy import JSON, inspect, text

    async with engine.begin() as connection:
        table_exists = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).has_table("runtime_task_nodes")
        )
        if not table_exists:
            return
        columns = await connection.run_sync(
            lambda sync_connection: {
                str(column["name"])
                for column in inspect(sync_connection).get_columns("runtime_task_nodes")
            }
        )
        additions = (("input_json", JSON()), ("budget_cost", sql_integer_id()))
        for name, column_type in additions:
            if name in columns:
                continue
            compiled_type = column_type.compile(dialect=connection.dialect)
            quoted_name = connection.dialect.identifier_preparer.quote(name)
            await connection.execute(
                text(f"ALTER TABLE runtime_task_nodes ADD COLUMN {quoted_name} {compiled_type}")
            )
            _logger.info("runtime task node column added: column=%s", name)


async def provision_asset_database(engine: "AsyncEngine") -> None:
    await provision_sql(engine, build_asset_sql_metadata())


__all__ = ["provision_asset_database", "provision_runtime_database"]
