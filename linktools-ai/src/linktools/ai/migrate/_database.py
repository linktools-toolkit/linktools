#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and provision the database schema owned by linktools-ai."""

from typing import TYPE_CHECKING

from linktools.core import environ

from ..storage import build_sql_schema_metadata

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine


_logger = environ.get_logger("ai.migrate")


async def provision_database(engine: "AsyncEngine") -> None:
    """Create all current database tables for an explicit deployment step."""
    plan = build_schema_metadata()
    await _provision_schema(engine, plan[0])
    _logger.info("database schema provisioned: manifest=%s table_count=%s", plan[1], len(plan[0].tables))


def build_schema_metadata() -> "tuple[MetaData, str]":
    """Build the complete SQL metadata and its frozen manifest digest."""
    return build_sql_schema_metadata()


async def _provision_schema(engine: "AsyncEngine", metadata: "MetaData") -> None:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)


__all__ = ["build_schema_metadata", "provision_database"]
