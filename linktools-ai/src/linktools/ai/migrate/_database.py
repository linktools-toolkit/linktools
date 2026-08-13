#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit Runtime and Asset SQL provisioning orchestration."""

from typing import TYPE_CHECKING

from linktools.core import environ

from ..adapter import build_runtime_sql_metadata
from ..asset import build_asset_sql_metadata
from ..runtime import RuntimeStoragePlan
from ..storage import ObjectStore, provision_sql

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


_logger = environ.get_logger("ai.migrate")


async def provision_runtime_database(engine: "AsyncEngine", *, plan: RuntimeStoragePlan = RuntimeStoragePlan.all()) -> None:
    metadata = build_runtime_sql_metadata(plan)
    await provision_sql(engine, metadata)
    _logger.info("Runtime SQL schema provisioned: tables=%s", len(metadata.tables))


async def provision_asset_database(engine: "AsyncEngine", *, object_store: ObjectStore | None = None) -> None:
    metadata = build_asset_sql_metadata(object_store=object_store)
    await provision_sql(engine, metadata)
    _logger.info("Asset SQL schema provisioned: tables=%s", len(metadata.tables))


__all__ = ["provision_asset_database", "provision_runtime_database"]
