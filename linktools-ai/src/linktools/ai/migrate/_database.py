#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit Runtime and Asset schema provisioning."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..asset import build_asset_sql_metadata
from ..runtime.state import RuntimeDomain, build_runtime_sql_metadata
from ..storage import provision_sql

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
    await provision_sql(engine, metadata)


async def provision_asset_database(engine: "AsyncEngine") -> None:
    await provision_sql(engine, build_asset_sql_metadata())


__all__ = ["provision_asset_database", "provision_runtime_database"]
