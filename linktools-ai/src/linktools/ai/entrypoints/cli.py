#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local-coding composition root with explicit SQLite initialization."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.core import environ

if TYPE_CHECKING:
    from ..storage.database import StorageDatabase

logger = environ.get_logger("ai.entrypoints.cli")


async def build_local_storage(path: "str | Path") -> "StorageDatabase":
    from ..storage.database import build_sqlite_storage
    from ..storage.initialization import initialize_storage

    storage = build_sqlite_storage(path)
    await initialize_storage(storage)
    logger.info("local storage initialized path=%s", path)
    return storage


def create_local_storage(path: "str | Path") -> "StorageDatabase":
    return asyncio.run(build_local_storage(path))


__all__ = ["build_local_storage", "create_local_storage"]
