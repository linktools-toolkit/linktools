"""Explicit storage initialization -- the only construction-time I/O path.

``initialize_storage`` creates the schema (``Base.metadata.create_all``) and is
idempotent. Building a ``StorageDatabase`` does not touch the database; this
function does. It is the single place a fresh SQLite file or server schema is
brought into existence.
"""

from __future__ import annotations

from .database import StorageDatabase
from .sqlalchemy.base import Base


async def initialize_storage(storage: StorageDatabase) -> None:
    async with storage.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


__all__ = ["initialize_storage"]
