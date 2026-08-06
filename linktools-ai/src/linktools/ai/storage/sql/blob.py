#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Content-addressed blob helpers shared by SQL-backed domain stores."""

import hashlib
from typing import TYPE_CHECKING

from sqlalchemy import select

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

    from .dialects import SqlAlchemyDialect


async def put_blob(
    session: "AsyncSession",
    dialect: "SqlAlchemyDialect",
    model: type,
    content: bytes,
) -> str:
    """Insert a blob if absent and return its SHA-256 object id."""
    object_id = hashlib.sha256(content).hexdigest()
    await dialect.insert_ignore_conflict(
        session,
        model=model,
        values={"sha256": object_id, "content": content},
        index_elements=("sha256",),
    )
    return object_id


async def put_blobs(
    session: "AsyncSession",
    dialect: "SqlAlchemyDialect",
    model: type,
    contents: "Iterable[bytes]",
) -> "list[str]":
    """Insert a batch of deduplicated blobs and return object ids in order."""
    values: "list[dict[str, object]]" = []
    object_ids: "list[str]" = []
    seen: "set[str]" = set()
    for content in contents:
        object_id = hashlib.sha256(content).hexdigest()
        object_ids.append(object_id)
        if object_id not in seen:
            values.append({"sha256": object_id, "content": content})
            seen.add(object_id)
    if values:
        await dialect.insert_ignore_conflict_many(
            session,
            model=model,
            rows=values,
            index_elements=("sha256",),
        )
    return object_ids


async def read_blob(session: "AsyncSession", model: type, object_id: str) -> "bytes | None":
    """Read a content-addressed blob by its object id."""
    row = await session.scalar(select(model.content).where(model.sha256 == object_id))
    return row


__all__ = ["put_blob", "put_blobs", "read_blob"]
