#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic content-addressed blob write/read helpers.

A backend whose ORM defines a content-addressed blob row (``sha256`` unique,
``content`` bytes) uses these to dedup identical bytes on write (insert-ignore
on the sha256 unique key) and to read a blob back by its sha256. The row class
is passed in -- the storage layer enforces the content-addressed convention but
owns no table; the business backend declares the row and its table.

``put_blob`` returns the sha256 hex it wrote. When the backend's content etag is
also ``sha256(content).hexdigest()``, the returned hex equals that etag, so a
caller can store it as a history pointer (``object_id``) and have
``object_id == info.etag`` hold for free."""


from typing import TYPE_CHECKING
import hashlib
from sqlalchemy import select

if TYPE_CHECKING:
    from .dialects import SqlAlchemyDialect
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import DeclarativeBase


async def put_blob(
    session: "AsyncSession",
    dialect: "SqlAlchemyDialect",
    model: "type[DeclarativeBase]",
    content: bytes,
) -> str:
    """Insert-ignore ``content`` into ``model``'s table keyed by its sha256 hex;
    return the hex. Identical content shares one row (dedup). The returned hex
    is the caller's ``object_id``."""
    sha256 = hashlib.sha256(content).hexdigest()
    await dialect.insert_ignore_conflict(
        session,
        model=model,
        values={"sha256": sha256, "content": content},
        index_elements=("sha256",),
    )
    return sha256


async def read_blob(
    session: "AsyncSession",
    model: "type[DeclarativeBase]",
    object_id: str,
) -> "bytes | None":
    """Return the content of the blob addressed by ``object_id`` (sha256 hex), or
    ``None`` when no such row exists. The caller decides whether ``None`` is
    corruption (a missing blob referenced by a change row) or a normal miss."""
    row = await session.scalar(select(model).where(model.sha256 == object_id))
    return None if row is None else row.content


__all__ = ["put_blob", "read_blob"]
