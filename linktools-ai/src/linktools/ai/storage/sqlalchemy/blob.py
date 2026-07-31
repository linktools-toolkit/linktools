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
    from collections.abc import Sequence

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


async def put_blobs(
    session: "AsyncSession",
    dialect: "SqlAlchemyDialect",
    model: "type[DeclarativeBase]",
    contents: "Sequence[bytes]",
) -> "list[str]":
    """Batch insert-ignore ``contents`` into ``model``'s table keyed by each
    content's sha256 hex; return the list of hexes (the ``object_id`` for each
    content, in input order). Identical content shares one row (dedup). One
    multi-row statement regardless of how many distinct contents are passed.
    Empty input returns ``[]`` with no SQL."""
    if not contents:
        return []
    object_ids = [hashlib.sha256(content).hexdigest() for content in contents]
    # Dedup by sha256 so a batch with repeated content writes each distinct
    # blob once; the unique constraint would no-op the dup anyway, but sending
    # only distinct rows keeps the statement smaller.
    distinct = {sha: content for sha, content in zip(object_ids, contents)}
    await dialect.insert_ignore_conflict_many(
        session,
        model=model,
        rows=[
            {"sha256": sha, "content": content}
            for sha, content in distinct.items()
        ],
        index_elements=("sha256",),
    )
    return object_ids


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


__all__ = ["put_blob", "put_blobs", "read_blob"]
