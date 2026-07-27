#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy table models for the storage.object kernel.

A private ``Base`` (unrelated to any domain schema). ``key_hash`` (not ``key``
itself) carries the uniqueness constraint for the same reason the legacy asset
table hashes its path: a long key under a multi-byte charset can exceed
MySQL's index key-length limit. Enterprise-DB conformance (spec D1-D8): every
table has a surrogate ``BigInteger`` ``id`` PK, ``ai_`` table prefix, built-in
``created_at``/``updated_at`` via ``TimestampMixin``, and ``uk_``/``ix_``
short index names."""

from datetime import datetime

from sqlalchemy import BINARY, DateTime, Index, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ...sqlalchemy.conventions import (
    BIGSERIAL,
    TABLE_PREFIX,
    TimestampMixin,
    timestamp_indexes,
)


class Base(DeclarativeBase):
    pass


# Unique-constraint names (short, per the enterprise index-naming rule).
STORAGE_OBJECT_KEY_CONSTRAINT = "uk_key_hash"
STORAGE_OBJECT_IDEMPOTENCY_CONSTRAINT = "uk_key_hash"
STORAGE_OBJECT_VERSION_CONSTRAINT = "uk_key_hash_version"


class StorageObjectRow(TimestampMixin, Base):
    """Current-state row per key. A tombstone is a live row with
    ``tombstone=True`` (mirrors the Memory/Filesystem backends: no separate
    whiteout table), so the unique key constraint naturally covers both live
    and deleted state."""

    __tablename__ = f"{TABLE_PREFIX}storage_objects"
    __table_args__ = (
        UniqueConstraint("key_hash", name=STORAGE_OBJECT_KEY_CONSTRAINT),
        # A MySQL deployment prefix-lengths `key` in this index (see
        # migrations/init_schema.sql) -- kept out of the vendor-neutral core.
        Index("ix_key", "key"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(1024))
    key_hash: Mapped[bytes] = mapped_column(BINARY(32))
    etag: Mapped[str] = mapped_column(String(64))
    version: Mapped[int]
    content_type: Mapped["str | None"] = mapped_column(String(255), nullable=True)
    size: Mapped[int]
    content: Mapped[bytes] = mapped_column(LargeBinary)
    modified_at: Mapped[datetime] = mapped_column(DateTime)
    metadata_json: Mapped[str] = mapped_column(Text)
    tombstone: Mapped[bool]
    commit_revision: Mapped[int]


class StorageObjectVersionRow(TimestampMixin, Base):
    """Append-only per-key history. Never updated or deleted -- one row per
    version, including tombstone versions -- so history is intrinsic rather
    than reconstructed."""

    __tablename__ = f"{TABLE_PREFIX}storage_object_versions"
    __table_args__ = (
        UniqueConstraint("key_hash", "version", name=STORAGE_OBJECT_VERSION_CONSTRAINT),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(1024))
    key_hash: Mapped[bytes] = mapped_column(BINARY(32))
    version: Mapped[int]
    etag: Mapped[str] = mapped_column(String(64))
    content_type: Mapped["str | None"] = mapped_column(String(255), nullable=True)
    size: Mapped[int]
    content: Mapped["bytes | None"] = mapped_column(LargeBinary, nullable=True)
    modified_at: Mapped[datetime] = mapped_column(DateTime)
    metadata_json: Mapped[str] = mapped_column(Text)
    tombstone: Mapped[bool]
    commit_revision: Mapped[int]


class StorageObjectIdempotencyRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}storage_object_idempotency"
    __table_args__ = (
        UniqueConstraint("key_hash", name=STORAGE_OBJECT_IDEMPOTENCY_CONSTRAINT),
        # A MySQL deployment prefix-lengths `key` in this index (see
        # migrations/init_schema.sql) -- kept out of the vendor-neutral core.
        Index("ix_key", "key"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    key_hash: Mapped[bytes] = mapped_column(BINARY(32))
    key: Mapped[str] = mapped_column(String(1024))
    request_hash: Mapped[str] = mapped_column(String(64))
    operation: Mapped[str] = mapped_column(String(32))
    result_key_hash: Mapped["bytes | None"] = mapped_column(BINARY(32), nullable=True)
    result_key: Mapped["str | None"] = mapped_column(String(1024), nullable=True)
    result_version: Mapped["int | None"] = mapped_column(nullable=True)
    commit_revision: Mapped[int] = mapped_column(default=0)
    result_json: Mapped["str | None"] = mapped_column(Text, nullable=True)


class StorageObjectRevisionRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}storage_object_revision"
    __table_args__ = (Index("ix_value", "value"), *timestamp_indexes())

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    value: Mapped[int]


class StorageSchemaVersionRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}storage_schema_version"
    __table_args__ = (UniqueConstraint("component", name="uk_component"), *timestamp_indexes())

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    component: Mapped[str] = mapped_column(String(128))
    version: Mapped[int]
    checksum: Mapped[str] = mapped_column(String(128))


__all__: "list[str]" = [
    "Base",
    "StorageObjectRow",
    "StorageObjectVersionRow",
    "StorageObjectIdempotencyRow",
    "StorageObjectRevisionRow",
    "STORAGE_OBJECT_KEY_CONSTRAINT",
    "STORAGE_OBJECT_IDEMPOTENCY_CONSTRAINT",
    "STORAGE_OBJECT_VERSION_CONSTRAINT",
    "StorageSchemaVersionRow",
]
