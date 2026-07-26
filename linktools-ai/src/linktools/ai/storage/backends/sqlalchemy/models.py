#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy table models for the storage.object kernel.

A private ``Base`` (not the legacy asset/run/event metadata) -- the object
kernel's tables are independent and unrelated to any domain schema. ``key_hash``
(not ``key`` itself) carries the uniqueness constraint for the same reason the
legacy asset table hashes its path: a long key under a multi-byte charset can
exceed MySQL's index key-length limit."""

from datetime import datetime

from sqlalchemy import Index, LargeBinary, String, Text, UniqueConstraint, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


STORAGE_OBJECT_KEY_CONSTRAINT = "uq_storage_objects_key_hash"
STORAGE_OBJECT_IDEMPOTENCY_CONSTRAINT = "uq_storage_object_idempotency_key_hash"
STORAGE_OBJECT_VERSION_CONSTRAINT = "uq_storage_object_versions_key_hash_version"


class StorageObjectRow(Base):
    """Current-state row per key. A tombstone is a live row with
    ``tombstone=True`` (mirrors the Memory/Filesystem backends: no separate
    whiteout table), so the unique key constraint naturally covers both live
    and deleted state."""

    __tablename__ = "storage_objects"
    __table_args__ = (
        UniqueConstraint("key_hash", name=STORAGE_OBJECT_KEY_CONSTRAINT),
        Index("ix_storage_objects_key_hash", "key_hash"),
        Index("ix_storage_objects_key_prefix", "key", mysql_length={"key": 191}),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(1024))
    key_hash: Mapped[bytes] = mapped_column(LargeBinary(32))
    etag: Mapped[str] = mapped_column(String(64))
    version: Mapped[int]
    content_type: Mapped["str | None"] = mapped_column(String(255), nullable=True)
    size: Mapped[int]
    content: Mapped[bytes] = mapped_column(LargeBinary)
    modified_at: Mapped[datetime]
    metadata_json: Mapped[str] = mapped_column(Text)
    tombstone: Mapped[bool]
    commit_revision: Mapped[int]


class StorageObjectVersionRow(Base):
    """Append-only per-key history. Never updated or deleted -- one row per
    version, including tombstone versions -- so history is intrinsic rather
    than reconstructed."""

    __tablename__ = "storage_object_versions"
    __table_args__ = (
        UniqueConstraint(
            "key_hash", "version", name=STORAGE_OBJECT_VERSION_CONSTRAINT
        ),
        Index("ix_storage_object_versions_key_hash", "key_hash"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(1024))
    key_hash: Mapped[bytes] = mapped_column(LargeBinary(32))
    version: Mapped[int]
    etag: Mapped[str] = mapped_column(String(64))
    content_type: Mapped["str | None"] = mapped_column(String(255), nullable=True)
    size: Mapped[int]
    content: Mapped["bytes | None"] = mapped_column(LargeBinary, nullable=True)
    modified_at: Mapped[datetime]
    metadata_json: Mapped[str] = mapped_column(Text)
    tombstone: Mapped[bool]
    commit_revision: Mapped[int]


class StorageObjectIdempotencyRow(Base):
    __tablename__ = "storage_object_idempotency"
    __table_args__ = (
        UniqueConstraint(
            "key_hash", name=STORAGE_OBJECT_IDEMPOTENCY_CONSTRAINT
        ),
        Index("ix_storage_object_idempotency_key_hash", "key_hash"),
        Index(
            "ix_storage_object_idempotency_key_prefix",
            "key",
            mysql_length={"key": 191},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key_hash: Mapped[bytes] = mapped_column(LargeBinary(32))
    key: Mapped[str] = mapped_column(String(1024))
    request_hash: Mapped[str] = mapped_column(String(64))
    operation: Mapped[str] = mapped_column(String(32))
    result_key_hash: Mapped["bytes | None"] = mapped_column(LargeBinary(32), nullable=True)
    result_key: Mapped["str | None"] = mapped_column(String(1024), nullable=True)
    result_version: Mapped["int | None"] = mapped_column(nullable=True)
    commit_revision: Mapped[int] = mapped_column(default=0)
    result_json: Mapped["str | None"] = mapped_column(Text, nullable=True)


class StorageObjectRevisionRow(Base):
    __tablename__ = "storage_object_revision"

    id: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[int]


class StorageSchemaVersionRow(Base):
    __tablename__ = "storage_schema_version"
    component: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[int]
    checksum: Mapped[str] = mapped_column(String(128))
    updated_at: Mapped[datetime] = mapped_column(DateTime)


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
