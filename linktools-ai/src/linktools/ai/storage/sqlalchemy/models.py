#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy table models for Resource storage. deleted_at/whiteout_version on
ResourceRow encode the whiteout tombstone in the same row/table as live resources,
rather than a separate whiteouts table, so the unique path constraint naturally
covers both live and deleted state."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ResourceRow(Base):
    __tablename__ = "ai_resources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(32))
    etag: Mapped[str] = mapped_column(String(64))
    version: Mapped[int]
    content_type: Mapped["str | None"] = mapped_column(String(255), nullable=True)
    size: Mapped[int]
    content: Mapped[bytes] = mapped_column(LargeBinary)
    modified_at: Mapped[datetime]
    metadata_json: Mapped[str] = mapped_column(Text)
    deleted_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    whiteout_version: Mapped["int | None"] = mapped_column(nullable=True)


class IdempotencyRow(Base):
    __tablename__ = "ai_resource_idempotency"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    result_json: Mapped["str | None"] = mapped_column(Text, nullable=True)


class RevisionRow(Base):
    __tablename__ = "ai_resource_revision"

    id: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[int]


class RunRow(Base):
    __tablename__ = "ai_runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    root_run_id: Mapped[str] = mapped_column(String(128), index=True)
    parent_run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    runnable_id: Mapped[str] = mapped_column(String(255))
    runnable_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    input_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped["str | None"] = mapped_column(Text, nullable=True)
    error_json: Mapped["str | None"] = mapped_column(Text, nullable=True)
    version: Mapped[int]
    created_at: Mapped[datetime]
    started_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    finished_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text)


class RunCheckpointRow(Base):
    __tablename__ = "ai_run_checkpoints"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_run_checkpoint_run_sequence"),)

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    sequence: Mapped[int]
    format: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[int]
    payload: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime]
    metadata_json: Mapped[str] = mapped_column(Text)


class SessionRow(Base):
    __tablename__ = "ai_sessions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    parent_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    metadata_json: Mapped[str] = mapped_column(Text)


class SessionMessageRow(Base):
    __tablename__ = "ai_session_messages"
    __table_args__ = (UniqueConstraint("session_id", "sequence", name="uq_session_message_session_sequence"),)

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    sequence: Mapped[int]
    role: Mapped[str] = mapped_column(String(32))
    content_json: Mapped[str] = mapped_column(Text)
    run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime]
    metadata_json: Mapped[str] = mapped_column(Text)


class EventRow(Base):
    __tablename__ = "ai_events"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_event_run_sequence"),)

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    sequence: Mapped[int]
    occurred_at: Mapped[datetime]
    root_run_id: Mapped[str] = mapped_column(String(128))
    parent_run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    session_id: Mapped[str] = mapped_column(String(128))
    runnable_id: Mapped[str] = mapped_column(String(255))
    payload_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text)


class SwarmRunRow(Base):
    __tablename__ = "ai_swarm_runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    round: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    total_cost: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    metadata_json: Mapped[str] = mapped_column(Text)


class SwarmTaskRow(Base):
    __tablename__ = "ai_swarm_tasks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    swarm_run_id: Mapped[str] = mapped_column(String(128), index=True)
    parent_task_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    assigned_agent_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    dependencies_json: Mapped[str] = mapped_column(Text)
    input_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped["str | None"] = mapped_column(Text, nullable=True)
    error_json: Mapped["str | None"] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer)
    claimed_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    lease_expires_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
