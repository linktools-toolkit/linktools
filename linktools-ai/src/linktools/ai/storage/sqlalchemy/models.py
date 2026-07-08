#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy table models for Resource storage. deleted_at/whiteout_version on
ResourceRow encode the whiteout tombstone in the same row/table as live resources,
rather than a separate whiteouts table, so the unique path constraint naturally
covers both live and deleted state."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, LargeBinary, String, Text, UniqueConstraint
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


class ToolIdempotencyRow(Base):
    """Persistent tool-call idempotency records (review doc §11). The unique
    constraint on (scope, key) is the natural primary key for the
    IdempotencyStore Protocol -- it backs ``reserve``'s "find-or-create"
    semantics (IntegrityError on the race -> SELECT the winner -> hash-check).
    Named ``ToolIdempotencyRow`` (not ``IdempotencyRow``) because that class
    name is already taken by resource-side idempotency above."""

    __tablename__ = "ai_idempotency"
    __table_args__ = (
        UniqueConstraint("scope", "key", name="uq_idempotency_scope_key"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    scope: Mapped[str] = mapped_column(String(128), index=True)
    key: Mapped[str] = mapped_column(String(512))
    request_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    result_json: Mapped["str | None"] = mapped_column(Text, nullable=True)
    error_text: Mapped["str | None"] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]
    completed_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    expires_at: Mapped["datetime | None"] = mapped_column(nullable=True)


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
    # G3/review3 §8.4: the uniqueness (and sequence-reservation) boundary is
    # the STREAM, not the run -- stream_id is a distinct column so a future
    # session/audit/root-run/swarm stream can coexist with a run's own stream
    # without colliding on (run_id, sequence). Every current caller still
    # passes stream_id == run_id, so this is a schema formalization, not a
    # behavior change.
    __table_args__ = (UniqueConstraint("stream_id", "sequence", name="uq_event_stream_sequence"),)

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    stream_id: Mapped[str] = mapped_column(String(128), index=True)
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
    # Phase-5A: child RunRecord id of the current/most-recent execution.
    # nullable for backward compat with rows written before this column existed.
    active_run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)


class SwarmTaskAttemptRow(Base):
    """One execution attempt of a SwarmTask (review doc §19.2). Mirrors the
    SwarmTaskAttempt domain model. Indexed on task_id for fast list_attempts."""

    __tablename__ = "ai_swarm_task_attempts"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), index=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    agent_id: Mapped[str] = mapped_column(String(128))
    attempt: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    error_json: Mapped["str | None"] = mapped_column(Text, nullable=True)


class MemoryRow(Base):
    __tablename__ = "ai_memories"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    content: Mapped[str] = mapped_column(Text)
    category: Mapped["str | None"] = mapped_column(String(64), nullable=True, index=True)
    confidence: Mapped["float | None"] = mapped_column(Float, nullable=True)
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    metadata_json: Mapped[str] = mapped_column(Text)


class ApprovalRow(Base):
    __tablename__ = "ai_approvals"
    # Package 3 (actionable-fix-spec §6): the database-level dedupe backstop.
    # (run_id, tool_call_id) IS the natural dedup key -- a pydantic-ai
    # tool_call_id is minted fresh per invocation, so it never needs to be
    # "released" after a terminal (approved/rejected) resolution for reuse by
    # a genuinely different approval. A plain UNIQUE constraint (no separate
    # dedupe_key column, no active/terminal partial-index games) is therefore
    # both simpler and sufficient: create_or_get_pending()'s SELECT-then-
    # INSERT is only a fast path -- this constraint is what actually prevents
    # two concurrent callers from ever committing two rows for the same key.
    __table_args__ = (
        UniqueConstraint("run_id", "tool_call_id", name="uq_approval_run_tool_call"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    tool_call_id: Mapped[str] = mapped_column(String(128))
    tool_name: Mapped[str] = mapped_column(String(255))
    reason: Mapped["str | None"] = mapped_column(Text, nullable=True)
    arguments_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    resolved_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    resolved_by: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text)
