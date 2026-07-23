#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy table models for Asset and reliable-task storage. deleted_at/whiteout_version on
AssetRow encode the whiteout tombstone in the same row/table as live assets,
rather than a separate whiteouts table, so the unique path constraint naturally
covers both live and deleted state."""

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# Stable, named unique-constraint identifiers so per-dialect integrity-error
# classifiers (storage/sqlalchemy/dialects/) can map a violation back to the
# asset path key vs. the idempotency key without parsing free-form messages.
ASSET_PATH_CONSTRAINT = "uq_ai_assets_tenant_path"
ASSET_IDEMPOTENCY_CONSTRAINT = "uq_ai_asset_idempotency_tenant_key"


class AssetRow(Base):
    __tablename__ = "ai_assets"
    __table_args__ = (UniqueConstraint("path", name=ASSET_PATH_CONSTRAINT),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(1024), index=True)
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


class AssetIdempotencyRow(Base):
    __tablename__ = "ai_asset_idempotency"
    __table_args__ = (UniqueConstraint("key", name=ASSET_IDEMPOTENCY_CONSTRAINT),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(512), index=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    result_json: Mapped["str | None"] = mapped_column(Text, nullable=True)


class ToolIdempotencyRow(Base):
    """Persistent tool-call idempotency records. The unique
    constraint on (scope, key) is the natural primary key for the
    IdempotencyStore Protocol -- it backs ``reserve``'s "find-or-create"
    semantics (IntegrityError on the race -> SELECT the winner -> hash-check).
    Named ``ToolIdempotencyRow`` (not ``AssetIdempotencyRow``) because that class
    name is already taken by asset-side idempotency above."""

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
    # Fencing fields for the claim/owner/generation/lease model.
    owner_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    generation: Mapped[int] = mapped_column(default=0)
    claimed_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    lease_expires_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    receipt_artifact_id: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    binding_fingerprint: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    result_processor_revision: Mapped["str | None"] = mapped_column(String(128), nullable=True)


class AssetRevisionRow(Base):
    __tablename__ = "ai_asset_revision"

    id: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[int]


class RunRow(Base):
    __tablename__ = "ai_runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    root_run_id: Mapped[str] = mapped_column(String(128), index=True)
    parent_run_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, index=True
    )
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
    # Cancel-request audit (nullable: absent on older rows and on runs never
    # cancelled).
    cancel_requested_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    cancel_requested_by: Mapped["str | None"] = mapped_column(
        String(128), nullable=True
    )
    worker_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    execution_token: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    heartbeat_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    manifest_id: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    resumability: Mapped["str | None"] = mapped_column(String(32), nullable=True)
    cancel_reason: Mapped["str | None"] = mapped_column(Text, nullable=True)


class RunCheckpointRow(Base):
    __tablename__ = "ai_run_checkpoints"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_run_checkpoint_run_sequence"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    sequence: Mapped[int]
    format: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[int]
    payload: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime]
    metadata_json: Mapped[str] = mapped_column(Text)


class RunCheckpointCounterRow(Base):
    """Per-run monotonic counter the Store increments inside the append
    transaction so concurrent appends for the same run never collide on
    sequence (the unique constraint on (run_id, sequence) is the backstop)."""

    __tablename__ = "ai_run_checkpoint_counters"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    last_sequence: Mapped[int]


class RunDefinitionRow(Base):
    """The immutable RunDefinitionSnapshot persisted at run creation so resume
    can restore the exact original spec + identity ()."""

    __tablename__ = "ai_run_definitions"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    runnable_type: Mapped[str] = mapped_column(String(32))
    runnable_id: Mapped[str] = mapped_column(String(255))
    serialized_spec_json: Mapped[str] = mapped_column(Text)
    spec_fingerprint: Mapped[str] = mapped_column(String(64))
    user_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    tenant_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    workspace: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime]
    manifest_json: Mapped["str | None"] = mapped_column(Text, nullable=True)
    resumability: Mapped["str | None"] = mapped_column(
        String(32), nullable=True
    )


class SessionRow(Base):
    __tablename__ = "ai_sessions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    parent_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    # Principal the session belongs to (). Nullable: legacy rows and
    # unowned (single-user CLI) sessions stay NULL. create_all adds the columns
    # for fresh databases; existing databases need an ALTER TABLE.
    user_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    tenant_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    metadata_json: Mapped[str] = mapped_column(Text)


class SessionMessageRow(Base):
    __tablename__ = "ai_session_messages"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "sequence", name="uq_session_message_session_sequence"
        ),
    )

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
    # the uniqueness (and sequence-reservation) boundary is
    # the STREAM, not the run -- stream_id is a distinct column so a future
    # session/audit/root-run/swarm stream can coexist with a run's own stream
    # without colliding on (run_id, sequence). Every current caller still
    # passes stream_id == run_id, so this is a schema formalization, not a
    # behavior change.
    __table_args__ = (
        UniqueConstraint("stream_id", "sequence", name="uq_event_stream_sequence"),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    stream_id: Mapped[str] = mapped_column(String(128), index=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    sequence: Mapped[int]
    occurred_at: Mapped[datetime]
    root_run_id: Mapped[str] = mapped_column(String(128))
    parent_run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    session_id: Mapped[str] = mapped_column(String(128))
    runnable_id: Mapped[str] = mapped_column(String(255))
    event_type: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    payload_json: Mapped[str] = mapped_column(Text)
    # Free-form per-event metadata (e.g. commit_id for commit-scoped dedup of
    # critical events). Nullable: rows written before this column existed (and
    # events with no metadata) store NULL -> read back as an empty mapping.
    metadata_json: Mapped["str | None"] = mapped_column(Text, nullable=True)


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
    # Child RunRecord id of the current/most-recent execution.
    # nullable for rows written before this column existed (data migration).
    active_run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)


class SwarmTaskAttemptRow(Base):
    """One execution attempt of a SwarmTask. Mirrors the
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

    # tenant_id is the hard isolation boundary. NULL is tolerated only for
    # legacy rows persisted before tenant-scoping: those rows are read back
    # with a synthesized legacy tenant and never match a real tenant's search
    # (NULL != 'tenant-a' in SQL), so old data is quarantined without a
    # migration script. The three composite indexes back the common scoped
    # search shapes (tenant+user / tenant+workspace / tenant+session).
    __table_args__ = (
        Index("ix_memory_tenant_user", "tenant_id", "user_id"),
        Index("ix_memory_tenant_workspace", "tenant_id", "workspace_id"),
        Index("ix_memory_tenant_session", "tenant_id", "session_id"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tenant_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    content: Mapped[str] = mapped_column(Text)
    category: Mapped["str | None"] = mapped_column(
        String(64), nullable=True, index=True
    )
    confidence: Mapped["float | None"] = mapped_column(Float, nullable=True)
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    metadata_json: Mapped[str] = mapped_column(Text)
    user_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    workspace_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    session_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)


class ApprovalRow(Base):
    __tablename__ = "ai_approvals"
    # the database-level dedupe backstop.
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
    # The REAL arguments are never persisted (may carry secrets). Only the
    # redacted audit copy + the identity hash are stored.
    redacted_arguments_json: Mapped[str] = mapped_column(Text)
    arguments_hash: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    resolved_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    resolved_by: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text)
    tenant_id: Mapped["str | None"] = mapped_column(String(128), nullable=True, index=True)
    descriptor_fingerprint: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    handler_revision: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    provider_revision: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    policy_revision: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    capability_revision: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)


# --- Reliable-task tables. Complex policy/context fields
# are stored as JSON so the row holds the indexed query columns + an envelope. ---


class TaskJobRow(Base):
    __tablename__ = "ai_jobs"
    __table_args__ = (
        Index("ix_ai_jobs_status_created", "status", "created_at"),
        Index("ix_ai_jobs_tenant_status", "tenant_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[str] = mapped_column(String(32))
    tenant_id: Mapped[str] = mapped_column(String(128))
    root_task_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    input_artifact_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    output_artifact_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    started_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    data_json: Mapped[str] = mapped_column(Text)


class TaskRow(Base):
    __tablename__ = "ai_tasks"
    __table_args__ = (
        Index("ix_ai_tasks_job_status", "job_id", "status"),
        Index("ix_ai_tasks_status_available", "status", "available_at"),
        Index(
            "ix_ai_tasks_handler_status_available", "handler", "status", "available_at"
        ),
        Index("ix_ai_tasks_lease_expires", "lease_expires_at"),
        UniqueConstraint("job_id", "key", name="uq_ai_tasks_job_key"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(128))
    parent_task_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    key: Mapped[str] = mapped_column(String(255))
    handler: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32))
    input_artifact_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    output_artifact_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer)
    available_at: Mapped[datetime] = mapped_column(DateTime)
    lease_owner: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    fencing_token: Mapped[int] = mapped_column(Integer)
    active_attempt_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    timeout_seconds: Mapped["float | None"] = mapped_column(Float, nullable=True)
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    data_json: Mapped[str] = mapped_column(Text)


class TaskAttemptRow(Base):
    __tablename__ = "ai_task_attempts"
    __table_args__ = (
        Index("ix_ai_attempts_task_attempt", "task_id", "attempt"),
        Index("ix_ai_attempts_run", "run_id"),
        UniqueConstraint("task_id", "attempt", name="uq_ai_attempts_task_attempt"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128))
    job_id: Mapped[str] = mapped_column(String(128))
    attempt: Mapped[int] = mapped_column(Integer)
    worker_id: Mapped[str] = mapped_column(String(128))
    fencing_token: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32))
    run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    failure_kind: Mapped["str | None"] = mapped_column(String(64), nullable=True)
    error_type: Mapped["str | None"] = mapped_column(String(255), nullable=True)
    error_message: Mapped["str | None"] = mapped_column(Text, nullable=True)
    data_json: Mapped[str] = mapped_column(Text)


class TaskTransitionRow(Base):
    __tablename__ = "ai_task_transitions"
    __table_args__ = (Index("ix_ai_transitions_job", "job_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(128))
    task_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    attempt_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    from_status: Mapped["str | None"] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(String(64))
    occurred_at: Mapped[datetime] = mapped_column(DateTime)
    data_json: Mapped[str] = mapped_column(Text, default="{}")


class TaskSignalRow(Base):
    __tablename__ = "ai_task_signals"
    __table_args__ = (Index("ix_ai_signals_job_name", "job_id", "name"),)

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(String(255))
    correlation_key: Mapped[str] = mapped_column(String(255))
    payload_artifact_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime)
    consumed_by_task_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True
    )
    data_json: Mapped[str] = mapped_column(Text, default="{}")


class EvalRunRow(Base):
    __tablename__ = "ai_eval_runs"
    __table_args__ = (Index("ix_ai_eval_runs_suite", "suite_id"),)

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    suite_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    started_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    # target, baseline_target, metadata live in the envelope.
    data_json: Mapped[str] = mapped_column(Text, default="{}")


class EvalResultRow(Base):
    __tablename__ = "ai_eval_results"
    __table_args__ = (Index("ix_ai_eval_results_run", "eval_run_id"),)

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    eval_run_id: Mapped[str] = mapped_column(String(128))
    case_id: Mapped[str] = mapped_column(String(128))
    run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    job_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    task_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    output_artifact_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    snapshot_artifact_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    error_type: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    error_message: Mapped["str | None"] = mapped_column(Text, nullable=True)
    # scores + metrics live in the envelope.
    data_json: Mapped[str] = mapped_column(Text, default="{}")


class ArtifactRecordRow(Base):
    """ArtifactRecord metadata (the lineage half of an artifact; the content
    blob lives out-of-band -- on the filesystem via FilesystemArtifactBlobStore,
    never in this table). Query columns (artifact_id / tenant_id / sha256 /
    producer_kind / producer_id / run_id) are indexed for the tenant gate, orphan
    sweep, and parent/provenance lookups; the full record envelope is the
    ``data_json`` column decoded via the public record codec."""

    __tablename__ = "ai_artifact_records"
    __table_args__ = (
        Index("ix_ai_artifact_records_tenant", "tenant_id"),
        Index("ix_ai_artifact_records_producer", "producer_kind", "producer_id"),
        Index("ix_ai_artifact_records_run", "run_id"),
    )

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    # Parent/provenance index columns: derived from the record's
    # ArtifactProvenance at put() time so the store can look up records by
    # producer / run without a JSON scan. parent_artifact_ids stays in data_json
    # (a multi-valued field; a join table would be the production-grade index).
    producer_kind: Mapped["str | None"] = mapped_column(String(64), nullable=True)
    producer_id: Mapped["str | None"] = mapped_column(String(255), nullable=True)
    run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    data_json: Mapped[str] = mapped_column(Text, default="{}")
