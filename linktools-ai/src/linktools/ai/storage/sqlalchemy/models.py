#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy table models for reliable-task storage and the other legacy
domain stores. Assets live on their own DeclarativeBase now
(storage/backends/sqlalchemy/models.py); this module no longer defines any
asset-related table."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BINARY,
    DECIMAL,
    DateTime,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .conventions import (
    BIGSERIAL,
    TABLE_PREFIX,
    TimestampMixin,
    sha256_hash,
    timestamp_indexes,
)


class Base(DeclarativeBase):
    pass


class ToolIdempotencyRow(TimestampMixin, Base):
    """Persistent tool-call idempotency records. The (scope, key) natural key
    backs ``reserve``'s "find-or-create" semantics; uniqueness is carried by
    ``scope_key_hash`` (sha256(scope+key)) so the wide key column stays out of
    the unique index (IntegrityError on the race -> SELECT the winner ->
    hash-check)."""

    __tablename__ = f"{TABLE_PREFIX}idempotency"
    __table_args__ = (
        UniqueConstraint("idempotency_id", name="uk_idempotency_id"),
        UniqueConstraint("scope_key_hash", name="uk_scope_key_hash"),
        Index("ix_scope", "scope"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    idempotency_id: Mapped[str] = mapped_column(String(128))
    scope: Mapped[str] = mapped_column(String(128))
    key: Mapped[str] = mapped_column(String(512))
    scope_key_hash: Mapped[bytes] = mapped_column(BINARY(32))
    request_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    result_json: Mapped["str | None"] = mapped_column(Text, nullable=True)
    error_message: Mapped["str | None"] = mapped_column(Text, nullable=True)
    completed_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    # Fencing fields for the claim/owner/generation/lease model.
    owner_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    generation: Mapped[int] = mapped_column(default=0)
    claimed_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    lease_expires_at: Mapped["datetime | None"] = mapped_column(nullable=True)
    receipt_artifact_id: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    binding_fingerprint: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    result_processor_revision: Mapped["str | None"] = mapped_column(String(128), nullable=True)


class RunRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}runs"
    __table_args__ = (
        UniqueConstraint("run_id", name="uk_run_id"),
        Index("ix_root_run_id", "root_run_id"),
        Index("ix_parent_run_id", "parent_run_id"),
        Index("ix_session_id", "session_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128))
    root_run_id: Mapped[str] = mapped_column(String(128))
    parent_run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    session_id: Mapped[str] = mapped_column(String(128))
    runnable_id: Mapped[str] = mapped_column(String(255))
    runnable_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int]
    started_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    # Cancel-request audit (nullable: absent on older rows and on runs never
    # cancelled).
    cancel_requested_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    cancel_requested_by: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    worker_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    execution_token: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    heartbeat_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    manifest_id: Mapped["str | None"] = mapped_column(String(256), nullable=True)
    resumability: Mapped["str | None"] = mapped_column(String(32), nullable=True)
    # data_json envelope: input/result/error/cancel_reason folded together to
    # satisfy the >2-text-column lint rule; metadata_json stays separate.
    data_json: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text)


class RunCheckpointRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}run_checkpoints"
    __table_args__ = (
        UniqueConstraint("checkpoint_id", name="uk_checkpoint_id"),
        UniqueConstraint("run_id", "sequence", name="uk_run_id_sequence"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    checkpoint_id: Mapped[str] = mapped_column(String(128))
    run_id: Mapped[str] = mapped_column(String(128))
    sequence: Mapped[int]
    format: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[int]
    payload: Mapped[bytes] = mapped_column(LargeBinary)
    metadata_json: Mapped[str] = mapped_column(Text)


class RunCheckpointCounterRow(TimestampMixin, Base):
    """Per-run monotonic counter the Store increments inside the append
    transaction so concurrent appends for the same run never collide on
    sequence (the unique constraint on (run_id, sequence) is the backstop)."""

    __tablename__ = f"{TABLE_PREFIX}run_checkpoint_counters"
    __table_args__ = (
        UniqueConstraint("run_id", name="uk_run_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128))
    last_sequence: Mapped[int]


class RunDefinitionRow(TimestampMixin, Base):
    """The immutable RunDefinitionSnapshot persisted at run creation so resume
    can restore the exact original spec + identity."""

    __tablename__ = f"{TABLE_PREFIX}run_definitions"
    __table_args__ = (
        UniqueConstraint("run_id", name="uk_run_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128))
    runnable_type: Mapped[str] = mapped_column(String(32))
    runnable_id: Mapped[str] = mapped_column(String(255))
    serialized_spec_json: Mapped[str] = mapped_column(Text)
    spec_fingerprint: Mapped[str] = mapped_column(String(64))
    user_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    tenant_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    workspace: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    manifest_json: Mapped["str | None"] = mapped_column(Text, nullable=True)
    resumability: Mapped["str | None"] = mapped_column(String(32), nullable=True)


class SessionRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}sessions"
    __table_args__ = (
        UniqueConstraint("session_id", name="uk_session_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128))
    parent_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    # Principal the session belongs to. Nullable: legacy rows and unowned
    # (single-user CLI) sessions stay NULL.
    user_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    tenant_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int]
    metadata_json: Mapped[str] = mapped_column(Text)


class SessionMessageRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}session_messages"
    __table_args__ = (
        UniqueConstraint("message_id", name="uk_message_id"),
        UniqueConstraint("session_id", "sequence", name="uk_session_id_sequence"),
        # commit-scoped idempotency for a batch append: uniqueness on
        # (session_id, commit_hash, batch_index) so a retried
        # append_messages_once does not duplicate the batch. commit_hash is
        # sha256(commit_id) for commit appends, sha256(message_id) for
        # non-commit appends (unique per row, so they never collide even
        # though commit_id is NULL for them).
        UniqueConstraint(
            "session_id",
            "commit_hash",
            "batch_index",
            name="uk_session_id_commit_hash_batch_index",
        ),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(128))
    session_id: Mapped[str] = mapped_column(String(128))
    sequence: Mapped[int]
    role: Mapped[str] = mapped_column(String(32))
    data_json: Mapped[str] = mapped_column(Text)
    run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text)
    # commit_id is NULL for non-commit appends (audit-only); the uniqueness
    # carrier is commit_hash below.
    commit_id: Mapped["str | None"] = mapped_column(String(200), nullable=True)
    commit_hash: Mapped[bytes] = mapped_column(BINARY(32))
    # 0-based position within the commit batch; 0 for non-commit appends.
    batch_index: Mapped[int] = mapped_column(Integer, default=0)


class EventRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uk_event_id"),
        UniqueConstraint("stream_id", "sequence", name="uk_stream_id_sequence"),
        # commit-scoped idempotency: uniqueness on (stream_id, commit_hash,
        # event_type). event_type is part of the key because one commit may
        # append several distinct events (pause emits ApprovalRequested +
        # RunPaused). commit_hash = sha256(commit_id) for commit appends,
        # sha256(event_id) for non-commit appends (unique per row).
        UniqueConstraint(
            "stream_id", "commit_hash", "event_type",
            name="uk_stream_id_commit_hash_event_type",
        ),
        Index("ix_run_id", "run_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(128))
    stream_id: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(String(128))
    sequence: Mapped[int]
    occurred_at: Mapped[datetime]
    root_run_id: Mapped[str] = mapped_column(String(128))
    parent_run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    session_id: Mapped[str] = mapped_column(String(128))
    runnable_id: Mapped[str] = mapped_column(String(255))
    event_type: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    data_json: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped["str | None"] = mapped_column(Text, nullable=True)
    commit_id: Mapped["str | None"] = mapped_column(String(200), nullable=True)
    commit_hash: Mapped[bytes] = mapped_column(BINARY(32))


class SwarmRunRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}swarm_runs"
    __table_args__ = (
        UniqueConstraint("swarm_run_id", name="uk_swarm_run_id"),
        Index("ix_run_id", "run_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    swarm_run_id: Mapped[str] = mapped_column(String(128), comment="Swarm run id")
    run_id: Mapped[str] = mapped_column(String(128), comment="Driving run id")
    round: Mapped[int] = mapped_column(Integer, comment="Swarm round")
    status: Mapped[str] = mapped_column(String(32), comment="Status")
    version: Mapped[int] = mapped_column(
        Integer, comment="Version (optimistic lock)"
    )
    input_tokens: Mapped[int] = mapped_column(Integer, comment="Input tokens")
    output_tokens: Mapped[int] = mapped_column(Integer, comment="Output tokens")
    total_cost: Mapped[str] = mapped_column(Text, comment="Total cost")
    metadata_json: Mapped[str] = mapped_column(Text, comment="Metadata (JSON)")
    # The run-level owner fence (SwarmCommitPolicy compares a commit's supplied
    # token against this persisted value). None until the swarm's `start`
    # commit stamps it; a reclaim rotates it to the new owner's token.
    execution_token: Mapped["str | None"] = mapped_column(
        String(256), nullable=True, comment="Execution token"
    )
    execution_owner_id: Mapped["str | None"] = mapped_column(
        String(256), nullable=True, comment="Execution owner id"
    )
    execution_generation: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Execution generation"
    )


class SwarmStepRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}swarm_tasks"
    __table_args__ = (
        UniqueConstraint("task_id", name="uk_task_id"),
        Index("ix_swarm_run_id", "swarm_run_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(128), comment="Task id")
    swarm_run_id: Mapped[str] = mapped_column(String(128), comment="Swarm run id")
    parent_task_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Parent task id"
    )
    assigned_agent_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Assigned agent id"
    )
    status: Mapped[str] = mapped_column(String(32), comment="Status")
    attempts: Mapped[int] = mapped_column(Integer, comment="Attempt count")
    version: Mapped[int] = mapped_column(
        Integer, comment="Version (optimistic lock)"
    )
    claimed_at: Mapped["datetime | None"] = mapped_column(
        DateTime, nullable=True, comment="Claimed at"
    )
    lease_expires_at: Mapped["datetime | None"] = mapped_column(
        DateTime, nullable=True, comment="Lease expires at"
    )
    # Child RunRecord id of the current/most-recent execution.
    active_run_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Active run id"
    )
    # Folds description + dependencies_json + input_json + result_json +
    # error_json (D4: >2 large TEXT columns). SwarmStep has no top-level
    # metadata field (unlike SwarmRun) -- no metadata_json column here.
    data_json: Mapped[str] = mapped_column(Text, comment="Payload (JSON)")


class SwarmStepAttemptRow(TimestampMixin, Base):
    """One execution attempt of a SwarmStep. Mirrors the
    SwarmStepAttempt domain model."""

    __tablename__ = f"{TABLE_PREFIX}swarm_task_attempts"
    __table_args__ = (
        UniqueConstraint("attempt_id", name="uk_attempt_id"),
        Index("ix_task_id", "task_id"),
        Index("ix_run_id", "run_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    attempt_id: Mapped[str] = mapped_column(String(128), comment="Attempt id")
    task_id: Mapped[str] = mapped_column(String(128), comment="Task id")
    run_id: Mapped[str] = mapped_column(String(128), comment="Run id")
    agent_id: Mapped[str] = mapped_column(String(128), comment="Agent id")
    attempt: Mapped[int] = mapped_column(Integer, comment="Attempt number")
    status: Mapped[str] = mapped_column(String(32), comment="Status")
    started_at: Mapped[datetime] = mapped_column(DateTime, comment="Started at")
    finished_at: Mapped["datetime | None"] = mapped_column(
        DateTime, nullable=True, comment="Finished at"
    )
    error_json: Mapped["str | None"] = mapped_column(
        Text, nullable=True, comment="Error (JSON)"
    )


class MemoryRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}memories"

    # Search is content LIKE-driven, not indexed-lookup-driven, so the prior
    # (tenant_id, user_id)/(tenant_id, workspace_id)/(tenant_id, session_id)
    # composite indexes were never the query path -- dropped rather than
    # padded out to satisfy the lint's index-count ceiling.
    __table_args__ = (
        UniqueConstraint("memory_id", name="uk_memory_id"),
        Index("ix_tenant_id", "tenant_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(String(128), comment="Memory id")
    # tenant_id is the hard isolation boundary. NULL is tolerated only for
    # legacy rows persisted before tenant-scoping: those rows are read back
    # with a synthesized legacy tenant and never match a real tenant's search
    # (NULL != 'tenant-a' in SQL), so old data is quarantined without a
    # migration script.
    tenant_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Tenant id"
    )
    owner_id: Mapped[str] = mapped_column(String(128), comment="Owner id")
    content: Mapped[str] = mapped_column(Text, comment="Memory content")
    category: Mapped["str | None"] = mapped_column(
        String(64), nullable=True, comment="Category"
    )
    confidence: Mapped["Decimal | None"] = mapped_column(
        DECIMAL(5, 4), nullable=True, comment="Confidence [0,1]"
    )
    version: Mapped[int] = mapped_column(
        Integer, comment="Version (optimistic lock)"
    )
    metadata_json: Mapped[str] = mapped_column(Text, comment="Metadata (JSON)")
    user_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="User id"
    )
    workspace_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Workspace id"
    )
    session_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Session id"
    )


class ApprovalRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}approvals"
    # (run_id, tool_call_id) IS the natural dedup key -- a pydantic-ai
    # tool_call_id is minted fresh per invocation, so it never needs to be
    # "released" after a terminal (approved/rejected) resolution for reuse by
    # a genuinely different approval. A plain UNIQUE constraint (no separate
    # dedupe_key column, no active/terminal partial-index games) is therefore
    # both simpler and sufficient: create_or_get_pending()'s SELECT-then-
    # INSERT is only a fast path -- this constraint is what actually prevents
    # two concurrent callers from ever committing two rows for the same key.
    __table_args__ = (
        UniqueConstraint("approval_id", name="uk_approval_id"),
        UniqueConstraint("run_id", "tool_call_id", name="uk_run_id_tool_call_id"),
        Index("ix_tenant_id", "tenant_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    approval_id: Mapped[str] = mapped_column(String(128), comment="Approval id")
    run_id: Mapped[str] = mapped_column(String(128), comment="Run id")
    tool_call_id: Mapped[str] = mapped_column(String(128), comment="Tool call id")
    tool_name: Mapped[str] = mapped_column(String(255), comment="Tool name")
    arguments_hash: Mapped[str] = mapped_column(
        String(128), comment="Arguments hash"
    )
    status: Mapped[str] = mapped_column(String(32), comment="Status")
    version: Mapped[int] = mapped_column(
        Integer, comment="Version (optimistic lock)"
    )
    resolved_at: Mapped["datetime | None"] = mapped_column(
        DateTime, nullable=True, comment="Resolved at"
    )
    resolved_by: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Resolved by"
    )
    tenant_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Tenant id"
    )
    descriptor_fingerprint: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Descriptor fingerprint"
    )
    handler_revision: Mapped["str | None"] = mapped_column(
        String(256), nullable=True, comment="Handler revision"
    )
    provider_revision: Mapped["str | None"] = mapped_column(
        String(256), nullable=True, comment="Provider revision"
    )
    policy_revision: Mapped["str | None"] = mapped_column(
        String(256), nullable=True, comment="Policy revision"
    )
    capability_revision: Mapped["str | None"] = mapped_column(
        String(256), nullable=True, comment="Capability revision"
    )
    schema_version: Mapped[int] = mapped_column(
        Integer, default=1, comment="Schema version"
    )
    # Folds `reason` + `redacted_arguments_json` (the real arguments are never
    # persisted -- only this redacted audit copy + arguments_hash are stored).
    data_json: Mapped[str] = mapped_column(Text, comment="Payload (JSON)")
    metadata_json: Mapped[str] = mapped_column(Text, comment="Metadata (JSON)")


# --- Reliable-task tables. Complex policy/context fields
# are stored as JSON so the row holds the indexed query columns + an envelope. ---


class TaskJobRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}jobs"
    __table_args__ = (
        UniqueConstraint("job_id", name="uk_job_id"),
        Index("ix_status_created_at", "status", "created_at"),
        Index("ix_tenant_id_status", "tenant_id", "status"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(128), comment="Job id")
    status: Mapped[str] = mapped_column(String(32), comment="Status")
    tenant_id: Mapped[str] = mapped_column(String(128), comment="Tenant id")
    root_task_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Root task id"
    )
    input_artifact_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Input artifact id"
    )
    output_artifact_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Output artifact id"
    )
    version: Mapped[int] = mapped_column(
        Integer, comment="Version (optimistic lock)"
    )
    started_at: Mapped["datetime | None"] = mapped_column(
        DateTime, nullable=True, comment="Started at"
    )
    finished_at: Mapped["datetime | None"] = mapped_column(
        DateTime, nullable=True, comment="Finished at"
    )
    data_json: Mapped[str] = mapped_column(Text, comment="Payload (JSON)")


class TaskRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}tasks"
    __table_args__ = (
        UniqueConstraint("task_id", name="uk_task_id"),
        UniqueConstraint("job_key_hash", name="uk_job_key_hash"),
        Index("ix_job_id_status", "job_id", "status"),
        Index("ix_status_available_at", "status", "available_at"),
        # A MySQL deployment prefix-lengths `handler` in this index (see
        # migrations/init_schema.sql) -- kept out of the vendor-neutral core.
        Index("ix_handler_status_available_at", "handler", "status", "available_at"),
        Index("ix_lease_expires_at", "lease_expires_at"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(128), comment="Task id")
    job_id: Mapped[str] = mapped_column(String(128), comment="Job id")
    parent_task_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Parent task id"
    )
    key: Mapped[str] = mapped_column(String(255), comment="Task key")
    # Carries the (job_id, key) uniqueness -- sha256(job_id+key) -- so the
    # wide natural key stays out of the unique index (D7).
    job_key_hash: Mapped[bytes] = mapped_column(
        BINARY(32), comment="Job+key hash"
    )
    handler: Mapped[str] = mapped_column(String(255), comment="Handler")
    status: Mapped[str] = mapped_column(String(32), comment="Status")
    input_artifact_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Input artifact id"
    )
    output_artifact_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Output artifact id"
    )
    attempt_count: Mapped[int] = mapped_column(Integer, comment="Attempt count")
    available_at: Mapped[datetime] = mapped_column(DateTime, comment="Available at")
    lease_owner: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Lease owner"
    )
    lease_expires_at: Mapped["datetime | None"] = mapped_column(
        DateTime, nullable=True, comment="Lease expires at"
    )
    fencing_token: Mapped[int] = mapped_column(Integer, comment="Fencing token")
    active_attempt_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Active attempt id"
    )
    timeout_ms: Mapped["int | None"] = mapped_column(
        Integer, nullable=True, comment="Timeout (ms)"
    )
    version: Mapped[int] = mapped_column(
        Integer, comment="Version (optimistic lock)"
    )
    data_json: Mapped[str] = mapped_column(Text, comment="Payload (JSON)")


class TaskAttemptRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}task_attempts"
    __table_args__ = (
        UniqueConstraint("attempt_id", name="uk_attempt_id"),
        UniqueConstraint("task_id", "attempt", name="uk_task_id_attempt"),
        Index("ix_run_id", "run_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    attempt_id: Mapped[str] = mapped_column(String(128), comment="Attempt id")
    task_id: Mapped[str] = mapped_column(String(128), comment="Task id")
    job_id: Mapped[str] = mapped_column(String(128), comment="Job id")
    attempt: Mapped[int] = mapped_column(Integer, comment="Attempt number")
    worker_id: Mapped[str] = mapped_column(String(128), comment="Worker id")
    fencing_token: Mapped[int] = mapped_column(Integer, comment="Fencing token")
    status: Mapped[str] = mapped_column(String(32), comment="Status")
    run_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Run id"
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, comment="Started at")
    finished_at: Mapped["datetime | None"] = mapped_column(
        DateTime, nullable=True, comment="Finished at"
    )
    failure_kind: Mapped["str | None"] = mapped_column(
        String(64), nullable=True, comment="Failure kind"
    )
    error_type: Mapped["str | None"] = mapped_column(
        String(255), nullable=True, comment="Error type"
    )
    error_message: Mapped["str | None"] = mapped_column(
        Text, nullable=True, comment="Error message"
    )
    data_json: Mapped[str] = mapped_column(Text, comment="Payload (JSON)")


class TaskTransitionRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}task_transitions"
    __table_args__ = (Index("ix_job_id", "job_id"), *timestamp_indexes())

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(128), comment="Job id")
    task_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Task id"
    )
    attempt_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Attempt id"
    )
    from_status: Mapped["str | None"] = mapped_column(
        String(32), nullable=True, comment="From status"
    )
    to_status: Mapped[str] = mapped_column(String(32), comment="To status")
    reason: Mapped[str] = mapped_column(String(64), comment="Reason")
    occurred_at: Mapped[datetime] = mapped_column(DateTime, comment="Occurred at")
    data_json: Mapped[str] = mapped_column(
        Text, default="{}", comment="Context (JSON)"
    )


class TaskSignalRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}task_signals"
    __table_args__ = (
        UniqueConstraint("signal_id", name="uk_signal_id"),
        # A MySQL deployment prefix-lengths `name` in this index (see
        # migrations/init_schema.sql) -- kept out of the vendor-neutral core.
        Index("ix_job_id_name", "job_id", "name"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String(128), comment="Signal id")
    job_id: Mapped[str] = mapped_column(String(128), comment="Job id")
    name: Mapped[str] = mapped_column(String(255), comment="Signal name")
    correlation_key: Mapped[str] = mapped_column(
        String(255), comment="Correlation key"
    )
    payload_artifact_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Payload artifact id"
    )
    consumed_by_task_id: Mapped["str | None"] = mapped_column(
        String(128), nullable=True, comment="Consumed by task id"
    )
    data_json: Mapped[str] = mapped_column(
        Text, default="{}", comment="Context (JSON)"
    )


class EvalRunRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}eval_runs"
    __table_args__ = (
        UniqueConstraint("eval_run_id", name="uk_eval_run_id"),
        Index("ix_suite_id", "suite_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    eval_run_id: Mapped[str] = mapped_column(String(128))
    suite_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    started_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    # target, baseline_target, metadata live in the envelope.
    data_json: Mapped[str] = mapped_column(Text, default="{}")


class EvalResultRow(TimestampMixin, Base):
    __tablename__ = f"{TABLE_PREFIX}eval_results"
    __table_args__ = (
        UniqueConstraint("result_id", name="uk_result_id"),
        Index("ix_eval_run_id", "eval_run_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    result_id: Mapped[str] = mapped_column(String(128))
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


class ArtifactRecordRow(TimestampMixin, Base):
    """ArtifactRecord metadata (the lineage half of an artifact; the content
    blob lives out-of-band -- on the filesystem via FilesystemArtifactBlobStore,
    never in this table). Query columns (artifact_id / tenant_id / content_hash
    / producer_kind / producer_id / run_id) back the tenant gate, content dedup,
    and parent/provenance lookups; the full record envelope is the ``data_json``
    column decoded via the public record codec."""

    __tablename__ = f"{TABLE_PREFIX}artifact_records"
    __table_args__ = (
        UniqueConstraint("artifact_id", name="uk_artifact_id"),
        Index("ix_content_hash", "content_hash"),
        Index("ix_updated_at", "updated_at"),
        Index("ix_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    artifact_id: Mapped[str] = mapped_column(String(128))
    tenant_id: Mapped[str] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(64))
    # Parent/provenance columns: derived from the record's ArtifactProvenance
    # at put() time. Trimmed to 3 plain indexes to meet the 1/3 index-count
    # rule, so producer/run/tenant lookups scan (sha256 dedup is indexed).
    producer_kind: Mapped["str | None"] = mapped_column(String(64), nullable=True)
    producer_id: Mapped["str | None"] = mapped_column(String(255), nullable=True)
    run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    data_json: Mapped[str] = mapped_column(Text, default="{}")
