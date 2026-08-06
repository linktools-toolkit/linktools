#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registration of the non-asset tables in the published SQL schema."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.errors import ErrorCode, LinktoolsAIError
from ..storage.database import SqlSchemaRegistry
from ..storage.names import storage_name

try:
    from sqlalchemy import (
        BigInteger,
        Column,
        DateTime,
        Double,
        Index,
        Integer,
        JSON,
        Numeric,
        String,
        Table,
        Text,
        UniqueConstraint,
    )
except ModuleNotFoundError as error:
    if error.name == "sqlalchemy":
        raise LinktoolsAIError(
            ErrorCode.OPTIONAL_DEPENDENCY_MISSING,
            "SQLAlchemy is required for SQL schema registration",
        ) from error
    raise

if TYPE_CHECKING:
    from sqlalchemy.sql.schema import Table as TableType


@dataclass(frozen=True, slots=True)
class SqlRuntimeTables:
    sessions: "TableType"
    session_turns: "TableType"
    executions: "TableType"
    execution_snapshots: "TableType"
    execution_trace_steps: "TableType"
    execution_events: "TableType"
    execution_evaluations: "TableType"
    artifacts: "TableType"
    task_plans: "TableType"
    task_executions: "TableType"
    memories: "TableType"


class SqlRuntimeSchema:
    """Register the legacy runtime tables without performing database I/O."""

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> SqlRuntimeTables:
        integer_id = BigInteger().with_variant(Integer, "sqlite")
        sessions = Table(
            storage_name("sessions"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("session_id", String(255), nullable=False),
            Column("tenant_id", String(255), nullable=True),
            Column("user_id", String(255), nullable=True),
            Column("next_turn_sequence", Integer, nullable=False),
            Column("latest_completed_run_id", String(255), nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("session_id", name=storage_name("sessions_uk_session_id")),
            Index(storage_name("sessions_ix_tenant_id_user_id"), "tenant_id", "user_id"),
            Index(storage_name("sessions_ix_updated_at"), "updated_at"),
            Index(storage_name("sessions_ix_created_at"), "created_at"),
        )
        session_turns = Table(
            storage_name("session_turns"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("session_id", String(255), nullable=False),
            Column("sequence", Integer, nullable=False),
            Column("execution_id", String(255), nullable=False),
            Column("input", JSON, nullable=True),
            Column("delta_messages", JSON, nullable=True),
            Column("status", String(32), nullable=False),
            Column("capture_state", String(32), nullable=False),
            Column("completed_at", DateTime(timezone=True), nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("session_id", "sequence", name=storage_name("session_turns_uk_session_id_sequence")),
            UniqueConstraint("execution_id", name=storage_name("session_turns_uk_execution_id")),
            Index(storage_name("session_turns_ix_updated_at"), "updated_at"),
            Index(storage_name("session_turns_ix_created_at"), "created_at"),
        )
        executions = Table(
            storage_name("executions"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("execution_id", String(255), nullable=False),
            Column("session_id", String(255), nullable=False),
            Column("kind", String(40), nullable=False),
            Column("runnable_id", String(255), nullable=False),
            Column("runnable_type", String(40), nullable=False),
            Column("session_turn_sequence", Integer, nullable=True),
            Column("parent_execution_id", String(255), nullable=True),
            Column("root_execution_id", String(255), nullable=False),
            Column("status", String(40), nullable=False),
            Column("definition", JSON, nullable=False),
            Column("definition_hash", String(64), nullable=False),
            Column("data", JSON, nullable=False),
            Column("owner", String(255), nullable=True),
            Column("fence", Integer, nullable=False),
            Column("lease_expires_at", DateTime(timezone=True), nullable=True),
            Column("cancel_requested_at", DateTime(timezone=True), nullable=True),
            Column("snapshot_revision", Integer, nullable=False),
            Column("trace_sequence", Integer, nullable=False),
            Column("event_sequence", Integer, nullable=False),
            Column("tenant_id", String(255), nullable=True),
            Column("user_id", String(255), nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("execution_id", name=storage_name("executions_uk_execution_id")),
            Index(storage_name("executions_ix_session_id"), "session_id"),
            Index(storage_name("executions_ix_root_execution_id"), "root_execution_id"),
            Index(storage_name("executions_ix_lease_expires_at"), "lease_expires_at"),
            Index(storage_name("executions_ix_updated_at"), "updated_at"),
            Index(storage_name("executions_ix_created_at"), "created_at"),
        )
        execution_snapshots = Table(
            storage_name("execution_snapshots"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("execution_id", String(255), nullable=False),
            Column("revision", Integer, nullable=False),
            Column("resume_messages", JSON, nullable=False),
            Column("outcome", JSON, nullable=False),
            Column("status", String(32), nullable=False),
            Column("trace_end_sequence", Integer, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("execution_id", name=storage_name("execution_snapshots_uk_execution_id")),
            Index(storage_name("execution_snapshots_ix_updated_at"), "updated_at"),
            Index(storage_name("execution_snapshots_ix_created_at"), "created_at"),
        )
        execution_trace_steps = Table(
            storage_name("execution_trace_steps"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("execution_id", String(255), nullable=False),
            Column("sequence", Integer, nullable=False),
            Column("kind", String(40), nullable=False),
            Column("payload", JSON, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("execution_id", "sequence", name=storage_name("execution_trace_steps_uk_execution_id_sequence")),
            Index(storage_name("execution_trace_steps_ix_updated_at"), "updated_at"),
            Index(storage_name("execution_trace_steps_ix_created_at"), "created_at"),
        )
        execution_events = Table(
            storage_name("execution_events"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("execution_id", String(255), nullable=False),
            Column("sequence", Integer, nullable=False),
            Column("type", String(120), nullable=False),
            Column("payload", JSON, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("execution_id", "sequence", name=storage_name("execution_events_uk_execution_id_sequence")),
            Index(storage_name("execution_events_ix_updated_at"), "updated_at"),
            Index(storage_name("execution_events_ix_created_at"), "created_at"),
        )
        execution_evaluations = Table(
            storage_name("execution_evaluations"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("evaluation_id", String(255), nullable=False),
            Column("execution_id", String(255), nullable=False),
            Column("evaluator", String(255), nullable=False),
            Column("score", Double, nullable=True),
            Column("result", JSON, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("evaluation_id", name=storage_name("execution_evaluations_uk_evaluation_id")),
            Index(storage_name("execution_evaluations_ix_evaluator"), "evaluator"),
            Index(storage_name("execution_evaluations_ix_execution_id_created_at"), "execution_id", "created_at"),
            Index(storage_name("execution_evaluations_ix_updated_at"), "updated_at"),
            Index(storage_name("execution_evaluations_ix_created_at"), "created_at"),
        )
        artifacts = Table(
            storage_name("artifacts"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("artifact_id", String(128), nullable=False),
            Column("sha256", String(64), nullable=False),
            Column("media_type", String(128), nullable=False),
            Column("size", Integer, nullable=False),
            Column("tenant_id", String(128), nullable=False),
            Column("producer_kind", String(64), nullable=False),
            Column("producer_id", String(128), nullable=False),
            Column("run_id", String(128), nullable=True),
            Column("session_id", String(128), nullable=True),
            Column("parent_artifact_ids", JSON, nullable=False),
            Column("provenance_metadata", JSON, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("tenant_id", "artifact_id", name=storage_name("artifacts_uk_tenant_id_artifact_id")),
            Index(storage_name("artifacts_ix_artifact_id"), "artifact_id"),
            Index(storage_name("artifacts_ix_sha256"), "sha256"),
            Index(storage_name("artifacts_ix_updated_at"), "updated_at"),
            Index(storage_name("artifacts_ix_created_at"), "created_at"),
        )
        task_plans = Table(
            storage_name("task_plans"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("plan_id", String(255), nullable=False),
            Column("payload", JSON, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("plan_id", name=storage_name("task_plans_uk_plan_id")),
            Index(storage_name("task_plans_ix_updated_at"), "updated_at"),
            Index(storage_name("task_plans_ix_created_at"), "created_at"),
        )
        task_executions = Table(
            storage_name("task_executions"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("execution_id", String(255), nullable=False),
            Column("plan_id", String(255), nullable=False),
            Column("node_id", String(255), nullable=False),
            Column("status", String(32), nullable=False),
            Column("owner", String(255), nullable=True),
            Column("fence", Integer, nullable=False),
            Column("attempt", Integer, nullable=False),
            Column("result", JSON, nullable=True),
            Column("error", JSON, nullable=True),
            Column("lease_expires_at", DateTime(timezone=True), nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("execution_id", name=storage_name("task_executions_uk_execution_id")),
            Index(storage_name("task_executions_ix_plan_id"), "plan_id"),
            Index(storage_name("task_executions_ix_updated_at"), "updated_at"),
            Index(storage_name("task_executions_ix_created_at"), "created_at"),
        )
        memories = Table(
            storage_name("memories"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("memory_id", String(128), nullable=False),
            Column("tenant_id", String(128), nullable=True),
            Column("owner_id", String(128), nullable=False),
            Column("content", Text, nullable=False),
            Column("category", String(64), nullable=True),
            Column("confidence", Numeric(5, 4), nullable=True),
            Column("version", Integer, nullable=False),
            Column("metadata_json", Text, nullable=False),
            Column("user_id", String(128), nullable=True),
            Column("workspace_id", String(128), nullable=True),
            Column("session_id", String(128), nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("memory_id", name=storage_name("memories_uk_memory_id")),
            Index(storage_name("memories_ix_tenant_id"), "tenant_id"),
            Index(storage_name("memories_ix_updated_at"), "updated_at"),
            Index(storage_name("memories_ix_created_at"), "created_at"),
        )
        tables = (
            (sessions, "adapter.session"),
            (session_turns, "adapter.session"),
            (executions, "adapter.execution"),
            (execution_snapshots, "adapter.execution"),
            (execution_trace_steps, "adapter.execution"),
            (execution_events, "adapter.execution"),
            (execution_evaluations, "adapter.evaluation"),
            (artifacts, "adapter.artifact"),
            (task_plans, "adapter.task"),
            (task_executions, "adapter.task"),
            (memories, "adapter.memory"),
        )
        for table, owner in tables:
            registry.add_table(table, owner=owner)
        return SqlRuntimeTables(
            sessions,
            session_turns,
            executions,
            execution_snapshots,
            execution_trace_steps,
            execution_events,
            execution_evaluations,
            artifacts,
            task_plans,
            task_executions,
            memories,
        )


__all__ = ["SqlRuntimeSchema", "SqlRuntimeTables"]
