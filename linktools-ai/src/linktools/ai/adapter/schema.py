#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized SQL tables owned by the Runtime SQL adapter."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..storage.database import SqlSchemaRegistry
from ..storage.names import storage_name
from ..core.errors import ErrorCode, LinktoolsAIError

runtime_metadata: object | None = None
step_metadata: object | None = None

if TYPE_CHECKING:
    from sqlalchemy import CHAR, MetaData, String, Table


@dataclass(frozen=True, slots=True)
class SqlRuntimeTables:
    tables: "dict[str, Table]"

    def __getitem__(self, name: str) -> "Table":
        return self.tables[name]


class SqlRuntimeSchema:
    """Register one concrete table for each persisted Runtime fact family."""

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> SqlRuntimeTables:
        global runtime_metadata
        try:
            from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, JSON, LargeBinary, String, Table, UniqueConstraint
            from sqlalchemy.dialects import mysql
        except ModuleNotFoundError as error:
            raise LinktoolsAIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING, "SQLAlchemy is required for SQL Runtime storage") from error

        runtime_metadata = registry.metadata
        integer_id = BigInteger().with_variant(Integer, "sqlite")
        names = (
            "sessions", "executions", "results", "idempotency", "execution_events",
            "task_graphs", "task_nodes", "evaluations", "memories", "artifacts", "approvals",
            "external_results", "operation_counters", "operation_ledger", "tool_operations", "blobs", "blob_chunks",
        )
        physical_names = {
            "sessions": "runtime_sessions",
            "executions": "runtime_executions",
            "results": "runtime_results",
            "idempotency": "runtime_idempotency",
            "execution_events": "runtime_events",
            "task_graphs": "runtime_tasks",
            "task_nodes": "runtime_task_nodes",
            "evaluations": "runtime_evaluations",
            "memories": "runtime_memories",
            "artifacts": "runtime_artifacts",
            "approvals": "runtime_approvals",
            "external_results": "runtime_externals",
            "operation_counters": "runtime_operation_counters",
            "operation_ledger": "runtime_operations",
            "tool_operations": "runtime_tools",
            "blobs": "runtime_blobs",
            "blob_chunks": "runtime_blob_chunks",
        }
        tables: dict[str, Table] = {}
        for name in names:
            table = Table(
                storage_name(physical_names[name]),
                registry.metadata,
                Column("id", integer_id, primary_key=True, autoincrement=True),
                Column("namespace_key", _hex64(), nullable=False),
                Column("tenant_id", _text_key(128), nullable=False),
                Column("record_id", _text_key(256), nullable=False),
                Column("session_id", _text_key(256), nullable=True),
                Column("parent_execution_id", _text_key(256), nullable=True),
                Column("source_execution_id", _text_key(256), nullable=True),
                Column("base_execution_id", _text_key(256), nullable=True),
                Column("lineage_kind", _text_key(64), nullable=False, default="RUN"),
                Column("sequence", BigInteger, nullable=False, default=0),
                Column("revision", BigInteger, nullable=False, default=0),
                Column("status", _text_key(64), nullable=False, default=""),
                Column("payload", JSON, nullable=False),
                Column("created_at", DateTime(timezone=True), nullable=False),
                Column("updated_at", DateTime(timezone=True), nullable=False),
                UniqueConstraint("namespace_key", "tenant_id", "record_id", name=storage_name(f"{name}_uk_identity")),
                mysql_engine="InnoDB",
                mysql_charset="utf8mb4",
                mysql_collate="utf8mb4_bin",
                extend_existing=True,
            )
            if name == "sessions":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "session_id", name=storage_name("runtime_sessions_uk_session")))
            if name == "executions":
                Index(storage_name("runtime_executions_ix_session_status"), table.c.namespace_key, table.c.tenant_id, table.c.session_id, table.c.status)
                Index(storage_name("runtime_executions_ix_source"), table.c.namespace_key, table.c.tenant_id, table.c.source_execution_id)
                Index(storage_name("runtime_executions_ix_base"), table.c.namespace_key, table.c.tenant_id, table.c.base_execution_id)
                Index(storage_name("runtime_executions_ix_parent_created"), table.c.namespace_key, table.c.tenant_id, table.c.parent_execution_id, table.c.created_at)
            if name == "tool_operations":
                table.append_column(Column("run_id", _text_key(256), nullable=True))
                table.append_column(Column("tool_call_id", _text_key(256), nullable=True))
                table.append_column(Column("owner", _text_key(256), nullable=True))
                table.append_column(Column("fence", BigInteger, nullable=True))
                table.append_column(Column("lease_expires_at", DateTime(timezone=True), nullable=True))
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "run_id", "tool_call_id", name=storage_name("tool_operations_uk_call")))
            if name == "task_nodes":
                table.append_column(Column("owner", String(512), nullable=True))
                table.append_column(Column("fence", BigInteger, nullable=False, default=0))
                table.append_column(Column("lease_expires_at", DateTime(timezone=True), nullable=True))
            if name == "operation_counters":
                table.append_column(Column("resource_kind", _text_key(64), nullable=False, default=""))
                table.append_column(Column("resource_id", _text_key(256), nullable=False, default=""))
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "resource_kind", "resource_id", name=storage_name("operation_counters_uk_partition")))
            if name == "idempotency":
                table.append_column(Column("scope", _text_key(64), nullable=False, default=""))
                table.append_column(Column("key_hash", _hex64(), nullable=False, default=""))
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "scope", "key_hash", name=storage_name("idempotency_uk_key")))
            if name == "operation_ledger":
                table.append_column(Column("resource_kind", _text_key(64), nullable=False, default=""))
                table.append_column(Column("resource_id", _text_key(256), nullable=False, default=""))
            if name == "blobs":
                table.append_column(Column("digest", _hex64(), nullable=False, default=""))
                table.append_column(Column("size", BigInteger, nullable=False, default=0))
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "digest", name=storage_name("runtime_blobs_uk_digest")))
            if name == "blob_chunks":
                table.append_column(Column("digest", _hex64(), nullable=False, default=""))
                table.append_column(Column("chunk_index", BigInteger, nullable=False, default=0))
                table.append_column(Column("content", LargeBinary().with_variant(mysql.LONGBLOB(), "mysql"), nullable=False, default=b""))
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "digest", "chunk_index", name=storage_name("runtime_blob_chunks_uk_chunk")))
            if name == "sessions":
                table.append_column(Column("profile", _text_key(64), nullable=False, default="local-coding"))
                table.append_column(Column("head_execution_id", _text_key(256), nullable=True))
            if name == "executions":
                table.append_column(Column("agent_run_sequence", BigInteger, nullable=False, default=0))
            registry.add_table(table, owner="adapter.sql")
            tables[name] = table
        return SqlRuntimeTables(tables)


def _text_key(length: int) -> "String":
    from sqlalchemy import String
    from sqlalchemy.dialects import mysql
    return String(length).with_variant(mysql.VARCHAR(length, charset="utf8mb4", collation="utf8mb4_bin"), "mysql")


def _hex64() -> "CHAR":
    from sqlalchemy import CHAR
    from sqlalchemy.dialects import mysql
    return CHAR(64).with_variant(mysql.CHAR(64, charset="ascii", collation="ascii_bin"), "mysql")


def ensure_step_metadata() -> "MetaData":
    global step_metadata
    if step_metadata is None:
        try:
            from sqlalchemy import MetaData
        except ModuleNotFoundError as error:
            raise LinktoolsAIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING, "SQLAlchemy is required for SQL Step storage") from error
        step_metadata = MetaData()
    return step_metadata


def new_step_metadata() -> "MetaData":
    global step_metadata
    try:
        from sqlalchemy import MetaData
    except ModuleNotFoundError as error:
        raise LinktoolsAIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING, "SQLAlchemy is required for SQL Step storage") from error
    metadata = MetaData()
    if step_metadata is None:
        step_metadata = metadata
    return metadata


__all__ = ["SqlRuntimeSchema", "SqlRuntimeTables", "ensure_step_metadata", "new_step_metadata", "runtime_metadata", "step_metadata"]
