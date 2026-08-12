#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized SQL tables owned by the Runtime SQL adapter."""

from typing import TYPE_CHECKING

from ..storage import (
    SqlSchemaRegistry,
    sql_blob,
    sql_digest,
    sql_index,
    sql_integer_id,
    sql_table_options,
    sql_text_key,
    storage_name,
    register_sql_schema_contributor,
)

runtime_metadata: object | None = None
step_metadata: object | None = None

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Table


class SqlRuntimeSchema:
    """Register one concrete table for each persisted Runtime fact family."""

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> "dict[str, Table]":
        global runtime_metadata
        from sqlalchemy import (
            JSON,
            BigInteger,
            Column,
            DateTime,
            Index,
            Table,
            UniqueConstraint,
        )
        runtime_metadata = registry.metadata
        integer_id = sql_integer_id()
        names = (
            "sessions", "executions", "results", "idempotency", "evaluation_idempotency", "execution_events",
            "task_graphs", "task_nodes", "evaluations", "memories", "artifacts", "approvals",
            "external_results", "operation_counters", "operation_ledger", "tool_operations", "blobs", "blob_chunks",
            "recovery_checkpoints",
        )
        physical_names = {
            "sessions": "runtime_sessions",
            "executions": "runtime_executions",
            "results": "runtime_results",
            "idempotency": "runtime_idempotency",
            "evaluation_idempotency": "runtime_evaluation_idempotency",
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
            "recovery_checkpoints": "runtime_recovery_checkpoints",
        }
        tables: dict[str, Table] = {}
        for name in names:
            columns = [
                Column("id", integer_id, primary_key=True, autoincrement=True),
                Column("namespace_key", sql_digest(), nullable=False),
                Column("tenant_id", sql_text_key(128), nullable=False),
                Column("record_id", sql_text_key(256), nullable=False),
                Column("session_id", sql_text_key(256), nullable=name != "sessions"),
                Column("parent_execution_id", sql_text_key(256), nullable=True),
                Column("source_execution_id", sql_text_key(256), nullable=True),
                Column("base_execution_id", sql_text_key(256), nullable=True),
                Column("lineage_kind", sql_text_key(64), nullable=False, default="RUN"),
                Column("sequence", BigInteger, nullable=False, default=0),
                Column("revision", BigInteger, nullable=False, default=0),
                Column("status", sql_text_key(64), nullable=False, default=""),
                Column("payload", JSON, nullable=False),
            ]
            if name == "sessions":
                columns.append(Column("profile", sql_text_key(64), nullable=False, default=""))
            if name == "executions":
                columns.append(Column("agent_run_sequence", BigInteger, nullable=False, default=0))
            if name == "tool_operations":
                columns.extend((
                    Column("run_id", sql_text_key(256), nullable=False),
                    Column("tool_call_id", sql_text_key(256), nullable=False),
                    Column("call_key", sql_digest(), nullable=False, default=""),
                    Column("owner", sql_text_key(256), nullable=True),
                    Column("fence", BigInteger, nullable=True),
                    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
                ))
            if name == "task_nodes":
                columns.extend((
                    Column("owner", sql_text_key(256), nullable=True),
                    Column("fence", BigInteger, nullable=False, default=0),
                    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
                ))
            if name == "operation_counters":
                columns.extend((
                    Column("resource_kind", sql_text_key(64), nullable=False, default=""),
                    Column("resource_id", sql_text_key(256), nullable=False, default=""),
                    Column("partition_key", sql_digest(), nullable=False, default=""),
                ))
            if name in {"idempotency", "evaluation_idempotency"}:
                columns.extend((
                    Column("scope", sql_text_key(64), nullable=False, default=""),
                    Column("key_hash", sql_digest(), nullable=False, default=""),
                    Column("identity_key", sql_digest(), nullable=False, default=""),
                ))
            if name == "operation_ledger":
                columns.extend((
                    Column("resource_kind", sql_text_key(64), nullable=False, default=""),
                    Column("resource_id", sql_text_key(256), nullable=False, default=""),
                ))
            if name == "blobs":
                columns.extend((
                    Column("digest", sql_digest(), nullable=False, default=""),
                    Column("size", BigInteger, nullable=False, default=0),
                ))
            if name == "blob_chunks":
                columns.extend((
                    Column("digest", sql_digest(), nullable=False, default=""),
                    Column("chunk_index", BigInteger, nullable=False, default=0),
                    Column("chunk_key", sql_digest(), nullable=False, default=""),
                    Column("content", sql_blob(), nullable=False, default=b""),
                ))
            columns.extend((
                Column("updated_at", DateTime(timezone=True), nullable=False),
                Column("created_at", DateTime(timezone=True), nullable=False),
            ))
            table = Table(
                storage_name(physical_names[name]),
                registry.metadata,
                *columns,
                **sql_table_options(),
                extend_existing=True,
            )
            if name == "sessions":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "session_id", name="uk_namespace_key_tenant_id_session_id"))
            if name == "executions":
                sql_index(Index("ix_namespace_key_tenant_id_session_id", table.c.namespace_key, table.c.tenant_id, table.c.session_id))
                sql_index(Index("ix_namespace_key_tenant_id_source_execution_id", table.c.namespace_key, table.c.tenant_id, table.c.source_execution_id))
                sql_index(Index("ix_namespace_key_tenant_id_base_execution_id", table.c.namespace_key, table.c.tenant_id, table.c.base_execution_id))
                sql_index(Index("ix_namespace_key_tenant_id_parent_execution_id", table.c.namespace_key, table.c.tenant_id, table.c.parent_execution_id))
            table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "record_id", name="uk_namespace_key_tenant_id_record_id"))
            sql_index(Index("ix_updated_at", table.c.updated_at))
            sql_index(Index("ix_created_at", table.c.created_at))
            if name == "tool_operations":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "call_key", name="uk_namespace_key_tenant_id_call_key"))
            if name == "operation_counters":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "partition_key", name="uk_namespace_key_tenant_id_partition_key"))
            if name in {"idempotency", "evaluation_idempotency"}:
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "identity_key", name="uk_namespace_key_tenant_id_identity_key"))
            if name == "blob_chunks":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "chunk_key", name="uk_namespace_key_tenant_id_chunk_key"))
            registry.add_table(table, owner="adapter.sql")
            tables[name] = table
        return tables


register_sql_schema_contributor("adapter.sql", SqlRuntimeSchema.register_schema)


def ensure_step_metadata() -> "MetaData":
    global step_metadata
    if step_metadata is None:
        from sqlalchemy import MetaData
        step_metadata = MetaData()
    return step_metadata


def new_step_metadata() -> "MetaData":
    global step_metadata
    from sqlalchemy import MetaData
    metadata = MetaData()
    if step_metadata is None:
        step_metadata = metadata
    return metadata


__all__ = ["SqlRuntimeSchema", "ensure_step_metadata", "new_step_metadata", "runtime_metadata", "step_metadata"]
