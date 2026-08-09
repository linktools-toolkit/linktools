#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized SQL tables owned by the Runtime SQL adapter."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..storage import SqlSchemaRegistry
from ..storage import storage_name

runtime_metadata: object | None = None
step_metadata: object | None = None

if TYPE_CHECKING:
    from sqlalchemy import CHAR, Index, MetaData, String, Table


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
        from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, JSON, LargeBinary, String, Table, UniqueConstraint
        from sqlalchemy.dialects import mysql

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
            columns = [
                Column("id", integer_id, primary_key=True, autoincrement=True),
                Column("namespace_key", _hex64(), nullable=False),
                Column("tenant_id", _text_key(128), nullable=False),
                Column("record_id", _text_key(256), nullable=False),
                Column("session_id", _text_key(256), nullable=name != "sessions"),
                Column("parent_execution_id", _text_key(256), nullable=True),
                Column("source_execution_id", _text_key(256), nullable=True),
                Column("base_execution_id", _text_key(256), nullable=True),
                Column("lineage_kind", _text_key(64), nullable=False, default="RUN"),
                Column("sequence", BigInteger, nullable=False, default=0),
                Column("revision", BigInteger, nullable=False, default=0),
                Column("status", _text_key(64), nullable=False, default=""),
                Column("payload", JSON, nullable=False),
            ]
            if name == "sessions":
                columns.extend((
                    Column("profile", _text_key(64), nullable=False, default=""),
                    Column("head_execution_id", _text_key(256), nullable=True),
                ))
            if name == "executions":
                columns.append(Column("agent_run_sequence", BigInteger, nullable=False, default=0))
            if name == "tool_operations":
                columns.extend((
                    Column("run_id", _text_key(256), nullable=False),
                    Column("tool_call_id", _text_key(256), nullable=False),
                    Column("call_key", _hex64(), nullable=False, default=""),
                    Column("owner", _text_key(256), nullable=True),
                    Column("fence", BigInteger, nullable=True),
                    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
                ))
            if name == "task_nodes":
                columns.extend((
                    Column("owner", String(512), nullable=True),
                    Column("fence", BigInteger, nullable=False, default=0),
                    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
                ))
            if name == "operation_counters":
                columns.extend((
                    Column("resource_kind", _text_key(64), nullable=False, default=""),
                    Column("resource_id", _text_key(256), nullable=False, default=""),
                    Column("partition_key", _hex64(), nullable=False, default=""),
                ))
            if name == "idempotency":
                columns.extend((
                    Column("scope", _text_key(64), nullable=False, default=""),
                    Column("key_hash", _hex64(), nullable=False, default=""),
                    Column("identity_key", _hex64(), nullable=False, default=""),
                ))
            if name == "operation_ledger":
                columns.extend((
                    Column("resource_kind", _text_key(64), nullable=False, default=""),
                    Column("resource_id", _text_key(256), nullable=False, default=""),
                ))
            if name == "blobs":
                columns.extend((
                    Column("digest", _hex64(), nullable=False, default=""),
                    Column("size", BigInteger, nullable=False, default=0),
                ))
            if name == "blob_chunks":
                columns.extend((
                    Column("digest", _hex64(), nullable=False, default=""),
                    Column("chunk_index", BigInteger, nullable=False, default=0),
                    Column("chunk_key", _hex64(), nullable=False, default=""),
                    Column("content", LargeBinary().with_variant(mysql.LONGBLOB(), "mysql"), nullable=False, default=b""),
                ))
            columns.extend((
                Column("updated_at", DateTime(timezone=True), nullable=False),
                Column("created_at", DateTime(timezone=True), nullable=False),
            ))
            table = Table(
                storage_name(physical_names[name]),
                registry.metadata,
                *columns,
                mysql_engine="InnoDB",
                mysql_charset="utf8mb4",
                mysql_collate="utf8mb4_bin",
                extend_existing=True,
            )
            if name == "sessions":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "session_id", name="uk_namespace_key_tenant_id_session_id"))
            if name == "executions":
                _mysql_index(Index("ix_namespace_key_tenant_id_session_id", table.c.namespace_key, table.c.tenant_id, table.c.session_id))
                _mysql_index(Index("ix_namespace_key_tenant_id_source_execution_id", table.c.namespace_key, table.c.tenant_id, table.c.source_execution_id))
                _mysql_index(Index("ix_namespace_key_tenant_id_base_execution_id", table.c.namespace_key, table.c.tenant_id, table.c.base_execution_id))
                _mysql_index(Index("ix_namespace_key_tenant_id_parent_execution_id", table.c.namespace_key, table.c.tenant_id, table.c.parent_execution_id))
            table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "record_id", name="uk_namespace_key_tenant_id_record_id"))
            _mysql_index(Index("ix_updated_at", table.c.updated_at))
            _mysql_index(Index("ix_created_at", table.c.created_at))
            if name == "tool_operations":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "call_key", name="uk_namespace_key_tenant_id_call_key"))
            if name == "operation_counters":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "partition_key", name="uk_namespace_key_tenant_id_partition_key"))
            if name == "idempotency":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "identity_key", name="uk_namespace_key_tenant_id_identity_key"))
            if name == "blobs":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "digest", name="uk_namespace_key_tenant_id_digest"))
            if name == "blob_chunks":
                table.append_constraint(UniqueConstraint("namespace_key", "tenant_id", "chunk_key", name="uk_namespace_key_tenant_id_chunk_key"))
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
    return CHAR(64).with_variant(mysql.CHAR(64, charset="utf8mb4", collation="utf8mb4_bin"), "mysql")


def _mysql_index(index: "Index") -> "Index":
    index.info["ddl_dialect"] = "mysql"
    return index.ddl_if(dialect="mysql")


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


__all__ = ["SqlRuntimeSchema", "SqlRuntimeTables", "ensure_step_metadata", "new_step_metadata", "runtime_metadata", "step_metadata"]
