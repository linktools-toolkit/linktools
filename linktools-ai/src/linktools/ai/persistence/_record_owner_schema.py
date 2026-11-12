#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL columns shared by independently owned optimistic domain rows."""

from typing import TYPE_CHECKING

from ..core import RESOURCE_ID_MAX_BYTES
from ..storage import (
    sql_audit_columns,
    sql_audit_indexes,
    sql_blob,
    sql_id_column,
    sql_query_index,
    sql_sha256,
    sql_state,
    sql_table_options,
    sql_unique,
)
from .state import STORE_KEY_MAX_BYTES, RELATION_KEY_MAX_BYTES, RECORD_KEY_MAX_BYTES, PARENT_KEY_MAX_BYTES

if TYPE_CHECKING:
    from sqlalchemy import MetaData


def build_record_owner_table(metadata: "MetaData", *, name: str, comment: str) -> None:
    """Build one physical owner with the common optimistic record contract."""
    from sqlalchemy import TIMESTAMP, BigInteger, JSON, Column, LargeBinary, String, Table, Text, text
    from sqlalchemy.dialects import mysql

    if name in metadata.tables:
        return
    columns = [
        sql_id_column(),
        Column(
            "store_key",
            LargeBinary(STORE_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(STORE_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the physical State store partition.",
        ),
        Column(
            "key",
            LargeBinary(RECORD_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(RECORD_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the persisted runtime record.",
        ),
        Column(
            "scope_key",
            LargeBinary(RELATION_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(RELATION_KEY_MAX_BYTES), "mysql"),
            nullable=True,
            comment="Bounded binary grouping identity used by the record kind's primary list query.",
        ),
        Column(
            "subject_id",
            LargeBinary(RESOURCE_ID_MAX_BYTES).with_variant(mysql.VARBINARY(RESOURCE_ID_MAX_BYTES), "mysql"),
            nullable=True,
            comment="Exact UTF-8 subject identifier used for lookups within a record scope.",
        ),
        Column(
            "parent_key",
            LargeBinary(PARENT_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(PARENT_KEY_MAX_BYTES), "mysql"),
            nullable=True,
            comment="Bounded binary parent identity used by hierarchical list queries.",
        ),
        Column(
            "kind",
            String(32),
            nullable=False,
            comment="Stable persisted record kind such as session, execution, task_node, or agent_run.",
        ),
        Column(
            "sort_key",
            sql_blob(),
            nullable=False,
            comment="Stable canonical ordering token used for deterministic keyset pagination.",
        ),
        Column(
            "state",
            sql_state(),
            nullable=True,
            comment="Current SQL-queryable state for record kinds with persisted state-machine semantics.",
        ),
        Column(
            "storage_revision",
            BigInteger,
            nullable=False,
            comment="Physical compare-and-swap revision incremented by every record mutation.",
        ),
        Column(
            "lease_owner",
            Text,
            nullable=True,
            comment="Current durable lease owner for record kinds using claim and fencing semantics.",
        ),
        Column(
            "lease_fence",
            BigInteger,
            nullable=False,
            server_default=text("0"),
            comment="Monotonic fencing token for the current durable record lease.",
        ),
        Column(
            "lease_expires_at",
            TIMESTAMP(timezone=True).with_variant(mysql.TIMESTAMP(fsp=6), "mysql"),
            nullable=True,
            comment="Database-authoritative expiry time of the current durable lease.",
        ),
        Column(
            "payload_json",
            JSON(),
            nullable=False,
            comment="Canonical domain JSON encoded as a JSON string preserving numeric representations.",
        ),
        Column("payload_bytes", sql_blob(), nullable=True,
               comment="Optional binary content body owned by this record; ordinary domain state remains JSON."),
        Column("key_locator", sql_sha256(), nullable=False,
               comment="Single SHA-256 lookup projection of the complete key; raw identity must also match."),
        Column("scope_locator", sql_sha256(), nullable=True,
               comment="Single SHA-256 lookup projection of the complete scope_key; raw identity must also match."),
        Column("parent_locator", sql_sha256(), nullable=True,
               comment="Single SHA-256 lookup projection of the complete parent_key; raw identity must also match."),
        Column("sort_prefix", LargeBinary(128).with_variant(mysql.VARBINARY(128), "mysql"), nullable=False,
               comment="First 128 bytes of the full sort key; non-unique query projection only."),
        *sql_audit_columns(),
    ]
    omitted = {"kind", "payload_bytes"} if name == "ai_step_tool_operations" else set()
    records = Table(name, metadata, *(column for column in columns if column.name not in omitted),
                    comment=comment, **sql_table_options())
    sql_unique(records, "store_key", "key_locator")
    if "kind" in records.c:
        sql_query_index(records, "store_key", "kind", "sort_prefix")
    if name != "ai_step_tool_operations":
        sql_query_index(records, "scope_locator", "sort_prefix")
    sql_query_index(records, "scope_locator", "subject_id")
    sql_query_index(records, "parent_locator", "sort_prefix")
    sql_audit_indexes(records)


__all__ = ["build_record_owner_table"]
