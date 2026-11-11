#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical SQL metadata for Runtime StateStore primitives."""

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
from ._plan import PersistenceDomain, PersistencePlan
from .state import STORE_KEY_MAX_BYTES, STREAM_KEY_MAX_BYTES, ALIAS_KEY_MAX_BYTES, OPERATION_KEY_MAX_BYTES, RELATION_KEY_MAX_BYTES, RECORD_KEY_MAX_BYTES, PARENT_KEY_MAX_BYTES

if TYPE_CHECKING:
    from sqlalchemy import MetaData


_RECORD_COMMENT = "Current durable state for runtime and step resources persisted as versioned records."
_ALIAS_COMMENT = (
    "Secondary unique lookup identities that resolve to canonical runtime records."
)
_FACT_COMMENT = (
    "Immutable ordered runtime facts including events, checkpoints, and effects."
)
_SEQUENCE_COMMENT = (
    "Durable monotonic counters used to allocate ordered runtime sequence numbers."
)
_OPERATION_COMMENT = "Ordered durable operation ledger for replay, result recovery, status transition, and compaction."


def build_runtime_sql_metadata(
    plan: "PersistencePlan | frozenset[PersistenceDomain]",
    *,
    metadata: "MetaData | None" = None,
) -> "MetaData":
    """Build Runtime primitive tables and the selected business-domain tables."""
    from sqlalchemy import (
        TIMESTAMP,
        BigInteger,
        JSON,
        Boolean,
        Column,
        LargeBinary,
        MetaData,
        String,
        Table,
        Text,
        text,
    )
    from sqlalchemy.dialects import mysql

    if metadata is None:
        metadata = MetaData()
    trace_selected = (PersistenceDomain.TRACE in plan if isinstance(plan, frozenset)
                      else plan.route(PersistenceDomain.TRACE).retention.value == "durable")
    if trace_selected:
        from ._trace_schema import build_trace_sql_metadata
        build_trace_sql_metadata(metadata=metadata)
    if "ai_state_records" in metadata.tables:
        return metadata
    if isinstance(plan, frozenset) and not plan:
        raise ValueError("at least one PersistenceDomain is required")
    if isinstance(plan, PersistencePlan) and not any(
        plan.route(domain).retention.value == "durable" for domain in PersistenceDomain
    ):
        raise ValueError("at least one durable PersistenceDomain is required")

    records = Table(
        "ai_state_records",
        metadata,
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
        *_audit(),
        comment=_RECORD_COMMENT,
        **sql_table_options(),
    )
    sql_unique(records, "store_key", "key_locator")
    sql_query_index(records, "store_key", "kind", "sort_prefix")
    sql_query_index(records, "scope_locator", "sort_prefix")
    sql_query_index(records, "scope_locator", "subject_id")
    sql_query_index(records, "parent_locator", "sort_prefix")
    sql_audit_indexes(records)

    aliases = Table(
        "ai_state_aliases",
        metadata,
        sql_id_column(),
        Column(
            "store_key",
            LargeBinary(STORE_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(STORE_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the physical State store partition.",
        ),
        Column(
            "alias_key",
            LargeBinary(ALIAS_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(ALIAS_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Binary secondary lookup address with a maximum of 768 bytes.",
        ),
        Column(
            "record_key",
            LargeBinary(RECORD_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(RECORD_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the runtime record resolved by this alias.",
        ),
        Column("alias_locator", sql_sha256(), nullable=False,
               comment="Single SHA-256 lookup projection of the complete alias_key; raw identity must also match."),
        Column("record_locator", sql_sha256(), nullable=False,
               comment="Single SHA-256 lookup projection of the complete record_key; raw identity must also match."),
        *_audit(),
        comment=_ALIAS_COMMENT,
        **sql_table_options(),
    )
    sql_unique(aliases, "store_key", "alias_locator")
    sql_query_index(aliases, "store_key", "record_locator")
    sql_audit_indexes(aliases)

    facts = Table(
        "ai_state_facts",
        metadata,
        sql_id_column(),
        Column(
            "store_key",
            LargeBinary(STORE_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(STORE_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the physical State store partition.",
        ),
        Column(
            "stream_key",
            LargeBinary(STREAM_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(STREAM_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the append-only fact stream.",
        ),
        Column(
            "sequence",
            BigInteger,
            nullable=False,
            comment="Strictly increasing position of the fact within its stream.",
        ),
        Column(
            "owner_key",
            LargeBinary(RECORD_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(RECORD_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the runtime record that owns this fact.",
        ),
        Column(
            "kind",
            String(32),
            nullable=False,
            comment="Fact kind such as execution_event, step_event, step_checkpoint, or step_effect.",
        ),
        Column(
            "subject_id",
            LargeBinary(RESOURCE_ID_MAX_BYTES).with_variant(mysql.VARBINARY(RESOURCE_ID_MAX_BYTES), "mysql"),
            nullable=True,
            comment="Exact UTF-8 subject identifier grouping facts such as one tool call.",
        ),
        Column(
            "state",
            sql_state(),
            nullable=True,
            comment="Queryable fact state such as checkpoint completeness or tool-effect lifecycle state.",
        ),
        Column(
            "payload_json",
            JSON(),
            nullable=False,
            comment="Canonical immutable fact JSON encoded as a JSON string preserving numeric representations.",
        ),
        Column("stream_locator", sql_sha256(), nullable=False,
               comment="Single SHA-256 lookup projection of the complete stream_key; raw identity must also match."),
        Column("owner_locator", sql_sha256(), nullable=False,
               comment="Single SHA-256 lookup projection of the complete owner_key; raw identity must also match."),
        *_audit(),
        comment=_FACT_COMMENT,
        **sql_table_options(),
    )
    sql_unique(facts, "store_key", "stream_locator", "sequence")
    sql_query_index(facts, "owner_locator")
    sql_query_index(facts, "stream_locator", "subject_id", "sequence")
    sql_audit_indexes(facts)

    sequences = Table(
        "ai_state_sequences",
        metadata,
        sql_id_column(),
        Column(
            "store_key",
            LargeBinary(STORE_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(STORE_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the physical State store partition.",
        ),
        Column(
            "stream_key",
            LargeBinary(STREAM_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(STREAM_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the monotonic sequence counter.",
        ),
        Column(
            "value",
            BigInteger,
            nullable=False,
            comment="Last committed value allocated by the sequence.",
        ),
        Column("stream_locator", sql_sha256(), nullable=False,
               comment="Single SHA-256 lookup projection of the complete stream_key; raw identity must also match."),
        *_audit(),
        comment=_SEQUENCE_COMMENT,
        **sql_table_options(),
    )
    sql_unique(sequences, "store_key", "stream_locator")
    sql_audit_indexes(sequences)

    operations = Table(
        "ai_state_operations",
        metadata,
        sql_id_column(),
        Column(
            "store_key",
            LargeBinary(STORE_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(STORE_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the physical State store partition.",
        ),
        Column(
            "operation_key",
            LargeBinary(OPERATION_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(OPERATION_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Binary operation address, at most 768 bytes.",
        ),
        Column(
            "stream_key",
            LargeBinary(STREAM_KEY_MAX_BYTES).with_variant(mysql.VARBINARY(STREAM_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact bounded binary identity of the ordered operation stream.",
        ),
        Column(
            "sequence",
            BigInteger,
            nullable=False,
            comment="Strictly increasing position of the operation within its stream.",
        ),
        Column(
            "state",
            sql_state(),
            nullable=False,
            comment="Current durable operation state used by replay and completion logic.",
        ),
        Column(
            "compactable",
            Boolean,
            nullable=False,
            comment="Whether a terminal operation may be removed by ledger compaction.",
        ),
        Column(
            "payload_json",
            JSON(),
            nullable=False,
            comment="Canonical operation JSON encoded as a JSON string preserving numeric representations.",
        ),
        Column("operation_locator", sql_sha256(), nullable=False,
               comment="Single SHA-256 lookup projection of the complete operation_key; raw identity must also match."),
        Column("stream_locator", sql_sha256(), nullable=False,
               comment="Single SHA-256 lookup projection of the complete stream_key; raw identity must also match."),
        *_audit(),
        comment=_OPERATION_COMMENT,
        **sql_table_options(),
    )
    sql_unique(operations, "store_key", "operation_locator")
    sql_unique(operations, "store_key", "stream_locator", "sequence")
    sql_query_index(operations, "stream_locator", "state", "sequence")
    sql_audit_indexes(operations)
    return metadata


def required_runtime_sql_tables(plan: PersistencePlan) -> frozenset[str]:
    """Return the stable Runtime table set for a durable plan."""
    if not any(
        plan.route(domain).retention.value == "durable" for domain in PersistenceDomain
    ):
        return frozenset()
    tables = {
        "ai_state_records",
        "ai_state_aliases",
        "ai_state_facts",
        "ai_state_sequences",
        "ai_state_operations",
    }
    if plan.route(PersistenceDomain.TRACE).retention.value == "durable":
        tables.add("ai_trace_events")
    return frozenset(tables)


def _audit() -> tuple[object, object]:
    return sql_audit_columns()


__all__ = [
    "build_runtime_sql_metadata",
    "required_runtime_sql_tables",
]
