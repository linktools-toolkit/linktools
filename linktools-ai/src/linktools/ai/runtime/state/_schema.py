#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical DBA-reviewed SQL metadata for Runtime StateStore primitives."""

from typing import TYPE_CHECKING

from ...storage import (
    sql_audit_columns,
    sql_audit_indexes,
    sql_id_column,
    sql_query_index,
    sql_sha256,
    sql_sort_key,
    sql_state,
    sql_table_options,
    sql_unique,
)
from ._plan import RuntimeDomain, RuntimeStatePlan

if TYPE_CHECKING:
    from sqlalchemy import MetaData


_RECORD_COMMENT = "Current durable state for runtime and step resources persisted as versioned records."
_ALIAS_COMMENT = "Secondary unique lookup identities that resolve to canonical runtime records."
_FACT_COMMENT = "Immutable ordered execution and step facts including events, snapshots, and tool effects."
_SEQUENCE_COMMENT = "Durable monotonic counters used to allocate ordered runtime and step sequence numbers."
_OPERATION_COMMENT = "Ordered durable operation ledger for replay, result recovery, status transition, and compaction."


def build_runtime_sql_metadata(
    plan: "RuntimeStatePlan | frozenset[RuntimeDomain]",
    *,
    metadata: "MetaData | None" = None,
) -> "MetaData":
    """Build the five canonical Runtime primitive tables."""
    from sqlalchemy import JSON, BigInteger, Boolean, Column, MetaData, String, Table, Text, TIMESTAMP, text
    from sqlalchemy.dialects import mysql

    if metadata is None:
        metadata = MetaData()
    if "ai_state_records" in metadata.tables:
        return metadata
    if isinstance(plan, frozenset) and not plan:
        raise ValueError("at least one RuntimeDomain is required")
    if isinstance(plan, RuntimeStatePlan) and not any(
        plan.route(domain).retention.value == "durable" for domain in RuntimeDomain
    ):
        raise ValueError("at least one durable RuntimeDomain is required")

    digest = sql_sha256()
    records = Table(
        "ai_state_records",
        metadata,
        sql_id_column(),
        Column(
            "key_digest", digest, nullable=False, comment="Canonical SHA-256 identity of the persisted runtime record."
        ),
        Column(
            "partition_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 partition for records of the same tenant, runtime domain, and record kind.",
        ),
        Column(
            "scope_digest",
            digest,
            nullable=True,
            comment="Canonical SHA-256 grouping key used by the record kind's primary list query.",
        ),
        Column(
            "parent_digest",
            digest,
            nullable=True,
            comment="Canonical SHA-256 identity of the logical parent used by hierarchical list queries.",
        ),
        Column(
            "kind",
            String(32),
            nullable=False,
            comment="Stable persisted record kind such as session, execution, task_node, or step_run.",
        ),
        Column(
            "sort_key",
            sql_sort_key(),
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
            "storage_version",
            BigInteger,
            nullable=False,
            comment="Internal optimistic-concurrency version incremented by every physical record mutation.",
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
            JSON,
            nullable=False,
            comment="Versioned canonical domain payload not needed by SQL identity, query, CAS, or lease predicates.",
        ),
        *_audit(),
        comment=_RECORD_COMMENT,
        **sql_table_options(),
    )
    sql_unique(records, "key_digest")
    sql_query_index(records, "partition_digest", "sort_key")
    sql_query_index(records, "scope_digest", "sort_key")
    sql_query_index(records, "scope_digest", "state", "sort_key")
    sql_query_index(records, "parent_digest", "sort_key")
    sql_audit_indexes(records)

    aliases = Table(
        "ai_state_aliases",
        metadata,
        sql_id_column(),
        Column(
            "alias_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of a secondary unique runtime lookup key.",
        ),
        Column(
            "record_key_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the runtime record resolved by this alias.",
        ),
        *_audit(),
        comment=_ALIAS_COMMENT,
        **sql_table_options(),
    )
    sql_unique(aliases, "alias_digest")
    sql_query_index(aliases, "record_key_digest")
    sql_audit_indexes(aliases)

    facts = Table(
        "ai_state_facts",
        metadata,
        sql_id_column(),
        Column(
            "stream_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the append-only fact stream.",
        ),
        Column(
            "sequence",
            BigInteger,
            nullable=False,
            comment="Strictly increasing position of the fact within its stream.",
        ),
        Column(
            "owner_key_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the runtime record that owns this fact.",
        ),
        Column(
            "kind",
            String(32),
            nullable=False,
            comment="Fact kind such as execution_event, step_event, step_snapshot, or step_effect.",
        ),
        Column(
            "subject_digest",
            digest,
            nullable=True,
            comment="Canonical SHA-256 grouping identity for multiple facts of one logical subject such as a tool call.",
        ),
        Column(
            "state",
            sql_state(),
            nullable=True,
            comment="Queryable fact state such as snapshot completeness or tool-effect lifecycle state.",
        ),
        Column("payload_json", JSON, nullable=False, comment="Versioned canonical immutable fact payload."),
        *_audit(),
        comment=_FACT_COMMENT,
        **sql_table_options(),
    )
    sql_unique(facts, "stream_digest", "sequence")
    sql_query_index(facts, "owner_key_digest")
    sql_query_index(facts, "stream_digest", "subject_digest", "sequence")
    sql_audit_indexes(facts)

    sequences = Table(
        "ai_state_sequences",
        metadata,
        sql_id_column(),
        Column(
            "key_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the monotonic sequence counter.",
        ),
        Column("value", BigInteger, nullable=False, comment="Last committed value allocated by the sequence."),
        *_audit(),
        comment=_SEQUENCE_COMMENT,
        **sql_table_options(),
    )
    sql_unique(sequences, "key_digest")
    sql_audit_indexes(sequences)

    operations = Table(
        "ai_state_operations",
        metadata,
        sql_id_column(),
        Column("key_digest", digest, nullable=False, comment="Canonical SHA-256 identity of the durable operation."),
        Column(
            "stream_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the ordered operation stream.",
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
            JSON,
            nullable=False,
            comment="Versioned canonical operation request, result, error, resource identity, and semantic timestamps.",
        ),
        *_audit(),
        comment=_OPERATION_COMMENT,
        **sql_table_options(),
    )
    sql_unique(operations, "key_digest")
    sql_unique(operations, "stream_digest", "sequence")
    sql_query_index(operations, "stream_digest", "state", "sequence")
    sql_audit_indexes(operations)
    return metadata


def required_runtime_sql_tables(plan: RuntimeStatePlan) -> frozenset[str]:
    """Return the stable Runtime table set for a durable plan."""
    if not any(plan.route(domain).retention.value == "durable" for domain in RuntimeDomain):
        return frozenset()
    return frozenset(
        {
            "ai_state_records",
            "ai_state_aliases",
            "ai_state_facts",
            "ai_state_sequences",
            "ai_state_operations",
        }
    )


def _audit() -> tuple[object, object]:
    return sql_audit_columns()


__all__ = [
    "build_runtime_sql_metadata",
    "required_runtime_sql_tables",
]
