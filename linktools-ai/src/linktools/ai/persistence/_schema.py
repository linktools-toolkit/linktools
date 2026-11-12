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
from .state import STORE_KEY_MAX_BYTES, STREAM_KEY_MAX_BYTES, OPERATION_KEY_MAX_BYTES, RELATION_KEY_MAX_BYTES, RECORD_KEY_MAX_BYTES, PARENT_KEY_MAX_BYTES

if TYPE_CHECKING:
    from sqlalchemy import MetaData


_RECORD_COMMENT = "Current durable state for runtime and step resources persisted as versioned records."
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
    if isinstance(plan, frozenset):
        # Execution's default Trace route owns retained history in this schema.
        trace_selected = bool(plan.intersection({PersistenceDomain.EXECUTION, PersistenceDomain.TRACE}))
    else:
        trace_selected = plan.route(PersistenceDomain.TRACE).retention.value == "durable"
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

    from ._record_owner_schema import build_record_owner_table
    build_record_owner_table(metadata, name="ai_state_records", comment=_RECORD_COMMENT)
    build_record_owner_table(metadata, name="ai_step_tool_operations",
                             comment="Authoritative Step tool effect ledger, including dispatch fencing and durable receipts.")
    from ._run_admission_schema import build_run_admission_sql_metadata
    build_run_admission_sql_metadata(metadata)


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
        "ai_state_counters",
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
        "ai_state_facts",
        "ai_state_counters",
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
