#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL metadata for the public upstream step-persistence records."""

from typing import TYPE_CHECKING

from ..storage import (
    sql_audit_columns,
    sql_audit_indexes,
    sql_id_column,
    sql_query_index,
    sql_sha256,
    sql_sort_key,
    sql_state,
    sql_table_options,
    sql_text_key,
    sql_unique,
)

if TYPE_CHECKING:
    from sqlalchemy import Column, MetaData, Table


def build_step_sql_metadata(*, metadata: "MetaData | None" = None) -> "MetaData":
    """Build scoped upstream run, event, snapshot and message tables."""
    from sqlalchemy import BigInteger, Boolean, Column, JSON, MetaData, SmallInteger, String, Table
    from sqlalchemy import text as sql_text
    from sqlalchemy.dialects.mysql import SMALLINT

    if metadata is None:
        metadata = MetaData()
    if "ai_step_runs" in metadata.tables:
        return metadata

    def text(name: str, comment: str, *, nullable: bool = True) -> "Column":
        return Column(name, sql_sort_key(), nullable=nullable, comment=comment)

    def digest(name: str, comment: str, *, nullable: bool = False) -> "Column":
        return Column(name, sql_sha256(), nullable=nullable, comment=comment)

    def table(name: str, comment: str, *columns: "Column") -> "Table":
        result = Table(
            name, metadata, sql_id_column(),
            *columns, *sql_audit_columns(), comment=comment,
            **sql_table_options(),
        )
        sql_audit_indexes(result)
        return result

    def run_columns() -> "tuple[Column, Column]":
        return (
            digest("run_digest", "Physical lookup digest of the original run identifier."),
            text("run_id", "Original upstream run identifier.", nullable=False),
        )

    runs = table(
        "ai_step_runs", "Upstream agent run lineage and registration records.",
        digest("store_digest", "Physical namespace and tenant partition."),
        *run_columns(),
        text("conversation_id", "Original upstream conversation identifier."),
        digest("conversation_digest", "Physical conversation lookup digest.", nullable=True),
        text("parent_run_id", "Original upstream parent run identifier."),
        digest("parent_digest", "Physical parent run lookup digest.", nullable=True),
        text("agent_name", "Upstream agent name."),
        text("metadata", "Upstream string metadata encoded as JSON.", nullable=False),
        Column("started_at", String(64), nullable=False, comment="Original ISO 8601 run start time."),
        text("registration_id", "Upstream registration invocation identifier."),
    )
    sql_unique(runs, "store_digest", "run_digest")
    sql_query_index(runs, "store_digest", "conversation_digest")
    sql_query_index(runs, "store_digest", "parent_digest")

    events = table(
        "ai_step_events", "Append-only upstream agent execution boundary events.",
        digest("store_digest", "Physical namespace and tenant partition."),
        *run_columns(),
        Column("kind", sql_state(), nullable=False, comment="Upstream boundary event kind."),
        Column("step_index", BigInteger, nullable=False, comment="Upstream graph step index."),
        Column("timestamp", String(64), nullable=False, comment="Original ISO 8601 event time."),
        text("conversation_id", "Original upstream conversation identifier."),
        text("parent_run_id", "Original upstream parent run identifier."),
        text("agent_name", "Upstream agent name."),
        text("tool_call_id", "Upstream tool call identifier."),
        text("tool_name", "Upstream tool name."),
        text("error", "Upstream boundary error text."),
        Column("metadata", JSON(none_as_null=True), nullable=True,
               comment="Upstream string metadata encoded as JSON, unless referenced from the native run."),
        Column("run_fields", SmallInteger().with_variant(SMALLINT(unsigned=True), "mysql"),
               nullable=False, server_default=sql_text("0"),
               comment="Native run field references: conversation 1, parent 2, agent 4, metadata 8."),
        text("idempotency_key", "Upstream boundary replay key."),
        digest("key_digest", "SHA-256 identity of the scoped run and replay key or append occurrence."),
    )
    sql_unique(events, "key_digest")
    sql_query_index(events, "store_digest", "run_digest")

    snapshots = table(
        "ai_step_snapshots", "Upstream complete or interrupted message-history snapshots.",
        digest("store_digest", "Physical namespace and tenant partition."),
        *run_columns(),
        Column("step_index", BigInteger, nullable=False, comment="Upstream graph step index."),
        text("conversation_id", "Original upstream conversation identifier."),
        text("parent_run_id", "Original upstream parent run identifier."),
        text("agent_name", "Upstream agent name."),
        Column("timestamp", String(64), nullable=False, comment="Original ISO 8601 snapshot time."),
        Column("state", sql_state(), nullable=False, comment="Upstream complete or interrupted state."),
        Column("messages", sql_text_key(1024), nullable=False,
               comment="Bounded message descriptor and native checkpoint state without message bodies."),
        digest("sequence_root", "Immutable message sequence root retained by this snapshot.", nullable=True),
        text("idempotency_key", "Upstream boundary replay key."),
        digest("key_digest", "SHA-256 identity of the scoped run and replay key or append occurrence."),
        Column("pruned", Boolean, nullable=False, default=False, comment="Retired snapshot identity without retained message references."),
    )
    sql_unique(snapshots, "key_digest")
    sql_query_index(snapshots, "store_digest", "run_digest")

    messages = table(
        "ai_step_messages", "Deduplicated upstream ModelMessage JSON bodies.",
        digest("store_digest", "Physical namespace and tenant partition."),
        digest("sha256", "SHA-256 of the exact stored upstream message JSON."),
        text("message", "One upstream ModelMessage JSON body.", nullable=False),
    )
    sql_unique(messages, "store_digest", "sha256")

    sequences = table(
        "ai_step_sequences", "Shared immutable message sequence nodes for native snapshots.",
        digest("store_digest", "Physical namespace and tenant partition."),
        digest("sha256", "SHA-256 of the immutable sequence node."),
        digest("message_sha256", "SHA-256 of the referenced upstream message JSON."),
        Column("tree_count", BigInteger, nullable=False, comment="Number of ordered message occurrences in this tree."),
        digest("left_sha256", "Immutable left subtree root.", nullable=True),
        digest("right_sha256", "Immutable right subtree root.", nullable=True),
        digest("next_sha256", "Next immutable tree in the sequence forest.", nullable=True),
    )
    sql_unique(sequences, "store_digest", "sha256")
    return metadata


__all__ = ["build_step_sql_metadata"]
