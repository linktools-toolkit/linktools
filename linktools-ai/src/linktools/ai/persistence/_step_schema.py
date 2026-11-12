#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL metadata for the public upstream step-persistence records."""

from typing import TYPE_CHECKING
from collections.abc import Iterable

from ..core import RESOURCE_ID_MAX_BYTES
from ..steps import NATIVE_STORE_KEY_MAX_BYTES, NATIVE_REPLAY_KEY_MAX_BYTES
from ..storage import (
    sql_audit_columns,
    sql_audit_indexes,
    sql_blob,
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
    from ._plan import PersistenceDomain
    from sqlalchemy import Column, MetaData, Table


def build_step_sql_metadata(*, metadata: "MetaData | None" = None) -> "MetaData":
    """Build scoped upstream run, event, snapshot and message tables."""
    from sqlalchemy import BigInteger, Boolean, Column, JSON, LargeBinary, MetaData, SmallInteger, String, Table
    from sqlalchemy import text as sql_text
    from sqlalchemy.dialects.mysql import SMALLINT, VARBINARY

    if metadata is None:
        metadata = MetaData()

    def text(name: str, comment: str, *, nullable: bool = True) -> "Column":
        return Column(name, sql_sort_key(), nullable=nullable, comment=comment)

    def identifier(name: str, comment: str, *, nullable: bool = True) -> "Column":
        return Column(name, sql_text_key(RESOURCE_ID_MAX_BYTES), nullable=nullable, comment=comment)

    def partition() -> "Column":
        return Column("store_key", LargeBinary(NATIVE_STORE_KEY_MAX_BYTES).with_variant(
            VARBINARY(NATIVE_STORE_KEY_MAX_BYTES), "mysql"), nullable=False,
            comment="Exact bounded binary native namespace and tenant partition.")

    def replay() -> "Column":
        return Column("idempotency_key", LargeBinary(NATIVE_REPLAY_KEY_MAX_BYTES).with_variant(
            VARBINARY(NATIVE_REPLAY_KEY_MAX_BYTES), "mysql"), nullable=False,
                      comment="Exact UTF-8 replay key or a reserved unique occurrence key for an unkeyed append.")

    def digest(name: str, comment: str, *, nullable: bool = False) -> "Column":
        return Column(name, sql_sha256(), nullable=nullable, comment=comment)

    def table(name: str, comment: str, *columns: "Column") -> "Table":
        if any(column.name == "store_key" for column in columns):
            columns += (digest("store_locator", "Single SHA-256 lookup projection of the complete store_key; raw identity must also match."),)
        result = Table(
            name, metadata, sql_id_column(),
            *columns, *sql_audit_columns(), comment=comment,
            sqlite_autoincrement=True,
            **sql_table_options(),
        )
        sql_audit_indexes(result)
        return result

    if "ai_step_runs" not in metadata.tables:
        runs = table(
            "ai_step_runs", "Upstream agent run lineage and registration records.",
            partition(),
            identifier("run_id", "Original upstream run identifier.", nullable=False),
            identifier("conversation_id", "Original upstream conversation identifier."),
            identifier("parent_run_id", "Original upstream parent run identifier."),
            text("agent_name", "Upstream agent name."),
            text("metadata", "Upstream string metadata encoded as JSON.", nullable=False),
            Column("started_at", String(64), nullable=False, comment="Original ISO 8601 run start time."),
            text("registration_id", "Upstream registration invocation identifier."),
            Column("managed_start_key", LargeBinary(NATIVE_REPLAY_KEY_MAX_BYTES).with_variant(
                VARBINARY(NATIVE_REPLAY_KEY_MAX_BYTES), "mysql"), nullable=True,
                comment="Exact first managed start identity; NULL means metadata admission only."),
        )
        sql_unique(runs, "store_locator", "run_id")
        sql_query_index(runs, "store_locator", "conversation_id")
        sql_query_index(runs, "store_locator", "parent_run_id")

    if "ai_step_events" not in metadata.tables:
        events = table(
            "ai_step_events", "Append-only upstream agent execution boundary events.",
            partition(),
            identifier("run_id", "Original upstream run identifier.", nullable=False),
            Column("kind", sql_state(), nullable=False, comment="Upstream boundary event kind."),
            Column("outcome", sql_state(), nullable=True, comment="Owned lifecycle outcome, absent for ordinary upstream events."),
            Column("step_index", BigInteger, nullable=False, comment="Upstream graph step index."),
            Column("timestamp", String(64), nullable=False, comment="Original ISO 8601 event time."),
            identifier("conversation_id", "Original upstream conversation identifier."),
            identifier("parent_run_id", "Original upstream parent run identifier."),
            text("agent_name", "Upstream agent name."),
            text("tool_call_id", "Upstream tool call identifier."),
            text("tool_name", "Upstream tool name."),
            text("error", "Upstream boundary error text."),
            Column("metadata", JSON(none_as_null=True), nullable=True,
                   comment="Upstream string metadata encoded as JSON, unless referenced from the native run."),
            Column("run_fields", SmallInteger().with_variant(SMALLINT(unsigned=True), "mysql"),
                   nullable=False, server_default=sql_text("0"),
                   comment="Native run field references: conversation 1, parent 2, agent 4, metadata 8."),
            digest("body_sha256", "Integrity digest of the managed lifecycle source body.", nullable=True),
            digest("fields_sha256", "Integrity digest of managed lifecycle fields excluding metadata.", nullable=True),
            replay(),
        )
        sql_unique(events, "store_locator", "run_id", "idempotency_key")
        # Replay order differs from occurrence order; this index serves ordered run reads.
        sql_query_index(events, "store_locator", "run_id")

    if "ai_step_snapshots" not in metadata.tables:
        snapshots = table(
            "ai_step_snapshots", "Upstream complete or interrupted message-history snapshots.",
            partition(),
            identifier("run_id", "Original upstream run identifier.", nullable=False),
            Column("step_index", BigInteger, nullable=False, comment="Upstream graph step index."),
            identifier("conversation_id", "Original upstream conversation identifier."),
            identifier("parent_run_id", "Original upstream parent run identifier."),
            text("agent_name", "Upstream agent name."),
            Column("timestamp", String(64), nullable=False, comment="Original ISO 8601 snapshot time."),
            Column("state", sql_state(), nullable=False, comment="Upstream complete or interrupted state."),
            Column("messages", sql_text_key(1024), nullable=False,
                   comment="Bounded message descriptor and native checkpoint state without message bodies."),
            digest("sequence_root", "Immutable message sequence root retained by this snapshot.", nullable=True),
            replay(),
            Column("pruned", Boolean, nullable=False, default=False, comment="Retired snapshot identity without retained message references."),
        )
        sql_unique(snapshots, "store_locator", "run_id", "idempotency_key")
        sql_query_index(snapshots, "store_locator", "run_id", "pruned")

    if "ai_step_messages" not in metadata.tables:
        messages = table(
            "ai_step_messages", "Immutable typed content bytes: native messages and accepted execution payloads in disjoint partitions.",
            partition(),
            digest("sha256", "SHA-256 of the canonical uncompressed bytes for the partition content type."),
            Column("message", sql_blob(), nullable=True, comment="Raw or compressed canonical content bytes, absent for native attachment manifests."),
            Column("compression", sql_state(), nullable=False, comment="Message body compression codec: raw or zlib."),
            Column("raw_size", BigInteger, nullable=False, comment="Canonical uncompressed content byte count for bounded decompression."),
            Column("payload_json", JSON(none_as_null=True), nullable=True,
                   comment="Typed attachment manifest without attachment bytes, absent for ordinary message bodies."),
            Column("gc_revision", BigInteger, nullable=False, server_default=sql_text("0"),
                   comment="Optimistic content revision used by the owning partition."),
        )
        sql_unique(messages, "store_locator", "sha256")

    if "ai_step_sequences" not in metadata.tables:
        sequences = table(
            "ai_step_sequences", "Shared immutable message sequence nodes for native snapshots.",
            partition(),
            digest("sha256", "SHA-256 of the immutable sequence node."),
            digest("message_sha256", "SHA-256 of the referenced upstream message JSON."),
            Column("tree_count", BigInteger, nullable=False, comment="Number of ordered message occurrences in this tree."),
            digest("left_sha256", "Immutable left subtree root.", nullable=True),
            digest("right_sha256", "Immutable right subtree root.", nullable=True),
            digest("next_sha256", "Next immutable tree in the sequence forest.", nullable=True),
        )
        sql_unique(sequences, "store_locator", "sha256")

    return metadata


def build_step_domain_sql_metadata(domains: "Iterable[PersistenceDomain]", *, metadata: "MetaData | None" = None) -> "MetaData":
    """Keep the canonical native tables required by the selected runtime domains."""
    from sqlalchemy import MetaData
    from ._plan import PersistenceDomain

    selected = frozenset(domains)
    metadata = MetaData() if metadata is None else metadata
    if not selected.intersection({PersistenceDomain.CONVERSATION, PersistenceDomain.EXECUTION,
                                  PersistenceDomain.TRACE, PersistenceDomain.CHECKPOINT}):
        return metadata
    existing = set(metadata.tables)
    build_step_sql_metadata(metadata=metadata)
    if PersistenceDomain.CHECKPOINT not in selected:
        keep = {"ai_step_runs", "ai_step_messages", "ai_step_sequences"}
        if selected.intersection({PersistenceDomain.EXECUTION, PersistenceDomain.TRACE}):
            keep.update({"ai_step_runs", "ai_step_events"})
        for name in set(metadata.tables) - existing - keep:
            metadata.remove(metadata.tables[name])
    return metadata


__all__ = ["build_step_sql_metadata", "build_step_domain_sql_metadata"]
