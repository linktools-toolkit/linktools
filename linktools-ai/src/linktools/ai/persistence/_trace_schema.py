#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical SQL metadata for ordered Trace events and their query fields."""

from typing import TYPE_CHECKING

from ..storage import (
    sql_audit_columns,
    sql_audit_indexes,
    sql_blob,
    sql_id_column,
    sql_query_index,
    sql_state,
    sql_table_options,
    sql_unique,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData


def build_trace_sql_metadata(*, metadata: "MetaData | None" = None) -> "MetaData":
    from sqlalchemy import BigInteger, Column, JSON, LargeBinary, MetaData, String, Table
    from sqlalchemy.dialects.mysql import VARBINARY

    if metadata is None:
        metadata = MetaData()
    if "ai_trace_events" in metadata.tables:
        return metadata

    def binary(length: int) -> LargeBinary:
        return LargeBinary(length).with_variant(VARBINARY(length), "mysql")

    events = Table(
        "ai_trace_events", metadata, sql_id_column(),
        Column("store_key", binary(128), nullable=False,
               comment="Exact Trace store partition identity."),
        Column("run_id", binary(128), nullable=False,
               comment="Original UTF-8 agent run identifier."),
        Column("agent_conversation_id", binary(128), nullable=True,
               comment="Inline event conversation identity; Native references retain source ownership."),
        Column("parent_agent_run_id", binary(128), nullable=True,
               comment="Inline event parent run identity; Native references retain source ownership."),
        Column("agent_id", sql_blob(), nullable=True,
               comment="Untruncated inline agent identifier as UTF-8 bytes; null for Native references."),
        Column("sequence", BigInteger, nullable=False,
               comment="Committed position within the run event stream."),
        Column("event_index", BigInteger, nullable=False,
               comment="Original event index reported by its producer."),
        Column("idempotency_key", sql_blob(), nullable=True,
               comment="Untruncated inline event replay key as UTF-8 bytes; null for Native references."),
        Column("kind", sql_state(), nullable=False,
               comment="Trace event type."),
        Column("step_index", BigInteger, nullable=False,
               comment="Original agent graph step index."),
        Column("timestamp", String(64), nullable=False,
               comment="Original event timestamp encoded as ISO 8601."),
        Column("order_key", binary(64), nullable=False,
               comment="UTC event timestamp and sequence for stable chronological paging."),
        Column("request_identity", binary(128), nullable=True,
               comment="Original model request correlation string; leading zeros remain significant."),
        Column("request_number", binary(128), nullable=True,
               comment="Canonical decimal request number used only for numeric filtering."),
        Column("message_identity", binary(128), nullable=True,
               comment="Successful response message identity; other message metadata remains in the body."),
        Column("tool_call_id", binary(128), nullable=True,
               comment="Original UTF-8 tool call identifier."),
        Column("tool_name", sql_blob(), nullable=True,
               comment="Untruncated inline tool name as UTF-8 bytes; null for Native references."),
        Column("history_identity", binary(128), nullable=True,
               comment="Eligible raw request or tool identity for exact history role lookup."),
        Column("error_code", binary(128), nullable=True,
               comment="Tool error code used to distinguish unresolved external effects."),
        Column("error", sql_blob(), nullable=True,
               comment="Untruncated inline error text as UTF-8 bytes; null for Native references."),
        Column("payload_json", JSON(), nullable=False,
               comment="Residual event metadata and extensions, or a Native source reference."),
        *sql_audit_columns(),
        comment="Immutable Trace events with direct history correlation and chronological queries.",
        **sql_table_options(),
    )
    sql_unique(events, "store_key", "run_id", "sequence")
    for field in ("order_key", "history_identity", "message_identity"):
        sql_query_index(events, "store_key", "run_id", field)
    sql_audit_indexes(events)
    return metadata


__all__ = ["build_trace_sql_metadata"]
