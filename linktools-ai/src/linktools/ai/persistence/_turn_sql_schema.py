#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical owners of source-bound SQL turn coordination."""

from typing import TYPE_CHECKING

from ..storage import sql_audit_columns, sql_audit_indexes, sql_id_column, sql_sha256, sql_table_options, sql_unique

if TYPE_CHECKING:
    from sqlalchemy import MetaData


TURN_SLOT_TABLE = "ai_turn_slots"
TURN_ADMISSION_TABLE = "ai_turn_admissions"
RUNTIME_TURN_SCOPE = "runtime-pending-turn"
TURN_SESSION_MAX_BYTES = 128
TURN_REQUEST_MAX_BYTES = 64
TURN_EXECUTION_MAX_BYTES = 128


def build_turn_sql_metadata(metadata: "MetaData") -> None:
    from sqlalchemy import BigInteger, Boolean, Column, DateTime, LargeBinary, SmallInteger, String, Table, text
    from sqlalchemy.dialects import mysql

    def binary(size: int) -> LargeBinary:
        return LargeBinary(size).with_variant(mysql.VARBINARY(size), "mysql")

    slot = Table(
        TURN_SLOT_TABLE, metadata, sql_id_column(),
        Column("store_key", binary(32), nullable=False, comment="Physical StateStore partition."),
        Column("session_kind", SmallInteger, nullable=False, comment="Conversation or standalone turn slot."),
        Column("session_key", binary(TURN_SESSION_MAX_BYTES), nullable=False, comment="Exact coordination session identity."),
        Column("active_execution_id", binary(TURN_EXECUTION_MAX_BYTES), nullable=True, comment="Current admission owner, or NULL for an idle slot."),
        Column("lease_token", binary(32), nullable=True, comment="Exact active lease token, or NULL for an idle slot."),
        Column("lease_fence", BigInteger, nullable=False, server_default=text("0"), comment="Lease fencing generation."),
        Column("lease_expires_at", DateTime(timezone=True).with_variant(mysql.DATETIME(fsp=6), "mysql"), nullable=True, comment="Source-clock lease deadline."),
        Column("revision", BigInteger, nullable=False, server_default=text("0"), comment="Physical slot revision for guarded writes."),
        *sql_audit_columns(),
        comment="Stable source turn lease slot, including the idle fence high-water mark.",
        **sql_table_options(),
    )
    sql_unique(slot, "store_key", "session_kind", "session_key")
    admission = Table(
        TURN_ADMISSION_TABLE, metadata, sql_id_column(),
        Column("store_key", binary(32), nullable=False, comment="Physical StateStore partition."),
        Column("session_kind", SmallInteger, nullable=False, comment="Conversation or standalone turn slot."),
        Column("execution_id", binary(TURN_EXECUTION_MAX_BYTES), nullable=False, comment="Exact coordination execution identity."),
        Column("session_key", binary(TURN_SESSION_MAX_BYTES), nullable=False, comment="Exact coordination session identity."),
        Column("retired_request_key", binary(TURN_REQUEST_MAX_BYTES), nullable=True, comment="Original request identity, retained only after retirement."),
        Column("route_kind", SmallInteger, nullable=False, comment="Active request or retired execution route."),
        Column("route_key", binary(TURN_EXECUTION_MAX_BYTES), nullable=False, comment="Exact active request or retired execution identity."),
        Column("request_digest", sql_sha256(), nullable=False, comment="Digest of the admitted request."),
        Column("owner_id", binary(32), nullable=False, comment="Producer that acquired the admission."),
        Column("lease_fence", BigInteger, nullable=False, comment="Lease fencing generation."),
        Column("cancel_requested", Boolean, nullable=False, comment="Whether the current admission was cancelled."),
        Column("receipt_id", binary(32), nullable=False, comment="Latest source mutation receipt identifier."),
        Column("receipt_kind", String(8), nullable=False, comment="Latest source mutation operation."),
        Column("control_id", BigInteger, nullable=True,
               comment="Local row address of the execution authority; rebuilt on restore."),
        *sql_audit_columns(),
        comment="Source turn admission and its latest operation receipt, retained after release.",
        **sql_table_options(),
    )
    sql_unique(admission, "store_key", "execution_id")
    sql_unique(admission, "store_key", "route_kind", "route_key")
    sql_audit_indexes(slot)
    sql_audit_indexes(admission)


__all__ = [
    "TURN_SLOT_TABLE", "TURN_ADMISSION_TABLE", "RUNTIME_TURN_SCOPE", "TURN_SESSION_MAX_BYTES",
    "TURN_REQUEST_MAX_BYTES", "TURN_EXECUTION_MAX_BYTES", "build_turn_sql_metadata",
]
