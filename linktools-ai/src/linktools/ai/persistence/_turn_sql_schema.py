#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical owners of source-bound SQL turn coordination."""

from typing import TYPE_CHECKING

from ..storage import sql_audit_columns, sql_audit_indexes, sql_id_column, sql_table_options, sql_unique
from .state import STORE_KEY_MAX_BYTES

if TYPE_CHECKING:
    from sqlalchemy import MetaData


TURN_SLOT_TABLE = "ai_turn_slots"
TURN_ADMISSION_TABLE = "ai_turn_admissions"
TURN_SCOPE_MAX_BYTES = 3 * (128 + 2)
TURN_SESSION_MAX_BYTES = 265
TURN_REQUEST_MAX_BYTES = 257
TURN_EXECUTION_MAX_BYTES = 128


def build_turn_sql_metadata(metadata: "MetaData") -> None:
    from sqlalchemy import BigInteger, Boolean, Column, DateTime, LargeBinary, String, Table, Text, text
    from sqlalchemy.dialects import mysql

    def binary(size: int) -> LargeBinary:
        return LargeBinary(size).with_variant(mysql.VARBINARY(size), "mysql")

    slot = Table(
        TURN_SLOT_TABLE, metadata, sql_id_column(),
        Column("store_key", binary(STORE_KEY_MAX_BYTES), nullable=False, comment="Physical StateStore partition."),
        Column("scope_key", binary(TURN_SCOPE_MAX_BYTES), nullable=False, comment="Exact namespace, tenant and local coordination scope."),
        Column("session_key", binary(TURN_SESSION_MAX_BYTES), nullable=False, comment="Exact coordination session identity."),
        Column("active_execution_id", binary(TURN_EXECUTION_MAX_BYTES), nullable=True, comment="Current admission owner, or NULL for an idle slot."),
        Column("lease_token", LargeBinary, nullable=True, comment="Exact active lease token, or NULL for an idle slot."),
        Column("lease_fence", BigInteger, nullable=False, server_default=text("0"), comment="Lease fencing generation."),
        Column("lease_expires_at", DateTime(timezone=True).with_variant(mysql.DATETIME(fsp=6), "mysql"), nullable=True, comment="Source-clock lease deadline."),
        Column("revision", BigInteger, nullable=False, server_default=text("0"), comment="Physical slot revision for guarded writes."),
        *sql_audit_columns(),
        comment="Stable source turn lease slot, including the idle fence high-water mark.",
        **sql_table_options(),
    )
    sql_unique(slot, "store_key", "scope_key", "session_key")
    admission = Table(
        TURN_ADMISSION_TABLE, metadata, sql_id_column(),
        Column("store_key", binary(STORE_KEY_MAX_BYTES), nullable=False, comment="Physical StateStore partition."),
        Column("scope_key", binary(TURN_SCOPE_MAX_BYTES), nullable=False, comment="Exact namespace, tenant and local coordination scope."),
        Column("execution_id", binary(TURN_EXECUTION_MAX_BYTES), nullable=False, comment="Exact coordination execution identity."),
        Column("session_key", binary(TURN_SESSION_MAX_BYTES), nullable=False, comment="Exact coordination session identity."),
        Column("request_key", binary(TURN_REQUEST_MAX_BYTES), nullable=False, comment="Exact request identity retained after retirement."),
        Column("live_request_key", binary(TURN_REQUEST_MAX_BYTES), nullable=True, comment="Exact request identity while active, otherwise NULL."),
        Column("request_digest", Text, nullable=False, comment="Digest of the admitted request."),
        Column("owner_id", Text, nullable=False, comment="Producer that acquired the admission."),
        Column("lease_fence", BigInteger, nullable=False, comment="Lease fencing generation."),
        Column("cancel_requested", Boolean, nullable=False, comment="Whether the current admission was cancelled."),
        Column("receipt_id", Text, nullable=False, comment="Latest source mutation receipt identifier."),
        Column("receipt_kind", String(8), nullable=False, comment="Latest source mutation operation."),
        *sql_audit_columns(),
        comment="Source turn admission and its latest operation receipt, retained after release.",
        **sql_table_options(),
    )
    sql_unique(admission, "store_key", "scope_key", "execution_id")
    sql_unique(admission, "store_key", "scope_key", "live_request_key")
    sql_audit_indexes(slot)
    sql_audit_indexes(admission)


__all__ = [
    "TURN_SLOT_TABLE", "TURN_ADMISSION_TABLE", "TURN_SCOPE_MAX_BYTES", "TURN_SESSION_MAX_BYTES",
    "TURN_REQUEST_MAX_BYTES", "TURN_EXECUTION_MAX_BYTES", "build_turn_sql_metadata",
]
