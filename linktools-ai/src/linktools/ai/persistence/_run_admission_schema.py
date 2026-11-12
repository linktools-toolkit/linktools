#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step-owned local run membership and immutable registration projections."""

from typing import TYPE_CHECKING

from ..storage import (
    sql_audit_columns, sql_audit_indexes, sql_blob, sql_id_column,
    sql_sha256, sql_table_options, sql_unique,
)
from .state import STORE_KEY_MAX_BYTES, RECORD_KEY_MAX_BYTES, RELATION_KEY_MAX_BYTES, PARENT_KEY_MAX_BYTES

if TYPE_CHECKING:
    from sqlalchemy import MetaData


def build_run_admission_sql_metadata(metadata: "MetaData") -> None:
    """Keep local publication atomic without copying the canonical run body."""
    from sqlalchemy import BigInteger, Column, JSON, LargeBinary, Table
    from sqlalchemy.dialects.mysql import VARBINARY

    if "ai_step_run_admissions" in metadata.tables:
        return
    table = Table(
        "ai_step_run_admissions", metadata, sql_id_column(),
        Column("store_key", LargeBinary(STORE_KEY_MAX_BYTES).with_variant(VARBINARY(STORE_KEY_MAX_BYTES), "mysql"),
               nullable=False, comment="Exact namespace, tenant and domain partition admitting this run."),
        Column("key", LargeBinary(RECORD_KEY_MAX_BYTES).with_variant(VARBINARY(RECORD_KEY_MAX_BYTES), "mysql"),
               nullable=False, comment="Canonical local run admission identity, independent of the registration owner."),
        Column("scope_key", LargeBinary(RELATION_KEY_MAX_BYTES).with_variant(VARBINARY(RELATION_KEY_MAX_BYTES), "mysql"),
               nullable=True, comment="Conversation lookup identity derived from the validated domain projection."),
        Column("parent_key", LargeBinary(PARENT_KEY_MAX_BYTES).with_variant(VARBINARY(PARENT_KEY_MAX_BYTES), "mysql"),
               nullable=True, comment="Parent-run lookup identity derived from the validated domain projection."),
        Column("sort_key", sql_blob(), nullable=False,
               comment="Stable ordering by projected start time and complete admission identity."),
        Column("storage_revision", BigInteger, nullable=False,
               comment="Optimistic local publication and release fence, independent of canonical registration."),
        Column("payload_json", JSON(), nullable=False,
               comment="Typed canonical registration reference and actual domain projection differences; no copied run body."),
        Column("key_locator", sql_sha256(), nullable=False,
               comment="SHA-256 lookup projection of the admission key; complete identity must also match."),
        Column("scope_locator", sql_sha256(), nullable=True,
               comment="SHA-256 lookup projection of the projected conversation scope."),
        Column("parent_locator", sql_sha256(), nullable=True,
               comment="SHA-256 lookup projection of the projected parent-run identity."),
        Column("sort_prefix", LargeBinary(128).with_variant(VARBINARY(128), "mysql"), nullable=False,
               comment="First 128 sort-key bytes used only for ordered lookup."),
        *sql_audit_columns(),
        comment="Step-owned domain run admission; canonical run metadata remains with its unique registration owner.",
        **sql_table_options(),
    )
    sql_unique(table, "store_key", "key_locator")
    # The unique store-key prefix bounds all lists to this typed domain owner.
    # Scope/parent filtering and ordering deliberately use that bounded scan.
    sql_audit_indexes(table)


__all__ = ["build_run_admission_sql_metadata"]
