#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical Runtime SQL metadata and reviewed MySQL DDL must stay aligned."""

from pathlib import Path

from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.runtime.state._schema import build_runtime_sql_metadata
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateIndex


def test_mysql_runtime_sort_key_ddl_matches_canonical_metadata() -> None:
    metadata = build_runtime_sql_metadata(frozenset({RuntimeDomain.CONVERSATION}))
    records = metadata.tables["ai_state_records"]
    dialect = mysql.dialect()

    assert (
        records.c.sort_key.type.compile(dialect=dialect)
        == "LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
    )

    mysql_indexes = {
        index.name: index
        for index in records.indexes
        if index.info.get("ddl_dialect") == "mysql"
    }
    for name in (
        "ix_partition_digest_sort_key",
        "ix_scope_digest_sort_key",
        "ix_scope_digest_state_sort_key",
        "ix_parent_digest_sort_key",
    ):
        ddl = str(CreateIndex(mysql_indexes[name]).compile(dialect=dialect))
        assert "sort_key(128)" in ddl

    migration = (
        Path(__file__).resolve().parents[2]
        / "linktools-ai"
        / "migrations"
        / "init_schema.sql"
    ).read_text(encoding="utf-8")
    assert (
        "sort_key LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL"
        in migration
    )
    assert "sort_key VARCHAR(128)" not in migration
    for fragment in (
        "ix_partition_digest_sort_key (partition_digest, sort_key(128))",
        "ix_scope_digest_sort_key (scope_digest, sort_key(128))",
        "ix_scope_digest_state_sort_key (scope_digest, state, sort_key(128))",
        "ix_parent_digest_sort_key (parent_digest, sort_key(128))",
    ):
        assert fragment in migration
