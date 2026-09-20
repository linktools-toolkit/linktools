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

    assert "partition_digest" not in records.c
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
        "ix_store_digest_kind_sort_key",
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
    assert "partition_digest" not in migration
    for fragment in (
        "ix_store_digest_kind_sort_key (store_digest, kind, sort_key(128))",
        "ix_scope_digest_sort_key (scope_digest, sort_key(128))",
        "ix_scope_digest_state_sort_key (scope_digest, state, sort_key(128))",
        "ix_parent_digest_sort_key (parent_digest, sort_key(128))",
    ):
        assert fragment in migration



def test_runtime_state_identity_indexes_are_store_scoped() -> None:
    metadata = build_runtime_sql_metadata(frozenset({RuntimeDomain.CONVERSATION}))
    expected = {
        "ai_state_records": {
            "uk_store_digest_key_digest",
            "ix_store_digest_kind_key_digest",
        },
        "ai_state_aliases": {
            "uk_store_digest_alias_digest",
            "ix_record_key_digest",
        },
        "ai_state_facts": {
            "uk_store_digest_stream_digest_sequence",
            "ix_owner_key_digest",
            "ix_stream_digest_subject_digest_sequence",
        },
        "ai_state_sequences": {"uk_store_digest_key_digest"},
        "ai_state_operations": {
            "uk_store_digest_key_digest",
            "uk_store_digest_stream_digest_sequence",
            "ix_stream_digest_state_sequence",
        },
    }
    forbidden = {
        "ai_state_records": {"uk_key_digest", "ix_partition_digest_sort_key"},
        "ai_state_aliases": {"uk_alias_digest", "ix_store_digest_alias_digest"},
        "ai_state_facts": {
            "uk_stream_digest_sequence",
            "ix_store_digest_stream_digest_sequence",
        },
        "ai_state_sequences": {"uk_key_digest", "ix_store_digest_key_digest"},
        "ai_state_operations": {
            "uk_key_digest",
            "uk_stream_digest_sequence",
            "ix_store_digest_key_digest",
        },
    }

    for table_name, names in expected.items():
        table = metadata.tables[table_name]
        mysql_names = {
            index.name
            for index in table.indexes
            if index.info.get("ddl_dialect") == "mysql"
        }
        assert names <= mysql_names
        assert not (forbidden[table_name] & mysql_names)
