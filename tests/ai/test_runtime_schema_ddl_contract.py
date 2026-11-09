#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical Runtime SQL metadata and reviewed MySQL DDL must stay aligned."""

import re
from pathlib import Path

from linktools.ai.migrate import build_sql_schema_metadata
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateIndex


def test_mysql_ddl_declares_all_indexes_in_each_create_table_and_matches_metadata() -> None:
    metadata = build_sql_schema_metadata()
    migration = (
        Path(__file__).resolve().parents[2]
        / "linktools-ai" / "migrations" / "init_schema.sql"
    ).read_text(encoding="utf-8")
    migration = re.sub(r"^--[^\n]*", "", migration, flags=re.MULTILINE)
    index_pattern = r"(UNIQUE )?(?:KEY|INDEX) (\w+) \(((?:[^()]|\([^()]*\))+)\)"
    table_indexes: dict[str, list[str]] = {}
    for raw_statement in migration.split(";"):
        statement = " ".join(raw_statement.split())
        if not statement:
            continue
        create_table = re.fullmatch(r"CREATE TABLE (\w+) \((.*)\)\s*ENGINE=.*", statement)
        assert create_table is not None, statement
        name, body = create_table.groups()
        assert name not in table_indexes
        indexes = [
            f"{unique}INDEX {index_name} ({columns})"
            for unique, index_name, columns in re.findall(
                rf"(?:^|,)\s*{index_pattern}", body,
            )
        ]
        assert "INDEX ix_updated_at (updated_at)" in indexes, name
        assert "INDEX ix_created_at (created_at)" in indexes, name
        table_indexes[name] = indexes

    assert set(table_indexes) == set(metadata.tables)
    dialect = mysql.dialect()
    dialect.server_version_info = (5, 7, 44)
    for name, table in metadata.tables.items():
        expected = [
            " ".join(str(CreateIndex(index).compile(dialect=dialect)).split())
            .removeprefix("CREATE ").replace(f" ON {name} ", " ")
            for index in table.indexes
            if index.info.get("ddl_dialect") == "mysql"
        ]
        assert sorted(table_indexes[name]) == sorted(expected), name


def test_mysql_runtime_sort_key_ddl_matches_canonical_metadata() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
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
    ):
        assert fragment in migration



def test_runtime_storage_mysql_indexes_match_reviewed_contract() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
    expected = {
        "ai_state_records": {
            "uk_store_digest_key_digest",
            "ix_store_digest_kind_sort_key",
            "ix_scope_digest_sort_key",
            "ix_updated_at",
            "ix_created_at",
        },
        "ai_state_aliases": {
            "uk_store_digest_alias_digest",
            "ix_record_key_digest",
            "ix_updated_at",
            "ix_created_at",
        },
        "ai_state_facts": {
            "uk_store_digest_stream_digest_sequence",
            "ix_owner_key_digest",
            "ix_stream_digest_subject_digest_sequence",
            "ix_updated_at",
            "ix_created_at",
        },
        "ai_state_sequences": {
            "uk_store_digest_key_digest",
            "ix_updated_at",
            "ix_created_at",
        },
        "ai_state_operations": {
            "uk_store_digest_key_digest",
            "uk_store_digest_stream_digest_sequence",
            "ix_stream_digest_state_sequence",
            "ix_updated_at",
            "ix_created_at",
        },
    }
    forbidden = {
        "ai_state_records": {
            "uk_key_digest",
            "ix_partition_digest_sort_key",
            "ix_scope_digest_state_sort_key",
            "ix_parent_digest_sort_key",
            "ix_store_digest_kind_key_digest",
        },
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
        assert mysql_names == names
        assert not (forbidden[table_name] & mysql_names)
        assert "ix_updated_at" in mysql_names
        assert "ix_created_at" in mysql_names
        if table_name == "ai_state_records":
            assert len(mysql_names) <= len(table.c) // 3


def test_runtime_storage_migration_contains_no_known_duplicate_indexes() -> None:
    migration = (
        Path(__file__).resolve().parents[2]
        / "linktools-ai"
        / "migrations"
        / "init_schema.sql"
    ).read_text(encoding="utf-8")

    for fragment in (
        "KEY ix_store_digest_alias_digest ",
        "KEY ix_store_digest_stream_digest_sequence ",
        "KEY ix_store_digest_key_digest ",
    ):
        assert fragment not in migration


def test_runtime_fact_ddl_comments_match_canonical_metadata() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
    facts = metadata.tables["ai_state_facts"]
    migration = (
        Path(__file__).resolve().parents[2]
        / "linktools-ai"
        / "migrations"
        / "init_schema.sql"
    ).read_text(encoding="utf-8")
    assert f"COMMENT='{facts.comment}'" in migration
    assert f"COMMENT '{facts.c.state.comment}'" in migration
