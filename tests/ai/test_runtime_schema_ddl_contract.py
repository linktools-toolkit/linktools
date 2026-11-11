#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical Runtime SQL metadata and reviewed MySQL DDL must stay aligned."""

import re
from pathlib import Path
from typing import cast

import pytest

from linktools.ai.migrate import build_sql_schema_metadata, provision_runtime_database
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
    # Semicolons inside SQL comment literals are column data, not delimiters.
    for raw_statement in re.split(r";(?=(?:[^']*'[^']*')*[^']*$)", migration):
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

    assert records.c.subject_digest.nullable
    assert records.c.subject_digest.type.compile(dialect=dialect) == "CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
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
        "ix_parent_digest_sort_key",
    ):
        ddl = str(CreateIndex(mysql_indexes[name]).compile(dialect=dialect))
        assert "sort_key(128)" in ddl

    assert not mysql_indexes["ix_scope_digest_subject_digest"].unique
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
        "ix_parent_digest_sort_key (parent_digest, sort_key(128))",
        "ix_scope_digest_subject_digest (scope_digest, subject_digest)",
    ):
        assert fragment in migration


def test_snapshot_active_lookup_uses_full_partition_index() -> None:
    from sqlalchemy import create_engine, false, select

    metadata = build_sql_schema_metadata()
    snapshots = metadata.tables["ai_step_snapshots"]
    index_columns = {
        tuple(column.name for column in index.columns)
        for index in snapshots.indexes
    }
    assert ("store_digest", "run_id", "pruned") in index_columns
    assert ("store_digest", "run_id") not in index_columns
    # Other append-only tables do not need the snapshot retirement dimension.
    assert ("store_digest", "run_id") in {
        tuple(column.name for column in index.columns)
        for index in metadata.tables["ai_step_events"].indexes
    }

    engine = create_engine("sqlite://")
    try:
        snapshots.create(engine)
        with engine.begin() as connection:
            connection.execute(snapshots.insert(), [{
                "store_digest": "s" * 64,
                "run_id": "run", "step_index": index, "timestamp": "timestamp",
                "state": "interrupted", "messages": "{}",
                "key_digest": f"{index:064x}", "pruned": index < 2000,
            } for index in range(2008)])
            query = select(snapshots.c.id).where(
                snapshots.c.store_digest == "s" * 64,
                snapshots.c.run_id == "run",
                snapshots.c.pruned == false(),
            ).order_by(snapshots.c.id.desc()).limit(2)
            compiled = str(query.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True}))
            plan = " ".join(row[-1] for row in connection.exec_driver_sql("EXPLAIN QUERY PLAN " + compiled))
            assert "store_digest=? AND run_id=? AND pruned=?" in plan
            assert "TEMP B-TREE" not in plan
            assert len(connection.execute(query).all()) == 2
    finally:
        engine.dispose()



def test_runtime_storage_mysql_indexes_match_reviewed_contract() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
    expected = {
        "ai_state_records": {
            "uk_store_digest_key_digest",
            "ix_store_digest_kind_sort_key",
            "ix_scope_digest_sort_key",
            "ix_scope_digest_subject_digest",
            "ix_parent_digest_sort_key",
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


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", (PersistenceDomain.TRACE, PersistenceDomain.EXECUTION, PersistenceDomain.CONVERSATION))
@pytest.mark.parametrize("dialect_name", ("sqlite", "postgresql", "mysql"))
async def test_selected_native_schema_preserves_canonical_dialect_ddl(
    monkeypatch: pytest.MonkeyPatch, domain: PersistenceDomain, dialect_name: str,
) -> None:
    from sqlalchemy import MetaData, create_mock_engine
    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy.schema import ExecutableDDLElement

    from linktools.ai.migrate import _database
    from linktools.ai.storage import InMemoryObjectStore

    captured: list[MetaData] = []

    async def capture(_engine: AsyncEngine, metadata: MetaData) -> None:
        captured.append(metadata)

    monkeypatch.setattr(_database, "provision_sql", capture)
    await provision_runtime_database(
        cast(AsyncEngine, object()), domains=(domain,), object_store=InMemoryObjectStore(),
    )
    selected = captured[0]
    assert set(selected.tables) == {
        "ai_state_records", "ai_state_aliases", "ai_state_facts",
        "ai_state_sequences", "ai_state_operations", "ai_step_messages", "ai_step_sequences", "ai_native_content_edges",
    } | ({"ai_step_runs", "ai_step_events"} if domain is not PersistenceDomain.CONVERSATION else set())

    def compile_ddl(metadata: MetaData) -> list[str]:
        statements: list[str] = []

        def emit(statement: ExecutableDDLElement, *_args: object, **_kwargs: object) -> None:
            statements.append(str(statement.compile(dialect=engine.dialect)))

        engine = create_mock_engine(f"{dialect_name}://", emit)
        metadata.create_all(
            engine, tables=[metadata.tables[name] for name in selected.tables],
            checkfirst=False,
        )
        return sorted(statements)

    # Mock-engine emission honors dialect conditions, unlike direct CreateIndex compilation.
    assert compile_ddl(selected) == compile_ddl(build_sql_schema_metadata())


@pytest.mark.parametrize("domains", (
    (PersistenceDomain.CONVERSATION, PersistenceDomain.EXECUTION, PersistenceDomain.CHECKPOINT),
    (PersistenceDomain.EXECUTION, PersistenceDomain.CONVERSATION, PersistenceDomain.CHECKPOINT),
    (PersistenceDomain.CHECKPOINT, PersistenceDomain.TRACE, PersistenceDomain.CONVERSATION),
))
def test_native_domain_schema_composes_without_replacing_canonical_tables(domains) -> None:
    from sqlalchemy import MetaData
    from linktools.ai.persistence.schema import build_step_domain_sql_metadata, build_step_sql_metadata

    metadata = MetaData()
    retained = {}
    for domain in domains:
        build_step_domain_sql_metadata((domain,), metadata=metadata)
        for name, table in retained.items():
            assert metadata.tables[name] is table
        retained.update(metadata.tables)
    complete = build_step_sql_metadata()
    assert set(metadata.tables) == set(complete.tables)
    for name, table in metadata.tables.items():
        assert set(index.name for index in table.indexes) == set(index.name for index in complete.tables[name].indexes)
        assert len(table.indexes) == len(complete.tables[name].indexes)
