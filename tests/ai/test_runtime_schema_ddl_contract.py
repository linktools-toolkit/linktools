#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical Runtime SQL metadata and reviewed MySQL DDL must stay aligned."""

import re
from pathlib import Path
from typing import cast

import pytest

from linktools.ai.steps import native_store_key, native_store_locator
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


@pytest.mark.parametrize("dialect_name", ("mysql", "postgresql", "sqlite"))
def test_trace_indexes_keep_history_association_and_chronological_paths(dialect_name: str) -> None:
    from sqlalchemy import create_mock_engine
    from sqlalchemy.schema import ExecutableDDLElement

    table = build_runtime_sql_metadata(frozenset({PersistenceDomain.TRACE})).tables["ai_trace_events"]
    statements: list[str] = []

    def emit(statement: ExecutableDDLElement, *_args: object, **_kwargs: object) -> None:
        statements.append(" ".join(str(statement.compile(dialect=engine.dialect)).split()))

    engine = create_mock_engine(f"{dialect_name}://", emit)
    table.create(engine, checkfirst=False)
    # Count emitted indexes, including those implied by PRIMARY and UNIQUE,
    # rather than counting both dialect variants present in the metadata.
    indexes = [statement for statement in statements if re.match(r"CREATE (?:UNIQUE )?INDEX ", statement)]
    constraints = re.findall(r"\b(?:PRIMARY KEY|UNIQUE) \([^()]+\)", " ".join(statements))
    assert len(table.c) == 24
    assert len(indexes) + len(constraints) == 7
    assert 3 * (len(indexes) + len(constraints)) < len(table.c)
    assert "PRIMARY KEY (id)" in constraints
    unique_keys = [statement for statement in indexes if statement.startswith("CREATE UNIQUE INDEX ")]
    unique_keys.extend(constraint for constraint in constraints if constraint.startswith("UNIQUE "))
    assert len(unique_keys) == 1
    assert unique_keys[0].endswith("(store_key, run_id, sequence)")
    assert sorted(re.findall(r"\(([^()]+)\)$", statement)[0] for statement in indexes
                  if not statement.startswith("CREATE UNIQUE INDEX ")) == [
        "created_at", "store_key, run_id, history_identity", "store_key, run_id, message_identity",
        "store_key, run_id, order_key", "updated_at",
    ]
    for name in ("agent_conversation_id", "parent_agent_run_id"):
        assert table.c[name].nullable and table.c[name].type.length == 128
    for name in ("agent_id", "tool_name", "error", "idempotency_key"):
        assert table.c[name].nullable and table.c[name].type.length is None
        assert table.c[name].type.compile(dialect=engine.dialect) == {
            "mysql": "LONGBLOB", "postgresql": "BYTEA", "sqlite": "BLOB",
        }[dialect_name]
        assert all(name not in index.columns for index in table.indexes)


def test_mysql_runtime_sort_key_ddl_matches_canonical_metadata() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
    records = metadata.tables["ai_state_records"]
    dialect = mysql.dialect()

    assert records.c.subject_id.nullable
    assert records.c.subject_id.type.compile(dialect=dialect) == "VARBINARY(128)"
    assert "partition_digest" not in records.c
    assert (
        records.c.sort_key.type.compile(dialect=dialect)
        == "LONGBLOB"
    )

    assert records.c.sort_prefix.type.compile(dialect=dialect) == "VARBINARY(128)"

    mysql_indexes = {
        index.name: index
        for index in records.indexes
        if index.info.get("ddl_dialect") == "mysql"
    }
    for name in (
        "ix_store_key_kind_sort_prefix",
        "ix_scope_locator_sort_prefix",
        "ix_parent_locator_sort_prefix",
    ):
        ddl = str(CreateIndex(mysql_indexes[name]).compile(dialect=dialect))
        assert "sort_prefix)" in ddl
        assert "sort_key" not in ddl
        assert not mysql_indexes[name].unique

    assert not mysql_indexes["ix_scope_locator_subject_id"].unique
    migration = (
        Path(__file__).resolve().parents[2]
        / "linktools-ai"
        / "migrations"
        / "init_schema.sql"
    ).read_text(encoding="utf-8")
    assert (
        "sort_key LONGBLOB NOT NULL"
        in migration
    )
    assert "sort_key VARCHAR(128)" not in migration
    assert "partition_digest" not in migration
    for fragment in (
        "ix_store_key_kind_sort_prefix (store_key, kind, sort_prefix)",
        "ix_scope_locator_sort_prefix (scope_locator, sort_prefix)",
        "ix_parent_locator_sort_prefix (parent_locator, sort_prefix)",
        "ix_scope_locator_subject_id (scope_locator, subject_id)",
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
    assert ("store_locator", "run_id", "pruned") in index_columns
    assert ("store_locator", "run_id") not in index_columns
    # Other append-only tables do not need the snapshot retirement dimension.
    assert ("store_locator", "run_id", "idempotency_key") in {
        tuple(column.name for column in index.columns)
        for index in metadata.tables["ai_step_events"].indexes
    }

    engine = create_engine("sqlite://")
    try:
        snapshots.create(engine)
        with engine.begin() as connection:
            connection.execute(snapshots.insert(), [{
                "store_key": native_store_key("s", "tenant"),
                "store_locator": native_store_locator(native_store_key("s", "tenant")),
                "run_id": "run", "step_index": index, "timestamp": "timestamp",
                "state": "interrupted", "messages": "{}",
                "idempotency_key": str(index).encode("utf-8"), "pruned": pruned,
            } for index, pruned in enumerate((False, True, False, False))])
            query = select(snapshots.c.id, snapshots.c.step_index).where(
                snapshots.c.store_locator == native_store_locator(native_store_key("s", "tenant")),
                snapshots.c.store_key == native_store_key("s", "tenant"),
                snapshots.c.run_id == "run",
                snapshots.c.pruned == false(),
            ).order_by(snapshots.c.id.desc()).limit(2)
            compiled = str(query.compile(dialect=engine.dialect))
            plan = " ".join(row[-1] for row in connection.exec_driver_sql("EXPLAIN QUERY PLAN " + compiled, tuple(query.compile(dialect=engine.dialect).params[name] for name in query.compile(dialect=engine.dialect).positiontup)))
            assert "store_locator=? AND run_id=? AND pruned=?" in plan
            assert "TEMP B-TREE" not in plan
            first_page = connection.execute(query).all()
            assert [row.step_index for row in first_page] == [3, 2]
            next_page = connection.execute(query.where(snapshots.c.id < first_page[-1].id)).all()
            assert [row.step_index for row in next_page] == [0]
    finally:
        engine.dispose()



def test_runtime_storage_mysql_indexes_match_reviewed_contract() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
    expected = {
        "ai_state_records": {
            "uk_store_key_key_locator",
            "ix_store_key_kind_sort_prefix",
            "ix_scope_locator_sort_prefix",
            "ix_scope_locator_subject_id",
            "ix_parent_locator_sort_prefix",
            "ix_updated_at",
            "ix_created_at",
        },
        "ai_state_aliases": {
            "uk_store_key_alias_locator",
            "ix_store_key_record_locator",
            "ix_updated_at",
            "ix_created_at",
        },
        "ai_state_facts": {
            "uk_store_key_stream_locator_sequence",
            "ix_owner_locator",
            "ix_stream_locator_subject_id_sequence",
            "ix_updated_at",
            "ix_created_at",
        },
        "ai_state_sequences": {
            "uk_store_key_stream_locator",
            "ix_updated_at",
            "ix_created_at",
        },
        "ai_state_operations": {
            "uk_store_key_operation_locator",
            "uk_store_key_stream_locator_sequence",
            "ix_stream_locator_state_sequence",
            "ix_updated_at",
            "ix_created_at",
        },
    }
    forbidden = {
        "ai_state_records": {
            "uk_key_digest",
            "ix_partition_digest_sort_key",
            "ix_scope_key_state_sort_key",
            "ix_store_key_kind_key_digest",
        },
        "ai_state_aliases": {"uk_alias_key", "ix_store_key_alias_key"},
        "ai_state_facts": {
            "uk_stream_key_sequence",
            "ix_store_key_stream_key_sequence",
        },
        "ai_state_sequences": {"uk_key_digest", "ix_store_key_key_digest"},
        "ai_state_operations": {
            "uk_key_digest",
            "uk_stream_key_sequence",
            "ix_store_key_key_digest",
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
        "KEY ix_store_key_alias_key ",
        "KEY ix_store_key_stream_key_sequence ",
        "KEY ix_store_key_key_digest ",
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
        "ai_state_sequences", "ai_state_operations", "ai_step_runs", "ai_step_messages", "ai_step_sequences", "ai_native_content_edges",
    } | ({"ai_step_events", "ai_trace_events"} if domain is not PersistenceDomain.CONVERSATION else set())

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


@pytest.mark.parametrize("dialect_name", ("mysql", "postgresql", "sqlite"))
def test_operation_address_column_is_bounded_binary_with_scoped_unique_key(dialect_name: str) -> None:
    from sqlalchemy.dialects import postgresql, sqlite
    table = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION})).tables["ai_state_operations"]
    dialect = {"mysql": mysql, "postgresql": postgresql, "sqlite": sqlite}[dialect_name].dialect()
    assert table.c.operation_key.type.length == 768
    assert table.c.operation_key.type.compile(dialect=dialect) == {
        "mysql": "VARBINARY(768)", "postgresql": "BYTEA", "sqlite": "BLOB",
    }[dialect_name]
    assert any(index.unique and tuple(column.name for column in index.columns) == ("store_key", "operation_locator")
               for index in table.indexes)
    assert all(len(index.columns) <= 3 for index in table.indexes)


def test_runtime_alias_schema_uses_partitioned_binary_indexes() -> None:
    from sqlalchemy.dialects import postgresql, sqlite

    table = build_runtime_sql_metadata(frozenset({PersistenceDomain.CHECKPOINT})).tables["ai_state_aliases"]
    assert table.c.alias_key.type.compile(dialect=mysql.dialect()) == "VARBINARY(768)"
    assert table.c.alias_key.type.compile(dialect=postgresql.dialect()) == "BYTEA"
    assert table.c.alias_key.type.compile(dialect=sqlite.dialect()) == "BLOB"
    columns = {tuple(column.name for column in index.columns): index.unique for index in table.indexes}
    assert columns[("store_key", "alias_locator")]
    assert not columns[("store_key", "record_locator")]
    assert all(len(index.columns) <= 3 for index in table.indexes)
    assert table.c.store_key.type.length == 128
    assert table.c.alias_key.type.length == table.c.record_key.type.length == 768


@pytest.mark.parametrize("dialect_name", ("mysql", "postgresql", "sqlite"))
def test_state_stream_and_counter_schema_uses_complete_binary_keys(dialect_name: str) -> None:
    from sqlalchemy.dialects import postgresql, sqlite
    dialect = {"mysql": mysql, "postgresql": postgresql, "sqlite": sqlite}[dialect_name].dialect()
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
    for name in ("ai_state_facts", "ai_state_operations", "ai_state_sequences"):
        table = metadata.tables[name]
        assert table.c.stream_key.type.length == 768
        assert table.c.stream_key.type.compile(dialect=dialect) == {
            "mysql": "VARBINARY(768)", "postgresql": "BYTEA", "sqlite": "BLOB",
        }[dialect_name]
        assert all(len(index.columns) <= 3 for index in table.indexes)
    facts = metadata.tables["ai_state_facts"]
    assert any(tuple(column.name for column in index.columns) == ("stream_locator", "subject_id", "sequence")
               for index in facts.indexes)
    operations = metadata.tables["ai_state_operations"]
    assert any(tuple(column.name for column in index.columns) == ("stream_locator", "state", "sequence")
               for index in operations.indexes)


def test_mysql_all_table_columns_and_comments_match_reviewed_ddl() -> None:
    from sqlalchemy.schema import CreateColumn

    metadata = build_sql_schema_metadata()
    migration = (Path(__file__).resolve().parents[2] / "linktools-ai/migrations/init_schema.sql").read_text()
    declared = {}
    for name, body in re.findall(r"CREATE TABLE (\w+) \((.*?)\n\)\s*ENGINE=", migration, re.DOTALL):
        columns = {}
        for line in body.splitlines():
            matched = re.match(r"\s*`?(\w+)`? ((?:BIGINT|SMALLINT|INT|VARCHAR|VARBINARY|CHAR|TEXT|LONGTEXT|JSON|LONGBLOB|BOOLEAN|BOOL|DATETIME|TIMESTAMP|BLOB|DOUBLE|DECIMAL)\b.*)", line)
            if matched:
                columns[matched[1]] = matched[1] + " " + matched[2].removesuffix(",")
        declared[name] = columns
    assert set(declared) == set(metadata.tables)

    def semantic_column(value: str) -> tuple[str, str | None]:
        comment = re.search(r" COMMENT '((?:[^']|'')*)'", value)
        if comment:
            value = value[:comment.start()] + value[comment.end():]
        value = value.replace("`", "").replace(" BOOLEAN", " BOOL")
        # Explicit defaults and inherited table defaults have identical semantics.
        value = value.replace(" CHARACTER SET utf8mb4 COLLATE utf8mb4_bin", "")
        value = re.sub(r"(?<!NOT)\s+NULL(?=\s|$)", "", value)
        return " ".join(value.split()), None if comment is None else comment[1]

    for name, table in metadata.tables.items():
        assert list(declared[name]) == list(table.c.keys()), name
        for column in table.c:
            expected = str(CreateColumn(column).compile(dialect=mysql.dialect()))
            assert semantic_column(declared[name][column.name]) == semantic_column(expected), (name, column.name)


@pytest.mark.parametrize("dialect_name", ("mysql", "postgresql", "sqlite"))
def test_state_locator_indexes_fit_complete_binary_byte_budgets(dialect_name: str) -> None:
    from sqlalchemy import BigInteger, Integer, LargeBinary, String
    from sqlalchemy.dialects import postgresql, sqlite
    dialect = {"mysql": mysql, "postgresql": postgresql, "sqlite": sqlite}[dialect_name].dialect()
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION}))
    maxima = {}
    for table in metadata.tables.values():
        assert table.c.store_key.type.length == 128
        assert table.c.store_key.type.compile(dialect=dialect) == {
            "mysql": "VARBINARY(128)", "postgresql": "BYTEA", "sqlite": "BLOB",
        }[dialect_name]
        for index in table.indexes:
            assert len(index.columns) <= 3
            if "store_key" not in index.columns:
                continue
            size = 0
            for column in index.columns:
                if isinstance(column.type, LargeBinary):
                    size += column.type.length
                elif isinstance(column.type, String):
                    size += column.type.length * 4
                elif isinstance(column.type, (BigInteger, Integer)):
                    size += 8
                else:
                    raise AssertionError((table.name, column.name, column.type))
            maxima[tuple(index.columns.keys())] = size
            assert size + 40 < 2704 < 3072
    assert maxima[("store_key", "key_locator")] == 384
    assert maxima[("store_key", "alias_locator")] == 384
    assert maxima[("store_key", "operation_locator")] == 384
    assert maxima[("store_key", "stream_locator", "sequence")] == 392


@pytest.mark.parametrize("dialect_name", ("mysql", "postgresql", "sqlite"))
def test_all_business_indexes_bound_columns_and_keep_unique_identities_nonnullable(dialect_name: str) -> None:
    from sqlalchemy import LargeBinary, String, UniqueConstraint
    from sqlalchemy.dialects import postgresql, sqlite
    from sqlalchemy.schema import CreateTable

    dialect = {"mysql": mysql, "postgresql": postgresql, "sqlite": sqlite}[dialect_name].dialect()
    metadata = build_sql_schema_metadata()
    assert set(metadata.tables) == {
        "ai_asset_batch_receipts", "ai_asset_changes", "ai_asset_entries", "ai_asset_heads",
        "ai_metric_definitions", "ai_metric_observations", "ai_native_content_edges",
        "ai_object_chunks", "ai_objects", "ai_state_aliases", "ai_state_facts",
        "ai_state_operations", "ai_state_records", "ai_state_sequences", "ai_step_events",
        "ai_step_messages", "ai_step_runs", "ai_step_sequences", "ai_step_snapshots", "ai_trace_events",
    }
    for table in metadata.tables.values():
        assert table.comment and all(column.comment for column in table.c)
        assert tuple(column.name for column in table.c)[-2:] == ("updated_at", "created_at")
        assert str(CreateTable(table).compile(dialect=dialect))
        indexes = [(table.primary_key, True)]
        indexes.extend((index, index.unique) for index in table.indexes)
        indexes.extend((constraint, True) for constraint in table.constraints if isinstance(constraint, UniqueConstraint))
        for index, unique in indexes:
            assert 1 <= len(index.columns) <= 3, (table.name, index.name)
            for column in index.columns:
                if isinstance(column.type, (LargeBinary, String)):
                    assert column.type.length is not None and column.type.length <= 128, (table.name, column.name)
                if unique:
                    assert not column.nullable, (table.name, index.name, column.name)
        for index in table.indexes:
            assert not index.dialect_options["mysql"].get("length"), (table.name, index.name)


@pytest.mark.parametrize("dialect_name", ("mysql", "postgresql", "sqlite"))
def test_raw_address_capacity_and_locator_types_match_reviewed_storage_contract(dialect_name: str) -> None:
    from sqlalchemy import CHAR
    from sqlalchemy.dialects import postgresql, sqlite

    dialect = {"mysql": mysql, "postgresql": postgresql, "sqlite": sqlite}[dialect_name].dialect()
    metadata = build_sql_schema_metadata()
    state_raw = {
        "ai_state_records": ("key", "scope_key", "parent_key"),
        "ai_state_aliases": ("alias_key", "record_key"),
        "ai_state_facts": ("stream_key", "owner_key"),
        "ai_state_sequences": ("stream_key",),
        "ai_state_operations": ("operation_key", "stream_key"),
    }
    raw_columns = {(name, column): 768 for name, columns in state_raw.items() for column in columns}
    raw_columns.update({("ai_step_" + name, "store_key"): 274
                        for name in ("runs", "events", "snapshots", "messages", "sequences")})
    raw_columns.update({("ai_asset_entries", "asset_id"): 512, ("ai_objects", "object_key"): 1024})
    for (name, column_name), capacity in raw_columns.items():
        table = metadata.tables[name]
        column = table.c[column_name]
        assert column.type.length == capacity, (name, column_name)
        assert all(column_name not in index.columns for index in table.indexes), (name, column_name)
        if name.startswith(("ai_state_", "ai_step_")):
            assert column.type.compile(dialect=dialect) == {
                "mysql": f"VARBINARY({capacity})", "postgresql": "BYTEA", "sqlite": "BLOB",
            }[dialect_name]
    for name in state_raw:
        assert metadata.tables[name].c.store_key.type.length == 128
    for table in metadata.tables.values():
        for column in table.c:
            if column.name.endswith("_locator"):
                assert isinstance(column.type, CHAR) and column.type.length == 64
                if dialect_name == "mysql":
                    assert column.type.compile(dialect=dialect) == "CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
    records = metadata.tables["ai_state_records"]
    assert records.c.sort_key.type.length is None
    assert records.c.sort_prefix.type.length == 128 and not records.c.sort_prefix.nullable
    for name in ("ai_step_events", "ai_step_snapshots"):
        table = metadata.tables[name]
        assert table.c.idempotency_key.type.length == 128 and not table.c.idempotency_key.nullable
        assert any(index.unique and tuple(index.columns.keys()) == ("store_locator", "run_id", "idempotency_key")
                   for index in table.indexes)
    for name in ("ai_metric_definitions", "ai_metric_observations"):
        assert metadata.tables[name].c.namespace.type.length == 128
    assert metadata.tables["ai_asset_batch_receipts"].c.idempotency_key.type.length == 128
    for name, columns in (
        ("ai_asset_entries", ("namespace", "kind", "asset_locator")),
        ("ai_objects", ("store_id", "object_locator")),
    ):
        assert any(index.unique and tuple(index.columns.keys()) == columns for index in metadata.tables[name].indexes)
