#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics schema must compile for every supported SQL dialect."""

import pytest
from linktools.ai.migrate import build_sql_schema_metadata
from linktools.ai.observe import build_metrics_sql_metadata
from sqlalchemy import create_mock_engine
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateIndex, CreateTable, ExecutableDDLElement


@pytest.mark.parametrize("dialect_name", ("mysql", "postgresql", "sqlite"))
@pytest.mark.parametrize("combined_schema", (False, True), ids=("metrics", "combined"))
def test_metrics_schema_compiles_for_supported_sql_dialects(
    dialect_name: str, combined_schema: bool,
) -> None:
    metadata = build_sql_schema_metadata() if combined_schema else build_metrics_sql_metadata()
    tables: set[str] = set()
    indexes: set[tuple[str, str]] = set()

    def compile_ddl(statement: ExecutableDDLElement, *_args: object, **_kwargs: object) -> None:
        assert str(statement.compile(dialect=engine.dialect))
        if isinstance(statement, CreateTable):
            tables.add(statement.element.name)
        elif isinstance(statement, CreateIndex):
            index = statement.element
            name = engine.dialect.identifier_preparer.format_index(index)
            limit = engine.dialect.max_index_name_length or engine.dialect.max_identifier_length
            assert len(name.encode("utf-8")) <= limit
            if len(index.name.encode("utf-8")) <= limit:
                assert name == index.name
            identity = (index.table.name if dialect_name == "mysql" else "", name)
            assert identity not in indexes
            indexes.add(identity)

    engine = create_mock_engine(f"{dialect_name}://", compile_ddl)
    metadata.create_all(engine, checkfirst=False)

    assert tables == set(metadata.tables)
    assert {"ai_metric_definitions", "ai_metric_observations"} <= tables
    assert indexes
    observations = metadata.tables["ai_metric_observations"]
    definitions = metadata.tables["ai_metric_definitions"]
    assert "observation_digest" not in observations.c
    for table in (definitions, observations):
        assert "namespace_digest" not in table.c
        assert table.c.namespace.type.length == 256
        if dialect_name == "postgresql":
            assert 'COLLATE "C"' in str(table.c.namespace.type.compile(dialect=engine.dialect))
    if dialect_name == "postgresql":
        assert 'COLLATE "C"' in str(observations.c.observation_id.type.compile(dialect=engine.dialect))


@pytest.mark.parametrize("table_name", ("ai_metric_definitions", "ai_metric_observations"))
def test_metrics_mysql_ddl_matches_canonical_metadata(table_name: str) -> None:
    from pathlib import Path

    table = build_metrics_sql_metadata().tables[table_name]
    dialect = mysql.dialect()
    migration = (
        Path(__file__).resolve().parents[2] / "linktools-ai" / "migrations" / "init_schema.sql"
    ).read_text(encoding="utf-8")
    for column in table.c:
        if column.name in {"id", "updated_at", "created_at"}:
            continue
        nullability = "NULL" if column.nullable else "NOT NULL"
        assert f"{column.name} {column.type.compile(dialect=dialect)} {nullability}" in migration
        assert f"COMMENT '{column.comment}'" in migration
    indexes = {
        index.name: index
        for index in table.indexes if index.info.get("ddl_dialect") == "mysql"
    }
    expected = (
        {"uk_namespace_metric_name_revision": ("namespace", "metric_name", "revision")}
        if table_name == "ai_metric_definitions" else {
            "uk_namespace_observation_id": ("namespace", "observation_id"),
            "ix_namespace_kind_active_occurred_at": ("namespace", "kind", "active_occurred_at"),
            "ix_namespace_active_occurred_at": ("namespace", "active_occurred_at"),
        }
    )
    for name, columns in expected.items():
        assert tuple(column.name for column in indexes[name].columns) == columns
    for name, index in indexes.items():
        prefixes = index.dialect_options["mysql"]["length"] or {}
        assert prefixes == ({"namespace": 128} if name.startswith("ix_namespace_") else {})
        columns = (
            f"{column.name}({prefixes[column.name]})" if column.name in prefixes else column.name
            for column in index.columns
        )
        assert f"KEY {name} ({', '.join(columns)})" in migration
    assert f"COMMENT='{table.comment}'" in migration
