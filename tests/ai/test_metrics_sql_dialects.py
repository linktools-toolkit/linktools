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


def test_metrics_retention_mysql_ddl_matches_canonical_metadata() -> None:
    from pathlib import Path

    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    dialect = mysql.dialect()
    migration = (
        Path(__file__).resolve().parents[2] / "linktools-ai" / "migrations" / "init_schema.sql"
    ).read_text(encoding="utf-8")
    for name in ("namespace", "observation_id", "payload_json", "active_occurred_at", "pruned_at"):
        column = table.c[name]
        nullability = "NULL" if column.nullable else "NOT NULL"
        assert f"{name} {column.type.compile(dialect=dialect)} {nullability}" in migration
        assert f"COMMENT '{column.comment}'" in migration
    indexes = {
        index.name: tuple(column.name for column in index.columns)
        for index in table.indexes if index.info.get("ddl_dialect") == "mysql"
    }
    assert indexes["ix_namespace_digest_kind_active_occurred_at"] == (
        "namespace_digest", "kind", "active_occurred_at",
    )
    assert indexes["ix_namespace_digest_active_occurred_at"] == (
        "namespace_digest", "active_occurred_at",
    )
    for name, columns in indexes.items():
        assert f"KEY {name} ({', '.join(columns)})" in migration
    assert f"COMMENT='{table.comment}'" in migration
