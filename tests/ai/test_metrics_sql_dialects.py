#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics schema must compile for every supported SQL dialect."""

import pytest
from linktools.ai.observe import build_metrics_sql_metadata
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.schema import CreateTable


@pytest.mark.parametrize("dialect", (mysql.dialect(), postgresql.dialect()))
def test_metrics_schema_compiles_for_supported_sql_dialects(dialect: object) -> None:
    metadata = build_metrics_sql_metadata()

    ddl = tuple(str(CreateTable(table).compile(dialect=dialect)) for table in metadata.tables.values())

    assert len(ddl) == 2
    assert any("ai_metric_definitions" in statement for statement in ddl)
    assert any("ai_metric_observations" in statement for statement in ddl)


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
