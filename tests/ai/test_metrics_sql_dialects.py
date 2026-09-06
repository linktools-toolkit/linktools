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
