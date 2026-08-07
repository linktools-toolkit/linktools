#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dialect-independent SQL StepStore checks."""

from pydantic_ai_harness.step_persistence import StepStore

from linktools.ai.adapter.step import SqlMediaStore, SqlStepStore


def test_one_sql_step_store_implements_the_public_harness_protocol() -> None:
    assert isinstance(SqlStepStore.__new__(SqlStepStore), StepStore)
    assert hasattr(SqlMediaStore, "put")


def test_sql_table_names_and_namespace_columns_are_separate() -> None:
    source = open("linktools-ai/src/linktools/ai/adapter/step.py", encoding="utf-8").read()
    assert 'storage_name("step_runs")' in source
    assert 'storage_name("step_events")' in source
    assert "namespace_key" in source
    assert "namespace=self._namespace" not in source
