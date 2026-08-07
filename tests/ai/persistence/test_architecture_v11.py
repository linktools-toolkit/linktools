#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Structural deletion and import-boundary checks from the V11 contract."""

from pathlib import Path
import re


def test_removed_parallel_persistence_symbols_are_absent() -> None:
    root = Path("linktools-ai/src/linktools/ai")
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    for symbol in (
        "SessionTurnRecord",
        "TraceRecord",
        "TraceRepository",
        "trace_sequence",
        "TRACE_SEQUENCE_CONFLICT",
        "MODEL_RUN_STARTED",
        "TRACE_PERSISTENCE_FAILED",
        "AgentBackedSubagentProvider",
        "LocalRecordStore",
        "LocalExecutionRecord",
    ):
        assert re.search(rf"\b{re.escape(symbol)}\b", source) is None


def test_runtime_does_not_import_harness_or_sql_concrete_types() -> None:
    for path in Path("linktools-ai/src/linktools/ai/runtime").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "pydantic_ai_harness" not in source
        assert "adapter.sql" not in source
        assert "adapter.step" not in source
