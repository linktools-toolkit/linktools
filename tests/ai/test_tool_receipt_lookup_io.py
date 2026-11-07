#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery receipt existence reads reuse facts found in the current run."""

from collections import Counter
import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from sqlalchemy import event

from linktools.ai.persistence import Persistence, PersistenceStorage

from ._runtime_test_helpers import tool_run_context
from .test_native_tool_effect_owner import _bridge


async def _complete(state: Persistence, run_id: str, call_id: str) -> None:
    bridge = _bridge(state, run_id=run_id)
    decision = await bridge.begin(
        tool_run_context(), ToolCallPart("work", {}, tool_call_id=call_id),
        ToolDefinition(name="work"), {}, False,
    )
    await bridge.complete(decision, call_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("requested", (("current",), ("current", "prior", "missing")))
async def test_recovery_only_reads_unresolved_call_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, requested: tuple[str, ...],
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "receipts.db"))
    async with Persistence.open("native-effects", tenant_id="tenant", storage=storage) as state:
        await _complete(state, "original", "prior")
        await _complete(state, "resumed", "current")
        repository = state.checkpoint.tools
        lookup = repository.existing_call_ids
        reads: list[tuple[str, tuple[str, ...]]] = []

        async def record(run_id, call_ids, *, tenant_id):
            reads.append((run_id, tuple(call_ids)))
            return await lookup(run_id, call_ids, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "existing_call_ids", record)
        bridge = _bridge(state, run_id="resumed", source_run_id="original")
        assert await bridge.existing_call_ids(requested) == frozenset(requested) - {"missing"}
        assert reads == ([("resumed", requested)] if len(requested) == 1 else [
            ("resumed", requested), ("original", ("prior", "missing")),
        ])


@pytest.mark.asyncio
async def test_current_run_receipt_avoids_recovery_source_sql(tmp_path: Path, record_property) -> None:
    async with Persistence.open("native-effects", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "receipts.db")) as state:
        await _complete(state, "resumed", "current")
        repository = state.checkpoint.tools
        engine = repository.state_store.storage_group.context.engine.sync_engine
        statements: Counter[str] = Counter()

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements[statement.split()[0].upper()] += 1

        event.listen(engine, "before_cursor_execute", capture)
        try:
            expected = frozenset()
            for run_id in ("resumed", "original"):
                expected |= await repository.existing_call_ids(run_id, ("current",), tenant_id="tenant")
            baseline = statements.copy()
            statements.clear()
            actual = await _bridge(state, run_id="resumed", source_run_id="original").existing_call_ids(("current",))
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        record_property("sql_traffic", json.dumps({"baseline": baseline, "candidate": statements}))
        assert actual == expected == {"current"}
        assert statements["SELECT"] < baseline["SELECT"]
        assert statements["BEGIN"] < baseline["BEGIN"]
        assert not (set(statements) | set(baseline)) & {"INSERT", "UPDATE", "DELETE"}
