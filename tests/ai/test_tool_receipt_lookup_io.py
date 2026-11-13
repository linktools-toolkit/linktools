#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery receipt reads batch explicit durable operation references."""

from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from sqlalchemy import event

from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.execution._tool_identity import tool_operation_id
from linktools.ai.persistence.contracts import ToolOperationRecord

from ._runtime_test_helpers import tool_run_context
from .test_native_tool_effect_owner import _bridge


def _identity(run_id: str, call_id: str) -> str:
    return tool_operation_id(tenant_id="tenant", execution_id="execution",
                             agent_run_id=run_id, tool_call_id=call_id)


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
async def test_recovery_batches_explicit_operation_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, requested: tuple[str, ...],
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "receipts.db"))
    async with Persistence.open("native-effects", tenant_id="tenant", storage=storage) as state:
        await _complete(state, "original", "prior")
        await _complete(state, "resumed", "current")
        repository = state.checkpoint.tools
        lookup = repository.get_operations
        reads: list[tuple[str, ...]] = []
        identities = {"prior": _identity("original", "prior"), "current": _identity("resumed", "current")}

        async def record(operation_ids: Sequence[str], *, tenant_id: str) -> tuple[ToolOperationRecord, ...]:
            reads.append(tuple(operation_ids))
            return await lookup(operation_ids, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "get_operations", record)
        bridge = _bridge(state, run_id="resumed", source_run_id="original", operation_ids=identities)
        assert await bridge.existing_call_ids(requested) == frozenset(requested) - {"missing"}
        assert reads == [tuple(identities.get(call_id) or _identity("resumed", call_id) for call_id in requested)]
        assert len(await repository.list_by_execution("execution", tenant_id="tenant")) == 2


@pytest.mark.asyncio
async def test_explicit_receipts_batch_without_recovery_source_sql(tmp_path: Path, record_property) -> None:
    async with Persistence.open("native-effects", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "receipts.db")) as state:
        await _complete(state, "resumed", "current")
        await _complete(state, "original", "prior")
        identities = {"current": _identity("resumed", "current"), "prior": _identity("original", "prior")}
        repository = state.checkpoint.tools
        engine = repository.state_store.storage_group.context.engine.sync_engine
        statements: Counter[str] = Counter()

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements[statement.split()[0].upper()] += 1

        event.listen(engine, "before_cursor_execute", capture)
        try:
            expected = frozenset()
            for identity in identities.values():
                values = await repository.get_operations((identity,), tenant_id="tenant")
                expected |= frozenset(value.tool_call_id for value in values)
            baseline = statements.copy()
            statements.clear()
            actual = await _bridge(state, run_id="resumed", source_run_id="original",
                                   operation_ids=identities).existing_call_ids(("current", "prior"))
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        record_property("sql_traffic", json.dumps({"baseline": baseline, "candidate": statements}))
        assert actual == expected == {"current", "prior"}
        assert statements["SELECT"] < baseline["SELECT"]
        assert statements["BEGIN"] < baseline["BEGIN"]
        assert not (set(statements) | set(baseline)) & {"INSERT", "UPDATE", "DELETE"}
