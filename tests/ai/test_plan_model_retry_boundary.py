#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning adapter keeps model mistakes separate from infrastructure failures."""

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanItem as HarnessPlanItem,
)

from linktools.ai.capability import ToolCallRejected
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._harness import HarnessPlanStoreAdapter
from linktools.ai.runtime._harness_planning import HarnessPlanning

pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self, error: AIError | None = None) -> None:
        self.error = error
        self.calls = 0

    async def write_plan(self, items: object) -> None:
        del items
        self.calls += 1
        if self.error is not None:
            raise self.error


def _adapter(store: _Store) -> HarnessPlanStoreAdapter:
    adapter = object.__new__(HarnessPlanStoreAdapter)
    adapter._store = store  # type: ignore[attr-defined]
    return adapter


async def test_unsupported_subtask_plan_is_model_correctable() -> None:
    store = _Store()
    adapter = _adapter(store)
    item = HarnessPlanItem(content="child", parent_id="parent")

    with pytest.raises(ToolCallRejected, match="flat plan"):
        await adapter.set_items([item])

    assert store.calls == 0


async def test_planning_tool_rejects_duplicate_ids_before_store() -> None:
    capability = HarnessPlanning(
        store=InMemoryPlanStore(),
        tools=("write_plan",),
    )
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )
    toolset = capability.get_toolset()
    assert toolset is not None
    tools = await toolset.get_tools(context)
    duplicate = HarnessPlanItem(id="same", content="step")

    with pytest.raises(ToolCallRejected):
        await toolset.call_tool(
            "write_plan",
            {"items": [duplicate, duplicate.model_copy(deep=True)]},
            context,
            tools["write_plan"],
        )


async def test_runtime_plan_request_error_is_model_correctable() -> None:
    store = _Store(AIError(ErrorCode.REQUEST_FIELD_INVALID))
    adapter = _adapter(store)

    with pytest.raises(AIError) as raised:
        await adapter.set_items([HarnessPlanItem(content="step")])

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert store.calls == 1


async def test_runtime_plan_infrastructure_error_is_not_downgraded() -> None:
    store = _Store(AIError(ErrorCode.STORAGE_UNAVAILABLE))
    adapter = _adapter(store)

    with pytest.raises(AIError) as raised:
        await adapter.set_items([HarnessPlanItem(content="step")])

    assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
