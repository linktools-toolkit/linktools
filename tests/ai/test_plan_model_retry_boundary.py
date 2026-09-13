#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning adapter keeps model mistakes separate from infrastructure failures."""

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai_harness.planning import PlanItem as HarnessPlanItem

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._harness import HarnessPlanStoreAdapter

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

    with pytest.raises(ModelRetry, match="flat plan"):
        await adapter.set_items([item])

    assert store.calls == 0


async def test_runtime_plan_request_error_is_model_correctable() -> None:
    store = _Store(AIError(ErrorCode.REQUEST_FIELD_INVALID))
    adapter = _adapter(store)

    with pytest.raises(ModelRetry, match="Correct the plan"):
        await adapter.set_items([HarnessPlanItem(content="step")])

    assert store.calls == 1


async def test_runtime_plan_infrastructure_error_is_not_downgraded() -> None:
    store = _Store(AIError(ErrorCode.STORAGE_UNAVAILABLE))
    adapter = _adapter(store)

    with pytest.raises(AIError) as raised:
        await adapter.set_items([HarnessPlanItem(content="step")])

    assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
