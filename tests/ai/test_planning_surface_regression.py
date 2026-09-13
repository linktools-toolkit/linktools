#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness Planning must preserve the intentionally narrow LinkTools surface."""

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._plan import (
    PlanItem as RuntimePlanItem,
    RuntimePlanStore,
    _decode_payload,
    _validated_items,
)
from linktools.ai.runtime._capabilities import (
    compose_platform_capabilities,
)
from linktools.ai.runtime._harness_planning import HarnessPlanning
from linktools.ai.runtime._compaction import RuntimeCompactionPolicy
from linktools.ai.runtime.state._steps import (
    StagingStepStore,
)
from pydantic_ai.messages import CachePoint, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanItem,
    Planning,
    TaskStatus,
)

pytestmark = pytest.mark.asyncio


async def test_linktools_planning_registers_only_write_plan() -> None:
    capabilities = await compose_platform_capabilities(
        agent_name="agent",
        step_run_id="run",
        segment_sequence=1,
        history_id=None,
        memory_scope=None,
        step_store=StagingStepStore(),
        memory_store=None,
        ordinary_tool_policy=(),
        compaction_policy=RuntimeCompactionPolicy(),
        planning=True,
        context_target_tokens=None,
        parent_step_run_id=None,
        plan_store_resolver=lambda _ctx: None,  # type: ignore[return-value]
    )
    planning = next(
        capability
        for capability in capabilities
        if isinstance(capability, HarnessPlanning)
    )
    assert tuple(planning.get_toolset().tools) == ("write_plan",)


async def test_harness_planning_prompt_is_request_scoped_and_cache_safe() -> None:
    store = InMemoryPlanStore()
    await store.set_items([PlanItem(content="ship it", status=TaskStatus.in_progress)])
    capability = Planning(
        store=store,
        tools=("write_plan",),
        id="linktools-planning",
    )
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[ModelRequest(parts=[UserPromptPart("continue")])],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    captured: list[ModelRequest] = []

    async def handler(current: ModelRequestContext) -> ModelResponse:
        assert isinstance(current.messages[-1], ModelRequest)
        captured.append(current.messages[-1])
        return ModelResponse(parts=[])

    await capability.wrap_model_request(
        context,
        request_context=request_context,
        handler=handler,
    )

    assert len(captured) == 1
    tokens: list[object] = []
    for part in captured[0].parts:
        if not isinstance(part, UserPromptPart):
            continue
        if isinstance(part.content, str):
            tokens.append(part.content)
        else:
            tokens.extend(part.content)

    cache_index = next(
        index for index, item in enumerate(tokens) if isinstance(item, CachePoint)
    )
    plan_index = next(
        index for index, item in enumerate(tokens) if "ship it" in str(item)
    )
    assert cache_index < plan_index
    assert any("continue" in str(item) for item in tokens)


async def test_runtime_plan_persistence_adds_no_arbitrary_size_limit() -> None:
    items = [RuntimePlanItem("x" * 600) for _ in range(129)]

    assert len(_validated_items(items)) == 129


async def test_runtime_plan_persists_only_the_current_payload_shape() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="plan-shape", tenant_id="tenant")
    try:
        store = RuntimePlanStore(
            state.execution.executions.state_store,
            namespace="plan-shape",
            tenant_id="tenant",
            owner_kind="execution",
            owner_id="execution",
        )
        await store.write_plan([RuntimePlanItem("ship it")])
        record = await store._store.read(
            lambda transaction: transaction.get_record(store._key)
        )
        assert record is not None
        assert record.data == {
            "version": 1,
            "items": [{"content": "ship it", "status": "pending"}],
        }
        assert _decode_payload(record) == ([RuntimePlanItem("ship it")], 1)

        record.data["future"] = True
        with pytest.raises(AIError) as raised:
            _decode_payload(record)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        record.data.pop("future")

        record.data["version"] = True
        with pytest.raises(AIError) as raised:
            _decode_payload(record)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        record.data["version"] = 2
        with pytest.raises(AIError) as raised:
            _decode_payload(record)
        assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
        record.data["version"] = 1

        record.data["items"][0]["future"] = True
        with pytest.raises(AIError) as raised:
            _decode_payload(record)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
