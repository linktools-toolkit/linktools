#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness Planning must preserve the intentionally narrow LinkTools surface."""

from copy import deepcopy
from dataclasses import replace

import pytest
from linktools.ai.persistence import Persistence
from linktools.ai.core import PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
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
from linktools.ai.runtime._compaction import CompactionPolicy
from linktools.ai.persistence.runs import (
    StagingAgentRunStore,
)
from pydantic_ai import Agent
from pydantic_ai.messages import CachePoint, ModelRequest, ModelResponse, TextContent, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
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
        agent_id="agent",
        agent_run_id="run",
        agent_run_seq=1,
        history_id=None,
        memory_scope=None,
        run_store=StagingAgentRunStore(),
        memory_store=None,
        tool_policy=(),
        compaction_policy=CompactionPolicy(),
        limits=PromptLimits(),
        planning=True,
        context_target_tokens=None,
        parent_agent_run_id=None,
        plan_store_resolver=lambda _ctx: None,  # type: ignore[return-value]
    )
    planning = next(
        capability
        for capability in capabilities
        if isinstance(capability, HarnessPlanning)
    )
    toolset = planning.get_toolset()
    assert toolset is not None
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )
    assert tuple(await toolset.get_tools(context)) == ("write_plan",)


async def test_harness_planning_prompt_is_visible_without_rewriting_history_or_duplicate_reminders() -> None:
    store = InMemoryPlanStore()
    await store.set_items([PlanItem(content="ship it", status=TaskStatus.in_progress)])
    capability = Planning(store=store, tools=("write_plan",), id="linktools-planning")
    captured = []

    async def model(messages, info):
        captured.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(model), capabilities=[capability])
    supplied = [ModelRequest(parts=[UserPromptPart("earlier")])]
    original = deepcopy(supplied)
    first = await agent.run("continue", message_history=supplied)
    assert supplied == original
    history = first.all_messages()
    frozen_history = deepcopy(history)
    await agent.run("continue again", message_history=history)
    assert history == frozen_history
    await store.set_items([PlanItem(content="verify it", status=TaskStatus.in_progress)])
    await agent.run("continue with updated plan", message_history=history)
    assert history == frozen_history
    assert len(captured) == 3
    for messages, expected_plan in zip(captured, ("ship it", "ship it", "verify it"), strict=True):
        tokens = []
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    tokens.extend([part.content] if isinstance(part.content, str) else part.content)
            assert expected_plan not in (message.instructions or "")
        text = [token if isinstance(token, str) else token.content for token in tokens if isinstance(token, (str, TextContent))]
        assert sum(expected_plan in token for token in text) == 1
        assert any("continue" in token for token in text)
        # Older supported Harness uses a request-only reminder after a cache
        # point. New Harness appends a history reminder only when it changed,
        # leaving cache policy to the model. Neither may rewrite caller history.
        cache_points = [index for index, token in enumerate(tokens) if isinstance(token, CachePoint)]
        if cache_points:
            plan = next(index for index, token in enumerate(tokens) if expected_plan in str(token))
            assert max(cache_points) < plan


async def test_runtime_plan_persistence_adds_no_arbitrary_size_limit() -> None:
    items = [RuntimePlanItem("x" * 600) for _ in range(129)]

    assert len(_validated_items(items)) == 129


async def test_runtime_plan_persists_only_the_current_payload_shape() -> None:
    state = Persistence(PersistenceStorage.in_memory())
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

        data = dict(record.data)
        data["future"] = True
        corrupted = replace(record, data=data)
        with pytest.raises(AIError) as raised:
            _decode_payload(corrupted)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        data = dict(record.data)
        data["version"] = True
        corrupted = replace(record, data=data)
        with pytest.raises(AIError) as raised:
            _decode_payload(corrupted)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        data = dict(record.data)
        data["version"] = 2
        corrupted = replace(record, data=data)
        with pytest.raises(AIError) as raised:
            _decode_payload(corrupted)
        assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

        data = dict(record.data)
        items = data["items"]
        assert isinstance(items, list)
        first = items[0]
        assert isinstance(first, dict)
        first["future"] = True
        corrupted = replace(record, data=data)
        with pytest.raises(AIError) as raised:
            _decode_payload(corrupted)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
