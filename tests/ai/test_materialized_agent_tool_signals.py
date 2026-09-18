#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end model-facing tool signal composition regressions."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext

from linktools.ai.capability import AgentContext, CapabilityGroup, ToolCallFailed
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics, Observation
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.workspace import (
    Workspace,
    WorkspacePolicy,
    WorkspaceToolPermissionPolicy,
)


class _CompositionModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    vision = False
    fingerprint = "a" * 64
    semantic_payload: dict[str, JsonValue] = {
        "provider": "test",
        "model": "test",
    }

    def materialize(self) -> TestModel:
        return TestModel(
            call_tools=["business", "read_file", "capability_tool"],
            custom_output_text="done",
        )


class _CompositionModels:
    def snapshot(self) -> "_CompositionModels":
        return self

    def resolve(self, route_id: str) -> _CompositionModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _CompositionModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _CompositionModelBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _CompositionModelBinding.semantic_payload
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _CompositionModelBinding()


async def _business_tool(_ctx: RunContext[AgentContext[None]]) -> str:
    raise ToolCallFailed("business failed")


class _FailingCapability(AbstractCapability[AgentContext[None]]):
    id = "application.failing-capability"

    def get_toolset(self) -> FunctionToolset[AgentContext[None]]:
        async def capability_tool(
            _ctx: RunContext[AgentContext[None]],
        ) -> str:
            raise ToolCallFailed("capability failed")

        return FunctionToolset([capability_tool], id=self.id)


@pytest.mark.asyncio
async def test_materialized_agent_converts_all_model_facing_tool_signals(
    tmp_path: Path,
) -> None:
    application = CapabilityGroup[None]("application")
    application.tool(_business_tool, name="business", effect="replay_safe")
    application.capability(_FailingCapability())
    application.agent(
        "default",
        model="default",
        allow_tools=("business", "read_file"),
        allow_skills=(),
        allow_subagents=(),
        allow_capabilities=("application.failing-capability",),
        tool_retries=0,
        output_retries=0,
    )
    metric_store = InMemoryMetricStore()
    metrics = Metrics.from_store(metric_store, namespace="composition")
    workspace = Workspace.load(
        tmp_path,
        workspace_id="workspace",
        policy=WorkspacePolicy(
            tool_permissions=WorkspaceToolPermissionPolicy(default="deny")
        ),
    )
    state = RuntimeState.in_memory()
    started = datetime.now(timezone.utc) - timedelta(seconds=1)

    async with Runtime.open(
        workspace.workspace_id,
        models=_CompositionModels(),  # type: ignore[arg-type]
        state=state,
        capabilities=(CapabilityGroup.from_workspace(workspace), application),
        metrics=metrics,
    ) as runtime:
        result = await runtime.agent("default").run(
            "prompt",
            timeout_seconds=10,
        )
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "done"}

        history = await runtime.execution.history(
            result.execution_id,
            principal=runtime.default_principal,
        )
        tool_results = {
            item.tool_name: item.content
            for item in history.items
            if item.item_kind == "tool_result"
        }

    assert tool_results == {
        "business": "business failed",
        "read_file": (
            "Workspace policy does not allow this tool in the current run. "
            "Repeating the same call will not change the policy; use an allowed "
            "tool or another approach."
        ),
        "capability_tool": "capability failed",
    }
    observations_page = await metric_store.scan_observations(
        "composition",
        "linktools.tool.execution",
        started,
        datetime.now(timezone.utc) + timedelta(seconds=1),
        cursor=None,
        limit=10,
    )
    observations: tuple[Observation, ...] = observations_page.items
    assert {
        observation.dimensions["tool_name"] for observation in observations
    } == {"business", "read_file", "capability_tool"}
    assert len(observations) == 3
    assert {
        observation.error_code for observation in observations
    } == {ErrorCode.TOOL_EXECUTION_FAILED.value}
