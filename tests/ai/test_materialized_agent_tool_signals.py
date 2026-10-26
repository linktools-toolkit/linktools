#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""End-to-end model-facing tool signal composition regressions."""
from ._runtime_test_helpers import model_features

from ._runtime_test_helpers import workspace_features

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext

from linktools.ai.capability import AgentContext, ToolCallFailed
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics, Observation
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.capability import TraceCapability
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.workspace import (
    Workspace,
    WorkspacePolicy,
    ToolPermissionPolicy,
)


class _CompositionModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    contract: dict[str, JsonValue] = {
        "provider": "test",
        "model": "test",
    }

    def materialize(self) -> TestModel:
        return TestModel(
            call_tools=["business", "read_file", "capability_tool"],
            custom_output_text="done",
        )


class _CompositionModels(_CompositionModelBinding):
    pass


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
    workspace = Workspace.load(
        tmp_path,

        policy=WorkspacePolicy(
            tool_permissions=ToolPermissionPolicy(default="deny")
        ),
    )
    application = workspace_features("application", workspace=workspace)
    application.capability(TraceCapability(), scope="runtime")
    application.tool(_business_tool, name="business", effect_policy="replay_safe")
    application.capability(_FailingCapability(), scope="agent")
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
    state = PersistenceStorage.in_memory()
    started = datetime.now(timezone.utc) - timedelta(seconds=1)

    async with Persistence.open('default', storage=state) as persistence_98, Runtime.attach(persistence=persistence_98, metrics=metrics, features=model_features(_CompositionModels(), application)) as runtime:
        result = (await runtime.agents.get("default").run(
            "prompt",
            timeout_seconds=10,
        )).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "done"}

        history = await runtime.executions.history(
            result.execution_id,
            principal=runtime.default_principal,
            include_content=True,
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
