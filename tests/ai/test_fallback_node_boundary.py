#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A failed request finishes before an outer node recovery spends shared usage."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent, messages as sdk_messages
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import AgentContext, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.model import FallbackBinding
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage

from .test_runtime_fallback_failures import _Binding
from .test_runtime_fallback_partial_usage import _PartialThenFailure


pytestmark = pytest.mark.skipif(
    not hasattr(sdk_messages, "ModelRequestAttempt"),
    reason="installed SDK does not expose fallback attempt observations",
)


class _NestedRecoveryCapability(AbstractCapability[AgentContext[None]]):
    async def on_node_run_error(
        self, ctx: RunContext[AgentContext[None]], *, node: AgentNode[AgentContext[None]], error: Exception,
    ) -> NodeResult[AgentContext[None]]:
        del node

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            return ModelResponse(
                [TextPart("recovery analysis")], usage=RequestUsage(input_tokens=101, output_tokens=7),
            )

        await Agent(FunctionModel(respond)).run("analyze the failure", usage=ctx.usage)
        raise error


@pytest.mark.asyncio
async def test_nested_node_recovery_usage_does_not_enter_failed_request_observation() -> None:
    features = RuntimeFeatures("fallback-node-boundary")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.model("default", FallbackBinding(
        _Binding(_PartialThenFailure([], earlier_attempt=False)), fallback_on=FallbackExceptionGroup,
    ))
    features.capability(TraceCapability(), scope="runtime")
    features.capability(_NestedRecoveryCapability(), id="nested-node-recovery", scope="runtime")

    async with Runtime.open(
        "fallback-node-boundary", features=features, storage=RuntimeStorage.in_memory(),
    ) as runtime:
        execution = await runtime.agents.get().start("respond")
        result = (await execution.wait(timeout_seconds=10)).result
        interactions = (await execution.model_interactions(include_content=True)).items

    assert result.status is ExecutionStatus.FAILED
    assert result.error_code == "INTERNAL_ERROR"
    assert result.usage.input_tokens == 110
    assert result.usage.output_tokens == 10
    assert len(interactions) == 1
    interaction = interactions[0]
    assert interaction.status == "FAILED"
    assert interaction.error_code == result.error_code
    assert interaction.model_request_seq == result.safe_error_details["model_request_seq"] == 1
    assert interaction.usage.input_tokens == 9
    assert interaction.usage.output_tokens == 3
    assert interaction.usage.model_requests == 1
