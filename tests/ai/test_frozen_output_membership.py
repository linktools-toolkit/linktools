#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public capability hooks can validate an explicitly captured input set."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ApprovalRequired, ModelRetry
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.tools import RunContext

from linktools.ai.capability import AgentContext, ExecutionCapability, StepPersistenceCapability
from linktools.ai.core import ApprovalDecision, ExecutionStatus, Principal
from linktools.ai.errors import ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest, ExecutionInputContext
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


_SCHEMA = {
    "type": "object",
    "properties": {"sources": {"type": "array", "items": {"type": "string"}}},
    "required": ["sources"],
    "additionalProperties": False,
}


@dataclass
class _Membership(AbstractCapability[AgentContext[None]]):
    id: str = "test.output-membership"
    approval: bool = False

    async def before_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any, args: Any) -> Any:
        if self.approval and call.tool_name == "continue_review" and not ctx.tool_call_approved:
            raise ApprovalRequired()
        return args

    async def after_output_validate(
        self, ctx: RunContext[AgentContext[None]], *, output_context: Any, output: Any,
    ) -> Any:
        del output_context
        allowed = frozenset(ctx.deps.session_metadata["accepted_sources"])
        selected = output["sources"]
        if len(set(selected)) != len(selected) or not set(selected).issubset(allowed):
            raise ModelRetry("Use distinct source identities from the accepted input.")
        return output


def _features(model: Any, effects: list[str], *, retries: int = 1, approval: bool = False,
              checkpoints: bool = False) -> RuntimeFeatures:
    features = RuntimeFeatures("frozen-output-membership")
    features.capability(_Membership(approval=approval), revision=1, scope="agent")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    if checkpoints:
        features.capability(StepPersistenceCapability(), scope="runtime")

    def record_effect(context: AgentContext[None]) -> str:
        effects.append("recorded")
        return "recorded"

    def continue_review(context: AgentContext[None]) -> str:
        return "approved"

    features.tool(record_effect, effect_policy="replay_safe")
    features.tool(continue_review, effect_policy="replay_safe")
    features.agent("default", allow_tools=("record_effect", "continue_review"),
                   allow_skills=(), allow_subagents=(), output_schema=_SCHEMA,
                   output_retries=retries)
    return model_features(_Models(_UsageFunctionModel(model)), features)


@pytest.mark.asyncio
@pytest.mark.parametrize(("sources", "correct"), (
    (("source-0",), False),
    (("unknown",), False),
    (("source-0", "source-0"), False),
    (("unknown",), True),
    (("source-0", "source-0"), True),
))
async def test_membership_correction_uses_frozen_input_without_schema_enumeration(
    sources: tuple[str, ...], correct: bool,
) -> None:
    effects: list[str] = []
    attempts: list[list[str]] = []
    schemas: list[str] = []
    original = [f"source-{index}" for index in range(2)]
    context = ExecutionInputContext.from_messages((), session_metadata={"accepted_sources": original})
    original[:] = ["unknown"]

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        schema = json.dumps(info.output_tools[0].parameters_json_schema, sort_keys=True)
        schemas.append(schema)
        assert "source-1" not in schema and len(schema) < 1024
        assert "source-1" not in str(messages)
        if not any(isinstance(part, ToolReturnPart) and part.tool_name == "record_effect"
                   for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart("record_effect", {}, tool_call_id="effect")])
        if attempts:
            feedback = [part for message in messages for part in message.parts
                        if isinstance(part, ToolReturnPart) and part.outcome == "retried"
                        and part.tool_name == info.output_tools[0].name]
            assert feedback
            assert "Use distinct source identities" in feedback[-1].model_response_str()
        selected = ["source-0"] if attempts and correct else list(sources)
        attempts.append(selected)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"sources": selected})])

    async with Persistence.open("frozen-output-membership", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=_features(model, effects)) as runtime:
            execution = await runtime.agents.get().start("Review the admitted input.", input_context=context)
            result = (await execution.wait(timeout_seconds=15)).result
            valid = sources == ("source-0",)
            assert result.status is (ExecutionStatus.SUCCEEDED if valid or correct else ExecutionStatus.FAILED)
            if result.status is ExecutionStatus.SUCCEEDED:
                assert result.output == {"sources": ["source-0"]}
                assert result.error_code is None
            else:
                assert result.error_code == ErrorCode.MODEL_RESPONSE_INVALID.value
    assert effects == ["recorded"]
    assert len(attempts) == (1 if valid else 2)
    assert len(set(schemas)) == 1


@pytest.mark.asyncio
async def test_cancelling_semantic_correction_keeps_completed_tool_single() -> None:
    effects: list[str] = []
    correcting = asyncio.Event()
    attempts = 0
    context = ExecutionInputContext.from_messages((), session_metadata={"accepted_sources": ["source"]})

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        if not any(isinstance(part, ToolReturnPart) and part.tool_name == "record_effect"
                   for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart("record_effect", {}, tool_call_id="effect")])
        attempts += 1
        if attempts == 1:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"sources": ["foreign"]})])
        correcting.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled correction must not finish")

    async with Persistence.open("frozen-output-membership", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=_features(model, effects)) as runtime:
            execution = await runtime.agents.get().start("Review.", input_context=context)
            await asyncio.wait_for(correcting.wait(), 15)
            await execution.cancel()
            result = (await execution.wait(timeout_seconds=15)).result
            assert result.status is ExecutionStatus.CANCELLED
    assert effects == ["recorded"] and attempts == 2


@pytest.mark.asyncio
async def test_membership_reopens_with_frozen_input_and_does_not_repeat_completed_tool(tmp_path: Path) -> None:
    effects: list[str] = []
    attempts: list[str] = []
    original = ["admitted-source"]
    context = ExecutionInputContext.from_messages((), session_metadata={"accepted_sources": original})

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        completed = {part.tool_name for message in messages for part in message.parts
                     if isinstance(part, ToolReturnPart) and part.outcome != "retried"}
        if "record_effect" not in completed:
            return ModelResponse(parts=[ToolCallPart("record_effect", {}, tool_call_id="effect")])
        if "continue_review" not in completed:
            return ModelResponse(parts=[ToolCallPart("continue_review", {}, tool_call_id="approval")])
        selected = "admitted-source" if attempts else "replacement-source"
        attempts.append(selected)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"sources": [selected]})])

    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    principal = Principal("reviewer", "default", "service")
    async with Persistence.open("frozen-output-membership", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence, features=_features(model, effects, approval=True),
                                  auto_recover=False) as runtime:
            execution = await runtime.agents.get().start("Review.", input_context=context, principal=principal)
            execution_id = execution.execution_id
            async with asyncio.timeout(15):
                while not (pending := await runtime.approvals.list(execution_id, principal=principal)):
                    await asyncio.sleep(0.01)
            assert effects == ["recorded"]
            await runtime.handoff()
    original[:] = ["replacement-source"]
    context = ExecutionInputContext.from_messages((), session_metadata={"accepted_sources": original})
    assert context.session_metadata["accepted_sources"] == ["replacement-source"]
    async with Persistence.open("frozen-output-membership", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence, features=_features(model, effects, approval=True),
                                  auto_recover=False) as runtime:
            await runtime.approvals.decide(execution_id, ApprovalDecisionRequest(
                principal, pending[0].approval_id, "approve", ApprovalDecision.APPROVE,
            ))
            result = (await runtime.executions.wait(execution_id, principal=principal,
                                                    timeout_seconds=15)).result
            assert result.status is ExecutionStatus.SUCCEEDED, result
            assert result.output == {"sources": ["admitted-source"]}
    assert attempts == ["replacement-source", "admitted-source"]
    assert effects == ["recorded"]


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoints", (False, True))
@pytest.mark.parametrize("same_response", (False, True))
@pytest.mark.parametrize("handoffs", (1, 2))
async def test_membership_recovery_keeps_consumed_output_retry_budget(
    tmp_path: Path, checkpoints: bool, same_response: bool, handoffs: int,
) -> None:
    effects: list[str] = []
    attempts: list[str] = []
    context = ExecutionInputContext.from_messages((), session_metadata={"accepted_sources": ["source"]})

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        completed = [part.tool_name for message in messages for part in message.parts
                     if isinstance(part, ToolReturnPart) and part.outcome != "retried"]
        if "record_effect" not in completed:
            return ModelResponse(parts=[ToolCallPart("record_effect", {}, tool_call_id="effect")])
        approved = completed.count("continue_review")
        approval = ToolCallPart("continue_review", {}, tool_call_id=f"approval-{approved}")
        if len(attempts) > approved and approved < handoffs:
            return ModelResponse(parts=[approval])
        attempts.append("foreign")
        parts = [ToolCallPart(info.output_tools[0].name, {"sources": ["foreign"]})]
        if same_response and approved < handoffs:
            parts.append(approval)
        return ModelResponse(parts=parts)

    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    principal = Principal("reviewer", "default", "service")
    for index in range(handoffs + 1):
        async with Persistence.open("frozen-output-membership", storage=storage) as persistence:
            features = _features(model, effects, retries=handoffs, approval=True, checkpoints=checkpoints)
            async with Runtime.attach(persistence=persistence, features=features, auto_recover=False) as runtime:
                if index == 0:
                    execution = await runtime.agents.get().start("Review.", input_context=context, principal=principal)
                    execution_id = execution.execution_id
                else:
                    await runtime.approvals.decide(execution_id, ApprovalDecisionRequest(
                        principal, pending[0].approval_id, f"approve-{index}", ApprovalDecision.APPROVE,
                    ))
                if index < handoffs:
                    async with asyncio.timeout(15):
                        while not (pending := await runtime.approvals.list(execution_id, principal=principal)):
                            await asyncio.sleep(0.01)
                    assert len(attempts) == index + 1
                    await runtime.handoff()
                else:
                    result = (await runtime.executions.wait(execution_id, principal=principal,
                                                           timeout_seconds=15)).result
                    assert result.status is ExecutionStatus.FAILED, result
                    assert result.error_code == ErrorCode.MODEL_RESPONSE_INVALID.value
    assert attempts == ["foreign"] * (handoffs + 1)
    assert effects == ["recorded"]


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoints", (False, True))
async def test_new_execution_does_not_charge_previous_session_output_retries(
    tmp_path: Path, checkpoints: bool,
) -> None:
    attempts: dict[int, int] = {}
    turn = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert all(not message.metadata or "linktools.ai.checkpoint-state" not in message.metadata
                   for message in messages)
        attempts[turn] = attempts.get(turn, 0) + 1
        source = "foreign" if attempts[turn] == 1 else "source"
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"sources": [source]})])

    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    async with Persistence.open("frozen-output-membership", storage=storage) as persistence:
        features = _features(model, [], checkpoints=checkpoints)
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            session = await runtime.agents.get().create_session("review", metadata={"accepted_sources": ["source"]})
            for turn in range(2):
                execution = await session.start("Review.")
                result = (await execution.wait(timeout_seconds=15)).result
                assert result.status is ExecutionStatus.SUCCEEDED, result
                assert result.output == {"sources": ["source"]}
    assert sorted(attempts.values()) == [2, 2]


@pytest.mark.asyncio
async def test_recovery_rejects_overdrawn_output_budget_before_model_request(tmp_path: Path) -> None:
    requests = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests > 1:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"sources": ["source"]})])
        return ModelResponse(parts=[
            ToolCallPart(info.output_tools[0].name, {"sources": ["foreign"]}, tool_call_id="invalid-1"),
            ToolCallPart(info.output_tools[0].name, {"sources": ["foreign"]}, tool_call_id="invalid-2"),
            ToolCallPart("continue_review", {}, tool_call_id="approval"),
        ])

    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    principal = Principal("reviewer", "default", "service")
    context = ExecutionInputContext.from_messages((), session_metadata={"accepted_sources": ["source"]})
    for reopening in (False, True):
        async with Persistence.open("frozen-output-membership", storage=storage) as persistence:
            features = _features(model, [], approval=True, checkpoints=True)
            async with Runtime.attach(persistence=persistence, features=features, auto_recover=False) as runtime:
                if not reopening:
                    execution = await runtime.agents.get().start("Review.", input_context=context, principal=principal)
                    execution_id = execution.execution_id
                    async with asyncio.timeout(15):
                        while not (pending := await runtime.approvals.list(execution_id, principal=principal)):
                            await asyncio.sleep(0.01)
                    await runtime.handoff()
                else:
                    await runtime.approvals.decide(execution_id, ApprovalDecisionRequest(
                        principal, pending[0].approval_id, "approve", ApprovalDecision.APPROVE,
                    ))
                    result = (await runtime.executions.wait(execution_id, principal=principal,
                                                           timeout_seconds=15)).result
                    assert result.status is ExecutionStatus.FAILED, result
                    assert result.error_code == ErrorCode.MODEL_RESPONSE_INVALID.value
    assert requests == 1
