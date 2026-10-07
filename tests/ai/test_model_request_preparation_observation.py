#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recorded model requests match the prepared input sent to the provider."""

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, ReinjectSystemPrompt
from pydantic_ai.messages import InstructionPart, ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import ExecutionInputContext, Runtime, RuntimeStorage
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._message import decode_model_messages
from linktools.ai.runtime._metric_capability import ModelObservationCapability
from linktools.ai.runtime._model_interaction import StagedContextInline, model_identity, request_envelope
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.runtime.state._steps import StagingAgentRunStore

from ._runtime_test_helpers import _UsageFunctionModel
from .test_model_interaction_regressions import _interaction
from .test_task_mixed_node_reliability import _TaskTestModels


class _ChangeRequestAfterResponse(AbstractCapability[AgentContext[object]]):
    async def after_model_request(
        self,
        ctx: RunContext[AgentContext[object]],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        del ctx
        request_context.messages = [ModelRequest(parts=[UserPromptPart("after-only diagnostic")])]
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancel"))
async def test_imported_request_records_prepared_provider_input_while_running(
    tmp_path: Path,
    outcome: str,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    provider_requests: list[JsonValue] = []
    provider_envelopes: list[dict[str, JsonValue]] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        provider_requests.append(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise ValueError("provider failed")
        return ModelResponse(parts=[TextPart("done")])

    class PreparedModel(_UsageFunctionModel):
        async def request(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            envelope, _ = request_envelope(
                model_settings=model_settings,
                parameters=model_request_parameters,
                streaming=True,
            )
            provider_envelopes.append(envelope)
            return await super().request(messages, model_settings, model_request_parameters)

        def prepare_messages(
            self,
            messages: list[ModelMessage],
            model_request_parameters: ModelRequestParameters | None = None,
        ) -> list[ModelMessage]:
            last = messages[-1]
            assert isinstance(last, ModelRequest)
            return [*messages[:-1], replace(last, parts=[*last.parts, UserPromptPart("prepared-only input")])]

    selected_model = PreparedModel(model, model_name="prepared-provider")

    class PrepareRequest(AbstractCapability[AgentContext[object]]):
        async def before_model_request(
            self,
            ctx: RunContext[AgentContext[object]],
            request_context: ModelRequestContext,
        ) -> ModelRequestContext:
            del ctx
            request_context.model = selected_model
            request_context.model_id = "prepared-route"
            request_context.model_settings = {**(request_context.model_settings or {}), "temperature": 0.25}
            request_context.model_request_parameters = replace(
                request_context.model_request_parameters,
                instruction_parts=[InstructionPart(content="prepared-only instruction", dynamic=False)],
            )
            return request_context

    group = CapabilityGroup("prepared-input")
    group.capability(PrepareRequest(), id="prepare-request")
    group.capability(_ChangeRequestAfterResponse(), id="change-after-response")
    group.agent("default", model="default", system_prompt="candidate-only behavior",
                allow_tools=(), allow_skills=(), allow_subagents=())
    imported = ExecutionInputContext.from_messages((
        ModelRequest(parts=[SystemPromptPart("original-only behavior"), UserPromptPart("previous turn")]),
        ModelResponse(parts=[TextPart("previous response")]),
    ), replace_history_system_prompt=True)
    metric_store = InMemoryMetricStore()
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    metrics = Metrics.from_store(metric_store, namespace="prepared-input")
    async with Runtime.open("prepared-input", models=_TaskTestModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            capabilities=(group,), metrics=metrics) as runtime:
        execution = await runtime.agents.get().start("new question", input_context=imported)
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert "candidate-only behavior" in str(provider_requests[0])
            assert "original-only behavior" not in str(provider_requests[0])
            assert "prepared-only input" in str(provider_requests[0])
            running = (await execution.model_interactions(include_content=True)).items[0]
            assert running.status == "RUNNING"
            assert "candidate-only behavior" in str(running.request)
            assert "original-only behavior" not in str(running.request)
            assert running.request["messages"] == provider_requests[0]
            assert {key: running.request[key] for key in provider_envelopes[0]} == provider_envelopes[0]
            assert running.model == model_identity(selected_model, route_id="prepared-route")
            if outcome == "cancel":
                await execution.cancel()
        finally:
            release.set()
            result = (await execution.wait()).result
        assert result.status is {
            "success": ExecutionStatus.SUCCEEDED,
            "failure": ExecutionStatus.FAILED,
            "cancel": ExecutionStatus.CANCELLED,
        }[outcome]
        terminal = (await execution.model_interactions(include_content=True)).items[0]
        assert terminal.request == running.request
        assert "after-only diagnostic" not in str(terminal.request)
    observations = await metric_store.scan_observations(
        "prepared-input", kind="linktools.model.request", start=start,
        end=datetime.now(timezone.utc) + timedelta(seconds=1), cursor=None, limit=100,
    )
    assert len(observations.items) == 1
    assert observations.items[0].dimensions["selected_model_name"] == selected_model.model_name
    assert observations.items[0].dimensions["selected_model_system"] == selected_model.system


@pytest.mark.asyncio
async def test_nonstream_request_freezes_provider_input_before_after_hooks() -> None:
    provider_requests: list[JsonValue] = []

    async def provider(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        provider_requests.append(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
        return ModelResponse(parts=[TextPart("done")])

    class PreparedModel(FunctionModel):
        def prepare_messages(
            self,
            messages: list[ModelMessage],
            model_request_parameters: ModelRequestParameters | None = None,
        ) -> list[ModelMessage]:
            last = messages[-1]
            assert isinstance(last, ModelRequest)
            return [*messages[:-1], replace(last, parts=[*last.parts, UserPromptPart("prepared-only input")])]

    store = StagingAgentRunStore()
    recorder = AgentRunRecorder(store, execution_id="execution", agent_run_id="run")
    await recorder.register_agent_run(AgentRunRecord("run"))
    observation = ModelObservationCapability(
        None, source_namespace="prepared-input", tenant_id="default", execution_id="execution",
        session_id=None, agent_run_id="run", agent_id="agent", interaction_recorder=recorder,
    )
    agent = Agent(
        PreparedModel(provider), system_prompt="candidate-only behavior",
        capabilities=[observation, ReinjectSystemPrompt(replace_existing=True), _ChangeRequestAfterResponse()],
    )
    await agent.run("question", deps=SimpleNamespace(correlation={}), message_history=[
        ModelRequest(parts=[SystemPromptPart("original-only behavior"), UserPromptPart("old question")]),
        ModelResponse(parts=[TextPart("old answer")]),
    ])
    interaction = (await store.list_model_interactions(agent_run_id="run"))[0]
    messages: list[ModelMessage] = []
    for item in interaction.request_context.items:
        assert isinstance(item, StagedContextInline)
        messages.extend(decode_model_messages(store.staged_payload("run", item.payload_digest)))
    assert ModelMessagesTypeAdapter.dump_python(messages, mode="json") == provider_requests[0]
    assert "candidate-only behavior" in str(provider_requests[0])
    assert "prepared-only input" in str(provider_requests[0])
    assert "original-only behavior" not in str(provider_requests[0])
    assert "after-only diagnostic" not in str(provider_requests[0])
    assert json.loads(store.staged_payload("run", interaction.request_envelope_digest))["streaming"] is False


@pytest.mark.asyncio
async def test_request_preparation_is_single_use_and_preserves_lifecycle_identity() -> None:
    store = StagingAgentRunStore()
    pending = replace(_interaction(1), status="RUNNING", duration_ns=None, finished_at=None)
    store.stage_model_interaction(pending)
    prepared = replace(pending, model={"route_id": "prepared"}, request_envelope_digest="b" * 64)
    for invalid in (replace(prepared, step_index=2), replace(prepared, started_at=pending.started_at + timedelta(seconds=1))):
        with pytest.raises(AIError) as caught:
            store.prepare_model_interaction(invalid)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    with pytest.raises(AIError):
        store.stage_model_interaction(prepared)
    store.prepare_model_interaction(prepared)
    with pytest.raises(AIError):
        store.prepare_model_interaction(prepared)
    terminal = replace(prepared, status="CANCELLED", duration_ns=1, finished_at=prepared.started_at)
    store.stage_model_interaction(terminal)
    with pytest.raises(AIError):
        store.prepare_model_interaction(replace(prepared, model={"route_id": "late"}))
    assert await store.list_model_interactions(agent_run_id="run") == [terminal]
