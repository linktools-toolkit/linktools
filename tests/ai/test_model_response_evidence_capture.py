#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Response evidence survives invalid calls and asynchronous event handoff."""
from ._runtime_test_helpers import model_features

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.test import TestModel

from linktools.ai.capability import AgentContext, TraceCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._journal import MESSAGE_SEQ_METADATA_KEY, ModelRequestJournal
from linktools.ai.runtime._message import decode_model_messages
from linktools.ai.runtime._model_interaction import StagedContextInline, StagedContextSpan
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.runtime.state._steps import StagingAgentRunStore

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ("duplicate_id", "restored_conflict"))
async def test_model_response_evidence_capture_keeps_rejected_tool_response(invalid: str) -> None:
    store = StagingAgentRunStore()
    await store.initialize()
    run = AgentRunRecord("run")
    recorder = AgentRunRecorder(store, execution_id=None, agent_run_id="run")
    await recorder.register_agent_run(run)
    journal = ModelRequestJournal(
        source_namespace="response", tenant_id="tenant", execution_id="execution", agent_run_id="run",
    )
    if invalid == "restored_conflict":
        previous = journal.begin(1)
        await recorder.record_model_event(
            journal.finish(previous.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[
                ToolCallPart("work", {"value": "original"}, tool_call_id="call"),
            ]), include_observation=False,
        )
        recorder = AgentRunRecorder(store, execution_id=None, agent_run_id="run")
        await recorder.register_agent_run(run)
    request = ModelRequest(parts=[UserPromptPart("continue")])
    recorder.append_transcript_message(request)
    fact = journal.begin(2)
    model = TestModel()
    recorder.begin_model_interaction(fact, model, (request,), None, ModelRequestParameters(), False)
    recorder.prepare_model_interaction(fact, model, (request,), None, ModelRequestParameters(), False)
    parts = [ToolCallPart("work", {"value": "changed"}, tool_call_id="call")]
    if invalid == "duplicate_id":
        parts.append(ToolCallPart("work", {"value": "second"}, tool_call_id="call"))
    response = ModelResponse(parts=parts)
    before = recorder.transcript_messages()
    finished = journal.finish(fact.model_request_seq, status="SUCCEEDED")
    with pytest.raises(AIError) as rejected:
        recorder.finish_model_interaction(
            finished, model=model, response=response, status="SUCCEEDED", error_code=None,
            duration_ns=1, usage=response.usage,
        )
    assert rejected.value.code is (
        ErrorCode.STORAGE_INTEGRITY_ERROR if invalid == "duplicate_id" else ErrorCode.STORAGE_CONFLICT
    )
    assert recorder.transcript_messages() == before
    interactions = await store.list_model_interactions(agent_run_id="run")
    assert len(interactions) == 1
    interaction = interactions[0]
    assert interaction.status == "SUCCEEDED"
    assert interaction.response_context is not None
    item, = interaction.response_context.items
    assert isinstance(item, StagedContextInline)
    archived, = decode_model_messages(store.staged_payload("run", item.payload_digest))
    assert archived == response
    assert not any(event.event_type.startswith("TOOL_CALL_") for event in await store.list_events(agent_run_id="run"))
    await store.close()


@pytest.mark.asyncio
async def test_model_response_evidence_capture_binds_event_to_its_original_message() -> None:
    store = StagingAgentRunStore()
    await store.initialize()
    recorder = AgentRunRecorder(store, execution_id=None, agent_run_id="run")
    await recorder.register_agent_run(AgentRunRecord("run"))
    journal = ModelRequestJournal(
        source_namespace="response", tenant_id="tenant", execution_id="execution", agent_run_id="run",
    )
    request = ModelRequest(parts=[UserPromptPart("question")])
    recorder.append_transcript_message(request)
    fact = journal.begin(1)
    model = TestModel()
    recorder.begin_model_interaction(fact, model, (request,), None, ModelRequestParameters(), False)
    recorder.prepare_model_interaction(fact, model, (request,), None, ModelRequestParameters(), False)
    finished = journal.finish(fact.model_request_seq, status="SUCCEEDED")
    response = ModelResponse(parts=[TextPart("answer")])
    recorder.finish_model_interaction(
        finished, model=model, response=response, status="SUCCEEDED", error_code=None,
        duration_ns=1, usage=response.usage,
    )
    interaction, = await store.list_model_interactions(agent_run_id="run")
    item, = interaction.response_context.items
    assert isinstance(item, StagedContextSpan)
    assert recorder.transcript_messages()[item.start:item.end] == (response,)
    later = ModelRequest(parts=[UserPromptPart("next")])
    recorder.append_transcript_message(later)
    await recorder.record_model_event(
        finished, phase="completed", response=response, include_observation=False,
    )
    assert recorder.transcript_messages() == (request, response, later)
    event, = await store.list_events(agent_run_id="run")
    assert event.metadata[MESSAGE_SEQ_METADATA_KEY] == "2"
    await store.close()


@pytest.mark.asyncio
async def test_model_response_evidence_capture_blocks_duplicate_tool_effects() -> None:
    effects: list[str] = []

    async def work(_context: AgentContext[None]) -> str:
        effects.append("effect")
        return "done"

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[
            ToolCallPart("work", {}, tool_call_id="duplicate"),
            ToolCallPart("work", {}, tool_call_id="duplicate"),
        ])

    features = RuntimeFeatures("application")
    features.capability(TraceCapability(), scope="runtime")
    features.tool(work, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=("work",))
    rejected_error = None
    try:
        async with Runtime.open(
            "invalid-response", features=model_features(_Models(_UsageFunctionModel(model)), features),
            storage=RuntimeStorage.in_memory(),
        ) as runtime:
            execution = await runtime.agents.get().start("run work")
            with pytest.raises(AIError) as rejected:
                await execution.wait(timeout_seconds=10)
            rejected_error = rejected.value
            assert rejected_error.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert effects == []
    except AIError as closed:
        assert rejected_error is not None
        assert closed.code is ErrorCode.STORAGE_INTEGRITY_ERROR
