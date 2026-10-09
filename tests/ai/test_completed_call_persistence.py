#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unfinished model bodies stay local until a controlled completion boundary."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime, RuntimeHistory
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime._message import decode_model_messages
from linktools.ai.runtime._model_interaction import StagedContextInline, StagedContextProjection, StagedModelInteraction
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.runtime.state._step_archive import StagingAgentRunStore

from ._runtime_test_helpers import _wait_for_committed
from .test_incremental_projection_boundaries import _storage
from .test_live_history_readback_integration import _Models


@pytest.mark.asyncio
async def test_running_public_metadata_does_not_read_prepared_request_payloads(tmp_path: Path) -> None:
    state = _storage("memory", tmp_path)
    await state.initialize(namespace="metadata-only", tenant_id="tenant")
    try:
        prepared = StagedModelInteraction(
            "run", 1, 1, "agent", None, {"model": "prepared"},
            StagedContextProjection((StagedContextInline("a" * 64, 1),)),
            "b" * 64, None, "RUNNING", None, None, None,
            datetime.now(timezone.utc), None,
        )

        def read_payload(digest: str) -> bytes:
            raise AssertionError(f"running observation read a prepared body: {digest}")

        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        values = await archive.prepare_interactions(AgentRunRecord("run"), (prepared,), read_payload)
        assert len(values) == 1
        assert values[0].request_context is None and values[0].request_envelope is None
        assert values[0].model == prepared.model
        assert prepared.request_context is not None and prepared.request_envelope_digest is not None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_failed_replayed_request_keeps_every_received_raw_response_part() -> None:
    store = StagingAgentRunStore()
    await store.initialize()
    try:
        run = AgentRunRecord("run")
        journal = ModelRequestJournal(
            source_namespace="partial-replay", tenant_id="tenant", execution_id="execution", agent_run_id="run",
        )
        model = TestModel()
        call = ToolCallPart("work", {"value": "same"}, tool_call_id="replayed")
        recorder = AgentRunRecorder(store, execution_id="execution", agent_run_id="run")
        await recorder.register_agent_run(run)
        first = journal.begin(1)
        await recorder.record_model_event(
            journal.finish(first.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[call]), include_observation=False,
        )
        resumed = AgentRunRecorder(store, execution_id="execution", agent_run_id="run")
        await resumed.register_agent_run(run)
        request = ModelRequest(parts=[UserPromptPart("retry")])
        resumed.append_transcript_message(request)
        second = journal.begin(2)
        resumed.begin_model_interaction(second, model, (request,), None, ModelRequestParameters(), True)
        resumed.prepare_model_interaction(second, model, (request,), None, ModelRequestParameters(), True)
        resumed.stage_response_part(call, 0)
        resumed.stage_response_part(TextPart("unfinished response"), 1, complete=False)
        resumed.finish_model_interaction(
            journal.finish(second.model_request_seq, status="FAILED"), model=model,
            response=None, status="FAILED", error_code="MODEL_API_ERROR", duration_ns=1, usage=None,
        )
        interaction, = await store.list_model_interactions(agent_run_id="run")
        assert interaction.model_request_seq == second.model_request_seq
        payload, = interaction.response_context.items
        response, = decode_model_messages(store.staged_payload("run", payload.payload_digest))
        assert response.state == "interrupted"
        assert response.parts == [call, TextPart("unfinished response")]
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("completion", ("success", "provider_error", "cancel", "close"))
async def test_partial_response_is_published_only_at_request_or_shutdown_completion(
    tmp_path: Path, backend: str, completion: str,
) -> None:
    received = asyncio.Event()
    release = asyncio.Event()

    async def stream(messages, info):
        del messages, info
        yield "received "
        yield "partial text"
        received.set()
        await release.wait()
        if completion == "provider_error":
            raise ValueError("provider stopped after partial content")
        yield "; done"

    group = CapabilityGroup("completed-call-persistence")
    group.agent("default", model="default", allow_tools=())
    state = _storage(backend, tmp_path)
    execution_id = None
    principal = None
    try:
        async with Runtime.open(
            "completed-call-persistence", models=_Models(FunctionModel(stream_function=stream)),
            storage=state, capabilities=(group,),
        ) as runtime:
            execution = await runtime.agents.get("default").start("Respond slowly")
            execution_id = execution.execution_id
            principal = runtime.default_principal
            await asyncio.wait_for(received.wait(), 10)
            running = await _wait_for_committed(
                lambda: execution.model_interactions(include_content=True),
                lambda page: bool(page.items),
            )
            assert [item.status for item in running.items] == ["RUNNING"]
            assert running.items[0].response is None
            assert not any(item.item_kind == "assistant" for item in (
                await execution.history(include_content=True)
            ).items)
            async with RuntimeHistory.open(
                "completed-call-persistence", storage=_storage(backend, tmp_path),
            ) as reader:
                observed = await reader.model_interactions(
                    execution_id, principal=principal, include_content=True,
                )
                assert observed.items[0].response is None
                assert not any(item.item_kind == "assistant" for item in (
                    await reader.history(execution_id, principal=principal, include_content=True)
                ).items)
            if completion == "close":
                await runtime.close()
            else:
                if completion == "cancel":
                    await execution.cancel()
                else:
                    release.set()
                result = (await execution.wait(timeout_seconds=10)).result
                assert result.status is {
                    "success": ExecutionStatus.SUCCEEDED,
                    "provider_error": ExecutionStatus.FAILED,
                    "cancel": ExecutionStatus.CANCELLED,
                }[completion]
    finally:
        release.set()

    assert execution_id is not None and principal is not None
    async with RuntimeHistory.open(
        "completed-call-persistence", storage=_storage(backend, tmp_path),
    ) as reader:
        interactions = (await reader.model_interactions(
            execution_id, principal=principal, include_content=True,
        )).items
        assert len(interactions) == 1
        interaction = interactions[0]
        assert interaction.status == {
            "success": "SUCCEEDED", "provider_error": "FAILED",
            "cancel": "CANCELLED", "close": "CANCELLED",
        }[completion]
        assert interaction.finished_at is not None
        assert "received partial text" in json.dumps(interaction.response)
        if completion != "success":
            assert "interrupted" in json.dumps(interaction.response)
            assert "; done" not in json.dumps(interaction.response)
            assert interaction.usage is None
        history = (await reader.history(
            execution_id, principal=principal, include_content=True,
        )).items
        assert [item.content for item in history if item.item_kind == "assistant"] == [
            "received partial text; done" if completion == "success" else "received partial text",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("completion", ("provider_error", "cancel"))
async def test_interrupted_tool_arguments_remain_raw_without_tool_execution(
    tmp_path: Path, backend: str, completion: str,
) -> None:
    received = asyncio.Event()
    release = asyncio.Event()
    executed = []
    arguments = '{"value":"unfinished'

    async def work(_ctx: AgentContext[None], value: str) -> str:
        executed.append(value)
        return value

    async def stream(messages, info):
        del messages, info
        yield {0: DeltaToolCall(name="work", json_args='{"value":"', tool_call_id="call")}
        yield {0: DeltaToolCall(json_args="unfinished")}
        received.set()
        await release.wait()
        raise ValueError("tool arguments interrupted")

    group = CapabilityGroup("partial-tool-arguments")
    group.tool(work, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("work",))
    try:
        async with Runtime.open(
            "partial-tool-arguments", models=_Models(FunctionModel(stream_function=stream)),
            storage=_storage(backend, tmp_path), capabilities=(group,),
        ) as runtime:
            execution = await runtime.agents.get("default").start("Call work")
            execution_id = execution.execution_id
            principal = runtime.default_principal
            await asyncio.wait_for(received.wait(), 10)
            running = await _wait_for_committed(
                lambda: execution.model_interactions(include_content=True),
                lambda page: bool(page.items),
            )
            assert running.items[0].response is None
            assert not any(item.item_kind == "tool_call" for item in (
                await execution.history(include_content=True)
            ).items)
            if completion == "cancel":
                await execution.cancel()
            else:
                release.set()
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is (
                ExecutionStatus.CANCELLED if completion == "cancel" else ExecutionStatus.FAILED
            )
    finally:
        release.set()
    assert not executed
    async with RuntimeHistory.open(
        "partial-tool-arguments", storage=_storage(backend, tmp_path),
    ) as reader:
        interactions = (await reader.model_interactions(
            execution_id, principal=principal, include_content=True,
        )).items
        assert len(interactions) == 1
        response = interactions[0].response
        assert response["state"] == "interrupted"
        assert response["parts"][0]["args"] == arguments
        assert interactions[0].status == ("CANCELLED" if completion == "cancel" else "FAILED")
        history = (await reader.history(
            execution_id, principal=principal, include_content=True,
        )).items
        calls = [item for item in history if item.item_kind == "tool_call"]
        assert len(calls) == 1 and calls[0].tool_call_id == "call"
        assert arguments in json.dumps(calls[0].content).replace('\\"', '"')
        assert not any(item.item_kind == "tool_result" for item in history)
        trace = (await reader.trace(execution_id, principal=principal)).items
        assert all(item.payload["kind"] not in {"TOOL_CALL", "TOOL_RESULT"} for item in trace)
