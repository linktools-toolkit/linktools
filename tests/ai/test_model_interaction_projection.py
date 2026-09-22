#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract tests for model interaction projection primitives."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    InstructionPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime._attachment import (
    bind_tool_return_attachments,
    input_attachment_views,
)
from linktools.ai.runtime._capture import RuntimeCaptureStore
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime._message import decode_model_messages
from linktools.ai.runtime._model_interaction import (
    StagedContextInline,
    StagedContextProjection,
    StagedContextSpan,
    StagedModelInteraction,
    build_context_projection,
    model_identity,
    project_public_messages,
    request_envelope,
)
from linktools.ai.runtime.state._model_interaction_store import (
    ModelInteractionStagingStepStore,
)
from linktools.ai.runtime.state._step_contracts import ContinuableSnapshot, RunRecord
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.workspace import Workspace


class _TextModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    vision = False
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {
        "provider": "test",
        "model": "test",
    }

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _TextModels:
    def snapshot(self) -> "_TextModels":
        return self

    def resolve(self, route_id: str) -> _TextModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _TextModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _TextModelBinding:
        if route_id not in {None, "default"} or dict(payload) != {
            "provider": "test",
            "model": "test",
        }:
            raise AssertionError("unexpected model snapshot")
        return _TextModelBinding()


def _write_default_agent(root: Path) -> None:
    path = root / ".linktools" / "agents" / "default"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        AgentSpecCodec().encode(AgentSpec("default", model="default", allow_tools=()))
    )


def _journal() -> ModelRequestJournal:
    return ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="run",
    )


def _cancelled_interaction(sequence: int = 1) -> StagedModelInteraction:
    return StagedModelInteraction(
        run_id="run",
        step_index=1,
        request_sequence=sequence,
        purpose="agent",
        output_retry_index=None,
        model={"route_id": "default"},
        request_context=StagedContextProjection(()),
        request_envelope_digest="a" * 64,
        response_context=None,
        status="CANCELLED",
        error_code=None,
        duration_ns=0,
        usage=None,
    )


def test_context_projection_retains_spans_and_deduplicates_inline_payloads() -> None:
    source = tuple(
        ModelRequest(parts=[UserPromptPart(value)]) for value in ("one", "two")
    )
    payloads: dict[str, bytes] = {}

    def intern(value: bytes) -> tuple[str, int]:
        import hashlib

        digest = hashlib.sha256(value).hexdigest()
        payloads.setdefault(digest, value)
        return digest, len(value)

    summary = ModelRequest(parts=[UserPromptPart("summary")])
    projection = build_context_projection(source, (source[0], summary, summary), intern)

    assert len(projection.items) == 3
    assert isinstance(projection.items[0], StagedContextSpan)
    assert isinstance(projection.items[1], StagedContextInline)
    assert projection.items[1].payload_digest == next(iter(payloads))
    assert len(payloads) == 1


def test_public_model_messages_replace_binary_body_with_metadata() -> None:
    message = ModelRequest(
        parts=[UserPromptPart([BinaryContent(b"image", media_type="image/png")])]
    )
    projected = project_public_messages((message,))
    assert "data" not in projected[0]["parts"][0]["content"][0]  # type: ignore[index]
    assert projected[0]["parts"][0]["content"][0]["size"] == 5  # type: ignore[index]


def test_request_envelope_keeps_capability_visibility_inputs() -> None:
    envelope, raw = request_envelope(
        model_settings={"temperature": 0.2},
        parameters=ModelRequestParameters(
            deferred_capability_ids={"skill-b", "skill-a"},
            revealed_tool_names={"tool-b", "tool-a"},
            instruction_parts=[InstructionPart(content="system", dynamic=False)],
        ),
        streaming=True,
    )
    assert envelope["version"] == 1
    assert envelope["streaming"] is True
    parameters = envelope["parameters"]
    assert isinstance(parameters, Mapping)
    assert parameters["deferred_capability_ids"] == ["skill-a", "skill-b"]
    assert parameters["revealed_tool_names"] == ["tool-a", "tool-b"]
    instruction_parts = parameters["instruction_parts"]
    assert isinstance(instruction_parts, list) and instruction_parts
    assert isinstance(instruction_parts[0], Mapping)
    assert instruction_parts[0]["content"] == "system"
    assert b'"version":1' in raw


def test_model_identity_preserves_selection_route() -> None:
    identity = model_identity(TestModel(), route_id="tenant-alias")
    assert identity["route_id"] == "tenant-alias"


def test_cancelled_model_interaction_has_no_synthetic_error() -> None:
    interaction = _cancelled_interaction()
    assert interaction.status == "CANCELLED"
    assert interaction.error_code is None


@pytest.mark.asyncio
async def test_staging_interaction_identity_is_idempotent() -> None:
    store = ModelInteractionStagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    interaction = _cancelled_interaction()
    store.stage_model_interaction(interaction)
    store.stage_model_interaction(interaction)
    assert await store.list_model_interactions(run_id="run") == [interaction]
    with pytest.raises(AIError) as raised:
        store.stage_model_interaction(
            replace(
                interaction,
                status="FAILED",
                error_code=ErrorCode.MODEL_API_ERROR.value,
            )
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_model_request_records_attach_files_call_identity() -> None:
    import hashlib

    body = b"image"
    digest = hashlib.sha256(body).hexdigest()
    store = ModelInteractionStagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    capture = RuntimeCaptureStore(store, execution_id="execution", step_run_id="run")
    return_value = {
        "files": [
            {
                "path": "evidence.png",
                "media_type": "image/png",
                "size": len(body),
                "sha256": digest,
            }
        ]
    }
    result = bind_tool_return_attachments(
        "attach_files",
        "call-1",
        ToolReturn(
            return_value=return_value,
            content=[BinaryContent(body, media_type="image/png")],
        ),
    )
    assert isinstance(result, ToolReturn)
    assert result.content is not None and not isinstance(result.content, str)
    message = ModelRequest(
        parts=[
            ToolReturnPart(
                "attach_files",
                return_value,
                tool_call_id="call-1",
            ),
            UserPromptPart(result.content),
        ]
    )
    journal = _journal()
    fact = journal.begin(1)
    capture.begin_model_interaction(
        fact,
        TestModel(),
        (message,),
        None,
        ModelRequestParameters(),
        False,
    )
    finished = journal.finish(fact.request_sequence, status="SUCCEEDED")
    capture.finish_model_interaction(
        finished,
        model=TestModel(),
        response=ModelResponse(parts=[TextPart("done")]),
        status="SUCCEEDED",
        error_code=None,
        duration_ns=1,
        usage=None,
    )

    interaction = (await store.list_model_interactions(run_id="run"))[0]
    assert [value["fact"] for value in interaction.attachments] == [
        "accepted",
        "included_in_request",
    ]
    assert interaction.attachments[0]["attachment_id"] == (
        interaction.attachments[1]["attachment_id"]
    )
    assert all(value["call_id"] == "call-1" for value in interaction.attachments)
    assert all(
        value["input_identifier"] is None for value in interaction.attachments
    )
    assert interaction.attachments[1]["digest"] == digest


@pytest.mark.asyncio
async def test_model_request_preserves_duplicate_initial_attachment_identity() -> None:
    first = BinaryContent(
        b"same",
        media_type="image/png",
        identifier="input-a",
    )
    second = BinaryContent(
        b"same",
        media_type="image/png",
        identifier="input-b",
    )
    accepted = input_attachment_views((first, second))
    store = ModelInteractionStagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    capture = RuntimeCaptureStore(
        store,
        execution_id="execution",
        step_run_id="run",
        initial_attachments=accepted,
    )
    message = ModelRequest(parts=[UserPromptPart([first, second])])
    journal = _journal()
    fact = journal.begin(1)
    capture.begin_model_interaction(
        fact,
        TestModel(),
        (message,),
        None,
        ModelRequestParameters(),
        False,
    )
    finished = journal.finish(fact.request_sequence, status="SUCCEEDED")
    capture.finish_model_interaction(
        finished,
        model=TestModel(),
        response=ModelResponse(parts=[TextPart("done")]),
        status="SUCCEEDED",
        error_code=None,
        duration_ns=1,
        usage=None,
    )

    interaction = (await store.list_model_interactions(run_id="run"))[0]
    included = interaction.attachments
    assert len(included) == 2
    assert all(value["fact"] == "included_in_request" for value in included)
    assert [value["attachment_id"] for value in included] == [
        accepted[0]["attachment_id"],
        accepted[1]["attachment_id"],
    ]
    assert included[0]["attachment_id"] != included[1]["attachment_id"]
    assert [value["input_identifier"] for value in accepted] == [
        "input-a",
        "input-b",
    ]
    assert [value["input_identifier"] for value in included] == [
        "input-a",
        "input-b",
    ]


@pytest.mark.asyncio
async def test_success_request_does_not_stage_full_message_payloads() -> None:
    store = ModelInteractionStagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    capture = RuntimeCaptureStore(store, execution_id="execution", step_run_id="run")
    message = ModelRequest(parts=[UserPromptPart("hello")])
    journal = _journal()
    fact = journal.begin(1)
    capture.append_transcript_message(message)
    capture.begin_model_interaction(
        fact, TestModel(), (message,), None, ModelRequestParameters(), False, "alias"
    )
    assert len(store._payloads["run"]) == 1
    finished = journal.finish(fact.request_sequence, status="SUCCEEDED")
    capture.finish_model_interaction(
        finished,
        model=TestModel(),
        response=ModelResponse(parts=[TextPart("done")]),
        status="SUCCEEDED",
        error_code=None,
        duration_ns=1,
        usage=None,
    )
    staged = await store.list_model_interactions(run_id="run")
    assert len(staged) == 1
    assert all(
        isinstance(item, StagedContextSpan)
        for item in staged[0].request_context.items  # type: ignore[union-attr]
    )
    assert staged[0].response_context is not None  # type: ignore[union-attr]
    assert all(
        isinstance(item, StagedContextInline)
        for item in staged[0].response_context.items  # type: ignore[union-attr]
    )
    assert len(store._payloads["run"]) == 2


@pytest.mark.asyncio
async def test_failed_request_inlines_context_only_after_failure() -> None:
    store = ModelInteractionStagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    capture = RuntimeCaptureStore(store, execution_id="execution", step_run_id="run")
    message = ModelRequest(parts=[UserPromptPart("hello")])
    journal = _journal()
    fact = journal.begin(1)
    capture.append_transcript_message(message)
    capture.begin_model_interaction(
        fact, TestModel(), (message,), None, ModelRequestParameters(), False
    )
    assert len(store._payloads["run"]) == 1
    finished = journal.finish(fact.request_sequence, status="FAILED")
    capture.finish_model_interaction(
        finished,
        model=TestModel(),
        response=None,
        status="FAILED",
        error_code=ErrorCode.MODEL_API_ERROR.value,
        duration_ns=1,
        usage=None,
    )
    staged = await store.list_model_interactions(run_id="run")
    request = staged[0].request_context  # type: ignore[union-attr]
    assert all(isinstance(item, StagedContextSpan) for item in request.items)
    assert len(store._payloads["run"]) == 1


@pytest.mark.asyncio
async def test_interaction_capture_is_immutable_after_sdk_object_mutation() -> None:
    store = ModelInteractionStagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    capture = RuntimeCaptureStore(store, execution_id="execution", step_run_id="run")
    business = {"timestamp": "before", "nested": {"value": 1}}
    request = ModelRequest(
        parts=[
            UserPromptPart("hello"),
            ToolReturnPart("tool", business, tool_call_id="call-1"),
        ],
        instructions="before",
    )
    capture.append_transcript_message(request)
    journal = _journal()
    fact = journal.begin(1)
    capture.begin_model_interaction(
        fact,
        TestModel(),
        (request,),
        None,
        ModelRequestParameters(),
        False,
    )

    request.instructions = "after"
    business["timestamp"] = "after"
    business["nested"]["value"] = 2  # type: ignore[index]
    request.parts = [UserPromptPart("mutated")]
    response = ModelResponse(parts=[TextPart("done")])
    finished = journal.finish(fact.request_sequence, status="SUCCEEDED")
    capture.finish_model_interaction(
        finished,
        model=TestModel(),
        response=response,
        status="SUCCEEDED",
        error_code=None,
        duration_ns=1,
        usage=None,
    )
    response.parts = [TextPart("mutated response")]

    frozen_request = capture.transcript_messages()[0]
    assert isinstance(frozen_request, ModelRequest)
    assert frozen_request.instructions == "before"
    assert frozen_request.parts[0].content == "hello"  # type: ignore[attr-defined]
    frozen_tool = frozen_request.parts[1]
    assert isinstance(frozen_tool, ToolReturnPart)
    assert frozen_tool.content == {
        "timestamp": "before",
        "nested": {"value": 1},
    }

    interaction = (await store.list_model_interactions(run_id="run"))[0]
    assert interaction.response_context is not None  # type: ignore[union-attr]
    response_item = interaction.response_context.items[0]  # type: ignore[union-attr]
    assert isinstance(response_item, StagedContextInline)
    frozen_response = decode_model_messages(
        store.staged_payload("run", response_item.payload_digest)
    )
    assert frozen_response[0].parts[0].content == "done"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_parent_tool_result_round_trip_materializes_two_model_requests() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="interaction-tool-roundtrip", tenant_id="tenant")
    try:
        capture = RuntimeCaptureStore(
            state.steps,
            execution_id="execution",
            step_run_id="run",
        )
        await capture.register_run(
            RunRecord(
                "run",
                conversation_id="conversation",
                agent_name="parent",
            )
        )
        journal = ModelRequestJournal(
            source_namespace="interaction-tool-roundtrip",
            tenant_id="tenant",
            execution_id="execution",
            step_run_id="run",
        )

        first_request = ModelRequest(
            parts=[UserPromptPart("start")],
            conversation_id="conversation",
        )
        capture.append_transcript_message(first_request)
        first_fact = journal.begin(1)
        capture.begin_model_interaction(
            first_fact,
            TestModel(),
            (first_request,),
            None,
            ModelRequestParameters(),
            False,
        )
        first_response = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="delegate_task",
                    args={"subagent_id": "child", "task": "work"},
                    tool_call_id="call-1",
                )
            ],
            conversation_id="conversation",
        )
        first_finished = journal.finish(
            first_fact.request_sequence,
            status="SUCCEEDED",
        )
        capture.finish_model_interaction(
            first_finished,
            model=TestModel(),
            response=first_response,
            status="SUCCEEDED",
            error_code=None,
            duration_ns=1,
            usage=None,
        )
        capture.append_transcript_message(first_response)

        tool_result = ModelRequest(
            parts=[
                ToolReturnPart(
                    "delegate_task",
                    {"status": "ok", "result": "child done"},
                    tool_call_id="call-1",
                )
            ],
            conversation_id="conversation",
        )
        capture.append_transcript_message(tool_result)
        second_fact = journal.begin(2)
        capture.begin_model_interaction(
            second_fact,
            TestModel(),
            capture.transcript_messages(),
            None,
            ModelRequestParameters(),
            False,
        )
        second_response = ModelResponse(
            parts=[TextPart("final")],
            conversation_id="conversation",
        )
        second_finished = journal.finish(
            second_fact.request_sequence,
            status="SUCCEEDED",
        )
        capture.finish_model_interaction(
            second_finished,
            model=TestModel(),
            response=second_response,
            status="SUCCEEDED",
            error_code=None,
            duration_ns=1,
            usage=None,
        )
        capture.append_transcript_message(second_response)
        await capture.save_snapshot(
            ContinuableSnapshot(
                run_id="run",
                step_index=2,
                messages=list(capture.transcript_messages()),
                conversation_id="conversation",
                agent_name="parent",
                state="complete",
                transcript_message_count_before=0,
            )
        )

        await state.steps.materialize_recovery_snapshot(
            step_run_id="run",
            require_complete=True,
        )
        archive = state.steps.read_store(RuntimeDomain.RECOVERY)
        interactions = await archive.list_model_interactions(run_id="run")
        assert [value.request_sequence for value in interactions] == [1, 2]
        resolved = await archive.resolve_model_interactions(interactions)
        second_request, second_resolved_response, _envelope = resolved[1]
        assert len(second_request) == 3
        assert isinstance(second_request[-1], ModelRequest)
        part = second_request[-1].parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.tool_name == "delegate_task"
        assert part.tool_call_id == "call-1"
        assert part.content == {"status": "ok", "result": "child done"}
        assert second_resolved_response is not None
        assert second_resolved_response[0].parts[0].content == "final"  # type: ignore[attr-defined]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_compaction_request_uses_explicit_source_not_stale_projection() -> None:
    store = ModelInteractionStagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    capture = RuntimeCaptureStore(store, execution_id="execution", step_run_id="run")
    stale_source = (ModelRequest(parts=[UserPromptPart("stale")]),)
    capture.remember_context_projection(
        stale_source,
        (ModelRequest(parts=[UserPromptPart("stale projected")]),),
    )
    source = (
        ModelRequest(parts=[UserPromptPart("keep")]),
        ModelRequest(parts=[UserPromptPart("replace")]),
    )
    synthetic = ModelRequest(parts=[UserPromptPart("summarize")])
    capture.append_transcript_message(source[0])
    journal = _journal()
    fact = journal.begin(2, purpose="compaction")
    capture.begin_model_interaction(
        fact,
        TestModel(),
        (source[0], synthetic),
        None,
        ModelRequestParameters(),
        False,
        source_messages=source,
    )
    finished = journal.finish(fact.request_sequence, status="SUCCEEDED")
    capture.finish_model_interaction(
        finished,
        model=TestModel(),
        response=ModelResponse(parts=[TextPart("summary")]),
        status="SUCCEEDED",
        error_code=None,
        duration_ns=1,
        usage=None,
    )
    interaction = (await store.list_model_interactions(run_id="run"))[0]
    request = interaction.request_context  # type: ignore[union-attr]
    assert isinstance(request.items[0], StagedContextSpan)
    assert request.items[0] == StagedContextSpan(0, 1)
    assert isinstance(request.items[1], StagedContextInline)


async def _assert_public_interaction(runtime: Runtime[object]) -> None:
    execution = await runtime.agent("default").start("hello")
    result = await execution.wait()
    page = await execution.model_interactions(include_content=True)
    assert result.status == "SUCCEEDED"
    assert len(page.items) == 1
    assert page.items[0].purpose == "agent"
    assert page.items[0].status == "SUCCEEDED"
    assert page.items[0].request["messages"]
    assert page.items[0].response is not None


@pytest.mark.asyncio
async def test_execution_model_interactions_are_durable_and_public(tmp_path: Path) -> None:
    _write_default_agent(tmp_path)
    workspace = Workspace.load(tmp_path)
    async with Runtime.open(
        "default",
        models=_TextModels(),  # type: ignore[arg-type]
        state=RuntimeState.in_memory(),
        capabilities=(CapabilityGroup("workspace", workspace=workspace),),
        metrics=Metrics.in_memory(),
    ) as runtime:
        await _assert_public_interaction(runtime)


@pytest.mark.asyncio
async def test_execution_model_interactions_support_volatile_memory_state(
    tmp_path: Path,
) -> None:
    _write_default_agent(tmp_path)
    workspace = Workspace.load(tmp_path)
    async with Runtime.open(
        "default",
        models=_TextModels(),  # type: ignore[arg-type]
        state=RuntimeState.in_memory(),
        capabilities=(CapabilityGroup("workspace", workspace=workspace),),
        metrics=Metrics.in_memory(),
    ) as runtime:
        await _assert_public_interaction(runtime)
