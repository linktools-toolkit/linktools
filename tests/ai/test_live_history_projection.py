#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-local transcript identities and terminal boundary facts."""

from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)

from linktools.ai.core import (
    ExecutionStatus,
    HmacCursorSigner,
    agent_conversation_id,
    agent_run_id,
)
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._event import project_event_payload
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
)

from .test_history_projection_conformance import _record


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "denied", "retry", "cancelled"])
async def test_same_call_id_is_scoped_to_run_and_has_no_invented_result(
    outcome: str,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(
            _record(ExecutionStatus.STARTED, 2)
        )
        reader = StepExecutionHistoryReader(
            namespace="history",
            executions=state.execution.executions,
            store=state.run_store.read_store(RuntimeDomain.EXECUTION),
            staging_store=state.run_store,
            cursor_signer=HmacCursorSigner("history", b"history-key"),
        )
        conversation = agent_conversation_id(
            namespace="history", tenant_id="tenant", execution_id="execution"
        )
        recorders = []
        for sequence in (1, 2):
            run_id = agent_run_id(
                namespace="history",
                tenant_id="tenant",
                execution_id="execution",
                agent_run_seq=sequence,
            )
            recorder = AgentRunRecorder(
                state.run_store, execution_id="execution", agent_run_id=run_id
            )
            await recorder.register_agent_run(
                AgentRunRecord(
                    run_id,
                    agent_conversation_id=conversation,
                    agent_id="default",
                    metadata={
                        "agent_run_seq": str(sequence),
                        "agent_id": "default",
                    },
                    started_at=datetime.now(timezone.utc),
                )
            )
            call = ToolCallPart("echo", {"run": sequence}, tool_call_id="same-call")
            recorder.append_transcript_message(ModelResponse(parts=[call]))
            await recorder.record_tool_start(call, 1)
            if outcome == "cancelled":
                await recorder.record_event(
                    "TOOL_CALL_FAILED",
                    1,
                    tool_call_id="same-call",
                    tool_name="echo",
                    error="cancelled",
                )
            else:
                result = (
                    RetryPromptPart(
                        "retry this", tool_name="echo", tool_call_id="same-call"
                    )
                    if outcome == "retry"
                    else ToolReturnPart(
                        "echo", None, tool_call_id="same-call", outcome=outcome
                    )
                )
                recorder.stage_tool_result(result)
                await recorder.record_tool_result_boundary(result, 1)
            recorders.append(recorder)

        for sequence in (1, 2):
            page = await reader.history(
                "execution",
                tenant_id="tenant",
                cursor=None,
                limit=10,
                agent_run_seq=sequence,
                tool_call_id="same-call",
            )
            assert page.items[0].content == {"run": sequence}
            assert {item.agent_run_seq for item in page.items} == {sequence}
            assert {item.status for item in page.items} == (
                {"SUCCEEDED"} if outcome == "success" else {"FAILED"}
            )
            assert len(page.items) == (1 if outcome == "cancelled" else 2)
            if outcome != "cancelled":
                assert page.items[1].item_kind == (
                    "retry" if outcome == "retry" else "tool_result"
                )
                assert page.items[1].content == (
                    "retry this" if outcome == "retry" else None
                )
                assert page.items[1].content_included is True

        missing = await reader.history(
            "execution",
            tenant_id="tenant",
            cursor=None,
            limit=10,
            agent_run_seq=2,
            tool_call_id="missing",
        )
        assert missing.items == ()
        for recorder in recorders:
            recorder.finish_transcript()
            await recorder.save_checkpoint(
                AgentRunCheckpoint(
                    recorder.agent_run_id,
                    1,
                    list(recorder.transcript_messages()),
                    transcript_message_count_before=0,
                )
            )
            await state.run_store.flush_execution_projection(
                recorder.agent_run_id, execution_id="execution"
            )
        await state.run_store.release_staging_many(
            candidate_agent_run_ids=tuple(r.agent_run_id for r in recorders),
            execution_id="execution",
        )
        archived = await reader.history(
            "execution",
            tenant_id="tenant",
            cursor=None,
            limit=10,
            agent_run_seq=2,
            tool_call_id="same-call",
        )
        assert archived.items[0].content == {"run": 2}
        assert len(archived.items) == (1 if outcome == "cancelled" else 2)
    finally:
        await state.close()


@pytest.mark.parametrize(
    "event_type, metadata",
    [
        (
            "TOOL_CALL_STARTED",
            {
                "agent_run_seq": 1,
                "call_id": "call",
                "tool_name": "echo",
                "arguments_digest": "digest",
            },
        ),
        (
            "TOOL_CALL_FINISHED",
            {"agent_run_seq": 1, "call_id": "call", "status": "FAILED"},
        ),
        (
            "ASSISTANT_PART_COMPLETED",
            {
                "agent_run_seq": 1,
                "message_seq": 2,
                "part_index": 0,
                "part_type": "text",
            },
        ),
    ],
)
def test_default_boundary_metadata_keeps_locators_but_omits_unrequested_body(
    event_type: str, metadata: dict
) -> None:
    payload = {
        **metadata,
        "arguments": {"private": "body"},
        "content": "body",
        "result": "body",
    }
    assert project_event_payload(event_type, payload, include_content=False) == metadata
    assert project_event_payload(event_type, payload, include_content=True) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, count", [("history", 2), ("transcript", 2), ("trace", 1)]
)
async def test_read_keeps_run_visible_across_staging_archive_handoff(
    method: str, count: int
) -> None:
    import asyncio
    from .test_subagent_history_projection import (
        _record as make_record,
        _Store,
        _Executions,
    )

    entered = asyncio.Event()
    release = asyncio.Event()

    class Archive(_Store):
        published = False

        async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
            if not self.published:
                entered.set()
                await release.wait()
                self.published = True
                return None
            return await super().get_agent_run(agent_run_id=agent_run_id)

    class Staging:
        async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
            if archive.published:
                return None
            return await _Store.get_agent_run(archive, agent_run_id=agent_run_id)

        def staged_transcript(self, agent_run_id: str) -> None:
            return None

        async def list_events(self, *, agent_run_id: str) -> list:
            return []

    record = make_record("root", created_at=datetime.now(timezone.utc))
    archive = Archive((record,))
    reader = StepExecutionHistoryReader(
        namespace="history",
        executions=_Executions((record,)),
        store=archive,
        cursor_signer=HmacCursorSigner("history", b"key"),
        staging_store=Staging(),
    )
    read = asyncio.create_task(
        getattr(reader, method)("root", tenant_id="tenant", cursor=None, limit=100)
    )
    await asyncio.wait_for(entered.wait(), 5)
    release.set()
    first = await read
    second = await getattr(reader, method)(
        "root", tenant_id="tenant", cursor=None, limit=100
    )
    assert len(first.items) == len(second.items) == count


@pytest.mark.asyncio
async def test_interrupted_response_preserves_confirmed_native_and_text_part_positions() -> (
    None
):
    import asyncio
    from contextlib import asynccontextmanager
    from pydantic_ai.messages import BinaryContent, FilePart, TextPart
    from pydantic_ai.models import CompletedStreamedResponse
    from pydantic_ai.models.function import FunctionModel
    from linktools.ai.capability import CapabilityGroup
    from linktools.ai.core import ExecutionEventType
    from linktools.ai.runtime import Runtime
    from .test_live_history_readback_integration import _Models

    release = asyncio.Event()

    class Partial(CompletedStreamedResponse):
        async def _get_event_iterator(self):
            for part in self.response.parts:
                yield self._parts_manager.handle_part(vendor_part_id=None, part=part)
            await release.wait()
            raise RuntimeError("transport interrupted")

    class Model(FunctionModel):
        @asynccontextmanager
        async def request_stream(
            self, messages, model_settings, model_request_parameters, run_context=None
        ):
            yield Partial(
                ModelResponse(
                    parts=[
                        FilePart(BinaryContent(data=b"image", media_type="image/png")),
                        TextPart("confirmed"),
                        TextPart("incomplete"),
                    ]
                ),
                model_request_parameters=model_request_parameters,
                replay_events=True,
            )

    async def unused(messages, info):
        return ModelResponse(parts=[TextPart("unused")])

    group = CapabilityGroup("partial")
    group.agent("default", model="default")
    async with Runtime.open(
        "partial",
        models=_Models(Model(unused)),
        storage=RuntimeStorage.in_memory(),
        capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get("default").start("prompt")
        selector = None
        first = None
        expected = None

        async def observe(tree):
            nonlocal selector, first, expected
            if tree.event.event_type != ExecutionEventType.ASSISTANT_PART_COMPLETED:
                return
            payload = tree.event.payload
            selector = {
                key: payload[key]
                for key in ("agent_run_seq", "message_seq", "part_index")
            }
            assert selector["part_index"] == 1
            exact = await execution.history(include_content=True, **selector)
            assert [item.content for item in exact.items] == ["confirmed"]
            expected = (await execution.history(include_content=True)).items
            first = await execution.history(include_content=True, limit=1)
            release.set()

        result = await execution.wait(on_event=observe, timeout_seconds=5)
        assert result.result.status is ExecutionStatus.FAILED
        assert selector is not None and first is not None and expected is not None
        exact = await execution.history(include_content=True, **selector)
        assert [item.content for item in exact.items] == ["confirmed"]
        seen = list(first.items)
        cursor = first.next_cursor
        while cursor is not None:
            page = await execution.history(include_content=True, limit=1, cursor=cursor)
            seen.extend(page.items)
            cursor = page.next_cursor
        assert [(item.item_kind, item.part_index, item.content) for item in seen] == [
            (item.item_kind, item.part_index, item.content) for item in expected
        ]
        assert not any(item.content == "incomplete" for item in seen)


@pytest.mark.asyncio
async def test_attachment_inclusion_is_readable_from_running_interaction_without_body_resolution() -> (
    None
):
    from dataclasses import replace
    from types import SimpleNamespace
    from linktools.ai.runtime.state._contracts import StoredUserInput
    from linktools.ai.storage import StoredPayload
    from .test_attachment_facts import _Executions, _fact
    from .test_model_interaction_lifecycle_paging import (
        _HistoryStore,
        _StagingStore,
        _running_interaction,
    )

    interaction = _running_interaction("execution", 1, 1, datetime.now(timezone.utc))
    interaction = replace(
        interaction,
        attachments=(
            _fact(
                fact="included_in_request",
                attachment_id="a" * 64,
                digest="b" * 64,
                position=0,
            ),
        ),
    )
    run = AgentRunRecord(
        interaction.agent_run_id,
        agent_conversation_id=agent_conversation_id(
            namespace="history", tenant_id="tenant", execution_id="execution"
        ),
        metadata={"agent_run_seq": "1", "agent_id": "default"},
    )
    staging = _StagingStore()
    staging.runs[run.agent_run_id] = run
    staging.staged[run.agent_run_id] = {1: interaction}
    record = SimpleNamespace(
        execution_id="execution",
        status=ExecutionStatus.STARTED,
        binding_kind="agent",
        agent_run_seq=1,
        stored_user_input=StoredUserInput(
            "user-content-v1",
            StoredPayload.inline_json({"items": []}),
            {"version": 1, "attachments": []},
        ),
    )
    reader = StepExecutionHistoryReader(
        namespace="history",
        executions=_Executions(record),
        store=_HistoryStore(),
        staging_store=staging,
        cursor_signer=HmacCursorSigner("history", b"key"),
    )
    page = await reader.attachment_facts(
        "execution", tenant_id="tenant", cursor=None, limit=10
    )
    assert [
        (item.fact, item.attachment_id, item.model_request_seq) for item in page.items
    ] == [("included_in_request", "a" * 64, 1)]


@pytest.mark.asyncio
async def test_resumed_tool_reuses_started_fact_and_restored_arguments() -> None:
    from linktools.ai.runtime.state._step_archive import StagingAgentRunStore

    store = StagingAgentRunStore()
    run = AgentRunRecord("run")
    original = AgentRunRecorder(store, execution_id="execution", agent_run_id="run")
    await original.register_agent_run(run)
    call = ToolCallPart("echo", {"value": "original"}, tool_call_id="call")
    original.append_transcript_message(ModelResponse(parts=[call]))
    await original.record_tool_start(call, 1)
    await original.save_checkpoint(
        AgentRunCheckpoint(
            "run",
            1,
            list(original.transcript_messages()),
            transcript_message_count_before=0,
        )
    )
    resumed = AgentRunRecorder(store, execution_id="execution", agent_run_id="run")
    await resumed.register_agent_run(run)
    await resumed.record_tool_start(call, 1)
    result = ToolReturnPart("echo", "done", tool_call_id="call")
    resumed.stage_tool_result(result)
    await resumed.record_tool_result_boundary(result, 1)
    events = await store.list_events(agent_run_id="run")
    assert [event.event_type for event in events] == [
        "TOOL_CALL_STARTED",
        "TOOL_CALL_SUCCEEDED",
    ]
    transcript = store.staged_transcript("run")
    assert transcript.messages[0].parts[0].args_as_dict() == {"value": "original"}
    assert transcript.pending.parts[0].content == "done"
