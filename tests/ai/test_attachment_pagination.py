#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attachment acceptance survives request preparation and archive pagination."""

import asyncio
import hashlib
from pathlib import Path

import pytest
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, WrapModelRequestHandler
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.test import TestModel

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionEventType, ExecutionStatus, HmacCursorSigner
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionInputContext, ExecutionTreeEvent, Runtime, RuntimeHistory, RuntimeStorage
from linktools.ai.runtime._attachment import bind_tool_return_attachments
from linktools.ai.runtime._history_projection import _attachment_fact_cursor
from linktools.ai.runtime._metric_capability import ModelObservationCapability

from ._runtime_test_helpers import _UsageFunctionModel, _wait_for_committed
from .test_history_request_association import _history
from .test_live_history_readback_integration import _Models


def _attachment_request(call_id: str) -> ModelRequest:
    body = call_id.encode()
    result = bind_tool_return_attachments(
        "attach_files", call_id, ToolReturn(
            return_value={"files": [{
                "media_type": "image/png",
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }]},
            content=[BinaryContent(body, media_type="image/png")],
        ),
    )
    return ModelRequest(parts=[
        ToolReturnPart("attach_files", result.return_value, tool_call_id=call_id),
        UserPromptPart(result.content),
    ])


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_inclusion", [False, True])
async def test_request_preparation_keeps_acceptance_and_recalculates_inclusion(
    remove_inclusion: bool,
) -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        messages = [_attachment_request("first"), _attachment_request("second")]
        model = TestModel()
        parameters = ModelRequestParameters()
        fact = journal.begin(1)
        recorder.begin_model_interaction(fact, model, messages, None, parameters, False)
        await recorder.commit_history_boundary()
        before = await history.reader.attachment_facts(
            "execution", tenant_id="tenant", cursor=None, limit=100,
        )
        first = await history.reader.attachment_facts(
            "execution", tenant_id="tenant", cursor=None, limit=1,
        )
        assert first.items[0].fact == "accepted"
        assert first.next_cursor is not None

        prepared_messages = messages[:1] if remove_inclusion else messages
        recorder.prepare_model_interaction(
            fact, model, prepared_messages, None, parameters, False,
        )
        await recorder.commit_history_boundary()
        prepared = await history.reader.attachment_facts(
            "execution", tenant_id="tenant", cursor=None, limit=100,
        )
        assert sorted((item.fact, item.call_id) for item in prepared.items) == [
            ("accepted", "first"), ("accepted", "second"),
            ("included_in_request", "first"),
            *([] if remove_inclusion else [("included_in_request", "second")]),
        ]
        assert prepared.items[:2] == before.items
        continuation = await history.reader.attachment_facts(
            "execution", tenant_id="tenant", cursor=first.next_cursor, limit=100,
        )
        assert first.items + continuation.items == prepared.items

        recorder.finish_model_interaction(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"),
            model=model, response=ModelResponse(parts=[TextPart("done")]),
            status="SUCCEEDED", error_code=None, duration_ns=1, usage=None,
        )
        next_fact = journal.begin(2)
        recorder.begin_model_interaction(next_fact, model, messages, None, parameters, False)
        recorder.prepare_model_interaction(next_fact, model, messages, None, parameters, False)
        recorder.finish_model_interaction(
            journal.finish(next_fact.model_request_seq, status="SUCCEEDED"),
            model=model, response=ModelResponse(parts=[TextPart("done again")]),
            status="SUCCEEDED", error_code=None, duration_ns=1, usage=None,
        )
        await history.archive()
        archived_tail = await history.reader.attachment_facts(
            "execution", tenant_id="tenant", cursor=first.next_cursor, limit=100,
        )
        assert first.items + archived_tail.items == prepared.items
        fresh = await history.reader.attachment_facts(
            "execution", tenant_id="tenant", cursor=None, limit=100,
        )
        assert fresh.items[:len(prepared.items)] == prepared.items
        assert sorted((item.fact, item.call_id, item.model_request_seq)
                      for item in fresh.items[len(prepared.items):]) == [
            ("included_in_request", "first", 2), ("included_in_request", "second", 2),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["delete_earlier", "delete_last_read", "reorder"])
async def test_attachment_cursor_preserves_admission_through_prepared_positions(
    change: str,
) -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        messages = [_attachment_request(name) for name in ("first", "second", "third")]
        messages.sort(key=lambda message: message.parts[1].content[0].identifier)
        repeated_content = messages[-1].parts[1].content
        messages[-1].parts.append(UserPromptPart(repeated_content))
        model = TestModel()
        parameters = ModelRequestParameters()
        fact = journal.begin(1)
        recorder.begin_model_interaction(fact, model, messages, None, parameters, False)
        await recorder.commit_history_boundary()
        before = await history.reader.attachment_facts(
            "execution", tenant_id="tenant", cursor=None, limit=100,
        )
        first = await history.reader.attachment_facts(
            "execution", tenant_id="tenant", cursor=None, limit=2,
        )
        assert [item.fact for item in first.items] == ["accepted", "accepted"]
        assert first.next_cursor is not None
        assert len(before.items) == 3
        repeated_attachment = before.items[-1].attachment_id
        prepared_messages = (
            messages[1:] if change == "delete_earlier"
            else [messages[0], messages[2]] if change == "delete_last_read"
            else list(reversed(messages))
        )
        recorder.prepare_model_interaction(fact, model, prepared_messages, None, parameters, False)
        await recorder.commit_history_boundary()
        expected = [1, 2] if change != "reorder" else [0, 1]
        for archived in (False, True):
            if archived:
                recorder.finish_model_interaction(
                    journal.finish(fact.model_request_seq, status="SUCCEEDED"),
                    model=model, response=ModelResponse(parts=[TextPart("done")]),
                    status="SUCCEEDED", error_code=None, duration_ns=1, usage=None,
                )
                await history.archive()
            items = []
            cursor = first.next_cursor
            while cursor is not None:
                page = await history.reader.attachment_facts(
                    "execution", tenant_id="tenant", cursor=cursor, limit=1,
                )
                items.extend(page.items)
                assert len(items) <= 5
                cursor = page.next_cursor
            included = [item for item in items if item.fact == "included_in_request"
                        and item.attachment_id == repeated_attachment]
            assert [item.position for item in included] == expected
            fresh = await history.reader.attachment_facts(
                "execution", tenant_id="tenant", cursor=None, limit=100,
            )
            assert first.items + tuple(items) == fresh.items


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [
    (1, 1, [], "a" * 64, 0),
    (1, 1, {}, "a" * 64, 0),
    (1, 2, "accepted", "a" * 64, 0),
    (0, 1, "accepted", "a" * 64, 0),
    (1, 1, "accepted", "a" * 64, False),
])
async def test_attachment_cursor_rejects_malformed_fact_identity(after: tuple[object, ...]) -> None:
    async with _history() as history:
        cursor = _attachment_fact_cursor(
            "tenant", "execution", after, ((1, 1),),  # type: ignore[arg-type]
            HmacCursorSigner("history", b"history-key"),
        )
        with pytest.raises(AIError) as caught:
            await history.reader.attachment_facts(
                "execution", tenant_id="tenant", cursor=cursor, limit=1,
            )
        assert caught.value.code is ErrorCode.CURSOR_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_attachment", [False, True])
async def test_attachment_cursor_from_request_started_survives_archive_reopen(
    tmp_path: Path, initial_attachment: bool,
) -> None:
    release = asyncio.Event()

    class PausePreparation(AbstractCapability[AgentContext[object]]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position="innermost", wrapped_by=(ModelObservationCapability,))

        async def wrap_model_request(
            self, ctx: RunContext[AgentContext[object]], *, request_context: ModelRequestContext,
            handler: WrapModelRequestHandler,
        ) -> ModelResponse:
            await asyncio.wait_for(release.wait(), 5)
            return await handler(request_context)

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(parts=[TextPart("done")])

    imported = ExecutionInputContext.from_messages([
        ModelRequest(parts=[UserPromptPart("attach")]),
        ModelResponse(parts=[
            ToolCallPart("attach_files", {}, tool_call_id="call"),
            ToolCallPart("attach_files", {}, tool_call_id="second-call"),
        ]),
        _attachment_request("call"),
        _attachment_request("second-call"),
    ])
    group = CapabilityGroup("attachment-pagination")
    group.capability(PausePreparation(), id="pause-preparation")
    group.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    models.vision = True
    first = None
    started_items = ()
    async with Runtime.open(
        "attachment-pagination", models=models, storage=RuntimeStorage.filesystem(tmp_path),
        capabilities=(group,),
    ) as runtime:
        principal = runtime.default_principal
        prompt = ["summarize", BinaryContent(b"initial", media_type="image/png")] if initial_attachment else "summarize"
        execution = await runtime.agents.get().start(prompt, input_context=imported)

        async def observe(tree: ExecutionTreeEvent) -> None:
            nonlocal first, started_items
            if tree.event.event_type != ExecutionEventType.MODEL_REQUEST_STARTED:
                return
            count = 3 if initial_attachment else 2
            started = await _wait_for_committed(
                lambda: runtime.history.attachment_facts(execution.execution_id, principal=principal),
                lambda page: len(page.items) == count,
            )
            started_items = started.items
            assert [item.fact for item in started_items] == ["accepted"] * count
            first = await runtime.history.attachment_facts(execution.execution_id, principal=principal, limit=1)
            assert first.items == started_items[:1]
            assert first.next_cursor is not None
            release.set()

        try:
            result = await execution.wait(on_event=observe, timeout_seconds=10)
        finally:
            release.set()
        assert result.result.status is ExecutionStatus.SUCCEEDED
        assert first is not None
        prepared = await runtime.history.attachment_facts(execution.execution_id, principal=principal)
        assert prepared.items[:len(started_items)] == started_items
        assert len(prepared.items) == 2 * len(started_items)
        items = list(first.items)
        cursor = first.next_cursor
        while cursor is not None:
            tail = await runtime.history.attachment_facts(
                execution.execution_id, principal=principal, cursor=cursor, limit=1,
            )
            items.extend(tail.items)
            assert len(items) <= len(prepared.items)
            cursor = tail.next_cursor
        assert tuple(items) == prepared.items

    async with RuntimeHistory.open(
        "attachment-pagination", storage=RuntimeStorage.filesystem(tmp_path),
    ) as archived:
        items = list(first.items)
        cursor = first.next_cursor
        while cursor is not None:
            tail = await archived.attachment_facts(
                execution.execution_id, principal=principal, cursor=cursor, limit=1,
            )
            items.extend(tail.items)
            assert len(items) <= len(prepared.items)
            cursor = tail.next_cursor
        fresh = await archived.attachment_facts(execution.execution_id, principal=principal)
        assert tuple(items) == fresh.items == prepared.items
