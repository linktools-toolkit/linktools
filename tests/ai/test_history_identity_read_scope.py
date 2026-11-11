#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""History selectors and page lookahead bound associated metadata reads."""

from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain

from .test_history_request_association import _history


async def _append_calls(history, request: int, count: int) -> None:
    recorder, = history.recorders
    journal, = history.journals
    fact = journal.begin(request)
    assert fact.model_request_seq == request
    await recorder.record_model_event(fact, phase="started", include_observation=False)
    await recorder.record_model_event(
        journal.finish(request, status="SUCCEEDED"), phase="completed",
        response=ModelResponse(parts=[
            ToolCallPart("echo", {"index": index}, tool_call_id=f"call-{request}-{index}")
            for index in range(count)
        ]), include_observation=False,
    )
    results = []
    for index in range(count):
        result = ToolReturnPart("echo", index, tool_call_id=f"call-{request}-{index}")
        await recorder.record_tool_result_boundary(result, request + 1)
        results.append(result)
    recorder.append_transcript_message(ModelRequest(parts=results))
    await recorder.commit_history_boundary()


@pytest.mark.asyncio
async def test_history_identity_filters_do_not_resolve_unmatched_message_associations(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _history() as history:
        await _append_calls(history, 1, 9)
        await _append_calls(history, 2, 9)
        await history.archive()
        full = await history.reader.history("execution", tenant_id="tenant", cursor=None, limit=100)
        archive = history.state.run_store.read_store(PersistenceDomain.TRACE)
        original = archive.read_history_associations
        messages, calls = set(), set()

        async def capture(*, agent_run_id, message_seqs, tool_call_ids, event_high_water):
            messages.update(message_seqs)
            calls.update(tool_call_ids)
            return await original(agent_run_id=agent_run_id, message_seqs=message_seqs,
                                  tool_call_ids=tool_call_ids, event_high_water=event_high_water)

        monkeypatch.setattr(archive, "read_history_associations", capture)
        selected = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100, tool_call_id="call-2-8",
        )
        expected = tuple(item for item in full.items if item.tool_call_id == "call-2-8")
        assert selected.items == expected
        assert calls == {"call-2-8"}
        assert messages <= {item.message_seq for item in expected}
        messages.clear()
        calls.clear()
        missing = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=1, tool_call_id="missing",
        )
        assert missing.items == ()
        assert not messages and not calls


@pytest.mark.asyncio
async def test_history_small_page_does_not_read_a_later_corrupt_tool() -> None:
    async with _history() as history:
        await _append_calls(history, 1, 9)
        await history.archive()
        archive = history.state.run_store.read_store(PersistenceDomain.TRACE)
        recorder, = history.recorders
        key = archive._history_association_key(recorder.agent_run_id, "TOOL_CALL_STARTED", "call-1-8")

        async def corrupt(transaction):
            record = await transaction.get_record(key)
            assert record is not None
            await transaction.delete_record(key)
            await transaction.insert_record(replace(record, sort_key=b"invalid"))

        await archive.state_store.mutate(corrupt)
        page = await history.reader.history("execution", tenant_id="tenant", cursor=None, limit=1)
        assert [item.tool_call_id for item in page.items] == ["call-1-0"]
        assert page.next_cursor is not None
        selected = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100, tool_call_id="call-1-0",
        )
        assert [item.item_kind for item in selected.items] == ["tool_call", "tool_result"]
        with pytest.raises(AIError) as caught:
            await history.reader.history("execution", tenant_id="tenant", cursor=page.next_cursor, limit=100)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_history_request_filters_bound_tool_operation_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _history() as history:
        await _append_calls(history, 1, 9)
        await _append_calls(history, 2, 9)
        await history.archive()
        repository = history.state.checkpoint.tools
        history.reader._tool_operations = repository
        original = repository.get_by_call_ids
        calls = set()

        async def capture(agent_run_id, tool_call_ids, *, tenant_id):
            calls.update(tool_call_ids)
            return await original(agent_run_id, tool_call_ids, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "get_by_call_ids", capture)
        for selector in ({"model_request_seq": 2}, {"step_index": 2}):
            calls.clear()
            selected = await history.reader.history(
                "execution", tenant_id="tenant", cursor=None, limit=100, **selector,
            )
            assert len(selected.items) == 18
            assert calls == {f"call-2-{index}" for index in range(9)}


@pytest.mark.asyncio
async def test_history_empty_response_keeps_association_validation() -> None:
    async with _history() as history:
        recorder, = history.recorders
        recorder.append_transcript_message(ModelResponse(parts=[]))
        await recorder.record_event("MODEL_REQUEST_SUCCEEDED", 1, metadata={
            "linktools.ai.model_request_seq": "1", "linktools.ai.message_seq": "1",
            "linktools.ai.replayed_tool_call_indices": "[0]",
        })
        await history.archive()
        with pytest.raises(AIError) as caught:
            await history.reader.history("execution", tenant_id="tenant", cursor=None, limit=1)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", (
    {"tool_call_id": "call-2-8"}, {"tool_call_id": "missing"}, {"part_index": 99},
))
async def test_history_explicit_identity_excludes_unrelated_message_corruption(selector: dict[str, str | int]) -> None:
    async with _history() as history:
        await _append_calls(history, 1, 9)
        await _append_calls(history, 2, 9)
        await history.archive()
        expected = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100, **selector,
        )
        archive = history.state.run_store.read_store(PersistenceDomain.TRACE)
        recorder, = history.recorders
        key = archive._history_association_key(recorder.agent_run_id, "response", "1")

        async def corrupt(transaction):
            record = await transaction.get_record(key)
            assert record is not None
            await transaction.delete_record(key)
            await transaction.insert_record(replace(record, sort_key=b"invalid"))

        await archive.state_store.mutate(corrupt)
        selected = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100, **selector,
        )
        assert selected.items == expected.items
        with pytest.raises(AIError) as caught:
            await history.reader.history("execution", tenant_id="tenant", cursor=None, limit=100)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_history_inline_events_still_validate_when_identity_has_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _history() as history:
        await _append_calls(history, 1, 1)
        await history.archive()
        archive = history.state.run_store.read_store(PersistenceDomain.TRACE)
        original = archive.capture_history

        async def capture(agent_run_ids, *, include_pending=False):
            captures = dict(await original(agent_run_ids, include_pending=include_pending))
            for run_id, value in captures.items():
                events = await archive.list_event_range(agent_run_id=run_id, start=0, end=value.event_count)
                events[0] = replace(events[0], metadata={"linktools.ai.model_request_seq": "invalid"})
                captures[run_id] = replace(value, inline_events=tuple(events))
            return captures

        monkeypatch.setattr(archive, "capture_history", capture)
        with pytest.raises(AIError) as caught:
            await history.reader.history(
                "execution", tenant_id="tenant", cursor=None, limit=1, tool_call_id="missing",
            )
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
