#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized native SDK callbacks keep execution and observation separate."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext
from linktools.ai.core import ExecutionEventType, ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.execution import ExecutionTreeEvent
from linktools.ai.runtime._domains import RuntimeSessions
from linktools.ai.runtime._native_session import _NativeMetadataObserver, _NativeSessionTurn

from ._runtime_test_helpers import _wait_for_committed
from .test_official_runtime_recovery import _runtime


class _SessionAuthorization(TenantAuthorizationPolicy):
    """An explicit SQL session authorizer."""


def _model_with_tool(calls, *, before_finish=None):
    async def model(messages, _info):
        calls.append(messages)
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            if before_finish is not None:
                await before_finish.wait()
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("checked", {}, tool_call_id="exact-call")])
    return model


async def _open_chat(runtime):
    principal = Principal("operator", runtime.tenant_id)
    await runtime.sessions.create_turn_session("chat", principal=principal)
    return principal


@pytest.mark.asyncio
async def test_native_sdk_callback_captures_fast_tool_and_terminal_before_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, effects, events, subscriptions = [], [], [], []
    subscribe = RuntimeSessions.observe_turn_session

    def count_subscription(sessions, session_id, receive):
        subscriptions.append(session_id)
        return subscribe(sessions, session_id, receive)

    monkeypatch.setattr(RuntimeSessions, "observe_turn_session", count_subscription)

    async def checked(_context: AgentContext[None]) -> str:
        effects.append("ran")
        return "tool-private-result"

    async def observe(event: ExecutionTreeEvent) -> None:
        events.append(event)

    async with _runtime(tmp_path / "fast.db", _model_with_tool(calls), checked,
                        authorization=_SessionAuthorization()) as (runtime, persistence, _steps):
        principal = await _open_chat(runtime)
        wait_observed = _NativeSessionTurn._wait_observed

        async def wait_after_terminal(turn, observer, **options):
            # Ensure the producer finishes before either callback task starts.
            await _wait_for_committed(
                lambda: persistence.session_turns.get_turn(turn.key),
                lambda row: row is not None and row["state"] == "succeeded",
            )
            return await wait_observed(turn, observer, **options)

        monkeypatch.setattr(_NativeSessionTurn, "_wait_observed", wait_after_terminal)
        run = runtime.agents.get().run
        options = dict(session_id="chat", principal=principal, idempotency_key="one", on_event=observe)
        result = await run("private prompt", **options)
        assert result.result.status is ExecutionStatus.SUCCEEDED
        key = result.result.key
        assert result.cursor is None and result.observation_error is None
        assert [item.event.event_type for item in events] == [
            ExecutionEventType.TOOL_CALL_STARTED.value, ExecutionEventType.TOOL_CALL_FINISHED.value,
            ExecutionEventType.EXECUTION_SUCCEEDED.value,
        ]
        assert all(item.execution_id == item.root_execution_id == key.turn_id
                   and item.depth == 0 and item.cursor is None
                   and item.event.durable_seq is None for item in events)
        assert events[0].event.payload["tool_call_id"] == "exact-call"
        assert events[1].event.payload["status"] == "SUCCEEDED"
        assert "private prompt" not in str([item.event.payload for item in events])
        assert "tool-private-result" not in str([item.event.payload for item in events])
        assert await persistence.execution.executions.get(key.turn_id, tenant_id=runtime.tenant_id) is None

        events.clear()
        replay = await run("private prompt", **options)
        assert replay.result.key == key
        assert [item.event.event_type for item in events] == [ExecutionEventType.EXECUTION_SUCCEEDED.value]
    assert len(calls) == 2 and effects == ["ran"]
    assert subscriptions == ["chat", "chat"]  # one slot each for original run and replay


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", (False, True))
async def test_native_sdk_callback_exception_does_not_cancel_turn(tmp_path: Path, strict: bool) -> None:
    calls = []

    async def checked(_context: AgentContext[None]) -> str:
        return "value"

    async def observer(_event: ExecutionTreeEvent) -> None:
        raise RuntimeError("consumer failed")

    async with _runtime(tmp_path / "callback-error.db", _model_with_tool(calls), checked,
                        authorization=_SessionAuthorization()) as (runtime, _persistence, _steps):
        principal = await _open_chat(runtime)
        agent = runtime.agents.get()
        with pytest.raises(ObservationError) as error:
            await agent.run("question", session_id="chat", principal=principal, idempotency_key="one",
                            on_event=observer, raise_on_stream_error=strict, close_timeout_seconds=1.0)
        assert error.value.origin == "callback" and isinstance(error.value.__cause__, RuntimeError)
        resumed = await agent.start("question", session_id="chat", principal=principal, idempotency_key="one")
        assert (await resumed.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", (False, True))
async def test_native_sdk_callback_gap_obeys_stream_error_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool,
) -> None:
    calls = []
    allow_finish, injected = asyncio.Event(), asyncio.Event()
    receive = _NativeMetadataObserver._receive

    def drop_frame(observer, frame):
        if frame.get("type") == "tool" and not injected.is_set():
            injected.set()
            receive(observer, {"type": "gap", "turn_id": frame["turn_id"]})
            return
        receive(observer, frame)

    async def checked(_context: AgentContext[None]) -> str:
        return "value"

    async def observer(_event: ExecutionTreeEvent) -> None:
        pass

    monkeypatch.setattr(_NativeMetadataObserver, "_receive", drop_frame)
    async with _runtime(tmp_path / "stream-gap.db", _model_with_tool(calls, before_finish=allow_finish), checked,
                        authorization=_SessionAuthorization()) as (runtime, _persistence, _steps):
        principal = await _open_chat(runtime)
        agent = runtime.agents.get()
        running = asyncio.create_task(agent.run(
            "question", session_id="chat", principal=principal, idempotency_key="one", on_event=observer,
            raise_on_stream_error=strict, timeout_seconds=10,
        ))
        try:
            await asyncio.wait_for(injected.wait(), timeout=5)
            if strict:
                with pytest.raises(ObservationError) as error:
                    await running
                assert error.value.origin == "stream"
            else:
                allow_finish.set()
                observed = await running
                assert observed.result.status is ExecutionStatus.SUCCEEDED
                assert observed.observation_error is not None and observed.observation_error.origin == "stream"
        finally:
            allow_finish.set()
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
        turn = await agent.start("question", session_id="chat", principal=principal, idempotency_key="one")
        assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_native_raw_content_is_rejected_before_admission_without_changing_independent_execution(
    tmp_path: Path,
) -> None:
    calls, observed = [], []

    async def checked(_context: AgentContext[None]) -> str:
        return "value"

    async def observer(event: ExecutionTreeEvent) -> None:
        observed.append(event)

    async with _runtime(tmp_path / "raw-content.db", _model_with_tool(calls), checked,
                        authorization=_SessionAuthorization()) as (runtime, persistence, _steps):
        principal = await _open_chat(runtime)
        agent = runtime.agents.get()
        with pytest.raises(AIError) as error:
            await agent.run("question", session_id="chat", principal=principal, idempotency_key="one",
                            on_event=observer, include_event_content=True)
        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert error.value.safe_details["field"] == "include_event_content"
        assert (await persistence.session_turns.get_session("chat"))["active_turn_id"] is None
        assert not calls and not observed

        independent = await agent.run("question", principal=principal, on_event=observer,
                                      include_event_content=True, timeout_seconds=10)
        assert independent.result.status is ExecutionStatus.SUCCEEDED
        assert observed and all(item.root_execution_id == independent.result.execution_id for item in observed)
        assert await persistence.execution.executions.get(
            independent.result.execution_id, tenant_id=runtime.tenant_id,
        ) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", (False, True))
async def test_native_callback_emits_authoritative_terminal_once_without_live_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool,
) -> None:
    calls, events = [], []
    receive = _NativeMetadataObserver._receive

    def drop_local_terminal(observer, frame):
        if frame.get("type") != "terminal":
            receive(observer, frame)

    monkeypatch.setattr(_NativeMetadataObserver, "_receive", drop_local_terminal)

    async def checked(_context: AgentContext[None]) -> str:
        return "value"

    async def observe(event: ExecutionTreeEvent) -> None:
        events.append(event)

    async with _runtime(tmp_path / "missed-notice.db", _model_with_tool(calls), checked,
                        authorization=_SessionAuthorization()) as (runtime, _persistence, _steps):
        principal = await _open_chat(runtime)
        result = await runtime.agents.get().run(
            "question", session_id="chat", principal=principal, idempotency_key="one",
            on_event=observe, raise_on_stream_error=strict, timeout_seconds=10,
        )
        assert result.result.status is ExecutionStatus.SUCCEEDED
        assert result.observation_error is not None and result.observation_error.origin == "stream"
        assert result.observation_error.safe_details["coverage_complete"] is False
        assert [event.event.event_type for event in events] == [
            ExecutionEventType.TOOL_CALL_STARTED.value, ExecutionEventType.TOOL_CALL_FINISHED.value,
            ExecutionEventType.EXECUTION_SUCCEEDED.value,
        ]
