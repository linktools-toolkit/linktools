#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise local producer liveness without polling or a model provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from linktools.ai.core import ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime._native_session import _NativeMetadataObserver, _NativeSessionTurn
from linktools.ai.runtime.turn_clients import TurnObserver


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("stopped", "terminal", "stale", "remote"))
async def test_turn_observer_handles_stopped_producer(case):
    principal = Principal("local", "viewer")
    emit = None
    unsubscribe = Mock()

    def observe(session_id, callback):
        nonlocal emit
        assert session_id == "session"
        emit = callback
        return unsubscribe

    async def read(session_id, turn_id, *, principal, local_workspace, producer_stopped_generation=None):
        if case == "stopped":
            assert producer_stopped_generation == 2
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if case == "stale":
            assert producer_stopped_generation == 2
        if case in ("stale", "remote"):
            # A later genuine frame proves observation continued after the gap.
            emit({"type": "tool", "turn_id": "turn"})
        return {"status": "PENDING_START"}

    turn_result = AsyncMock(side_effect=read)
    runtime = SimpleNamespace(
        sessions=SimpleNamespace(observe_turn_session=observe),
        history=SimpleNamespace(turn_result=turn_result),
    )
    gap = {"type": "gap", "turn_id": "turn"}
    if case != "remote":
        gap.update(phase="producer_stopped", generation=2)
    terminal = {"type": "terminal", "turn_id": "turn", "state": "succeeded"}

    with TurnObserver(runtime, "session", principal, local_workspace=False) as observer:
        emit(gap)
        if case == "terminal":
            emit(terminal)
        events = observer.events("turn")
        try:
            assert await anext(events) == gap
            if case == "stopped":
                with pytest.raises(AIError) as failure:
                    await anext(events)
                assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            elif case == "terminal":
                assert await anext(events) == terminal
                with pytest.raises(StopAsyncIteration):
                    await anext(events)
            else:
                assert await anext(events) == {"type": "tool", "turn_id": "turn"}
        finally:
            await events.aclose()

    unsubscribe.assert_called_once_with()
    if case == "terminal":
        turn_result.assert_not_awaited()
    elif case == "remote":
        turn_result.assert_awaited_once_with("session", "turn", principal=principal, local_workspace=False)
    else:
        turn_result.assert_awaited_once_with(
            "session", "turn", principal=principal, local_workspace=False, producer_stopped_generation=2,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("preview", (True, False))
async def test_native_turn_observe_projects_live_status_without_extra_reads(preview):
    principal = Principal("local", "viewer")
    unsubscribe = Mock()
    frames = [
        {"type": "progress", "turn_id": "turn", "item": {"kind": "model", "key": "1:1", "status": "RUNNING"}},
        {
            "type": "thinking_completed", "turn_id": "turn", "model_key": "1:1",
            "attempt": 2, "part_index": 0, "elapsed_ms": 375,
        },
        {
            "type": "progress", "turn_id": "turn",
            "item": {"kind": "model", "key": "1:1", "status": "SUCCEEDED", "elapsed_ms": 410},
        },
    ]
    identity = {"turn_id": "turn", "turn_no": 1, "generation": 1, "revision": 2, "model_key": "model:1:1"}
    frames.extend((
        {**identity, "type": "preview_reset", "attempt": 2},
        {**identity, "type": "delta", "attempt": 3, "part_index": 0, "delta_kind": "text", "text": "replacement"},
    ))
    terminal = {"type": "terminal", "turn_id": "turn", "state": "succeeded"}
    subscribe = Mock(return_value=unsubscribe)

    async def read_current(key, actor):
        assert key == SessionTurnKey("session", "turn")
        assert actor == principal
        session_id, emit = subscribe.call_args.args
        assert session_id == "session"
        for frame in frames:
            emit({**frame, "turn_id": "other"})
            emit(frame)
        emit(terminal)
        return SimpleNamespace(status=ExecutionStatus.STARTED)

    read_result = AsyncMock(side_effect=read_current)
    gap_read = AsyncMock()
    runtime = SimpleNamespace(
        sessions=SimpleNamespace(observe_turn_session=subscribe),
        history=SimpleNamespace(turn_result=gap_read),
    )
    turn = _NativeSessionTurn(
        SessionTurnKey("session", "turn"), principal, runtime.sessions, read_result, runtime, False,
    )

    assert [frame async for frame in turn.observe(preview=preview)] == [*(frames if preview else ()), terminal]
    read_result.assert_awaited_once_with(turn.key, principal)
    gap_read.assert_not_awaited()
    unsubscribe.assert_called_once_with()

    # Metadata callbacks retain the existing tool/settlement view. Even when
    # the callback is slow, status previews cannot consume its queue budget.
    subscribe.reset_mock()
    with _NativeMetadataObserver(runtime, "session", principal, local_workspace=False) as observer:
        _, emit = subscribe.call_args.args
        tool = {"type": "tool", "turn_id": "turn", "item": {"key": "tool:1", "status": "RUNNING"}}
        emit(tool)
        for _ in range(256):
            emit(frames[0])
            emit(frames[1])
        emit(terminal)
        assert [frame async for frame in observer.events("turn")] == [tool, terminal]
    gap_read.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("append_only", (True, False))
@pytest.mark.parametrize("disruption", ("preview_reset", "gap", "overflow"))
async def test_preview_replacement_only_continues_for_reset_aware_observers(append_only, disruption):
    principal = Principal("local", "viewer")
    unsubscribe = Mock()
    subscribe = Mock(return_value=unsubscribe)
    read_back = AsyncMock()
    runtime = SimpleNamespace(
        sessions=SimpleNamespace(observe_turn_session=subscribe),
        history=SimpleNamespace(turn_result=read_back),
    )
    reset = {"type": "preview_reset", "turn_id": "turn", "model_key": "1:1", "attempt": 1}
    replacement = {"type": "delta", "turn_id": "turn", "model_key": "1:1", "attempt": 2, "text": "replacement"}
    terminal = {"type": "terminal", "turn_id": "turn", "state": "succeeded"}
    with TurnObserver(runtime, "session", principal, local_workspace=False) as observer:
        _, emit = subscribe.call_args.args
        if disruption == "overflow":
            for _ in range(257):
                emit(replacement)
        else:
            emit(reset if disruption == "preview_reset" else {"type": "gap"})
        emit(replacement)
        emit(terminal)
        frames = [frame async for frame in observer.events("turn", append_only=append_only)]
    if disruption == "preview_reset":
        assert frames == ([reset, terminal] if append_only else [reset, replacement, terminal])
        assert observer.incomplete is append_only
    else:
        assert frames == [{"type": "gap"}, terminal]
        assert observer.incomplete
    read_back.assert_not_awaited()
    unsubscribe.assert_called_once_with()
