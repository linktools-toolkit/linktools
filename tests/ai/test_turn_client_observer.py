#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise local producer liveness without polling or a model provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
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

    async def read(*args, **kwargs):
        if case in ("stale", "remote"):
            # A later genuine frame proves observation continued after the gap.
            emit({"type": "tool", "turn_id": "turn"})
        return {"status": "PENDING_START", "generation": 3 if case == "stale" else 2}

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
    else:
        turn_result.assert_awaited_once_with("session", "turn", principal=principal, local_workspace=False)
