#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Cancellation cleanup for the interactive ai run command.

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from linktools.commands.ai._output import _cancel_interrupted_execution, emit_result


@pytest.mark.asyncio
async def test_ai_run_interrupt_waits_for_execution_quiescence() -> None:
    calls: list[object] = []

    class Execution:
        execution_id = "execution"

        async def cancel(self, *, idempotency_key: str | None = None, force: bool = False):
            calls.append(("cancel", idempotency_key, force))
            return SimpleNamespace(cancelled=False)

        async def wait(self, *, timeout_seconds: float | None = None):
            calls.append(("wait", timeout_seconds))
            return SimpleNamespace()

    await _cancel_interrupted_execution(Execution())

    assert calls == [
        ("cancel", "ai-run-interrupt:execution", False),
        ("wait", None),
    ]


@pytest.mark.asyncio
async def test_ai_run_json_interrupt_cancels_owned_execution() -> None:
    import asyncio

    started = asyncio.Event()
    cancelled = asyncio.Event()

    class Execution:
        execution_id = "execution"

        def __init__(self) -> None:
            self.wait_calls = 0

        async def wait(self):
            self.wait_calls += 1
            if self.wait_calls == 1:
                started.set()
                await asyncio.Event().wait()
            return SimpleNamespace()

        async def cancel(self, *, idempotency_key=None, force=False):
            assert idempotency_key == "ai-run-interrupt:execution"
            assert force is False
            cancelled.set()
            return SimpleNamespace(cancelled=True)

    execution = Execution()

    class Agent:
        async def start(self, *args, **kwargs):
            del args, kwargs
            return execution

    class Runtime:
        @property
        def agents(self):
            return self

        def get(self):
            return Agent()

    from linktools.commands.ai._output import emit_result

    task = asyncio.create_task(
        emit_result(Runtime(), "prompt", "session", True, False, False)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert execution.wait_calls == 2


@pytest.mark.asyncio
async def test_ai_run_native_json_returns_canonical_turn_without_readback(capsys) -> None:
    import json

    subscription = []

    def subscribe(session_id, callback):
        assert session_id == "default"
        subscription.append(callback)
        return subscription.clear

    async def start_turn(session_id, prompt, **kwargs):
        assert session_id == "default" and prompt == "hello" and kwargs["agent"] == "default"
        subscription[0]({"type": "terminal", "turn_id": "canonical-turn", "state": "succeeded",
                         "output": {"text": "answer"}})
        return "canonical-turn"

    runtime = SimpleNamespace(
        default_principal=object(),
        sessions=SimpleNamespace(
            _uses_turn_sessions=True, create_turn_session=AsyncMock(), start_turn=start_turn,
            observe_turn_session=subscribe,
        ),
        history=SimpleNamespace(turn_result=AsyncMock()),
    )
    assert await emit_result(runtime, "hello", "default", True, False, None) == 0

    result = json.loads(capsys.readouterr().out)
    assert (result["execution_id"], result["status"], result["output"]) == ("canonical-turn", "SUCCEEDED", {"text": "answer"})
    runtime.history.turn_result.assert_not_awaited()
    assert subscription == []
