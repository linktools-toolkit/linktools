#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Cancellation cleanup for the interactive ai run command.

from types import SimpleNamespace

import pytest

from linktools.commands.ai.run import _cancel_interrupted_execution


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
        def agent(self):
            return Agent()

    from linktools.commands.ai.run import _emit_result

    task = asyncio.create_task(
        _emit_result(Runtime(), "prompt", "session", "memory", True, False, False)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert execution.wait_calls == 2
