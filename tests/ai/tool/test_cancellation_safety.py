#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§14.1 / §14.5: cancellation + interrupt safety. CancelledError and
KeyboardInterrupt are BaseExceptions, not business errors -- they must
propagate (to the Runtime / event loop) and never be swallowed or rewrapped as
a ToolError. The tool executor catches only ``except Exception``, so these
propagate naturally. The recording-retry-policy test locks this by proving the
executor's except branch is never entered for a BaseException (should_retry is
never consulted) -- it would fail if that catch widened to ``except
BaseException``. The atomic-write helper cleans up its temp file under
cancellation without swallowing it."""

import asyncio

import pytest

from linktools.ai.policy.engine import PolicyEngine, ToolContext, ToolRequest
from linktools.ai.tool.executor import ToolExecutor
from linktools.ai.tool.models import ToolDescriptor
from linktools.ai.tool.policy import EffectiveToolPolicy

_DESC = ToolDescriptor(
    name="t", source="test", category="misc", risk="low", mutating=False
)
_POLICY = EffectiveToolPolicy()


def test_execute_propagates_cancelled_error():
    # A handler that raises asyncio.CancelledError must surface it unchanged
    # (not wrapped in a ToolError / swallowed by the retry loop).
    executor = ToolExecutor(policy=PolicyEngine(rules=()))

    async def _handler():
        raise asyncio.CancelledError()

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            descriptor=_DESC,
            effective_policy=_POLICY,
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run())


def test_execute_propagates_keyboard_interrupt():
    # KeyboardInterrupt is a BaseException; the executor must not turn it into
    # a ToolError (the run should die, not be retried/normalized).
    executor = ToolExecutor(policy=PolicyEngine(rules=()))

    async def _handler():
        raise KeyboardInterrupt()

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            descriptor=_DESC,
            effective_policy=_POLICY,
        )

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(_run())


def test_atomic_write_propagates_cancelled_error_and_cleans_temp(
    tmp_path, monkeypatch
):
    # §14.1: the atomic-write helper must (a) propagate CancelledError and
    # (b) still remove its temp file -- the try/finally does both.
    from linktools.ai.storage.file._util import _atomic_write

    target = tmp_path / "out.bin"
    real_replace = __import__("os").replace

    def _replacing_raise(src, dst):  # noqa: ARG001
        # Only the atomic-write's own rename raises; leave other replaces alone.
        if isinstance(src, str) and src.endswith(".tmp") and str(tmp_path) in src:
            raise asyncio.CancelledError()
        return real_replace(src, dst)

    monkeypatch.setattr("linktools.ai.storage.file.atomic.os.replace", _replacing_raise)
    with pytest.raises(asyncio.CancelledError):
        _atomic_write(target, b"payload")
    # Target was never written...
    assert not target.exists()
    # ...and no temp file was leaked in tmp_path.
    leftover_temps = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftover_temps == []


def test_execute_does_not_consult_retry_policy_for_cancelled_error():
    # Stronger §14.1 guard: the executor's except branch must NOT be entered for
    # a BaseException. If it were (e.g. a widening of ``except Exception`` to
    # ``except BaseException``), the retry policy's should_retry would be
    # consulted with error=CancelledError. A recording policy that rejects any
    # BaseException consult catches that regression.
    from linktools.ai.tool.executor import ToolExecutor

    seen: "list[BaseException]" = []

    class _RecordingPolicy:
        def should_retry(self, *, error, attempt, policy, descriptor):  # noqa: ARG002
            seen.append(error)
            return False

    executor = ToolExecutor(policy=PolicyEngine(rules=()), retry_policy=_RecordingPolicy())

    async def _handler():
        raise asyncio.CancelledError()

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            descriptor=_DESC,
            effective_policy=_POLICY,
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run())
    # should_retry was never called -- the except branch never saw the
    # CancelledError.
    assert seen == []


def test_execute_does_not_consult_retry_policy_for_keyboard_interrupt():
    from linktools.ai.tool.executor import ToolExecutor

    seen: "list[BaseException]" = []

    class _RecordingPolicy:
        def should_retry(self, *, error, attempt, policy, descriptor):  # noqa: ARG002
            seen.append(error)
            return False

    executor = ToolExecutor(policy=PolicyEngine(rules=()), retry_policy=_RecordingPolicy())

    async def _handler():
        raise KeyboardInterrupt()

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            descriptor=_DESC,
            effective_policy=_POLICY,
        )

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(_run())
    assert seen == []
