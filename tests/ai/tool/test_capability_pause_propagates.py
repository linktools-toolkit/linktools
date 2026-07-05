#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``PolicyCapability.before_tool_execute`` must NOT catch ``RunPaused``.

Its catch list is ``ToolDeniedError`` + ``ToolApprovalRequiredError`` (both
``ToolError`` subclasses) -> ``SkipToolExecution``. ``RunPaused`` is a
``RunError`` (not a ``ToolError``), so when ``ToolExecutor.check`` raises it
(under ``pause_on_approval=True``) the capability MUST let it propagate
unchanged out of pydantic-ai's tool-execution stack to ``AgentRunner`` --
which checkpoints state, transitions the Run to WAITING_APPROVAL, and stops.

The cleanest proof is a stub executor whose ``.check`` raises ``RunPaused``
directly: if ``RunPaused`` reaches the caller of ``before_tool_execute``,
propagation holds; if it gets translated to ``SkipToolExecution``, the pause
signal is lost (the bug this test guards against)."""
import pytest
from pydantic_ai.messages import ToolCallPart

from linktools.ai.errors import RunPaused
from linktools.ai.policy.engine import ToolContext
from linktools.ai.tool.capability import PolicyCapability


class _ExecutorThatPauses:
    """Stub executor whose ``check`` raises ``RunPaused`` -- mimics
    ToolExecutor under ``pause_on_approval=True`` encountering a REQUIRE_APPROVAL
    decision."""

    def __init__(self, *, run_id: str, approval_id: str):
        self._run_id = run_id
        self._approval_id = approval_id

    async def check(self, request, context) -> None:
        raise RunPaused(run_id=self._run_id, approval_id=self._approval_id)


def _build_call_kwargs():
    """Build the ``(call, tool_def, args)`` kwargs ``before_tool_execute``
    needs (after the positional ``ctx``). ``tool_def`` only needs a ``.name``
    attribute; ``call`` only needs ``.tool_call_id``; ``args`` is passed
    through verbatim. ``before_tool_execute``'s signature is
    ``(ctx, *, call, tool_def, args)`` -- ``call``/``tool_def``/``args`` are
    keyword-only."""
    class _ToolDef:
        name = "rm"

    class _RunContext:
        pass

    return (
        _RunContext(),
        {
            "call": ToolCallPart(tool_name="rm", args={"path": "/"}, tool_call_id="tc-1"),
            "tool_def": _ToolDef(),
            "args": {"path": "/"},
        },
    )


@pytest.mark.asyncio
async def test_run_paused_propagates_through_before_tool_execute():
    """``RunPaused`` raised by ``executor.check`` must reach the caller of
    ``before_tool_execute`` unchanged (NOT translated to ``SkipToolExecution``)."""
    executor = _ExecutorThatPauses(run_id="r1", approval_id="a1")
    capability = PolicyCapability(
        executor=executor,  # type: ignore[arg-type]
        current_context=ToolContext(run_id="r1", session_id="s1"),
    )

    ctx, kwargs = _build_call_kwargs()
    with pytest.raises(RunPaused) as exc_info:
        await capability.before_tool_execute(ctx, **kwargs)

    # Propagation carries the ids AgentRunner needs -- not a translated shell.
    assert exc_info.value.run_id == "r1"
    assert exc_info.value.approval_id == "a1"


@pytest.mark.asyncio
async def test_run_paused_propagates_even_when_current_context_unset():
    """Propagation holds regardless of ``current_context`` state -- the
    capability must not synthesize a fallback that swallows ``RunPaused``."""
    executor = _ExecutorThatPauses(run_id="r2", approval_id="a2")
    capability = PolicyCapability(
        executor=executor,  # type: ignore[arg-type]
        current_context=None,
    )

    ctx, kwargs = _build_call_kwargs()
    with pytest.raises(RunPaused) as exc_info:
        await capability.before_tool_execute(ctx, **kwargs)

    assert exc_info.value.run_id == "r2"
    assert exc_info.value.approval_id == "a2"
