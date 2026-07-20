#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GovernedToolInvoker.execute() requires the finalized descriptor and effective
policy. The signature itself rejects a call that omits either (no internal
default Descriptor/Policy that could mis-classify a mutating tool), and the
retry decision honors descriptor.mutating + policy.idempotent."""

import asyncio

import pytest

from linktools.ai.errors import TransientToolError
from linktools.ai.governance.policy.engine import PolicyEngine, ToolContext, ToolRequest
from linktools.ai.storage.filesystem.idempotency import FilesystemIdempotencyStore
from linktools.ai.tool.executor import GovernedToolInvoker
from linktools.ai.tool.models import ToolDescriptor
from linktools.ai.tool.policy import EffectiveToolPolicy

_NON_MUTATING = ToolDescriptor(
    name="t", source="test", category="misc", risk="low", mutating=False
)
_MUTATING = ToolDescriptor(
    name="write", source="test", category="custom", risk="high", mutating=True
)


# ---------------------------------------------------------------------------
# 1. The signature requires descriptor and effective_policy (no default).
# ---------------------------------------------------------------------------


def test_execute_missing_descriptor_raises_type_error():
    """Omitting descriptor is a programming error -- the signature rejects it
    rather than letting the executor invent a default non-mutating descriptor."""
    executor = GovernedToolInvoker(policy=PolicyEngine(rules=()))

    async def _handler() -> str:
        return "x"

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            effective_policy=EffectiveToolPolicy(),
        )

    with pytest.raises(TypeError):
        asyncio.run(_run())


def test_execute_missing_effective_policy_raises_type_error():
    """Omitting effective_policy is a programming error -- the signature
    rejects it rather than letting the executor invent a default policy."""
    executor = GovernedToolInvoker(policy=PolicyEngine(rules=()))

    async def _handler() -> str:
        return "x"

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            descriptor=_NON_MUTATING,
        )

    with pytest.raises(TypeError):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. Retry safety: the descriptor's mutating flag and the policy's idempotent
#    flag drive the retry decision (a mutating non-idempotent tool is never
#    retried, even on a transient error with max_retries set).
# ---------------------------------------------------------------------------


def test_mutating_non_idempotent_tool_not_retried_on_transient_error():
    """A mutating, non-idempotent tool may have partially applied its effect --
    never retry it blind, even if the error is transient and max_retries > 0."""
    executor = GovernedToolInvoker(policy=PolicyEngine(rules=()))
    calls = {"n": 0}

    async def _handler() -> str:
        calls["n"] += 1
        raise TransientToolError("transient")

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="write", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            descriptor=_MUTATING,
            effective_policy=EffectiveToolPolicy(idempotent=False),
            max_retries=3,
        )

    with pytest.raises(TransientToolError):
        asyncio.run(_run())
    assert calls["n"] == 1, "mutating non-idempotent tool must not be retried"


def test_non_mutating_transient_error_is_retried():
    """A non-mutating tool whose handler raises a transient error is retried
    up to max_retries (the mutating-safety gate does not apply)."""
    executor = GovernedToolInvoker(policy=PolicyEngine(rules=()))
    calls = {"n": 0}

    async def _handler() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientToolError(f"transient {calls['n']}")
        return "ok"

    async def _run():
        return await executor.execute(
            ToolRequest(tool_name="read", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            descriptor=_NON_MUTATING,
            effective_policy=EffectiveToolPolicy(),
            max_retries=2,
        )

    assert asyncio.run(_run()) == "ok"
    assert calls["n"] == 3


def test_mutating_idempotent_tool_with_store_can_retry(tmp_path):
    """A mutating tool declared idempotent (with an IdempotencyStore wired) is
    safe to retry: the store records the FAILED attempt and the retry
    overwrites it on success."""
    store = FilesystemIdempotencyStore(root=tmp_path / "idem")
    executor = GovernedToolInvoker(policy=PolicyEngine(rules=()), idempotency_store=store)
    calls = {"n": 0}

    async def _handler() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientToolError("transient")
        return "ok"

    async def _run():
        return await executor.execute(
            ToolRequest(tool_name="write", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            descriptor=_MUTATING,
            effective_policy=EffectiveToolPolicy(idempotent=True),
            idempotency_key="op-1",
            max_retries=1,
        )

    assert asyncio.run(_run()) == "ok"
    assert calls["n"] == 2, "idempotent mutating tool must be retried"
