#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentProvider (spec §16): call_subagent authorization, depth limit,
global + package-scoped resolution, structured error on failure."""

import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.capability import CapabilityContext, CapabilityToolExposurePolicy
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.errors import (
    SubagentDepthExceededError, SubagentExecutionError, SubagentNotFoundError,
)
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.subagent import SubagentProvider, SubagentResult


def _agent(agent_id):
    return AgentSpec(
        id=agent_id, name=agent_id, model=ModelPolicy(primary="gpt-4"),
        instructions=PromptSpec(instructions="hi"),
    )


class _SubSrc:
    def __init__(self, ids):
        self._ids = ids

    async def list_ids(self):
        return self._ids

    async def get(self, agent_id):
        if agent_id not in self._ids:
            raise KeyError(agent_id)
        return _agent(agent_id)


class _Executor:
    def __init__(self, fail=()):
        self.calls = []
        self._fail = set(fail)

    async def execute(self, *, agent_spec, task, context, parent, scope, timeout_seconds):
        self.calls.append((
            agent_spec.id, task,
            parent.run_id if parent else None,
            parent.session_id if parent else None,
            scope,
        ))
        if agent_spec.id in self._fail:
            return SubagentResult(
                agent_id=agent_spec.id, session_id="cs", run_id="cr",
                status="failed", error={"reason": "boom"},
            )
        return SubagentResult(
            agent_id=agent_spec.id, session_id="cs", run_id="cr",
            status="succeeded", output=f"did:{task}",
        )


def _ctx():
    return CapabilityContext(
        agent_id="parent", exposure_policy=CapabilityToolExposurePolicy(),
        run_id="parent-run", session_id="parent-sess",
    )


@pytest.mark.asyncio
async def test_call_subagent_allowed_global():
    provider = SubagentProvider(subagent_provider=_SubSrc(("reviewer",)), executor=_Executor())
    bundle = await provider.resolve(CapabilityRef("subagent", "reviewer"), _ctx())
    call = bundle.toolsets[0].tools["call_subagent"].function
    out = await call("reviewer", "review this")
    assert out["status"] == "succeeded"
    assert out["output"] == "did:review this"
    assert out["session_id"]  # child session created by executor


@pytest.mark.asyncio
async def test_call_subagent_unauthorized_rejected():
    provider = SubagentProvider(subagent_provider=_SubSrc(("reviewer",)), executor=_Executor())
    bundle = await provider.resolve(CapabilityRef("subagent", "reviewer"), _ctx())
    call = bundle.toolsets[0].tools["call_subagent"].function
    with pytest.raises(SubagentNotFoundError):
        await call("ghost", "task")


@pytest.mark.asyncio
async def test_call_subagent_depth_exceeded():
    depth = {"d": 3}
    provider = SubagentProvider(
        subagent_provider=_SubSrc(("reviewer",)), executor=_Executor(),
        depth_provider=lambda: depth["d"],
    )
    ref = CapabilityRef("subagent", "reviewer", config={"max_depth": 3})
    bundle = await provider.resolve(ref, _ctx())
    call = bundle.toolsets[0].tools["call_subagent"].function
    with pytest.raises(SubagentDepthExceededError):
        await call("reviewer", "task")


@pytest.mark.asyncio
async def test_call_subagent_structured_error_on_failure():
    provider = SubagentProvider(
        subagent_provider=_SubSrc(("reviewer",)), executor=_Executor(fail=("reviewer",)),
    )
    bundle = await provider.resolve(CapabilityRef("subagent", "reviewer"), _ctx())
    call = bundle.toolsets[0].tools["call_subagent"].function
    out = await call("reviewer", "task")
    assert out["status"] == "failed"
    assert out["error"] == {"reason": "boom"}


@pytest.mark.asyncio
async def test_call_subagent_no_executor_raises_execution_error():
    provider = SubagentProvider(subagent_provider=_SubSrc(("reviewer",)), executor=None)
    bundle = await provider.resolve(CapabilityRef("subagent", "reviewer"), _ctx())
    call = bundle.toolsets[0].tools["call_subagent"].function
    with pytest.raises(SubagentExecutionError):
        await call("reviewer", "task")


@pytest.mark.asyncio
async def test_subagent_wildcard_authorizes_all():
    provider = SubagentProvider(
        subagent_provider=_SubSrc(("a", "b")), executor=_Executor())
    bundle = await provider.resolve(CapabilityRef("subagent", "*"), _ctx())
    call = bundle.toolsets[0].tools["call_subagent"].function
    await call("a", "t")
    await call("b", "t")
    with pytest.raises(SubagentNotFoundError):
        await call("c", "t")


def test_subagent_default_limits():
    from linktools.ai.subagent import (
        DEFAULT_MAX_DEPTH, DEFAULT_MAX_CONCURRENCY, DEFAULT_TIMEOUT_SECONDS,
    )
    assert DEFAULT_MAX_DEPTH == 3
    assert DEFAULT_MAX_CONCURRENCY == 1
    assert DEFAULT_TIMEOUT_SECONDS == 120
