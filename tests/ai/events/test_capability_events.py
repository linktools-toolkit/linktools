#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability-lifecycle event emission (contract Middleware/EventStore): the
EventStore wired on a CapabilityContext receives capability.resolve + skill +
package-resource + entrypoint events."""

import pytest

from linktools.ai.capability import (
    CapabilityAssembler, CapabilityContext, CapabilityToolExposurePolicy,
)
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.events.payloads import (
    CapabilityResolveCompleted, CapabilityResolveStarted, PackageResourceListed,
    PackageResourceRead, SkillListed, SkillRead,
)
from linktools.ai.package.provider import DirectoryPackageResourceProvider
from linktools.ai.package.scope import PackageScope
from linktools.ai.package.resource import ResourceRef
from linktools.ai.skill import SkillProvider
from linktools.ai.skill.toolset import build_skill_toolset


class _RecordingStore:
    def __init__(self):
        self.events = []

    async def append(self, *, stream_id, run_id, root_run_id, parent_run_id,
                     session_id, runnable_id, payload):
        self.events.append(type(payload).__name__)
        return payload


class _SkillSrc:
    async def list_ids(self):
        return ("sql",)

    async def get(self, sid):
        class _S:
            id = sid; name = sid; description = "d"; instructions = "x"; metadata = {}
        return _S()


def _ctx(store):
    return CapabilityContext(
        agent_id="a1", exposure_policy=CapabilityToolExposurePolicy(),
        run_id="r1", root_run_id="r1", session_id="s1", event_store=store,
    )


@pytest.mark.asyncio
async def test_capability_resolve_events_emitted():
    store = _RecordingStore()

    class _P:
        kind = "skill"

        async def resolve(self, ref, context):
            from linktools.ai.capability.bundle import CapabilityBundle
            return CapabilityBundle()

    asm = CapabilityAssembler({"skill": _P()})
    from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
    from linktools.ai.model.policy import ModelPolicy
    spec = AgentSpec(id="a1", name="a1", model=ModelPolicy(primary="m"),
                     instructions=PromptSpec(instructions="hi"),
                     tools=(ToolRef(name="*", kind="skill"),))
    await asm.assemble(spec, _ctx(store))
    assert "CapabilityResolveStarted" in store.events
    assert "CapabilityResolveCompleted" in store.events


@pytest.mark.asyncio
async def test_skill_operation_events_emitted():
    store = _RecordingStore()
    ctx = _ctx(store)
    provider = SkillProvider(_SkillSrc())
    await provider.resolve(CapabilityRef("skill", "*"), ctx)  # catalog resolution
    # The toolset emit fires only on tool invocation; exercise it directly.
    ts = build_skill_toolset(_SkillSrc(), authorized={"sql"}, emit=__import__(
        "linktools.ai.capability.provider", fromlist=["make_event_emitter"]).make_event_emitter(ctx))
    list_fn = ts.tools["list_skills"].function
    read_fn = ts.tools["read_skill"].function
    await list_fn()
    await read_fn("sql")
    assert "SkillListed" in store.events
    assert "SkillRead" in store.events


@pytest.mark.asyncio
async def test_no_event_store_is_safe():
    # Without an EventStore, resolution + tool calls must not raise.
    ctx = CapabilityContext(agent_id="a1", exposure_policy=CapabilityToolExposurePolicy())
    provider = SkillProvider(_SkillSrc())
    bundle = await provider.resolve(CapabilityRef("skill", "sql"), ctx)
    read_fn = next(md.handler for c in bundle.tool_contributions for md in c.tools
                   if md.descriptor.name == "read_skill")
    out = await read_fn("sql")
    assert out["content"] == "x"
