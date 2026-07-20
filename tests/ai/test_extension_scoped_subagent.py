#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extension-scoped subagent (contract/contract): call_subagent with a scenario
resolves via the EntrypointResolver and never pollutes the global namespace."""

import pytest

from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
from linktools.ai.capability.provider import CapabilityContext
from linktools.ai.capability.models import CapabilityRef
from linktools.ai.errors import SubagentNotFoundError
from linktools.ai.extension.resolver import DirectoryEntrypointResolver
from linktools.ai.subagent import SubagentProvider, SubagentResult


class _Executor:
    def __init__(self):
        self.resolved_ids = []

    async def execute(
        self, *, agent_spec, task, context, parent, scope, timeout_seconds
    ):
        self.resolved_ids.append((agent_spec.id, scope.extension_id if scope else None))
        return SubagentResult(
            agent_id=agent_spec.id,
            session_id="cs",
            run_id="cr",
            status="succeeded",
            output=agent_spec.id,
        )


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "skill-creator"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "grader.md").write_text(
        "---\nname: grader\nmodel:\n  primary: gpt-4o\n---\nGrade.\n", encoding="utf-8"
    )
    er = DirectoryEntrypointResolver({"skill-creator": root})
    return SubagentProvider(entrypoint_resolver=er, executor=_Executor())


def _ctx():
    return CapabilityContext(
        agent_id="parent",
        exposure_policy=CapabilityToolExposurePolicy(),
        run_id="pr",
        session_id="ps",
    )


@pytest.mark.asyncio
async def test_scoped_subagent_resolves_via_entrypoint(env):
    provider = env
    # Scoped calls require the scenario be declared on the ref (confinement).
    bundle = await provider.resolve(
        CapabilityRef(
            "subagent", "grader", config={"allowed_extensions": ["skill-creator"]}
        ),
        _ctx(),
    )
    call = next(md.handler for c in bundle.tool_contributions for md in c.tools)
    out = await call(
        "grader",
        "grade it",
        scope={"extension_id": "skill-creator", "extension_kind": "skill"},
    )
    assert out["status"] == "succeeded"
    assert out["output"] == "extension:skill-creator:agent:grader"


@pytest.mark.asyncio
async def test_scoped_subagent_undeclared_extension_rejected(env):
    provider = env
    # No allowed_extensions declared -> a scoped call to any scenario refused.
    bundle = await provider.resolve(CapabilityRef("subagent", "grader"), _ctx())
    call = next(md.handler for c in bundle.tool_contributions for md in c.tools)
    with pytest.raises(SubagentNotFoundError, match="extension scope not allowed"):
        await call("grader", "t", scope={"extension_id": "skill-creator"})


@pytest.mark.asyncio
async def test_scoped_subagent_missing_in_extension_raises(env):
    provider = env
    bundle = await provider.resolve(
        CapabilityRef(
            "subagent", "grader", config={"allowed_extensions": ["skill-creator"]}
        ),
        _ctx(),
    )
    call = next(md.handler for c in bundle.tool_contributions for md in c.tools)
    with pytest.raises(SubagentNotFoundError):
        # 'grader' is allowed by name + scenario, but the scenario no 'ghost'.
        await call("ghost", "t", scope={"extension_id": "skill-creator"})


@pytest.mark.asyncio
async def test_scoped_subagent_does_not_register_globally(env):
    # A scoped call resolves only through the entrypoint resolver; the global
    # subagent_provider is None, so an unscoped call cannot resolve.
    provider = env
    bundle = await provider.resolve(CapabilityRef("subagent", "grader"), _ctx())
    call = next(md.handler for c in bundle.tool_contributions for md in c.tools)
    from linktools.ai.errors import SubagentExecutionError

    with pytest.raises(SubagentExecutionError):
        await call("grader", "t")  # no scope, no global provider
