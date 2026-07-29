#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""call_extension_entrypoint must execute or raise -- never return a fake-success
'reserved' marker (contract)."""

import pytest

from linktools.ai.agent.tool.exposure import ToolExposurePolicy
from linktools.ai.agent.assembly.provider import AgentFeatureContext
from linktools.ai.agent.assembly.models import AgentFeatureRef
from linktools.ai.agent.extension.provider import ExtensionProvider


def _ctx(execution_tools=True):
    return AgentFeatureContext(
        agent_id="a1",
        execution_id="e1",
        root_execution_id="e1",
        parent_execution_id=None,
        session_id="s1",
        tenant_id="t1",
        user_id="u1",
        workspace=None,
        sandbox=None,
    )


@pytest.mark.asyncio
async def test_call_without_executor_raises_not_reserved():
    # expose_call_tool=True but no entrypoint_executor wired -> denial, not
    # a {"status": "reserved"} fake-success.
    provider = ExtensionProvider(
        entrypoint_resolver=object()
    )  # resolver set, executor None
    ref = AgentFeatureRef(
        "extension-entrypoint",
        "pkg",
        config={
            "allowed_kinds": ["agent"],
            "allowed_names": ["grader"],
            "expose_call_tool": True,
        },
    )
    bundle = await provider.resolve(ref, _ctx())
    call = next(
        md.handler
        for md in bundle.tools
        if md.descriptor.name == "call_extension_entrypoint"
    )
    with pytest.raises(Exception):  # ExtensionEntrypointDeniedError
        await call("pkg", "agent", "grader", "do it")


def test_no_reserved_marker_in_source():
    import pathlib

    src = pathlib.Path("linktools-ai/src/linktools/ai/agent/extension/toolset.py").read_text()
    assert '"status": "reserved"' not in src
    assert 'status="reserved"' not in src


@pytest.mark.asyncio
async def test_call_without_resolver_raises_not_nameerror():
    # resolver=None must raise ExtensionEntrypointNotFoundError, not NameError
    # from a missing import. Exercise the toolset
    # directly since ExtensionProvider guards resolver=None -> empty bundle.
    from linktools.ai.errors import ExtensionEntrypointNotFoundError
    from linktools.ai.agent.extension.scope import ExtensionScope
    from linktools.ai.agent.extension.toolset import build_extension_entrypoint_toolset

    ts = build_extension_entrypoint_toolset(
        resolver=None,
        allowed={"pkg": ExtensionScope("pkg")},
        allowed_kinds=("agent",),
        allowed_names=("grader",),
        expose_call_tool=True,
        executor=object(),
    )
    call = ts.handlers["call_extension_entrypoint"]
    with pytest.raises(ExtensionEntrypointNotFoundError):
        await call("pkg", "agent", "grader", "do it")


@pytest.mark.asyncio
async def test_call_extension_entrypoint_forwards_parent_identity_unmodified():
    """Regression: call_extension_entrypoint used to hardcode
    root_execution_id=parent_execution_id, truncating lineage to one hop whenever the
    entrypoint call is itself already nested under an existing chain. It must
    now forward the SAME ParentRunIdentity the caller built -- in particular
    parent.root_execution_id, never re-derived from parent.run_id here."""
    from linktools.ai.agent.extension.scope import ExtensionScope
    from linktools.ai.agent.extension.toolset import build_extension_entrypoint_toolset
    from linktools.ai.execution.identity import ParentRunIdentity
    from linktools.ai.agent.subagent.models import SubagentResult

    class _FakeResolver:
        async def resolve_agent(self, ref):
            from linktools.ai.agent.spec import AgentSpec, PromptSpec
            from linktools.ai.model.policy import ModelPolicy

            return AgentSpec(
                id=ref.name,
                name=ref.name,
                model=ModelPolicy(primary="m"),
                instructions=PromptSpec(instructions="hi"),
            )

    class _RecordingExecutor:
        def __init__(self):
            self.seen_parent = None

        async def execute(
            self, *, agent_spec, task, context, parent, scope, timeout_seconds
        ):
            self.seen_parent = parent
            return SubagentResult(
                agent_id=agent_spec.id,
                session_id="cs",
                run_id="cr",
                status="succeeded",
            )

    # Simulates: run A -> subagent B -> (nested) scenario call, so
    # the immediate parent is B (run_id="B") but the true root is A ("A-root").
    parent_identity = ParentRunIdentity(
        run_id="B",
        root_execution_id="A-root",
        session_id="session-B",
        user_id="u1",
        tenant_id="t1",
    )
    executor = _RecordingExecutor()
    ts = build_extension_entrypoint_toolset(
        resolver=_FakeResolver(),
        allowed={"pkg": ExtensionScope("pkg")},
        allowed_kinds=("agent",),
        allowed_names=("grader",),
        expose_call_tool=True,
        executor=executor,
        parent=parent_identity,
    )
    call = ts.handlers["call_extension_entrypoint"]
    await call("pkg", "agent", "grader", "do it")

    assert executor.seen_parent is parent_identity
    assert executor.seen_parent.root_execution_id == "A-root"
    assert executor.seen_parent.user_id == "u1"
