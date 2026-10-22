#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct capability registration is global unless Agent scope is explicit."""
from ._runtime_test_helpers import model_features

import pytest
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, PrefixTools
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import AssetCapability, ToolResponsesCapability, WorkspaceCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.workspace import Workspace

from .test_runtime_capability_scope import _Models, _immediate
from .test_tool_response_fixture import _manifest, _store


class _Global(AbstractCapability):
    id = 'policy.global'

    def __init__(self, calls):
        self.calls = calls

    async def before_run(self, ctx):
        self.calls.append(ctx.deps.execution_id)


class _AgentOnly(AbstractCapability):
    id = 'policy.agent'

    async def before_run(self, ctx):
        raise AssertionError('unselected Agent capability must not run')


@pytest.mark.asyncio
@pytest.mark.parametrize('container', ('plain', 'combined', 'wrapped'))
async def test_default_global_runs_with_empty_agent_selection_and_rejecting_filter(container):
    calls = []
    global_policy = _Global(calls)
    if container == 'combined':
        global_policy = CombinedCapability([global_policy], id='policies')
    elif container == 'wrapped':
        global_policy = PrefixTools(CombinedCapability([global_policy]), prefix='policy', id='policies')
    features = RuntimeFeatures('default-global', filter=lambda *_: False)
    for name in ('default', 'worker'):
        features.agent(name, allow_capabilities=(), allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(global_policy)
    features.capability(_AgentOnly(), scope='agent')
    features.model("default", _Models(FunctionModel(_immediate)))
    capture = await features.capture()
    assert next(item for item in capture.contributions if item.id == 'policy.agent').scope == 'agent'
    assert next(item for item in capture.contributions if item.id == global_policy.id).scope == 'runtime'
    async with Runtime.open('global-default',
                            storage=RuntimeStorage.in_memory(), features=capture) as runtime:
        for name in ('default', 'worker'):
            execution = await runtime.agents.get(name).start('question')
            await execution.wait(timeout_seconds=5)
            assert execution.execution_id in calls
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_default_runtime_bootstrap_and_workspace_do_not_globalize_assets(tmp_path):
    from linktools.ai.asset import AssetKey

    store, fixture = await _store(_manifest())
    try:
        await store.put(AssetKey('agent', 'loaded/AGENT.md'), b'---\nmodel: default\n---\n')
        features = RuntimeFeatures('bootstrap')
        features.capability(AssetCapability(store))
        features.capability(WorkspaceCapability(Workspace.load(tmp_path)))
        features.capability(ToolResponsesCapability(fixture))
        capture = await features.capture()
        assert {item.id for item in capture.contributions if item.scope == 'runtime'} == {
            'runtime.assets', 'runtime.workspace', 'runtime.tool_responses',
        }
        assert all(item.scope == 'agent' for item in capture.contributions if item.kind != 'capability')
        assert any(item.kind == 'agent' and item.id == 'loaded' for item in capture.contributions)
        assert any(item.kind == 'tool' for item in capture.contributions)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_explicit_agent_scope_still_rejects_runtime_only_features():
    store, fixture = await _store(_manifest())
    try:
        for capability in (AssetCapability(store), ToolResponsesCapability(fixture)):
            with pytest.raises(AIError) as caught:
                RuntimeFeatures('invalid').capability(capability, scope='agent')
            assert caught.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()
