#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical SQL session turns use the existing skill and MCP tool boundaries."""

from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai_harness.step_persistence import StepPersistence

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import AssetCapability, ExecutionCapability, SkillInput
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageOverlay

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _SessionHostPolicy(TenantAuthorizationPolicy):
    """Explicit policy for the isolated single-tenant test host."""


@asynccontextmanager
async def _open_runtime(tmp_path, features):
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open("canonical-tools", storage=PersistenceStorage.from_root(tmp_path)) as state:
        native = await state.create_step_store()
        try:
            features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
            async with Runtime.open(persistence=state, features=features, authorization=_SessionHostPolicy()) as runtime:
                yield state, runtime
        finally:
            await native.close()


@pytest.mark.asyncio
async def test_native_session_loads_a_skill_and_persists_only_preload_reference(tmp_path):
    backend = InMemoryAssetBackend()
    assets = AssetStore(StorageOverlay(backend, writer=backend))
    await assets.initialize()
    rule = b"---\nname: guide\ndescription: Test guide\n---\nPINNED_RULE"
    try:
        key = AssetKey("skill", "guide/SKILL.md")
        await assets.put(key, rule)
        original, = await assets.resolve_versions((key,))

        async def model(messages, info):
            returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
            if not returns:
                assert any(tool.name == "load_skill" for tool in info.function_tools)
                assert "PINNED_RULE" in str((messages, info))
                await assets.put(key, rule.replace(b"PINNED_RULE", b"CHANGED_RULE"))
                return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "guide"}, "skill")])
            assert returns[-1].content["instructions"].endswith("PINNED_RULE")
            return ModelResponse(parts=[TextPart("done")])

        features = RuntimeFeatures("canonical-skills", workspace_factory=None)
        features.capability(AssetCapability(assets), scope="runtime")
        features.agent("default", allow_tools=(), allow_skills=("guide",), preload_skills=("guide",), allow_subagents=())
        model_features(_Models(_UsageFunctionModel(model)), features)
        async with _open_runtime(tmp_path, features) as (state, runtime):
            actor = runtime.default_principal
            await runtime.sessions.create_turn_session("session", principal=actor, idempotency_key="create")
            turn = await runtime.sessions.start("session", "Load the guide", agent="default", principal=actor, idempotency_key="first")
            result = await turn.wait_final(timeout_seconds=10)
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.view["output"] == {"text": "done"}
            row = await state.session_turn_reader.get_turn(turn.key)
            pinned, = row["request_json"]["preload_skills"]
            assert SkillInput.from_payload(pinned).source.declaration.asset == original
            assert "PINNED_RULE" not in str(row["request_json"])
    finally:
        await assets.close()


@pytest.mark.asyncio
async def test_native_session_executes_selected_mcp_tool_once(tmp_path, monkeypatch):
    from .test_mcp_remote import _RemoteModels, _group, _serve, _spec

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    async with _serve() as remote:
        features = model_features(_RemoteModels(), _group(_spec(remote), workspace_factory=None))
        async with _open_runtime(tmp_path, features) as (_state, runtime):
            actor = runtime.default_principal
            await runtime.sessions.create_turn_session("session", principal=actor, idempotency_key="create")
            first = await runtime.sessions.start("session", "Echo", agent="default", principal=actor, idempotency_key="first")
            result = await first.wait_final(timeout_seconds=10)
            assert result.status is ExecutionStatus.SUCCEEDED
            replay = await runtime.sessions.start("session", "Echo", agent="default", principal=actor, idempotency_key="first")
            assert replay.key == first.key
            assert (await replay.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
            assert len(remote.effects) == 1
