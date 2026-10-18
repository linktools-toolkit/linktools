#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset loading is owned by one explicit Runtime-scoped capability."""

import asyncio
from dataclasses import replace

import pytest
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, PrefixTools

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import AssetCapability, CapabilityContribution, RuntimeFeatures
from linktools.ai.errors import AIError
from linktools.ai.spec import AgentSpec, MCPServerSpec, MCPServerSpecCodec
from linktools.ai.storage import StorageOverlay


async def _store():
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    return store


def _features(store, *, capability=None):
    features = RuntimeFeatures("source")
    assets = AssetCapability(store) if capability is None else capability
    features.capability(assets, scope="runtime")
    return features, assets


@pytest.mark.asyncio
async def test_asset_capability_is_the_single_loading_configuration():
    store = await _store()
    try:
        with pytest.raises(TypeError):
            RuntimeFeatures("old", assets=store)
        features = RuntimeFeatures("source")
        assert not hasattr(features, "loader") and not hasattr(features, "assets")
        assets = AssetCapability(store)
        with pytest.raises(AIError, match="requires Runtime scope"):
            features.capability(assets, scope="agent")
        features.capability(assets, scope="runtime")
        with pytest.raises(AIError, match="CAPABILITY_CONFLICT"):
            features.capability(AssetCapability(store), scope="runtime")
        assert (await features.capture()).asset_reader is not None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_asset_capability_captures_builtin_and_future_kinds_at_one_revision():
    store = await _store()
    try:
        await store.put(AssetKey("agent", "root/AGENT.md"), b"---\nmodel: default\n---\n")
        await store.put(AssetKey("skill", "guide/SKILL.md"), b"---\nname: guide\ndescription: Guide\n---\nOld instructions")
        await store.put(AssetKey("mcp", "server/mcp.json"), b'{"command":"python"}')
        await store.put(AssetKey("future-kind", "opaque:id"), b"custom bytes")
        await store.put(AssetKey("rule", "base.md"), b"captured rule")
        features, assets = _features(store)
        contexts = []

        class Loader:
            source_kind = "future-kind"

            async def load(self, context):
                contexts.append(context)
                assert await context.read(AssetKey("future-kind", "opaque:id")) == b"custom bytes"
                return (AgentSpec("custom"),)

        assets.loader("future-kind", Loader())
        capture = await features.capture()
        identities = {(item.kind, item.id) for item in capture.contributions}
        assert {("agent", "root"), ("agent", "custom"), ("skill", "guide"), ("mcp", "server")} <= identities
        assert len(contexts) == 1 and contexts[0].source_revision == capture.source_revision
        assert len(capture.instructions.documents) == 1
        assert capture.workspace is None
        skill = next(item.value for item in capture.contributions if item.kind == "skill")
        assert skill.source_ref.source_id == "source"
        filtered = replace(capture, filter=lambda *_: True)
        assert filtered.asset_reader is capture.asset_reader
        await store.put(AssetKey("skill", "guide/SKILL.md"), b"---\nname: guide\ndescription: Guide\n---\nNew instructions")
        assert b"Old instructions" in await capture.asset_reader.get(AssetKey("skill", "guide/SKILL.md"))
        with pytest.raises(AIError, match="SNAPSHOT_CONFLICT"):
            await capture.verify_source_revision()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_asset_capability_loader_snapshot_and_same_capture_handwritten_mcp():
    store = await _store()
    try:
        await store.put(AssetKey("mcp", "manual/server.py"), b"original")
        entered, release = asyncio.Event(), asyncio.Event()
        features, assets = _features(store)
        calls = []

        class Loader:
            async def load(self, context):
                entered.set()
                await release.wait()
                calls.append(context.group_id)
                return ()

        class Replacement:
            async def load(self, context):
                raise AssertionError("new loader entered earlier capture")

        assets.loader("custom", Loader())
        features.mcp(MCPServerSpec("manual", "python", ("resource:server.py",), AssetKey("mcp", "manual")))
        task = asyncio.create_task(features.capture())
        await entered.wait()
        assets.loader("custom", Replacement())
        release.set()
        capture = await task
        assert calls == ["source"]
        item = next(item for item in capture.contributions if item.kind == "mcp")
        versions = MCPServerSpecCodec().decode_binding_payload(item.contract, declaration=item.value)
        assert item.contract["asset_source_id"] == "source"
        assert await capture.asset_reader.read_versions(versions) == (b"original",)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_asset_capability_preserves_wrapped_combination_registration():
    store = await _store()
    try:
        class Other(AbstractCapability):
            id = "other"
        combined = CombinedCapability([AssetCapability(store), Other()], id="asset.bundle")
        features, _ = _features(store, capability=PrefixTools(combined, prefix="wrapped"))
        capture = await features.capture()
        assert capture.asset_reader is not None
        assert capture.workspace is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_asset_capability_rejects_loader_revision_drift_and_declaration_collisions():
    store = await _store()
    try:
        features, assets = _features(store)

        class MutatingLoader:
            async def load(self, context):
                await store.put(AssetKey("custom", "mutation"), b"changed")
                return ()

        assets.loader("custom", MutatingLoader())
        with pytest.raises(AIError, match="SNAPSHOT_CONFLICT"):
            await features.capture()

        class CollisionLoader:
            async def load(self, context):
                return (AgentSpec("duplicate"),)

        assets.loader("custom", CollisionLoader())
        features.agent("duplicate")
        with pytest.raises(AIError, match="CAPABILITY_CONFLICT"):
            await features.capture()
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", (False, True))
async def test_asset_loader_cannot_introduce_an_uncaptured_asset_source(wrapped):
    store = await _store()
    try:
        features, assets = _features(store)

        class Loader:
            async def load(self, context):
                nested = AssetCapability(store)
                if wrapped:
                    nested = CombinedCapability([nested.prefix_tools("nested")], id="nested.assets")
                return (CapabilityContribution.from_opaque(
                    "capability", nested.id, nested, scope="runtime",
                ),)

        assets.loader("custom", Loader())
        with pytest.raises(AIError, match="cannot introduce AssetCapability"):
            await features.capture()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_asset_bootstrap_is_consumed_before_native_agent_hooks():
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.runtime import Runtime, RuntimeStorage
    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    store = await _store()
    try:
        await store.put(AssetKey("agent", "default/AGENT.md"), b"---\nmodel: default\n---\n")

        class Bootstrap(AssetCapability):
            def get_instructions(self):
                raise AssertionError("bootstrap reached native Agent hooks")

        native_calls = []

        class Native(AbstractCapability):
            id = "native.instructions"

            def get_instructions(self):
                native_calls.append(True)
                return "retained native instructions"

        capability = CombinedCapability([
            Bootstrap(store).prefix_tools("asset"), Native(),
        ], id="assets.and.native").prefix_tools("outer")
        features, _ = _features(store, capability=capability)
        capture = await features.capture()

        async def answer(messages, info):
            return ModelResponse(parts=[TextPart("answer")])

        async with Runtime.open("asset-bootstrap", features=capture,
                                models=_Models(_UsageFunctionModel(answer)),
                                storage=RuntimeStorage.in_memory()) as runtime:
            assert (await runtime.agents.get().run("question")).result.output == {"text": "answer"}
            assert native_calls
    finally:
        await store.close()
