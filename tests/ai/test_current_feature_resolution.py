#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Every capability source participates in one current, versioned resolution."""

import asyncio
import json
from dataclasses import replace

import pytest
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    AssetCapability,
    CapabilityContribution,
    CapabilityLoadContext,
    SkillDefinition,
    SkillResource,
    SkillSourceRef,
    tool_metadata,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.spec import AgentSpec, MCPServerSpec, MCPServerSpecCodec, RepositoryInstructionDocument, SkillSpec
from linktools.ai.storage import StorageOverlay


class _Behavior(AbstractCapability):
    def __init__(self, name: str) -> None:
        self.id = name


def _entries(name: str, revision: int):
    tool = Tool(lambda: revision, name=name, metadata=tool_metadata(effect_policy="replay_safe", tool_class="business"))
    return (
        AgentSpec(name, system_prompt=f"Instructions {revision}", revision=revision),
        CapabilityContribution.from_opaque("tool", name, tool, revision=revision),
        MCPServerSpec(name, transport="sse", url=f"https://example.test/{revision}", revision=revision),
        SkillDefinition(SkillSpec(name, f"Skill {revision}", revision=revision)),
        CapabilityContribution.from_opaque("capability", name, _Behavior(name), revision=revision),
        RepositoryInstructionDocument(f"rule:{name}", ".", f"Rule {revision}"),
    )


class _MixedLoader:
    async def load(self, context: CapabilityLoadContext):
        result = []
        for entry in context.list(kind="custom"):
            revision = int(await context.read(entry.key))
            result.extend(_entries(entry.key.id, revision))
        return tuple(result)


async def _store() -> AssetStore:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    return store


def _features(store: AssetStore):
    features = RuntimeFeatures("current")
    assets = AssetCapability(store)
    features.capability(assets)
    return features, assets


def _selected(resolved):
    return tuple(item for item in resolved.contributions if item.id != "runtime.assets")


@pytest.mark.asyncio
async def test_mixed_loader_output_additions_replacements_and_removals_are_current() -> None:
    store = await _store()
    try:
        features, assets = _features(store)
        assets.loader("custom", _MixedLoader())
        empty = await features.resolve()
        assert _selected(empty) == ()
        for name, revision in (("first", 1), ("second", 2)):
            await store.put(AssetKey("custom", name), str(revision).encode())
        first = await features.resolve()
        assert {(item.kind, item.id, item.revision) for item in _selected(first)} == {
            (kind, name, revision)
            for kind in ("agent", "tool", "mcp", "skill", "capability")
            for name, revision in (("first", 1), ("second", 2))
        }
        await store.put(AssetKey("custom", "first"), b"3")
        await store.delete(AssetKey("custom", "second"))
        current = await features.resolve()
        assert {(item.kind, item.id, item.revision) for item in _selected(current)} == {
            (kind, "first", 3) for kind in ("agent", "tool", "mcp", "skill", "capability")
        }
        assert current.instructions.documents == (RepositoryInstructionDocument("rule:first", ".", "Rule 3"),)
        assert first.instructions.documents[0].content == "Rule 1"
        assert {item.revision for item in _selected(first)} == {1, 2}
        await store.delete(AssetKey("custom", "first"))
        removed = await features.resolve()
        assert _selected(removed) == ()
        assert removed.instructions.documents == ()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_builtin_loader_sources_change_without_replacing_the_provider() -> None:
    store = await _store()
    try:
        features, _ = _features(store)
        keys = (
            AssetKey("agent", "worker/AGENT.md"),
            AssetKey("skill", "guide/SKILL.md"),
            AssetKey("mcp", "server/mcp.json"),
            AssetKey("rule", "base.md"),
        )
        assert _selected(await features.resolve()) == ()
        for revision in (1, 2):
            for key, value in zip(keys, (
                f"---\nrevision: {revision}\n---\nAgent {revision}",
                f"---\nname: guide\ndescription: Guide\nrevision: {revision}\n---\nSkill {revision}",
                json.dumps({"url": f"https://example.test/{revision}", "revision": revision}),
                f"Rule {revision}",
            ), strict=True):
                await store.put(key, value.encode())
            current = await features.resolve()
            assert {(item.kind, item.id) for item in _selected(current)} == {
                ("agent", "worker"), ("skill", "guide"), ("mcp", "server"),
            }
            assert str(revision) in next(item.value.system_prompt for item in _selected(current) if item.kind == "agent")
            assert str(revision) in next(item.value.model_content for item in _selected(current) if item.kind == "skill")
            assert current.instructions.documents[0].content == f"Rule {revision}"
        for key in keys:
            await store.delete(key)
        assert _selected(await features.resolve()) == ()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_resolution_keeps_its_registrations_routes_and_loader_slots() -> None:
    store = await _store()
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        features, assets = _features(store)
        features.agent("initial")
        old_model = lambda: (_ for _ in ()).throw(AssertionError("must remain deferred"))
        new_model = lambda: (_ for _ in ()).throw(AssertionError("must remain deferred"))
        features.model("default", old_model)

        class BlockingLoader:
            async def load(self, context):
                entered.set()
                await release.wait()
                return _entries("old", 1)

        class ReplacementLoader:
            async def load(self, context):
                return _entries("new", 2)

        assets.loader("custom", BlockingLoader())
        pending = asyncio.create_task(features.resolve())
        await entered.wait()
        features.agent("added")
        features.model("default", new_model)
        features.model("added", new_model)
        assets.loader("custom", ReplacementLoader())

        class AddedLoader:
            async def load(self, context):
                return (AgentSpec("from-added-slot"),)

        assets.loader("additional", AddedLoader())
        release.set()
        first = await pending
        assert {item.id for item in _selected(first)} == {"initial", "old"}
        assert dict(first.models) == {"default": old_model}
        current = await features.resolve()
        assert {item.id for item in _selected(current)} == {"initial", "added", "new", "from-added-slot"}
        assert dict(current.models) == {"default": new_model, "added": new_model}
    finally:
        release.set()
        await store.close()


@pytest.mark.asyncio
async def test_one_resolution_reads_consistent_versions_despite_source_changes() -> None:
    store = await _store()
    try:
        features, assets = _features(store)
        resource = AssetKey("package", "shared/data.txt")
        await store.put(resource, b"first")
        features.mcp(MCPServerSpec("manual", "python", ("resource:data.txt",), AssetKey("package", "shared")))

        class ChangingLoader:
            changed = False

            async def load(self, context):
                before = await context.read(resource)
                if not self.changed:
                    self.changed = True
                    await store.put(resource, b"second")
                    await store.put(AssetKey("agent", "later/AGENT.md"), b"---\nmodel: default\n---\nLater")
                    await store.put(AssetKey("mcp", "later/mcp.json"), b'{"url":"https://example.test/later"}')
                assert await context.read(resource) == before
                assert await context.asset_reader.get(resource) == before
                reference, = context.bind_versions((resource,))
                return (SkillDefinition(
                    SkillSpec("mixed-skill", before.decode()),
                    SkillSourceRef(context.group_id, "shared", (SkillResource("data.txt", reference),)),
                ),)

        assets.loader("package", ChangingLoader())
        first = await features.resolve()
        skill = next(item.value for item in first.contributions if item.kind == "skill")
        assert skill.spec.content == "first"
        assert await first.asset_reader.read_versions((skill.source_ref.resources[0].asset,)) == (b"first",)
        mcp = next(item for item in first.contributions if item.kind == "mcp")
        refs = MCPServerSpecCodec().decode_binding_payload(mcp.contract, declaration=mcp.value)
        assert await first.asset_reader.read_versions(refs) == (b"first",)
        assert "later" not in {item.id for item in first.contributions}
        current = await features.resolve()
        assert {(item.kind, item.id) for item in current.contributions} >= {("agent", "later"), ("mcp", "later")}
        assert next(item.value.spec.content for item in current.contributions if item.kind == "skill") == "second"
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("skill", "mcp"))
@pytest.mark.parametrize("invalid", ("wrong_source", "stale", "unknown_key", "forged_digest"))
async def test_loader_resource_refs_must_belong_to_the_current_resolution(kind: str, invalid: str) -> None:
    store = await _store()
    try:
        features, assets = _features(store)
        resource = AssetKey("package", "shared/data.txt")
        await store.put(resource, b"first")
        ref, = await store.resolve_versions((resource,))
        if invalid == "stale":
            await store.put(resource, b"second")
        elif invalid == "unknown_key":
            ref = replace(ref, key=AssetKey("package", "shared/missing.txt"))
        elif invalid == "forged_digest":
            ref = replace(ref, digest="0" * 64)
        source = "another" if invalid == "wrong_source" else features.id

        class InvalidLoader:
            async def load(self, context):
                if kind == "skill":
                    return (SkillDefinition(SkillSpec("guide", "Guide"), SkillSourceRef(
                        source, "shared", (SkillResource("data.txt", ref),),
                    )),)
                spec = MCPServerSpec("server", "python", (), AssetKey("package", "shared"))
                return (CapabilityContribution.from_mcp_contract(
                    MCPServerSpecCodec().to_binding_payload(spec, (ref,), asset_source_id=source), spec,
                ),)

        assets.loader("package", InvalidLoader())
        with pytest.raises(AIError) as error:
            await features.resolve()
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_resolution_replaces_registered_agent_without_changing_previous_result() -> None:
    features = RuntimeFeatures("current")
    features.agent("same", instructions=("First",))
    first = await features.resolve()
    features.agent("same", instructions=("Current",), revision=2)
    current = await features.resolve()
    assert len(first.contributions) == len(current.contributions) == 1
    assert first.contributions[0].revision == 1
    assert current.contributions[0].revision == 2
    assert first.contributions[0].value.instructions == ("First",)
    assert current.contributions[0].value.instructions == ("Current",)


@pytest.mark.asyncio
async def test_direct_registrations_added_after_resolution_remain_available() -> None:
    features = RuntimeFeatures("current")
    first = await features.resolve()

    def business(context: object) -> str:
        return "current"

    features.agent("worker")
    features.tool(business)
    features.mcp(MCPServerSpec("server", transport="sse", url="https://example.test/mcp"))
    features.capability(_Behavior("behavior"), scope="agent")
    current = await features.resolve()
    assert first.contributions == ()
    assert {(item.kind, item.id) for item in current.contributions} == {
        ("agent", "worker"), ("tool", "business"), ("mcp", "server"), ("capability", "behavior"),
    }


@pytest.mark.asyncio
async def test_loader_host_capabilities_use_the_same_current_tool_projection(tmp_path) -> None:
    from linktools.ai.capability import FileSystem, MemoryCapability, ShellCapability
    from linktools.ai.workspace import Workspace

    current = [MemoryCapability(scope="first"), ShellCapability(), FileSystem(Workspace.load(tmp_path))]

    class Loader:
        async def load(self, context):
            return tuple(CapabilityContribution.from_opaque("capability", item.id, item, scope="runtime")
                         for item in current)

    store = await _store()
    try:
        features, assets = _features(store)
        assets.loader("custom", Loader())
        resolved = await features.resolve()
        expected = {tool.name for capability in current for tool in capability.tool_definitions()}
        assert {item.id for item in resolved.contributions if item.kind == "tool"} == expected
        assert resolved.workspace.root == tmp_path
        current[:] = [MemoryCapability(scope="second")]
        changed = await features.resolve()
        assert {item.id for item in changed.contributions if item.kind == "tool"} == {
            tool.name for tool in current[0].tool_definitions()
        }
        assert changed.workspace is None
        current.clear()
        assert not any(item.kind == "tool" for item in (await features.resolve()).contributions)
    finally:
        await store.close()
