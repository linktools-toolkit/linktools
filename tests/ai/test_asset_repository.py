#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store-backed CapabilityGroup discovery and freeze contract checks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pytest
from linktools.ai.asset import AssetInfo, AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    CapabilityContribution,
    CapabilityGroup,
    capability_fingerprint,
    contribution_semantic_contract,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import AgentSpec, AgentSpecCodec, MCPServerSpec, MCPServerSpecCodec, SkillSpec, SkillSpecCodec
from linktools.ai.storage import StorageOverlay


async def _store() -> AssetStore:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_builtin_loader_freezes_agent_skill_and_mcp_declarations() -> None:
    store = await _store()
    agent = AgentSpec("agent", model="model")
    skill = SkillSpec("skill", "instructions")
    mcp = MCPServerSpec("server", "python", ("-m", "server"))
    await store.put(AssetKey("agent", "agent"), AgentSpecCodec().encode(agent))
    await store.put(AssetKey("skill", "skill"), SkillSpecCodec().encode(skill))
    await store.put(AssetKey("mcp", "server"), MCPServerSpecCodec().encode(mcp))

    frozen = await CapabilityGroup.from_store("workspace", store).freeze()

    assert [(item.kind, item.id) for item in frozen] == [
        ("agent", "agent"),
        ("mcp", "server"),
        ("skill", "skill"),
    ]
    assert [item.value for item in frozen] == [agent, mcp, skill]
    assert all("semantic_revision" not in item.semantic_contract for item in frozen)


@pytest.mark.asyncio
async def test_builtin_loader_rejects_declaration_identity_mismatch() -> None:
    store = await _store()
    await store.put(
        AssetKey("agent", "expected"),
        AgentSpecCodec().encode(AgentSpec("actual", model="model")),
    )

    with pytest.raises(AIError) as error:
        await CapabilityGroup.from_store("workspace", store).freeze()

    assert error.value.code is ErrorCode.ASSET_CONTENT_MISMATCH


@pytest.mark.asyncio
async def test_store_group_requires_initialized_asset_store() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))

    with pytest.raises(AIError) as error:
        await CapabilityGroup.from_store("workspace", store).freeze()

    assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


class _CapturingLoader:
    def __init__(self) -> None:
        self.calls = 0
        self.entries: tuple[AssetInfo, ...] = ()
        self.contents: dict[AssetKey, bytes] = {}

    @property
    def id(self) -> str:
        return "capture"

    async def load(
        self,
        entries: Sequence[AssetInfo],
        contents: Mapping[AssetKey, bytes],
    ) -> "Sequence[CapabilityContribution[object]]":
        self.calls += 1
        self.entries = tuple(entries)
        self.contents = dict(contents)
        return ()


@pytest.mark.asyncio
async def test_custom_loader_receives_one_frozen_metadata_and_content_snapshot() -> None:
    store = await _store()
    await store.put(AssetKey("custom", "a"), b"a")
    await store.put(AssetKey("custom", "b"), b"b")
    loader = _CapturingLoader()
    group = CapabilityGroup.from_store("workspace", store)
    group.loader(loader)

    assert await group.freeze() == ()
    assert loader.calls == 1
    assert [entry.key for entry in loader.entries] == [AssetKey("custom", "a"), AssetKey("custom", "b")]
    assert loader.contents == {
        AssetKey("custom", "a"): b"a",
        AssetKey("custom", "b"): b"b",
    }


class _MutatingLoader:
    @property
    def id(self) -> str:
        return "mutating"

    async def load(
        self,
        entries: Sequence[AssetInfo],
        contents: Mapping[AssetKey, bytes],
    ) -> "Sequence[CapabilityContribution[object]]":
        del entries
        cast("dict[AssetKey, bytes]", contents)[AssetKey("custom", "a")] = b"changed"
        return ()


@pytest.mark.asyncio
async def test_custom_loader_cannot_mutate_shared_content_snapshot() -> None:
    store = await _store()
    await store.put(AssetKey("custom", "a"), b"a")
    group = CapabilityGroup.from_store("workspace", store)
    group.loader(_MutatingLoader())

    with pytest.raises(TypeError):
        await group.freeze()


@dataclass
class _DuplicateAgentLoader:
    @property
    def id(self) -> str:
        return "duplicate-agent"

    async def load(
        self,
        entries: Sequence[AssetInfo],
        contents: Mapping[AssetKey, bytes],
    ) -> "Sequence[CapabilityContribution[object]]":
        del entries, contents
        spec = AgentSpec("agent", model="other-model")
        contract = contribution_semantic_contract("agent", spec.id, spec)
        return (
            CapabilityContribution(
                "agent",
                spec.id,
                capability_fingerprint("agent", spec.id, contract),
                spec,
            ),
        )


@pytest.mark.asyncio
async def test_duplicate_candidate_identity_is_rejected_after_all_loaders_finish() -> None:
    store = await _store()
    await store.put(
        AssetKey("agent", "agent"),
        AgentSpecCodec().encode(AgentSpec("agent", model="model")),
    )
    group = CapabilityGroup.from_store("workspace", store)
    group.loader(_DuplicateAgentLoader())

    with pytest.raises(AIError) as error:
        await group.freeze()

    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


class _RaceStore(AssetStore):
    def __init__(self, backend: InMemoryAssetBackend) -> None:
        super().__init__(StorageOverlay(backend, writer=backend))
        self._raced = False

    async def get_many(self, keys: Sequence[AssetKey]) -> "tuple[bytes | None, ...]":
        values = await super().get_many(keys)
        if not self._raced:
            self._raced = True
            await self.put(AssetKey("skill", "late"), SkillSpecCodec().encode(SkillSpec("late", "late")))
        return values


@pytest.mark.asyncio
async def test_freeze_rejects_store_revision_change_during_snapshot() -> None:
    backend = InMemoryAssetBackend()
    store = _RaceStore(backend)
    await store.initialize()
    await store.put(AssetKey("skill", "first"), SkillSpecCodec().encode(SkillSpec("first", "first")))

    with pytest.raises(AIError) as error:
        await CapabilityGroup.from_store("workspace", store).freeze()

    assert error.value.code is ErrorCode.STORAGE_CONFLICT


def test_capability_group_does_not_expose_logical_asset_crud() -> None:
    group = CapabilityGroup[object]("group")
    assert not hasattr(group, "resolve")
    assert not hasattr(group, "put")
    assert not hasattr(group, "delete")
    assert not hasattr(group, "list")
