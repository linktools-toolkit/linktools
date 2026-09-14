#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store-backed CapabilityGroup discovery and freeze contract checks."""

from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    CapabilityContribution,
    CapabilityGroup,
    CapabilityLoadContext,
    SkillDefinition,
)
from linktools.ai.capability._group import (
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
    assert [item.value for item in frozen] == [agent, mcp, SkillDefinition(skill)]
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
        self.entries: tuple[object, ...] = ()
        self.read_value: bytes | None = None

    @property
    def id(self) -> str:
        return "capture"

    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[object]]":
        self.calls += 1
        self.entries = context.list()
        self.read_value = await context.read(AssetKey("custom", "a"))
        return ()


@pytest.mark.asyncio
async def test_custom_loader_receives_frozen_metadata_and_reads_explicit_keys_only() -> None:
    store = await _store()
    await store.put(AssetKey("custom", "a"), b"a")
    await store.put(AssetKey("custom", "b"), b"b")
    loader = _CapturingLoader()
    group = CapabilityGroup.from_store("workspace", store)
    group.loader(loader)

    assert await group.freeze() == ()
    assert loader.calls == 1
    assert [entry.key for entry in loader.entries] == [AssetKey("custom", "a"), AssetKey("custom", "b")]
    assert loader.read_value == b"a"


class _OutsideSnapshotLoader:
    @property
    def id(self) -> str:
        return "outside-snapshot"

    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[object]]":
        await context.read(AssetKey("custom", "missing"))
        return ()


@pytest.mark.asyncio
async def test_custom_loader_cannot_read_key_outside_frozen_metadata() -> None:
    store = await _store()
    await store.put(AssetKey("custom", "a"), b"a")
    group = CapabilityGroup.from_store("workspace", store)
    group.loader(_OutsideSnapshotLoader())

    with pytest.raises(AIError) as error:
        await group.freeze()
    assert error.value.code is ErrorCode.STORAGE_CONFLICT


@dataclass
class _DuplicateAgentLoader:
    @property
    def id(self) -> str:
        return "duplicate-agent"

    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[object]]":
        del context
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

    async def get_many(
        self,
        keys: "Sequence[AssetKey]",
    ) -> "tuple[bytes | None, ...]":
        values = await super().get_many(keys)
        if not self._raced:
            self._raced = True
            await self.put(AssetKey("skill", "late"), SkillSpecCodec().encode(SkillSpec("late", "late")))
        return values


@pytest.mark.asyncio
async def test_freeze_ignores_assets_added_after_the_captured_snapshot() -> None:
    backend = InMemoryAssetBackend()
    store = _RaceStore(backend)
    await store.initialize()
    await store.put(AssetKey("skill", "first"), SkillSpecCodec().encode(SkillSpec("first", "first")))

    frozen = await CapabilityGroup.from_store("workspace", store).freeze()

    assert [item.id for item in frozen] == ["first"]


class _BatchReadStore(AssetStore):
    def __init__(self, backend: InMemoryAssetBackend) -> None:
        super().__init__(StorageOverlay(backend, writer=backend))
        self.batch_reads: list[tuple[AssetKey, ...]] = []
        self.individual_reads = 0

    async def get_many(
        self,
        keys: "Sequence[AssetKey]",
    ) -> "tuple[bytes | None, ...]":
        self.batch_reads.append(tuple(keys))
        return await super().get_many(keys)

    async def get(self, key: AssetKey) -> "bytes | None":
        self.individual_reads += 1
        return await super().get(key)

    async def stat(self, key: AssetKey):
        self.individual_reads += 1
        return await super().stat(key)


@pytest.mark.asyncio
async def test_builtin_loader_batches_declaration_body_reads() -> None:
    backend = InMemoryAssetBackend()
    store = _BatchReadStore(backend)
    await store.initialize()
    await store.put(
        AssetKey("agent", "agent"),
        AgentSpecCodec().encode(AgentSpec("agent", model="model")),
    )
    await store.put(
        AssetKey("mcp", "server"),
        MCPServerSpecCodec().encode(MCPServerSpec("server", "python", ("-m", "server"))),
    )
    await store.put(
        AssetKey("skill", "skill"),
        SkillSpecCodec().encode(SkillSpec("skill", "instructions")),
    )

    frozen = await CapabilityGroup.from_store("workspace", store).freeze()

    assert [item.id for item in frozen] == ["agent", "server", "skill"]
    assert len(store.batch_reads) == 1
    assert set(store.batch_reads[0]) == {
        AssetKey("agent", "agent"),
        AssetKey("mcp", "server"),
        AssetKey("skill", "skill"),
    }
    assert store.individual_reads == 0


def test_capability_group_does_not_expose_logical_asset_crud() -> None:
    group = CapabilityGroup[object]("group")
    assert not hasattr(group, "resolve")
    assert not hasattr(group, "put")
    assert not hasattr(group, "delete")
    assert not hasattr(group, "list")
