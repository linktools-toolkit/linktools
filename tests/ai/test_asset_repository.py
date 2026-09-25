#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store-backed CapabilityGroup discovery and snapshot contract checks."""

from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from linktools.ai.asset import (
    AssetKey,
    AssetStore,
    AssetStoreReader,
    AssetVersionRef,
    InMemoryAssetBackend,
)
from linktools.ai.capability import (
    CapabilityContribution,
    CapabilityGroup,
    CapabilityLoadContext,
    SkillDefinition,
    SkillResourceVersion,
    SkillSourceRef,
)
from linktools.ai.capability._group import contribution_contract
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import AgentSpec, AgentSpecCodec, MCPServerSpec, MCPServerSpecCodec, SkillSpec, SkillSpecCodec
from linktools.ai.storage import (
    InMemoryObjectStore,
    StorageEntryRevision,
    StorageOverlay,
)


async def _store() -> AssetStore:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_builtin_loader_snapshots_agent_skill_and_mcp_declarations() -> None:
    store = await _store()
    agent = AgentSpec("agent", model_route="model")
    skill = SkillSpec("skill", "instructions")
    mcp = MCPServerSpec("server", "python", ("-m", "server"))
    await store.put(AssetKey("agent", "agent"), AgentSpecCodec().encode(agent))
    await store.put(AssetKey("skill", "skill"), SkillSpecCodec().encode(skill))
    await store.put(AssetKey("mcp", "server"), MCPServerSpecCodec().encode(mcp))

    snapshot = await CapabilityGroup("workspace", assets=store).capture()

    assert [(item.kind, item.id) for item in snapshot.contributions] == [
        ("agent", "agent"),
        ("mcp", "server"),
        ("skill", "skill"),
    ]
    assert [item.value for item in snapshot.contributions] == [
        agent,
        mcp,
        SkillDefinition(skill),
    ]
    assert all(
        "revision" in item.contract
        for item in snapshot.contributions
    )


@pytest.mark.asyncio
async def test_group_snapshot_exposes_only_read_only_asset_access() -> None:
    store = await _store()
    key = AssetKey("custom", "file")
    await store.put(key, b"contents")

    snapshot = await CapabilityGroup("workspace", assets=store).capture()
    reader = snapshot.asset_reader

    assert isinstance(reader, AssetStoreReader)
    assert reader is not None
    assert not hasattr(snapshot, "asset_store")
    assert not hasattr(reader, "put")
    assert await reader.get(key) == b"contents"
    frozen_version = (await reader.resolve_versions((key,)))[0]

    await store.put(key, b"changed")
    with pytest.raises(AIError) as error:
        await reader.get(key)
    assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT
    assert await reader.read_versions((frozen_version,)) == (b"contents",)


@pytest.mark.asyncio
async def test_asset_version_ref_reads_exact_historical_content() -> None:
    store = await _store()
    key = AssetKey("custom", "file")
    try:
        await store.put(key, b"first")
        first = (await store.resolve_versions((key,)))[0]
        await store.put(key, b"second")
        second = (await store.resolve_versions((key,)))[0]

        assert first != second
        assert first.layer_id == second.layer_id == "primary"
        assert await store.read_versions((first, second)) == (
            b"first",
            b"second",
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_asset_snapshot_rejects_wrong_object_store_owner() -> None:
    store = await _store()
    objects = InMemoryObjectStore("owner")
    try:
        key = AssetKey("custom", "file")
        await store.put(key, b"contents")
        reference = await store.snapshot((key,), object_store=objects)
        with pytest.raises(AIError) as error:
            AssetStore.from_snapshot(
                reference,
                object_store=InMemoryObjectStore("other"),
            )
        assert error.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_builtin_loader_rejects_declaration_identity_mismatch() -> None:
    store = await _store()
    await store.put(
        AssetKey("agent", "expected"),
        AgentSpecCodec().encode(AgentSpec("actual", model_route="model")),
    )

    with pytest.raises(AIError) as error:
        await CapabilityGroup("workspace", assets=store).capture()

    assert error.value.code is ErrorCode.ASSET_CONTENT_MISMATCH


@pytest.mark.asyncio
async def test_store_group_requires_initialized_asset_store() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))

    with pytest.raises(AIError) as error:
        await CapabilityGroup("workspace", assets=store).capture()

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
async def test_custom_loader_receives_snapshot_metadata_and_reads_explicit_keys_only() -> None:
    store = await _store()
    await store.put(AssetKey("custom", "a"), b"a")
    await store.put(AssetKey("custom", "b"), b"b")
    loader = _CapturingLoader()
    group = CapabilityGroup("workspace", assets=store)
    group.loader("custom", loader)

    assert (await group.capture()).contributions == ()
    assert loader.calls == 1
    assert [entry.key for entry in loader.entries] == [AssetKey("custom", "a"), AssetKey("custom", "b")]
    assert loader.read_value == b"a"


class _NoopLoader:
    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[object]]":
        del context
        return ()


@pytest.mark.asyncio
async def test_replacing_skill_loader_disables_builtin_skill_layout_validation() -> None:
    store = await _store()
    await store.put(AssetKey("skill", "a/SKILL.md"), b"custom")
    await store.put(AssetKey("skill", "a/x/SKILL.md"), b"custom")
    group = CapabilityGroup("workspace", assets=store)
    group.loader("skill", _NoopLoader())

    assert (await group.capture()).contributions == ()


class _ForeignSkillSourceLoader:
    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[object]]":
        del context
        return (
            CapabilityContribution.from_declaration(
                SkillDefinition(
                    SkillSpec("review", "review"),
                    SkillSourceRef("other", "review"),
                )
            ),
        )


@pytest.mark.asyncio
async def test_custom_loader_cannot_bind_skill_resources_to_another_group() -> None:
    store = await _store()
    group = CapabilityGroup("application", assets=store)
    group.loader("skill", _ForeignSkillSourceLoader())

    with pytest.raises(AIError) as error:
        await group.capture()

    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


class _PinnedSkillVersionLoader:
    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[object]]":
        return (
            CapabilityContribution.from_declaration(
                SkillDefinition(
                    SkillSpec("review", "review"),
                    SkillSourceRef(context.group_id, "review").with_asset_versions(
                        (
                            SkillResourceVersion(
                                "guide.md",
                                AssetVersionRef(
                                    AssetKey("skill", "review/guide.md"),
                                    "source",
                                    StorageEntryRevision(1),
                                    "0" * 64,
                                    1,
                                ),
                            ),
                        ),
                        "1" * 64,
                    ),
                )
            ),
        )


@pytest.mark.asyncio
async def test_custom_loader_cannot_prebind_skill_versions() -> None:
    store = await _store()
    group = CapabilityGroup("application", assets=store)
    group.loader("skill", _PinnedSkillVersionLoader())

    with pytest.raises(AIError) as error:
        await group.capture()

    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


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
async def test_custom_loader_cannot_read_key_outside_snapshot_metadata() -> None:
    store = await _store()
    await store.put(AssetKey("custom", "a"), b"a")
    group = CapabilityGroup("workspace", assets=store)
    group.loader("custom", _OutsideSnapshotLoader())

    with pytest.raises(AIError) as error:
        await group.capture()
    assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT


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
        spec = AgentSpec("agent", model_route="other-model")
        contract = contribution_contract("agent", spec.id, spec)
        return (
            CapabilityContribution(
                "agent",
                spec.id,
                spec,
            ),
        )


@pytest.mark.asyncio
async def test_duplicate_candidate_identity_is_rejected_after_all_loaders_finish() -> None:
    store = await _store()
    await store.put(
        AssetKey("agent", "agent"),
        AgentSpecCodec().encode(AgentSpec("agent", model_route="model")),
    )
    group = CapabilityGroup("workspace", assets=store)
    group.loader("custom", _DuplicateAgentLoader())

    with pytest.raises(AIError) as error:
        await group.capture()

    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


class _RaceStore(AssetStore):
    def __init__(self, backend: InMemoryAssetBackend) -> None:
        super().__init__(StorageOverlay(backend, writer=backend))
        self._raced = False

    async def read_versions(
        self,
        refs: "Sequence[AssetVersionRef]",
    ) -> "tuple[bytes, ...]":
        values = await super().read_versions(refs)
        if not self._raced:
            self._raced = True
            await self.put(AssetKey("skill", "late"), SkillSpecCodec().encode(SkillSpec("late", "late")))
        return values


@pytest.mark.asyncio
async def test_snapshot_rejects_assets_added_during_declaration_loading() -> None:
    backend = InMemoryAssetBackend()
    store = _RaceStore(backend)
    await store.initialize()
    await store.put(AssetKey("skill", "first"), SkillSpecCodec().encode(SkillSpec("first", "first")))

    with pytest.raises(AIError) as error:
        await CapabilityGroup("workspace", assets=store).capture()

    assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT


@pytest.mark.asyncio
async def test_group_snapshot_rejects_source_changes_before_admission() -> None:
    store = await _store()
    await store.put(
        AssetKey("agent", "agent"),
        AgentSpecCodec().encode(AgentSpec("agent", model_route="model")),
    )
    snapshot = await CapabilityGroup("workspace", assets=store).capture()
    await store.put(AssetKey("other", "late"), b"changed")

    with pytest.raises(AIError) as error:
        await snapshot.verify_source_revision()

    assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT
    updated = await CapabilityGroup("workspace", assets=store).capture()
    assert updated.source_revision != snapshot.source_revision


class _BatchReadStore(AssetStore):
    def __init__(self, backend: InMemoryAssetBackend) -> None:
        super().__init__(StorageOverlay(backend, writer=backend))
        self.version_reads: list[tuple[AssetVersionRef, ...]] = []
        self.individual_reads = 0

    async def read_versions(
        self,
        refs: "Sequence[AssetVersionRef]",
    ) -> "tuple[bytes, ...]":
        self.version_reads.append(tuple(refs))
        return await super().read_versions(refs)

    async def get(self, key: AssetKey) -> "bytes | None":
        self.individual_reads += 1
        return await super().get(key)

    async def stat(self, key: AssetKey):
        self.individual_reads += 1
        return await super().stat(key)


@pytest.mark.asyncio
async def test_asset_snapshot_reads_captured_versions_for_all_kinds() -> None:
    backend = InMemoryAssetBackend()
    store = _BatchReadStore(backend)
    objects = InMemoryObjectStore("snapshot")
    await store.initialize()
    try:
        agent = AssetKey("agent", "review")
        rule = AssetKey("rule", "review.md")
        await store.put(agent, b"agent")
        await store.put(rule, b"rule")

        reference = await store.snapshot((rule, agent), object_store=objects)

        assert tuple(tuple(ref.key for ref in batch) for batch in store.version_reads) == (
            (agent,),
            (rule,),
        )
        assert store.individual_reads == 0
        restored = AssetStore.from_snapshot(reference, object_store=objects)
        await restored.initialize()
        try:
            assert await restored.get_many((agent, rule)) == (b"agent", b"rule")
        finally:
            await restored.close()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_builtin_loader_batches_declaration_version_reads() -> None:
    backend = InMemoryAssetBackend()
    store = _BatchReadStore(backend)
    await store.initialize()
    await store.put(
        AssetKey("agent", "agent"),
        AgentSpecCodec().encode(AgentSpec("agent", model_route="model")),
    )
    await store.put(
        AssetKey("mcp", "server"),
        MCPServerSpecCodec().encode(MCPServerSpec("server", "python", ("-m", "server"))),
    )
    await store.put(
        AssetKey("skill", "skill"),
        SkillSpecCodec().encode(SkillSpec("skill", "instructions")),
    )

    snapshot = await CapabilityGroup("workspace", assets=store).capture()

    assert [item.id for item in snapshot.contributions] == [
        "agent",
        "server",
        "skill",
    ]
    assert len(store.version_reads) == 3
    assert tuple(tuple(ref.key for ref in batch) for batch in store.version_reads) == (
        (AssetKey("agent", "agent"),),
        (AssetKey("skill", "skill"),),
        (AssetKey("mcp", "server"),),
    )
    assert store.individual_reads == 0


def test_capability_group_does_not_expose_logical_asset_crud() -> None:
    group = CapabilityGroup[object]("group")
    assert not hasattr(group, "resolve")
    assert not hasattr(group, "put")
    assert not hasattr(group, "delete")
    assert not hasattr(group, "list")
