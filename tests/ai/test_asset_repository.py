#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logical AssetRepository behavior and Skill Markdown contract checks."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from linktools.ai.app import build_asset_repository
from linktools.ai.asset import (
    AssetInfo,
    AssetKey,
    AssetRef,
    AssetRepository,
    AssetStore,
    AssetTypeBinding,
    AssetTypeRegistry,
    AssetVariantBinding,
    DirectoryLayout,
    InMemoryAssetBackend,
    LocalDirectoryAssetBackend,
    SingleFileLayout,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import (
    SkillMarkdownSpecAdapter,
    SkillMarkdownSpecCodec,
    SkillSpec,
)
from linktools.ai.storage import (
    StorageEntryRevision,
    StorageOverlay,
    StorageResetResult,
)


@dataclass(frozen=True, slots=True)
class _Value:
    id: str
    content: str


class _Codec:
    def encode(self, value: _Value) -> bytes:
        return value.content.encode("utf-8")

    def decode(self, data: bytes) -> _Value:
        decoded = data.decode("utf-8")
        return _Value(decoded, decoded)


class _WrongTypeCodec:
    def encode(self, value: _Value) -> bytes:
        return value.content.encode("utf-8")

    def decode(self, data: bytes) -> str:
        return data.decode("utf-8")


class _BrokenCodec:
    def encode(self, value: _Value) -> bytes:
        return value.content.encode("utf-8")

    def decode(self, data: bytes) -> _Value:
        del data
        raise ValueError("broken")


class _RaceStore(AssetStore):
    def __init__(self, backend: InMemoryAssetBackend) -> None:
        super().__init__(StorageOverlay(backend, writer=backend))
        self._backend = backend
        self._raced = False

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_revision: StorageEntryRevision | None = None,
    ) -> AssetInfo:
        info = await super().put(key, value, expected_revision=expected_revision)
        if not self._raced:
            self._raced = True
            await self._backend.put(AssetKey(key.kind, "foo.md"), b"foo")
        return info


class _RecoveryRaceStore(_RaceStore):
    async def reset(
        self,
        key: AssetKey,
        *,
        expected_revision: StorageEntryRevision | None = None,
    ) -> StorageResetResult[AssetKey]:
        del key, expected_revision
        raise AIError(ErrorCode.STORAGE_READ_ONLY)


def _binding() -> AssetTypeBinding[object]:
    return AssetTypeBinding(
        "subagent",
        _Value,
        (
            AssetVariantBinding("file", SingleFileLayout(".md"), _Codec(), "subagent-file", 1),
            AssetVariantBinding("directory", DirectoryLayout("AGENT.md"), _Codec(), "subagent-directory", 1),
        ),
        "directory",
        lambda ref, value: value.id == ref.id,
        "exact-id-v1",
        True,
    )


async def _repo() -> tuple[AssetStore, AssetRepository]:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    registry = AssetTypeRegistry()
    registry.register(_binding())
    return store, AssetRepository(store, registry.freeze())


@pytest.mark.asyncio
async def test_repository_discovers_directory_scope_and_hides_descendants() -> None:
    store, repository = await _repo()
    await store.put(AssetKey("subagent", "team/foo/AGENT.md"), b"team/foo")
    await store.put(AssetKey("subagent", "team/foo/references/python/rules.md"), b"rules")
    await store.put(AssetKey("subagent", "team/foo/child/AGENT.md"), b"child")

    entries = (await repository.list(kind="subagent")).items
    assert len(entries) == 1
    assert entries[0].ref == AssetRef("subagent", "team/foo")
    assert entries[0].status.value == "RESOLVABLE"
    assert entries[0].variants == ("directory",)
    resolved = await repository.resolve(AssetRef("subagent", "team/foo"))
    assert resolved.variant == "directory"
    assert await resolved.scope.get("references/python/rules.md") == b"rules"
    assert await resolved.scope.get("child/AGENT.md") == b"child"
    assert all(item.path != "child/AGENT.md" or not item.is_entry for item in (await resolved.scope.list()).items)
    with pytest.raises(AIError) as error:
        await repository.resolve(AssetRef("subagent", "team/foo/child"))
    assert error.value.code is ErrorCode.STORAGE_NOT_FOUND


@pytest.mark.asyncio
async def test_repository_reports_layout_conflict_without_decoding_list_content() -> None:
    store, repository = await _repo()
    await store.put(AssetKey("subagent", "broken.md"), b"broken")
    await store.put(AssetKey("subagent", "same.md"), b"one")
    await store.put(AssetKey("subagent", "same/AGENT.md"), b"two")
    entries = (await repository.list(kind="subagent")).items
    assert entries[0].ref.id == "broken"
    assert entries[0].status.value == "RESOLVABLE"
    conflict = next(item for item in entries if item.ref.id == "same")
    assert conflict.status.value == "CONFLICT"
    assert conflict.variants == ("directory", "file")
    with pytest.raises(AIError) as error:
        await repository.resolve(AssetRef("subagent", "same"))
    assert error.value.code is ErrorCode.ASSET_LAYOUT_CONFLICT


@pytest.mark.asyncio
async def test_repository_typed_put_preserves_layout_and_fences_scope_entry() -> None:
    _store, repository = await _repo()
    ref = AssetRef("subagent", "new/item")
    value = _Value(ref.id, "content")
    created = await repository.put(ref, value)
    assert created.variant == "directory"
    assert created.entry.key.id == "new/item/AGENT.md"
    with pytest.raises(AIError) as error:
        await created.scope.put("AGENT.md", b"raw")
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    updated = await repository.put(ref, _Value(ref.id, "updated"))
    assert updated.variant == "directory"
    with pytest.raises(AIError) as error:
        await repository.put(ref, _Value(ref.id, "file"), variant="file")
    assert error.value.code is ErrorCode.ASSET_LAYOUT_CONFLICT


@pytest.mark.asyncio
async def test_repository_rejects_descendant_takeover_and_scope_paths() -> None:
    store, repository = await _repo()
    await store.put(AssetKey("subagent", "foo/bar.md"), b"bar")
    with pytest.raises(AIError) as error:
        await repository.put(AssetRef("subagent", "foo"), _Value("foo", "foo"))
    assert error.value.code is ErrorCode.ASSET_LAYOUT_CONFLICT
    await store.put(AssetKey("subagent", "root/AGENT.md"), b"root")
    resolved = await repository.resolve(AssetRef("subagent", "root"))
    with pytest.raises(AIError) as error:
        await resolved.scope.get("../outside")
    assert error.value.code is ErrorCode.ASSET_PATH_OUTSIDE_ROOT
    with pytest.raises(AIError) as error:
        await resolved.scope.get("/outside")
    assert error.value.code is ErrorCode.ASSET_PATH_ABSOLUTE


def test_registry_freeze_rejects_overlapping_single_file_layouts() -> None:
    registry = AssetTypeRegistry()
    with pytest.raises(AIError) as error:
        registry.register(
            AssetTypeBinding(
                "sample",
                _Value,
                (
                    AssetVariantBinding("short", SingleFileLayout(".md"), _Codec(), "short", 1),
                    AssetVariantBinding("long", SingleFileLayout(".agent.md"), _Codec(), "long", 1),
                ),
                "short",
            )
        )
    assert error.value.code is ErrorCode.ASSET_CODEC_CONFLICT
    frozen = AssetTypeRegistry()
    frozen.register(_binding())
    frozen.freeze()
    with pytest.raises(AIError):
        frozen.register(_binding())


def test_skill_markdown_codec_and_adapter_contract() -> None:
    content = "---\nname: foo\ndescription: A skill\n---\n# Instructions\n"
    codec = SkillMarkdownSpecCodec()
    value = codec.decode(content.encode())
    assert value == SkillSpec("foo", 1, content)
    logical = SkillMarkdownSpecAdapter().to_logical("team/foo", value)
    assert logical.id == "team/foo"
    assert codec.encode(value) == content.encode()
    with pytest.raises(AIError) as error:
        codec.decode(b"---\nname: foo\nname: bar\ndescription: x\n---\n")
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    with pytest.raises(AIError) as error:
        codec.decode(b"---\nname: foo\ndescription: x\nmetadata: null\n---\n")
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    with pytest.raises(AIError) as error:
        codec.encode(SkillSpec("bar", 1, content))
    assert error.value.code is ErrorCode.ASSET_CONTENT_MISMATCH
    with pytest.raises(AIError) as error:
        codec.encode(SkillSpec("foo", True, content))
    assert error.value.code is ErrorCode.ASSET_CONTENT_MISMATCH


@pytest.mark.asyncio
async def test_builtin_repository_supports_nested_skill_logical_id() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    repository = build_asset_repository(store)
    content = "---\nname: foo\ndescription: A skill\n---\n# Instructions\n"
    await store.put(AssetKey("skill", "team/backend/foo/SKILL.md"), content.encode())
    resolved = await repository.resolve(AssetRef("skill", "team/backend/foo"))
    assert resolved.spec.id == "team/backend/foo"
    assert resolved.scope.entry_path == "SKILL.md"

    extended = build_asset_repository(store, extra_bindings=(_binding(),))
    created = await extended.put(AssetRef("subagent", "team/reviewer"), _Value("team/reviewer", "reviewer"))
    assert created.variant == "directory"


@pytest.mark.asyncio
async def test_repository_cursor_prefix_and_content_change_semantics() -> None:
    store, repository = await _repo()
    for identifier in ("team/backend", "team/backend2", "team/frontend"):
        await store.put(AssetKey("subagent", identifier + ".md"), identifier.encode())
    first = await repository.list(kind="subagent", prefix="team/backend", limit=1)
    assert [item.ref.id for item in first.items] == ["team/backend"]
    assert first.next_cursor is None
    with pytest.raises(AIError) as error:
        await repository.list(kind="subagent", limit=0)
    assert error.value.code is ErrorCode.PAGE_LIMIT_INVALID
    paged = await repository.list(kind="subagent", limit=2)
    assert paged.next_cursor is not None
    await store.put(AssetKey("subagent", "team/backend.md"), b"changed")
    continued = await repository.list(kind="subagent", cursor=paged.next_cursor, limit=2)
    assert [item.ref.id for item in continued.items] == ["team/frontend"]
    await store.put(AssetKey("subagent", "another.md"), b"another")
    with pytest.raises(AIError) as error:
        await repository.list(kind="subagent", cursor=paged.next_cursor, limit=2)
    assert error.value.code is ErrorCode.ASSET_CURSOR_INVALID


@pytest.mark.asyncio
async def test_repository_type_identity_and_codec_errors_have_no_write_side_effect() -> None:
    store, repository = await _repo()
    ref = AssetRef("subagent", "identity")
    with pytest.raises(AIError) as error:
        await repository.put(ref, _Value("other", "value"))
    assert error.value.code is ErrorCode.ASSET_CONTENT_MISMATCH
    assert await store.get(AssetKey("subagent", "identity/AGENT.md")) is None

    registry = AssetTypeRegistry()
    registry.register(
        AssetTypeBinding(
            "wrong",
            _Value,
            (AssetVariantBinding("wrong", SingleFileLayout(""), _WrongTypeCodec(), "wrong", 1),),
            "wrong",
        )
    )
    registry.register(
        AssetTypeBinding(
            "broken",
            _Value,
            (AssetVariantBinding("broken", SingleFileLayout(""), _BrokenCodec(), "broken", 1),),
            "broken",
        )
    )
    extra = AssetRepository(store, registry.freeze())
    await store.put(AssetKey("wrong", "item"), b"item")
    with pytest.raises(AIError) as error:
        await extra.resolve(AssetRef("wrong", "item"))
    assert error.value.code is ErrorCode.ASSET_CONFIG_TYPE_INVALID
    await store.put(AssetKey("broken", "item"), b"item")
    with pytest.raises(AIError) as error:
        await extra.resolve(AssetRef("broken", "item"))
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.asyncio
async def test_repository_compensates_post_write_layout_conflict() -> None:
    backend = InMemoryAssetBackend()
    store = _RaceStore(backend)
    await store.initialize()
    registry = AssetTypeRegistry()
    registry.register(_binding())
    repository = AssetRepository(store, registry.freeze())
    with pytest.raises(AIError) as error:
        await repository.put(AssetRef("subagent", "foo"), _Value("foo", "foo"))
    assert error.value.code is ErrorCode.STORAGE_CONFLICT
    assert await store.get(AssetKey("subagent", "foo/AGENT.md")) is None
    assert await store.get(AssetKey("subagent", "foo.md")) == b"foo"


@pytest.mark.asyncio
async def test_repository_surfaces_recovery_required_when_compensation_fails() -> None:
    backend = InMemoryAssetBackend()
    store = _RecoveryRaceStore(backend)
    await store.initialize()
    registry = AssetTypeRegistry()
    registry.register(_binding())
    repository = AssetRepository(store, registry.freeze())
    with pytest.raises(AIError) as error:
        await repository.put(AssetRef("subagent", "foo"), _Value("foo", "foo"))
    assert error.value.code is ErrorCode.ASSET_RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_local_directory_backend_can_be_used_as_a_read_only_logical_layer(tmp_path: Path) -> None:
    root = tmp_path
    path = root / "subagent" / "foo" / "AGENT.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"foo")
    backend = LocalDirectoryAssetBackend(str(root))
    store = AssetStore(StorageOverlay(backend))
    await store.initialize()
    registry = AssetTypeRegistry()
    registry.register(_binding())
    repository = AssetRepository(store, registry.freeze())
    resolved = await repository.resolve(AssetRef("subagent", "foo"))
    assert resolved.spec.id == "foo"
    assert await resolved.scope.get("AGENT.md") == b"foo"
