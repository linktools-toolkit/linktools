#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo

from linktools.ai.asset import (
    AssetKey,
    AssetStore,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.spec import (
    RepositoryInstructionDocument,
    RepositoryInstructions,
)
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import (
    WorkspaceInstructionResolver,
    Workspace,
    WorkspacePolicy,
)
from . import _runtime_test_helpers as runtime_test_helpers


async def _capture_rules(root: Path) -> RepositoryInstructions:
    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(root / "assets"),
                path_adapter=PrefixAssetPathAdapter({"rule": "rules"}),
                kinds=("rule",),
            )
        )
    )
    await store.initialize()
    try:
        return (await CapabilityGroup("assets", assets=store).capture()).instructions
    finally:
        await store.close()


def _resolver(root: Path, *, policy: WorkspacePolicy | None = None) -> WorkspaceInstructionResolver:
    selected = WorkspacePolicy() if policy is None else policy
    rules = asyncio.run(_capture_rules(root))
    return WorkspaceInstructionResolver(root, selected, rules)


def test_repository_instruction_bundle_is_canonical_and_strict() -> None:
    documents = (
        RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "nested agent"),
        RepositoryInstructionDocument("rule:pkg/strict", "pkg", "nested rule"),
        RepositoryInstructionDocument("agents:AGENTS.md", ".", "root agent"),
        RepositoryInstructionDocument("rule:base", ".", "root rule"),
    )
    bundle = RepositoryInstructions(documents)

    assert [document.source for document in bundle.documents] == [
        "rule:base",
        "agents:AGENTS.md",
        "rule:pkg/strict",
        "agents:pkg/AGENTS.md",
    ]
    assert RepositoryInstructions.from_payload(bundle.to_payload()) == bundle
    assert RepositoryInstructions(bundle.documents) == bundle
    assert RepositoryInstructions(bundle.documents).render() == bundle.render()
    assert RepositoryInstructionDocument("rule:base", ".", "root rule") != (
        RepositoryInstructionDocument("rule:base", "pkg", "root rule")
    )

    payload = bundle.to_payload()
    assert isinstance(payload, dict)
    with pytest.raises(AIError) as version_error:
        RepositoryInstructions.from_payload({**payload, "version": 2})
    assert version_error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

    assert RepositoryInstructions.from_payload({**payload, "extra": True}) == bundle

    document = dict(payload["documents"][0])
    document["extra"] = True
    additive_payload = dict(payload)
    additive_payload["documents"] = [document, *payload["documents"][1:]]
    assert RepositoryInstructions.from_payload(additive_payload) == bundle

    reversed_payload = dict(payload)
    reversed_payload["documents"] = list(reversed(payload["documents"]))
    with pytest.raises(AIError) as order_error:
        RepositoryInstructions.from_payload(reversed_payload)
    assert order_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

    duplicate_payload = dict(payload)
    duplicate_payload["documents"] = [payload["documents"][0], payload["documents"][0]]
    with pytest.raises(AIError) as duplicate_error:
        RepositoryInstructions.from_payload(duplicate_payload)
    assert duplicate_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_repository_resolver_uses_target_ancestry_and_rules_before_agents(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("root-agent", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "AGENTS.md").write_text("nested-agent", encoding="utf-8")
    rules = tmp_path / "assets" / "rules"
    (rules / "python").mkdir(parents=True)
    (rules / "base.md").write_text("root-rule", encoding="utf-8")
    (rules / "python" / "strict.md").write_text(
        "---\nscope: pkg\n---\nnested-rule",
        encoding="utf-8",
    )

    resolver = _resolver(tmp_path)
    resolved = asyncio.run(resolver.resolve("pkg/sub"))
    assert [document.source for document in resolved.documents] == [
        "rule:base",
        "agents:AGENTS.md",
        "rule:python/strict",
        "agents:pkg/AGENTS.md",
    ]
    assert [document.content for document in resolved.documents] == [
        "root-rule",
        "root-agent",
        "nested-rule",
        "nested-agent",
    ]

    absolute = asyncio.run(resolver.resolve(tmp_path / "pkg" / "sub"))
    normalized = asyncio.run(resolver.resolve("pkg/./child/.."))
    assert absolute == resolved
    assert normalized.documents[-1].source == "agents:pkg/AGENTS.md"


def test_repository_resolver_rejects_invalid_targets_and_sources(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    with pytest.raises(AIError) as outside_error:
        asyncio.run(resolver.resolve(tmp_path.parent / "outside"))
    assert outside_error.value.code is ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT

    with pytest.raises(AIError) as nul_error:
        asyncio.run(resolver.resolve("bad\x00path"))
    assert nul_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

    (tmp_path / "AGENTS.md").write_bytes(b"\xff")
    with pytest.raises(AIError) as utf8_error:
        asyncio.run(resolver.resolve("."))
    assert utf8_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_repository_resolver_excludes_before_touching_candidate(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").mkdir()
    resolver = _resolver(tmp_path)
    excluded = asyncio.run(
        resolver.resolve(".", exclude_sources=frozenset({"agents:AGENTS.md"}))
    )
    assert excluded.documents == ()

    with pytest.raises(AIError) as source_error:
        asyncio.run(resolver.resolve("."))
    assert source_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

    with pytest.raises(AIError) as exclude_error:
        asyncio.run(resolver.resolve(".", exclude_sources=frozenset({"invalid"})))
    assert exclude_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_repository_instruction_source_limits_are_enforced(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("0123456789", encoding="utf-8")
    policy = WorkspacePolicy(max_repository_instruction_bytes=8)
    resolver = _resolver(tmp_path, policy=policy)
    with pytest.raises(AIError) as error:
        asyncio.run(resolver.resolve("."))
    assert error.value.code is ErrorCode.PROMPT_TOO_LARGE


def test_agents_symlink_must_resolve_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "shared.md"
    target.write_text("shared", encoding="utf-8")
    (tmp_path / "AGENTS.md").symlink_to(target.name)
    resolved = asyncio.run(_resolver(tmp_path).resolve("."))
    assert resolved.documents[0].content == "shared"

    (tmp_path / "AGENTS.md").unlink()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-agents.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        (tmp_path / "AGENTS.md").symlink_to(outside)
        with pytest.raises(AIError) as error:
            asyncio.run(_resolver(tmp_path).resolve("."))
        assert error.value.code is ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT
    finally:
        outside.unlink(missing_ok=True)


def test_non_regular_agents_file_is_rejected_before_read(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is unavailable on this platform")
    os.mkfifo(tmp_path / "AGENTS.md")
    with pytest.raises(AIError) as error:
        asyncio.run(_resolver(tmp_path).resolve("."))
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_rule_capture_is_recursive_scoped_and_frozen(tmp_path: Path) -> None:
    rules = tmp_path / "assets" / "rules"
    (rules / "nested").mkdir(parents=True)
    base = rules / "base.md"
    strict = rules / "nested" / "strict.md"
    base.write_text("base-v1", encoding="utf-8")
    strict.write_text(
        "---\nscope: src\nowner: security\nfuture-field: enabled\n---\nstrict",
        encoding="utf-8",
    )

    policy = WorkspacePolicy()
    rules = asyncio.run(_capture_rules(tmp_path))
    assert [(item.source, item.scope) for item in rules.documents] == [
        ("rule:base", "."),
        ("rule:nested/strict", "src"),
    ]
    resolver = WorkspaceInstructionResolver(tmp_path, policy, rules)
    root = asyncio.run(resolver.resolve("."))
    nested = asyncio.run(resolver.resolve("src/pkg"))
    assert [item.source for item in root.documents] == ["rule:base"]
    assert [item.source for item in nested.documents] == ["rule:base", "rule:nested/strict"]

    base.write_text("base-v2", encoding="utf-8")
    old = asyncio.run(resolver.resolve("."))
    fresh = asyncio.run(_resolver(tmp_path).resolve("."))
    assert old.documents[0].content == "base-v1"
    assert fresh.documents[0].content == "base-v2"


def test_rule_capture_rejects_invalid_rule_content(tmp_path: Path) -> None:
    rules = tmp_path / "assets" / "rules"
    rules.mkdir(parents=True)
    (rules / "bad.md").write_bytes(b"\xff")
    with pytest.raises(AIError) as content_error:
        asyncio.run(_capture_rules(tmp_path))
    assert content_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_rule_capture_rejects_invalid_scope_and_ignores_symlink(tmp_path: Path) -> None:
    rules = tmp_path / "assets" / "rules"
    rules.mkdir(parents=True)
    rule = rules / "bad.md"
    rule.write_text("---\nscope: ../outside\n---\nbad", encoding="utf-8")
    with pytest.raises(AIError) as scope_error:
        asyncio.run(_capture_rules(tmp_path))
    assert scope_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

    rule.unlink()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-rule.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        rule.symlink_to(outside)
        assert asyncio.run(_capture_rules(tmp_path)).documents == ()
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_rule_assets_follow_capability_capture_and_reject_duplicate_sources() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        key = AssetKey("rule", "nested/review.md")
        await store.put(key, b"---\nscope: src\n---\nfirst")
        capture = await CapabilityGroup("rules", assets=store).capture()
        assert capture.instructions.documents == (
            RepositoryInstructionDocument("rule:nested/review", "src", "first"),
        )

        await store.put(key, b"---\nscope: src\n---\nsecond")
        assert capture.instructions.documents[0].content == "first"
        with pytest.raises(AIError) as stale_error:
            await capture.verify_source_revision()
        assert stale_error.value.code is ErrorCode.SNAPSHOT_CONFLICT

        fresh = await CapabilityGroup("rules", assets=store).capture()
        assert fresh.instructions.documents[0].content == "second"
        duplicate = RepositoryInstructionDocument("rule:duplicate", ".", "same")
        with pytest.raises(AIError) as duplicate_error:
            RepositoryInstructions((duplicate, duplicate))
        assert duplicate_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_rule_asset_id_must_be_canonical() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(AssetKey("rule", "../outside.md"), b"unsafe")
        with pytest.raises(AIError) as error:
            await CapabilityGroup("rules", assets=store).capture()
        assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_runtime_reads_rules_from_asset_store_not_workspace_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / ".linktools" / "rules"
    legacy.mkdir(parents=True)
    (legacy / "legacy.md").write_text("legacy-rule", encoding="utf-8")

    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    await store.put(AssetKey("rule", "base.md"), b"asset-rule")
    observed: list[str] = []
    original = runtime_test_helpers._runtime_usage_model

    async def record_model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        observed.append(repr(messages))
        return await original(messages, info)

    monkeypatch.setattr(runtime_test_helpers, "_runtime_usage_model", record_model)
    try:
        async with Runtime.open(
            "rule-assets",
            models=runtime_test_helpers.RuntimeUsageModels(),  # type: ignore[arg-type]
            storage=RuntimeStorage.in_memory(),
            capabilities=(
                CapabilityGroup("workspace", workspace=Workspace.load(tmp_path)),
                CapabilityGroup("rules", assets=store),
            ),
        ) as runtime:
            result = await runtime.agent("default").run("hello", timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED
        assert any("asset-rule" in request for request in observed)
        assert all("legacy-rule" not in request for request in observed)
    finally:
        await store.close()
