#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import (
    LocalRepositoryInstructionResolver,
    LocalRuleCatalog,
    RepositoryInstructionDocument,
    RepositoryInstructions,
    WorkspacePolicy,
)


def _resolver(root: Path, *, policy: WorkspacePolicy | None = None) -> LocalRepositoryInstructionResolver:
    selected = WorkspacePolicy() if policy is None else policy
    catalog = asyncio.run(LocalRuleCatalog.load(root, selected))
    return LocalRepositoryInstructionResolver(root, selected, catalog)


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
    assert RepositoryInstructions(bundle.documents).digest == bundle.digest
    assert RepositoryInstructions(bundle.documents).render() == bundle.render()
    assert RepositoryInstructionDocument("rule:base", ".", "root rule").digest != (
        RepositoryInstructionDocument("rule:base", "pkg", "root rule").digest
    )

    payload = bundle.to_payload()
    assert isinstance(payload, dict)
    with pytest.raises(AIError) as version_error:
        RepositoryInstructions.from_payload({**payload, "version": 2})
    assert version_error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

    with pytest.raises(AIError) as extra_error:
        RepositoryInstructions.from_payload({**payload, "extra": True})
    assert extra_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

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
    rules = tmp_path / ".linktools" / "rules"
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


def test_rule_catalog_is_recursive_scoped_and_frozen(tmp_path: Path) -> None:
    rules = tmp_path / ".linktools" / "rules"
    (rules / "nested").mkdir(parents=True)
    base = rules / "base.md"
    strict = rules / "nested" / "strict.md"
    base.write_text("base-v1", encoding="utf-8")
    strict.write_text("---\nscope: src\n---\nstrict", encoding="utf-8")

    policy = WorkspacePolicy()
    catalog = asyncio.run(LocalRuleCatalog.load(tmp_path, policy))
    assert [(item.source, item.scope) for item in catalog.documents] == [
        ("rule:base", "."),
        ("rule:nested/strict", "src"),
    ]
    resolver = LocalRepositoryInstructionResolver(tmp_path, policy, catalog)
    root = asyncio.run(resolver.resolve("."))
    nested = asyncio.run(resolver.resolve("src/pkg"))
    assert [item.source for item in root.documents] == ["rule:base"]
    assert [item.source for item in nested.documents] == ["rule:base", "rule:nested/strict"]

    base.write_text("base-v2", encoding="utf-8")
    old = asyncio.run(resolver.resolve("."))
    fresh = asyncio.run(_resolver(tmp_path).resolve("."))
    assert old.documents[0].content == "base-v1"
    assert fresh.documents[0].content == "base-v2"


def test_rule_catalog_rejects_invalid_rule_shapes(tmp_path: Path) -> None:
    rules = tmp_path / ".linktools" / "rules"
    rules.mkdir(parents=True)
    (rules / "bad.md").mkdir()
    with pytest.raises(AIError) as directory_error:
        asyncio.run(LocalRuleCatalog.load(tmp_path, WorkspacePolicy()))
    assert directory_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_rule_catalog_rejects_invalid_scope_and_outside_symlink(tmp_path: Path) -> None:
    rules = tmp_path / ".linktools" / "rules"
    rules.mkdir(parents=True)
    rule = rules / "bad.md"
    rule.write_text("---\nscope: ../outside\n---\nbad", encoding="utf-8")
    with pytest.raises(AIError) as scope_error:
        asyncio.run(LocalRuleCatalog.load(tmp_path, WorkspacePolicy()))
    assert scope_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

    rule.unlink()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-rule.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        rule.symlink_to(outside)
        with pytest.raises(AIError) as symlink_error:
            asyncio.run(LocalRuleCatalog.load(tmp_path, WorkspacePolicy()))
        assert symlink_error.value.code is ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT
    finally:
        outside.unlink(missing_ok=True)
