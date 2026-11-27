#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository instruction partition recovery contracts."""

from types import SimpleNamespace
from dataclasses import replace
from datetime import datetime, timezone
from .test_execution_recovery_commands import _execution

import pytest
from linktools.ai.core import PersistenceDomain, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._recovery_coordinator import _RecoveryCoordinator
from linktools.ai.inputs import RepositoryInstructionPayload
from linktools.ai.persistence.contracts import ExecutionRecoveryDetail
from linktools.ai.spec import (
    RepositoryInstructionDocument,
    RepositoryInstructions,
)


class _Port:
    tenant_id = "tenant"

    def __init__(self, checkpoint: object, overlay: RepositoryInstructions) -> None:
        self.checkpoint = checkpoint
        self.overlay = overlay

    async def load_recovery_execution(
        self, execution_id: str, *, tenant_id: str
    ) -> object:
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        return self.checkpoint

    async def load_repository_instructions(self, reference: object) -> RepositoryInstructions:
        assert reference is self.checkpoint.recovery.repository_instruction_overlay
        return self.overlay


class _Resolver:
    async def resolve(self, path, *, exclude_sources, within=None):
        assert exclude_sources == {"agents:pkg/AGENTS.md", "agents:pkg/deep/AGENTS.md"}
        return RepositoryInstructions()


def _instructions() -> tuple[RepositoryInstructions, RepositoryInstructions]:
    first = RepositoryInstructionDocument(
        "agents:pkg/AGENTS.md",
        "pkg",
        "package rules",
    )
    second = RepositoryInstructionDocument(
        "agents:pkg/deep/AGENTS.md",
        "pkg/deep",
        "deep rules",
    )
    return RepositoryInstructions((first,)), RepositoryInstructions((first, second))


def _checkpoint(
    latest: RepositoryInstructions,
    *,
    latest_payload_digest: str | None = None,
) -> tuple[object, dict[str, str]]:
    arguments = {"path": "pkg/file.txt"}
    reference = RepositoryInstructionPayload(
        canonical_sha256(latest.to_payload()) if latest_payload_digest is None else latest_payload_digest,
        latest.to_payload(),
    ).to_reference(PersistenceDomain.CHECKPOINT)
    checkpoint = replace(_execution(datetime.now(timezone.utc)), recovery=ExecutionRecoveryDetail(
        agent_run_id="step",
        repository_instruction_overlay=reference,
    ))
    return checkpoint, arguments


@pytest.mark.asyncio
async def test_existing_tool_call_uses_latest_overlay_even_with_changed_arguments() -> None:
    first, latest = _instructions()
    checkpoint, _ = _checkpoint(latest)
    coordinator = _RecoveryCoordinator(
        _Port(checkpoint, latest),  # type: ignore[arg-type]
        _Resolver(),  # type: ignore[arg-type]
    )

    overlay = await coordinator.check_repository_instructions(
        execution=SimpleNamespace(execution_id="execution", tenant_id="tenant"),  # type: ignore[arg-type]
        initial=None,
        overlay=first,
        tool_name="read_file",
        tool_call_id="call-a",
        arguments={"path": "pkg/other.txt"},
        path_fields=("path",),
    )

    assert overlay == latest


@pytest.mark.asyncio
async def test_latest_overlay_uses_loaded_content_without_rechecking_digest() -> None:
    first, latest = _instructions()
    checkpoint, arguments = _checkpoint(
        latest,
        latest_payload_digest="0" * 64,
    )
    coordinator = _RecoveryCoordinator(
        _Port(checkpoint, latest),  # type: ignore[arg-type]
        _Resolver(),  # type: ignore[arg-type]
    )

    overlay = await coordinator.check_repository_instructions(
        execution=SimpleNamespace(execution_id="execution", tenant_id="tenant"),  # type: ignore[arg-type]
        initial=None,
        overlay=first,
        tool_name="read_file",
        tool_call_id="call-a",
        arguments=arguments,
        path_fields=("path",),
    )

    assert overlay == latest


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("initial", "paths", "bytes", "overlay"))
async def test_dynamic_instruction_union_commits_only_fitting_additions(tmp_path, kind) -> None:
    from linktools.ai.workspace import WorkspaceInstructionResolver, WorkspacePolicy

    root = RepositoryInstructionDocument("agents:AGENTS.md", ".", "根规则")
    child = RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "子规则")
    other = RepositoryInstructionDocument("agents:other/AGENTS.md", "other", "其他规则")
    documents = (child, other) if kind in {"paths", "overlay"} else (root, child)
    for document in documents:
        path = tmp_path / document.source.removeprefix("agents:")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document.content, encoding="utf-8")
    full = RepositoryInstructions(documents)
    policy = WorkspacePolicy(
        max_repository_instruction_documents=1 if kind != "bytes" else 2,
        max_repository_instruction_bytes=len(full.render().encode("utf-8")) - 1 if kind == "bytes" else 256 * 1024,
    )
    checkpoint = replace(_execution(datetime.now(timezone.utc)), recovery=ExecutionRecoveryDetail(agent_run_id="run"))

    persisted = None
    if kind == "overlay":
        persisted = RepositoryInstructions((child,))
        checkpoint = replace(checkpoint, recovery=replace(checkpoint.recovery,
            repository_instruction_overlay=RepositoryInstructionPayload.capture(persisted, persisted.to_payload()).to_reference(PersistenceDomain.CHECKPOINT)))

    committed = []

    class Port(_Port):
        async def commit_repository_instruction_overlay(self, current, overlay, **kwargs):
            assert current == self.checkpoint
            committed.append(overlay)
            self.overlay = overlay
            self.checkpoint = replace(current, recovery=replace(current.recovery,
                repository_instruction_overlay=RepositoryInstructionPayload.capture(
                    overlay, overlay.to_payload(),
                ).to_reference(PersistenceDomain.CHECKPOINT)))
            return self.checkpoint

    coordinator = _RecoveryCoordinator(
        Port(checkpoint, persisted),
        WorkspaceInstructionResolver(tmp_path, policy, RepositoryInstructions()),
    )
    arguments = {"path": ["pkg/file.txt", "other/file.txt"]} if kind == "paths" else {
        "path": "other/file.txt" if kind == "overlay" else "pkg/file.txt",
    }
    result = await coordinator.check_repository_instructions(
        execution=SimpleNamespace(execution_id="execution"),
        initial=None if kind in {"paths", "overlay"} else RepositoryInstructions((root,)),
        overlay=persisted, tool_name="read_file", tool_call_id="call",
        arguments=arguments, path_fields=("path",),
    )
    if kind == "paths":
        assert result == RepositoryInstructions((child,)) and committed == [result]
    else:
        assert result == persisted and committed == []


@pytest.mark.asyncio
async def test_pinned_sources_are_not_recounted_or_revalidated_without_discovery(tmp_path) -> None:
    from linktools.ai.workspace import WorkspaceInstructionResolver, WorkspacePolicy

    initial = RepositoryInstructions((RepositoryInstructionDocument("agents:AGENTS.md", ".", "pinned root"),))
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")  # A pinned source must not be read again.
    checkpoint = replace(_execution(datetime.now(timezone.utc)), recovery=ExecutionRecoveryDetail(agent_run_id="run"))
    coordinator = _RecoveryCoordinator(
        _Port(checkpoint, None),
        WorkspaceInstructionResolver(tmp_path, WorkspacePolicy(max_repository_instruction_bytes=1), RepositoryInstructions()),
    )
    overlay = await coordinator.check_repository_instructions(
        execution=SimpleNamespace(execution_id="execution"), initial=initial,
        overlay=None, tool_name="read_files", tool_call_id="call",
        arguments={"paths": ["first.txt", "second.txt"]}, path_fields=("paths",),
    )
    assert overlay is None
