#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository instruction partition recovery contracts."""

from types import SimpleNamespace

import pytest
from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._recovery_coordinator import _RecoveryCoordinator
from linktools.ai.runtime.state._contracts import (
    RecoveryCheckpointState,
    RepositoryInstructionBarrier,
)
from linktools.ai.workspace import (
    RepositoryInstructionDocument,
    RepositoryInstructions,
)


class _Port:
    tenant_id = "tenant"

    def __init__(self, checkpoint: object, overlay: RepositoryInstructions) -> None:
        self.checkpoint = checkpoint
        self.overlay = overlay

    async def load_recovery_checkpoint(
        self, execution_id: str, *, tenant_id: str
    ) -> object:
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        return self.checkpoint

    async def load_repository_instructions(self, reference: object) -> RepositoryInstructions:
        assert reference is self.checkpoint.repository_instruction_overlay
        return self.overlay


class _Resolver:
    async def resolve(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("historical barrier replay must not rediscover sources")


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
    first: RepositoryInstructions,
    latest: RepositoryInstructions,
    *,
    latest_barrier_digest: str | None = None,
) -> tuple[object, dict[str, str]]:
    arguments = {"path": "pkg/file.txt"}
    first_barrier = RepositoryInstructionBarrier(
        "step",
        "call-a",
        canonical_sha256(arguments),
        first.digest,
    )
    latest_barrier = RepositoryInstructionBarrier(
        "step",
        "call-b",
        canonical_sha256({"path": "pkg/deep/file.txt"}),
        latest.digest if latest_barrier_digest is None else latest_barrier_digest,
    )
    reference = SimpleNamespace(payload=SimpleNamespace(digest=latest.digest))
    checkpoint = SimpleNamespace(
        state=RecoveryCheckpointState.ACTIVE,
        step_run_id="step",
        repository_instruction_overlay=reference,
        repository_instruction_barriers=(first_barrier, latest_barrier),
    )
    return checkpoint, arguments


@pytest.mark.asyncio
async def test_historical_barrier_replay_keeps_latest_overlay() -> None:
    first, latest = _instructions()
    checkpoint, arguments = _checkpoint(first, latest)
    coordinator = _RecoveryCoordinator(
        _Port(checkpoint, latest),  # type: ignore[arg-type]
        _Resolver(),  # type: ignore[arg-type]
    )

    overlay, reconsider = await coordinator.check_repository_instructions(
        execution=SimpleNamespace(execution_id="execution", tenant_id="tenant"),  # type: ignore[arg-type]
        initial=None,
        overlay=first,
        tool_name="read_file",
        tool_call_id="call-a",
        arguments=arguments,
        path_fields=("path",),
    )

    assert overlay == latest
    assert reconsider is True


@pytest.mark.asyncio
async def test_historical_barrier_replay_rejects_changed_arguments() -> None:
    first, latest = _instructions()
    checkpoint, _ = _checkpoint(first, latest)
    coordinator = _RecoveryCoordinator(
        _Port(checkpoint, latest),  # type: ignore[arg-type]
        _Resolver(),  # type: ignore[arg-type]
    )

    with pytest.raises(AIError) as raised:
        await coordinator.check_repository_instructions(
            execution=SimpleNamespace(execution_id="execution", tenant_id="tenant"),  # type: ignore[arg-type]
            initial=None,
            overlay=first,
            tool_name="read_file",
            tool_call_id="call-a",
            arguments={"path": "pkg/other.txt"},
            path_fields=("path",),
        )

    assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
async def test_latest_overlay_must_match_latest_barrier_digest() -> None:
    first, latest = _instructions()
    checkpoint, arguments = _checkpoint(
        first,
        latest,
        latest_barrier_digest="0" * 64,
    )
    coordinator = _RecoveryCoordinator(
        _Port(checkpoint, latest),  # type: ignore[arg-type]
        _Resolver(),  # type: ignore[arg-type]
    )

    with pytest.raises(AIError) as raised:
        await coordinator.check_repository_instructions(
            execution=SimpleNamespace(execution_id="execution", tenant_id="tenant"),  # type: ignore[arg-type]
            initial=None,
            overlay=first,
            tool_name="read_file",
            tool_call_id="call-a",
            arguments=arguments,
            path_fields=("path",),
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
