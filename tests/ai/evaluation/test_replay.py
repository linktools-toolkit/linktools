#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay + snapshot validation tests (sections 25.2 / 25.3)."""

import asyncio

import pytest

from linktools.ai.artifact import ArtifactStore
from linktools.ai.storage.artifact_backends import build_artifact_store_from_assets
from linktools.ai.evaluation.models import EvalExecution, EvalTarget
from linktools.ai.evaluation.replay import (
    SnapshotValidationError,
    replay,
    validate_snapshot,
)
from linktools.ai.evaluation.snapshot import RunSnapshot
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.store import AssetStore


def _artifact(artifacts: ArtifactStore, content: bytes, tenant: str) -> str:
    async def _put() -> str:
        rec = await artifacts.put(content, media_type="text/plain", tenant_id=tenant)
        return rec.ref.id

    return asyncio.run(_put())


class _RecordingExecutor:
    def __init__(self):
        self.called_with = None

    async def execute(self, target, case):
        self.called_with = case
        return EvalExecution(case_id=case.id, run_id="new-run", output="replayed")


def test_validate_snapshot_rejects_missing_artifact() -> None:
    artifacts = build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))
    snap = RunSnapshot(
        run_id="r1",
        run_record_artifact_id="missing-sha",
        run_definition_artifact_id="also-missing",
        input_artifact_id="in-missing",
    )

    async def run():
        with pytest.raises(SnapshotValidationError, match="missing"):
            await validate_snapshot(snap, artifacts, tenant_id="t1")

    asyncio.run(run())


def test_validate_snapshot_passes_when_all_artifacts_exist() -> None:
    artifacts = build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))
    rr = _artifact(artifacts, b"rec", "t1")
    rd = _artifact(artifacts, b"def", "t1")
    inp = _artifact(artifacts, b"input", "t1")
    snap = RunSnapshot(
        run_id="r1",
        run_record_artifact_id=rr,
        run_definition_artifact_id=rd,
        input_artifact_id=inp,
    )

    async def run():
        # No raise -- all referenced artifacts exist for this tenant.
        await validate_snapshot(snap, artifacts, tenant_id="t1")

    asyncio.run(run())


def test_replay_validates_then_creates_new_execution() -> None:
    artifacts = build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))
    rr = _artifact(artifacts, b"rec", "t1")
    rd = _artifact(artifacts, b"def", "t1")
    inp = _artifact(artifacts, b"original-input", "t1")
    snap = RunSnapshot(
        run_id="r1",
        run_record_artifact_id=rr,
        run_definition_artifact_id=rd,
        input_artifact_id=inp,
    )
    executor = _RecordingExecutor()

    async def run():
        execution = await replay(
            snap,
            executor,
            EvalTarget(kind="agent", id="a1"),
            artifacts,
            tenant_id="t1",
        )
        # Replay produced a fresh execution from the snapshot's input.
        assert execution.run_id == "new-run"
        assert executor.called_with.input_artifact_id == inp
        assert executor.called_with.id == "replay-r1"

    asyncio.run(run())


def test_replay_refuses_on_invalid_snapshot() -> None:
    artifacts = build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))
    snap = RunSnapshot(
        run_id="r1",
        run_record_artifact_id="missing",
        run_definition_artifact_id="missing",
        input_artifact_id="missing",
    )
    executor = _RecordingExecutor()

    async def run():
        with pytest.raises(SnapshotValidationError):
            await replay(
                snap,
                executor,
                EvalTarget(kind="agent", id="a1"),
                artifacts,
                tenant_id="t1",
            )
        # The executor must never have run against an unvalidated snapshot.
        assert executor.called_with is None

    asyncio.run(run())
