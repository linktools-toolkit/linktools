#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime snapshot publication race regression coverage."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.runtime import (
    RuntimeSnapshot,
    SnapshotLimits,
    SnapshotTargetInspection,
)
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.storage import InMemoryObjectStore, ObjectRef


@pytest.mark.asyncio
async def test_same_snapshot_concurrent_restore_publishes_one_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryObjectStore("snapshot")
    state_ref = ObjectRef("snapshot", "state", "a" * 64, 0)
    snapshot_ref = ObjectRef("snapshot", "snapshot", "b" * 64, 0)
    manifest = {
        "kind": "runtime-snapshot",
        "format_version": 1,
        "namespace": "namespace",
        "tenant_id": "tenant",
        "state": {
            "store_id": state_ref.store_id,
            "key": state_ref.key,
            "digest": state_ref.digest,
            "size": state_ref.size,
        },
        "workspace": {"present": False, "entries": []},
        "metadata": {},
    }
    ready = asyncio.Event()
    arrivals = 0

    async def verified_manifest(
        cls,
        ref: ObjectRef,
        object_store,
        limits: SnapshotLimits,
    ):
        del cls, ref, object_store, limits
        return manifest

    async def restore_state(
        cls,
        ref: ObjectRef,
        *,
        object_store,
        root: str | Path,
        limits: SnapshotLimits,
    ) -> None:
        nonlocal arrivals
        del cls, ref, object_store, limits
        Path(root).mkdir(parents=True, exist_ok=False)
        arrivals += 1
        if arrivals == 2:
            ready.set()
        await ready.wait()

    async def inspect_target(
        cls,
        target: str | Path,
        ref: ObjectRef,
        *,
        object_store,
        limits: SnapshotLimits,
    ) -> SnapshotTargetInspection:
        del cls, target, object_store, limits
        return SnapshotTargetInspection("matching", None, ref.digest)

    monkeypatch.setattr(
        RuntimeSnapshot,
        "_verified_manifest",
        classmethod(verified_manifest),
    )
    monkeypatch.setattr(
        RuntimeState,
        "restore_snapshot",
        classmethod(restore_state),
    )
    monkeypatch.setattr(
        RuntimeSnapshot,
        "inspect_target",
        classmethod(inspect_target),
    )

    target = tmp_path / "runtime"
    first, second = await asyncio.gather(
        RuntimeSnapshot.restore(
            snapshot_ref,
            object_store=store,
            target=target,
            namespace="namespace",
            tenant_id="tenant",
            limits=SnapshotLimits(max_entries=100, max_bytes=1024 * 1024),
        ),
        RuntimeSnapshot.restore(
            snapshot_ref,
            object_store=store,
            target=target,
            namespace="namespace",
            tenant_id="tenant",
            limits=SnapshotLimits(max_entries=100, max_bytes=1024 * 1024),
        ),
    )

    assert first.generation == second.generation
    generations = tuple((target / "generations").iterdir())
    assert len(generations) == 1
    staging = target / ".staging"
    assert not staging.exists() or not tuple(staging.iterdir())
