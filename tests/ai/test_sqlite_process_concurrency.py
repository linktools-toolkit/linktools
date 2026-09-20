#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite process-concurrency regression coverage."""

import asyncio
import hashlib
import multiprocessing
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state._contracts import ArtifactRecord
from linktools.ai.storage import ObjectRef
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _local_metrics


_PROCESS_TIMEOUT_SECONDS = 15.0
_NAMESPACE = "sqlite-process-concurrency"
_TENANT_ID = "tenant"


def _join_processes(processes: tuple[multiprocessing.Process, ...]) -> None:
    deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
    for process in processes:
        process.start()
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    alive = tuple(process for process in processes if process.is_alive())
    if alive:
        for process in alive:
            process.terminate()
        for process in alive:
            process.join()
        pytest.fail("SQLite concurrency worker did not exit before the deadline")
    exit_codes = tuple(process.exitcode for process in processes)
    assert exit_codes == (0,) * len(processes)


def _local_metrics_pair_worker(root: str) -> None:
    async def run() -> None:
        workspace = Workspace.load(Path(root))
        await asyncio.gather(
            _local_metrics(workspace),
            _local_metrics(workspace),
        )

    asyncio.run(run())


def _local_metrics_worker(root: str, barrier) -> None:
    barrier.wait()

    async def run() -> None:
        await _local_metrics(Workspace.load(Path(root)))

    asyncio.run(run())


def _runtime_bootstrap_worker(database: str, barrier) -> None:
    barrier.wait()

    async def run() -> None:
        state = RuntimeState.sqlite(Path(database))
        await state.initialize(namespace=_NAMESPACE, tenant_id=_TENANT_ID)
        await state.close()

    asyncio.run(run())


def _artifact_write_worker(database: str, barrier) -> None:
    barrier.wait()

    async def run() -> None:
        state = RuntimeState.sqlite(Path(database))
        await state.initialize(namespace=_NAMESPACE, tenant_id=_TENANT_ID)
        try:
            created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
            for index in range(8):
                payload = f"artifact-{index}".encode("utf-8")
                await state.artifact.records.put_metadata(
                    ArtifactRecord(
                        artifact_id=f"artifact-{index}",
                        execution_id="execution",
                        producer="concurrency-test",
                        media_type="application/octet-stream",
                        object_ref=ObjectRef(
                            "runtime",
                            f"artifact/{index}",
                            hashlib.sha256(payload).hexdigest(),
                            len(payload),
                        ),
                        created_at=created_at,
                    )
                )
        finally:
            await state.close()

    asyncio.run(run())


def test_local_metrics_concurrent_initialization_does_not_block_event_loop(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_local_metrics_pair_worker,
        args=(str(workspace.root),),
    )

    _join_processes((process,))


def test_local_metrics_initialize_once_across_processes(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    processes = tuple(
        context.Process(
            target=_local_metrics_worker,
            args=(str(workspace.root), barrier),
        )
        for _ in range(2)
    )

    _join_processes(processes)

    database = workspace.storage_root / "runtime" / "metrics.db"
    assert database.is_file()


@pytest.mark.asyncio
async def test_runtime_sqlite_bootstrap_converges_across_processes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runtime.db"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    processes = tuple(
        context.Process(
            target=_runtime_bootstrap_worker,
            args=(str(database), barrier),
        )
        for _ in range(2)
    )

    _join_processes(processes)

    state = RuntimeState.sqlite(database)
    await state.initialize(
        namespace=_NAMESPACE,
        tenant_id=_TENANT_ID,
        read_only=True,
    )
    await state.close()


@pytest.mark.asyncio
async def test_runtime_sqlite_concurrent_idempotent_writes_converge(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runtime.db"
    initial = RuntimeState.sqlite(database)
    await initial.initialize(namespace=_NAMESPACE, tenant_id=_TENANT_ID)
    await initial.close()

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    processes = tuple(
        context.Process(
            target=_artifact_write_worker,
            args=(str(database), barrier),
        )
        for _ in range(2)
    )

    _join_processes(processes)

    state = RuntimeState.sqlite(database)
    await state.initialize(
        namespace=_NAMESPACE,
        tenant_id=_TENANT_ID,
        read_only=True,
    )
    try:
        for index in range(8):
            record = await state.artifact.records.get_metadata(
                f"artifact-{index}",
                tenant_id=_TENANT_ID,
            )
            assert record is not None
            assert record.execution_id == "execution"
    finally:
        await state.close()
