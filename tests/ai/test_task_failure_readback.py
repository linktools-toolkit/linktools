#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failure readback confirms the same semantic outcome as direct replay."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import TaskGraph, TaskNode

from ._task_test_helpers import admit_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("write_error", (None, ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN))
@pytest.mark.parametrize("changed", (None, "error_origin", "safe_error_details", "error_digest", "execution_id", "fence"))
async def test_failure_readback_preserves_direct_replay_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
    write_error: ErrorCode | None, changed: str | None,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "state.sqlite"))
    async with Persistence.open("failure-readback", storage=storage, tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", (TaskNode("node"),)))
        repository = state.task.tasks
        lease = await repository.claim("graph", "node", tenant_id="tenant", owner="owner", lease_seconds=30)
        requested = {"error_code": "FAILED", "error_digest": "a" * 64,
                     "error_origin": "node", "safe_error_details": {}, "execution_id": "execution"}
        await repository.fail(lease, tenant_id="tenant", **requested)
        if changed == "fence":
            lease = replace(lease, fence=lease.fence + 1)
        elif changed is not None:
            requested[changed] = {"error_origin": "execution", "safe_error_details": {"reason": "other"},
                                  "error_digest": "b" * 64, "execution_id": "other"}[changed]

        async def unknown(operation):
            raise AIError(write_error)

        if write_error is not None:
            monkeypatch.setattr(repository, "_mutate_with_event_retry", unknown)
        if changed is None:
            terminal = await repository.fail(lease, tenant_id="tenant", **requested)
            assert terminal.status is TaskStatus.FAILED
            assert terminal.error_digest == "a" * 64
            assert terminal.execution_id == "execution"
        else:
            with pytest.raises(AIError) as raised:
                await repository.fail(lease, tenant_id="tenant", **requested)
            expected = ErrorCode.TASK_FENCE_STALE if changed == "fence" else ErrorCode.TASK_RESULT_CONFLICT
            assert raised.value.code is expected
        persisted = (await repository.list_nodes("graph", tenant_id="tenant"))[0]
        assert persisted.error_origin == "node"
        assert persisted.safe_error_details == {}
        assert persisted.error_digest == "a" * 64
        assert persisted.execution_id == "execution"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("write_error", (ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN))
async def test_success_readback_does_not_reinterpret_retained_recovery_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, write_error: ErrorCode,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "state.sqlite"))
    async with Persistence.open("success-readback", storage=storage, tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", (TaskNode("node"),)))
        repository = state.task.tasks
        lease = await repository.claim("graph", "node", tenant_id="tenant", owner="owner", lease_seconds=30)
        await repository.mark_recovery_required(
            lease, tenant_id="tenant", execution_id="execution",
            error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value, error_digest="b" * 64,
            error_origin="node", safe_error_details={"reason": "unconfirmed"},
        )
        await repository.complete(
            None, tenant_id="tenant", graph_id="graph", node_id="node", expected_fence=lease.fence,
            execution_id="execution", result_digest="a" * 64,
        )
        current = (await repository.list_nodes("graph", tenant_id="tenant"))[0]
        assert current.error_origin == "node"
        assert current.safe_error_details == {"reason": "unconfirmed"}
        direct = await repository.complete(lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64)

        async def unknown(operation):
            raise AIError(write_error)

        monkeypatch.setattr(repository, "_mutate_with_event_retry", unknown)
        readback = await repository.complete(lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64)
        assert replace(readback, completed_at=direct.completed_at) == direct
