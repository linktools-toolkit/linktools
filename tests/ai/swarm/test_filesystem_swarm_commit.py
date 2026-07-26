#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemSwarmCommitCoordinator contract: same shape as the SQL impl --
every operation is idempotent by commit_id, same-id-different-payload
raises SwarmCommitConflictError, terminal complete is idempotent."""

import asyncio
import tempfile
from datetime import datetime, timezone

import pytest

from linktools.ai.swarm.commit import (
    CompleteSwarmCommand,
    StartSwarmCommand,
    SwarmCommitConflictError,
)
from linktools.ai.swarm.models import SwarmStatus, TokenUsage
from linktools.ai.swarm.persistence.filesystem import FilesystemSwarmStore
from linktools.ai.swarm.persistence.filesystem_commit import (
    FilesystemSwarmCommitCoordinator,
)


def _run(coro):
    return asyncio.run(coro)


def _now():
    return datetime.now(timezone.utc)


def _make(tmp_path):
    swarm_store = FilesystemSwarmStore(root=tmp_path / "swarm")
    transactions_root = tmp_path / "transactions"
    coordinator = FilesystemSwarmCommitCoordinator(
        swarm_store, transactions_root=transactions_root
    )
    return coordinator, swarm_store


def test_fs_start_is_idempotent(tmp_path):
    coordinator, _ = _make(tmp_path)

    async def _run_async():
        command = StartSwarmCommand(
            commit_id="start:swarm-fs-1",
            swarm_run_id="swarm-fs-1",
            expected_version=1,
            payload={
                "id": "swarm-fs-1",
                "run_id": "driving-fs-1",
                "round": 0,
                "status": SwarmStatus.PENDING,
                "version": 1,
                "token_usage": TokenUsage(),
                "cost": "0",
                "created_at": _now(),
                "updated_at": _now(),
            },
            event_context=None,
        )
        first = await coordinator.start(command)
        second = await coordinator.start(command)
        return first, second

    first, second = _run(_run_async())
    assert first == second


def test_fs_start_same_commit_id_different_payload_conflicts(tmp_path):
    coordinator, _ = _make(tmp_path)

    async def _run_async():
        base = {
            "id": "swarm-fs-x",
            "round": 0,
            "status": SwarmStatus.PENDING,
            "version": 1,
            "token_usage": TokenUsage(),
            "cost": "0",
            "created_at": _now(),
            "updated_at": _now(),
        }
        await coordinator.start(
            StartSwarmCommand(
                commit_id="start:swarm-fs-x",
                swarm_run_id="swarm-fs-x",
                expected_version=1,
                payload={**base, "run_id": "driving-a"},
                event_context=None,
            )
        )
        with pytest.raises(SwarmCommitConflictError):
            await coordinator.start(
                StartSwarmCommand(
                    commit_id="start:swarm-fs-x",
                    swarm_run_id="swarm-fs-x",
                    expected_version=1,
                    payload={**base, "run_id": "driving-b"},
                    event_context=None,
                )
            )

    _run(_run_async())


def test_fs_terminal_complete_is_idempotent(tmp_path):
    coordinator, swarm_store = _make(tmp_path)
    _run(
        coordinator.start(
            StartSwarmCommand(
                commit_id="start:swarm-fs-c",
                swarm_run_id="swarm-fs-c",
                expected_version=1,
                payload={
                    "id": "swarm-fs-c",
                    "run_id": "driving-fs-c",
                    "round": 0,
                    "status": SwarmStatus.PENDING,
                    "version": 1,
                    "token_usage": TokenUsage(),
                    "cost": "0",
                    "created_at": _now(),
                    "updated_at": _now(),
                },
                event_context=None,
            )
        )
    )
    _run(
        swarm_store.update_run(
            "swarm-fs-c", expected_version=1, status=SwarmStatus.RUNNING
        )
    )

    async def _run_async():
        command = CompleteSwarmCommand(
            commit_id="complete:swarm-fs-c",
            swarm_run_id="swarm-fs-c",
            expected_version=2,
            payload={},
            event_context=None,
        )
        first = await coordinator.complete(command)
        second = await coordinator.complete(command)
        return first, second

    first, second = _run(_run_async())
    assert first == second
    run = _run(swarm_store.get_run("swarm-fs-c"))
    assert run.status is SwarmStatus.SUCCEEDED


def test_fs_recovery_marks_inflight_commit_failed(tmp_path):
    """A crash leaving an in-flight swarm journal (the commit started but did
    not finish) reconciles at next startup: recovery marks the affected
    swarm_run FAILED (fail-closed) and clears the journal."""
    coordinator, swarm_store = _make(tmp_path)
    # Seed a swarm_run + an in-flight journal as if a complete commit had
    # started and crashed before the completion-log landed.
    _run(
        coordinator.start(
            StartSwarmCommand(
                commit_id="start:swarm-fs-r",
                swarm_run_id="swarm-fs-r",
                expected_version=1,
                payload={
                    "id": "swarm-fs-r",
                    "run_id": "driving-fs-r",
                    "round": 0,
                    "status": SwarmStatus.PENDING,
                    "version": 1,
                    "token_usage": TokenUsage(),
                    "cost": "0",
                    "created_at": _now(),
                    "updated_at": _now(),
                },
                event_context=None,
            )
        )
    )
    _run(
        swarm_store.update_run(
            "swarm-fs-r", expected_version=1, status=SwarmStatus.RUNNING
        )
    )
    # Simulate the crash mid-complete: hand-write the in-flight journal.
    import hashlib
    import json
    import os

    commit_id = "complete:swarm-fs-r"
    inflight_path = (
        tmp_path / "transactions" / "swarm_inflight"
        / f"{hashlib.sha256(commit_id.encode()).hexdigest()}.json"
    )
    inflight_path.parent.mkdir(parents=True, exist_ok=True)
    inflight_path.write_text(
        json.dumps(
            {
                "commit_id": commit_id,
                "operation": "complete",
                "swarm_run_id": "swarm-fs-r",
                "request_hash": "00",
            }
        ),
        encoding="utf-8",
    )
    _ = os  # silence linter

    _run(coordinator.recover_incomplete_commits())

    run = _run(swarm_store.get_run("swarm-fs-r"))
    assert run.status is SwarmStatus.FAILED
    assert not inflight_path.exists()
