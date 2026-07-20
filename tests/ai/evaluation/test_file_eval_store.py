#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemEvaluationStore persistence tests."""

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.evaluation.models import (
    EvalResult,
    EvalRun,
    EvalRunStatus,
    EvalTarget,
)
from linktools.ai.evaluation.store import (
    EvalResultConflictError,
    EvalRunNotFoundError,
)
from linktools.ai.storage.filesystem.evaluation import FilesystemEvaluationStore


def test_file_eval_store_round_trip(tmp_path) -> None:
    store = FilesystemEvaluationStore(tmp_path)

    async def run() -> None:
        now = datetime.now(timezone.utc)
        eval_run = EvalRun(
            id="er1",
            suite_id="s1",
            target=EvalTarget(kind="agent", id="a1"),
            status=EvalRunStatus.RUNNING,
            baseline_target=None,
            created_at=now,
            started_at=now,
        )
        await store.create_run(eval_run)
        fetched = await store.get_run("er1")
        assert fetched is not None
        # Enum status + nested EvalTarget round-trip through generic serde.
        assert fetched.status == EvalRunStatus.RUNNING
        assert fetched.target.id == "a1"
        updated = await store.transition_run(
            "er1", status=EvalRunStatus.SUCCEEDED, finished_at=now
        )
        assert updated.status == EvalRunStatus.SUCCEEDED
        result = EvalResult(
            id="er1-c1", eval_run_id="er1", case_id="c1", scores={"q": 0.9}
        )
        await store.append_result(result)
        with pytest.raises(EvalResultConflictError):
            await store.append_result(result)
        stored = await store.list_results("er1")
        assert len(stored) == 1
        assert stored[0].scores == {"q": 0.9}
        assert (await store.get_result("er1-c1")).case_id == "c1"
        with pytest.raises(EvalRunNotFoundError):
            await store.transition_run("missing", status=EvalRunStatus.SUCCEEDED)

    asyncio.run(run())


def test_file_eval_store_persists_across_instances(tmp_path) -> None:
    """A fresh FilesystemEvaluationStore over the same root sees prior writes (real
    on-disk persistence, not in-memory)."""
    now = datetime.now(timezone.utc)

    async def write() -> None:
        s = FilesystemEvaluationStore(tmp_path)
        await s.create_run(
            EvalRun(
                id="er1",
                suite_id="s1",
                target=EvalTarget(kind="agent", id="a1"),
                status=EvalRunStatus.RUNNING,
                baseline_target=None,
                created_at=now,
                started_at=now,
            )
        )

    async def read() -> None:
        s = FilesystemEvaluationStore(tmp_path)
        fetched = await s.get_run("er1")
        assert fetched is not None
        assert fetched.target.id == "a1"

    asyncio.run(write())
    asyncio.run(read())
