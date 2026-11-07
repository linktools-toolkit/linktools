#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched SQLite Task IO probes, excluding schema creation and fixture setup."""

import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence import _sql
from linktools.ai.persistence._task_repository import TaskRepositoryImpl
from linktools.ai.task import (
    LocalTaskGraphLauncher, Task, TaskGraph, TaskGraphAdmission, TaskGraphLimits,
    TaskGraphRequest, TaskNode, TaskNodeContext,
)
from linktools.ai.task.engine import TaskEngine


@contextmanager
def _measure(monkeypatch: pytest.MonkeyPatch):
    counts = Counter()
    writers = set()
    original = _sql._record_from_row
    original_scheduler = TaskRepositoryImpl.scheduler_state

    async def scheduler(repository, *args, **kwargs):
        counts["scheduler_passes"] += 1
        return await original_scheduler(repository, *args, **kwargs)

    def record_from_row(row):
        counts["record_rows"] += 1
        counts["record_text_binary_bytes"] += sum(
            len(value.encode("utf-8")) if isinstance(value, str) else len(value)
            for value in row.values() if isinstance(value, (str, bytes))
        )
        counts["record_payload_json_bytes"] += len(json.dumps(
            row["payload_json"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
        return original(row)

    def sql(connection, cursor, statement, parameters, context, many):
        word = statement.lstrip().split()[0].upper()
        counts[word] += 1
        if word == "SELECT" and "FROM ai_state_records" in statement:
            counts["record_selects"] += 1
        if word in {"INSERT", "UPDATE", "DELETE"}:
            writers.add(id(connection))

    def begin(connection):
        counts["begins"] += 1

    def commit(connection):
        counts["commits"] += 1
        if id(connection) in writers:
            counts["write_commits"] += 1
            writers.remove(id(connection))

    def rollback(connection):
        counts["rollbacks"] += 1
        writers.discard(id(connection))

    listeners = {"before_cursor_execute": sql, "begin": begin, "commit": commit, "rollback": rollback}
    with monkeypatch.context() as patch:
        patch.setattr(_sql, "_record_from_row", record_from_row)
        patch.setattr(TaskRepositoryImpl, "scheduler_state", scheduler)
        for name, callback in listeners.items():
            event.listen(Engine, name, callback)
        try:
            yield counts
        finally:
            for name, callback in listeners.items():
                event.remove(Engine, name, callback)


@pytest.mark.manual
@pytest.mark.asyncio
@pytest.mark.parametrize("width", [2, 200])
async def test_task_repository_io_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int,
) -> None:
    async with Persistence.open(
        "task-io-probe", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"),
        tenant_id="tenant",
    ) as state:
        target = TaskNode("target", ("source",))
        graph = TaskGraph("graph", (TaskNode("source"), target) + tuple(
            TaskNode(f"unrelated-{i}") for i in range(width - 2)
        ))
        request = TaskGraphRequest(
            graph, Principal("tester", "tenant"), "submit:graph",
            limits=TaskGraphLimits(max_nodes=width),
        )
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)
        repository = state.task.tasks
        lease = await repository.claim(
            "graph", "source", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        results = {}
        with _measure(monkeypatch) as counts:
            lease = await repository.renew(lease, tenant_id="tenant", lease_seconds=30)
        results["renew"] = dict(counts)
        with _measure(monkeypatch) as counts:
            terminal = await repository.complete(
                lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
            )
            assert terminal.status is TaskStatus.SUCCEEDED
        results["complete"] = dict(counts)
        with _measure(monkeypatch) as counts:
            dependencies, statuses = await LocalTaskGraphLauncher(
                repository, None, owner="worker",
            )._dependency_context("graph", target, tenant_id="tenant")
            assert set(dependencies) == set(statuses) == {"source"}
        results["dependencies"] = dict(counts)
        print("TASK_IO_METRICS " + json.dumps({"scope": "repository", "width": width, "metrics": results}, sort_keys=True))


@pytest.mark.manual
@pytest.mark.asyncio
@pytest.mark.parametrize("width", [2, 8])
@pytest.mark.parametrize("sample", [0, 1, 2])
async def test_task_engine_io_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, sample: int,
) -> None:
    seen = []

    async def work(context: TaskNodeContext[None]) -> str:
        seen.append(context.node_id)
        return context.node_id

    task = Task("io-task", work, effect_policy="none")
    graph = TaskGraph("graph", tuple(
        TaskNode(str(index), (str(index - 1),) if index else (), task=task)
        for index in range(width)
    ))
    async with Persistence.open(
        "engine-io-probe", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"),
        tenant_id="tenant",
    ) as state:
        with _measure(monkeypatch) as counts:
            async with TaskEngine.open(task, persistence=state) as engine:
                run = await engine.start(
                    graph, idempotency_key="run", principal=Principal("tester", "tenant"),
                    limits=TaskGraphLimits(max_concurrency=1),
                )
                assert (await run.wait(timeout_seconds=30)).result.status is TaskStatus.SUCCEEDED
                assert await run.result(str(width - 1)) == str(width - 1)
            assert seen == [str(index) for index in range(width)]
        print("TASK_IO_METRICS " + json.dumps({"scope": "engine_open_start_wait_result_close", "width": width, "sample": sample, "metrics": dict(counts)}, sort_keys=True))
