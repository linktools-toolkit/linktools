#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime recovery coverage for durable TaskGraph state."""

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from linktools.ai.core import Principal, TaskStatus, canonical_sha256
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime._planner import RuntimeTaskNodeRunner
from linktools.ai.task import (
    TaskDependencyResult,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskNode,
    TaskNodeRunResult,
)
from linktools.ai.workspace import Workspace
from sqlalchemy.ext.asyncio import create_async_engine


async def _run_digest(
    self: RuntimeTaskNodeRunner,
    node: TaskNode,
    *,
    graph_id: str,
    principal: Principal,
    dependency_results: Mapping[str, TaskDependencyResult],
) -> TaskNodeRunResult:
    del self, principal, dependency_results
    return TaskNodeRunResult(
        canonical_sha256({"graph_id": graph_id, "node_id": node.node_id})
    )


async def _cancel_noop(
    self: RuntimeTaskNodeRunner,
    node: TaskNode,
    *,
    graph_id: str,
    principal: Principal,
    dependency_results: Mapping[str, TaskDependencyResult],
) -> None:
    del self, node, graph_id, principal, dependency_results


@pytest.mark.asyncio
async def test_sqlite_runtime_open_recovers_expired_task_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    try:
        await provision_runtime_database(engine)
    finally:
        await engine.dispose()

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    graph = TaskGraph("reopen-expired", (TaskNode("root"),))
    request = TaskGraphRequest(
        graph,
        Principal("tester", "default"),
        "submit:reopen-expired",
        TaskGraphLimits(max_concurrency=1),
    )
    state = RuntimeState.sqlite(database)
    await state.initialize(
        namespace=workspace.workspace_id,
        tenant_id="default",
    )
    try:
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            graph,
        )
        await state.task.tasks.claim(
            graph.graph_id,
            "root",
            tenant_id="default",
            owner="dead-runtime",
            lease_seconds=1,
        )
    finally:
        await state.close()

    await asyncio.sleep(1.05)
    monkeypatch.setattr(RuntimeTaskNodeRunner, "run", _run_digest)
    monkeypatch.setattr(RuntimeTaskNodeRunner, "cancel", _cancel_noop)

    reopened = RuntimeState.sqlite(database)
    async with Runtime.open(
        workspace,
        models=ModelRegistry.openai(model="gpt-test"),
        state=reopened,
    ) as runtime:
        result = await runtime.task.wait_graph(
            graph.graph_id,
            principal=runtime.default_principal,
            timeout_seconds=10,
        )
        page = await reopened.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )

    assert result.status is TaskStatus.SUCCEEDED
    assert tuple(node.status for node in result.node_results) == (
        TaskStatus.SUCCEEDED,
    )
    assert page.items == ()
