#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime recovery coverage for durable TaskGraph state."""

import asyncio
from pathlib import Path

import pytest
from linktools.ai.core import (
    JsonValue,
    Principal,
    TaskStatus,
    canonical_json_bytes,
    canonical_sha256,
)
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.task import (
    Task,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskNode,
    TaskNodeContext,
)
from linktools.ai.storage import FilesystemObjectStore
from sqlalchemy.ext.asyncio import create_async_engine


async def _recover_node(context: TaskNodeContext[None]) -> JsonValue:
    return {"graph_id": context.graph_id, "node_id": context.node_id}


@pytest.mark.asyncio
async def test_sqlite_runtime_explicit_recovery_recovers_expired_task_lease(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    try:
        await provision_runtime_database(engine)
    finally:
        await engine.dispose()

    handler = Task("test.recovery", _recover_node, effect_policy="none")
    graph = TaskGraph("reopen-expired", (TaskNode("root", task=handler),))
    admission = TaskGraphAdmission.from_request(
        TaskGraphRequest(
            graph,
            Principal("tester", "default"),
            "submit:reopen-expired",
            TaskGraphLimits(max_concurrency=1),
        )
    )
    state = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    await state.initialize(
        namespace="default",
        tenant_id="default",
    )
    try:
        await state.task.admissions.admit(
            admission,
            graph,
        )
        snapshot_manifest: dict[str, JsonValue] = {
            "kind": "task-definition-capture",
            "format_version": 2,
            "namespace": "default",
            "tenant_id": admission.principal.tenant_id,
            "graph_id": admission.graph_id,
            "request_digest": admission.initial_request_digest,
            "roots": {},
            "bindings": {},
            "tasks": [
                {
                    "version": 1,
                    "id": "test.recovery",
                    "revision": 1,
                    "type": "function",
                    "effect_policy": "none",
                    "output_contract": {"kind": "json"},
                    "reconcile": False,
                }
            ],
            "expanders": [],
        }
        snapshot_payload = canonical_json_bytes(snapshot_manifest)
        snapshot_digest = canonical_sha256(snapshot_manifest)
        snapshot_key = (
            "v2/task-capture/"
            + canonical_sha256(
                {
                    "version": 2,
                    "namespace": "default",
                    "tenant_id": admission.principal.tenant_id,
                    "graph_id": admission.graph_id,
                    "request_digest": admission.initial_request_digest,
                }
            )
        )

        async def snapshot_chunks():
            yield snapshot_payload

        await state.object_store(RuntimeDomain.TASK).put(
            snapshot_key,
            snapshot_chunks(),
            expected_size=len(snapshot_payload),
            expected_digest=snapshot_digest,
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

    reopened = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    async with Runtime.open(
        "default",
        models=ModelRegistry.openai(model="gpt-test"),
        storage=reopened,
    ) as runtime:
        run = await runtime.tasks.bind(handler).get(graph.graph_id)
        await run.recover(idempotency_key="recover-expired-lease")
        result = await run.wait(timeout_seconds=10)
        page = await reopened.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )

    assert result.status is TaskStatus.SUCCEEDED
    assert tuple(node.status for node in result.node_results) == (
        TaskStatus.SUCCEEDED,
    )
    assert page.items == ()
