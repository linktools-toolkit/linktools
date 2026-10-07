#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime recovery coverage for durable TaskGraph state."""

import asyncio
from dataclasses import fields
from pathlib import Path

import pytest
from linktools.ai.core import (
    AuthorizationAction,
    JsonValue,
    Principal,
    ResourceRef,
    TaskStatus,
    canonical_json_bytes,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
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
    TaskNodeInfo,
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
            "kind": "task-graph-binding-capture",
            "format_version": 1,
            "namespace": "default",
            "tenant_id": admission.principal.tenant_id,
            "graph_id": admission.graph_id,
            "request_digest": admission.initial_request_digest,
            "tasks": [
                {
                    "version": 1,
                    "id": "test.recovery",
                    "revision": 1,
                    "type": "function",
                    "effect_policy": "none",
                    "output_contract": {"kind": "json"},
                    "cancel": False,
                    "reconcile": False,
                }
            ],
            "expanders": [],
        }
        snapshot_payload = canonical_json_bytes(snapshot_manifest)
        snapshot_digest = canonical_sha256(snapshot_manifest)
        snapshot_key = (
            "v1/task-graph-binding-capture/"
            + canonical_sha256(
                {
                    "version": 1,
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
        result = (await run.wait(timeout_seconds=10)).result
        page = await reopened.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )

    assert result.status is TaskStatus.SUCCEEDED
    assert tuple(node.status for node in result.node_states) == (
        TaskStatus.SUCCEEDED,
    )
    assert page.items == ()


@pytest.mark.asyncio
async def test_runtime_batch_recovery_keeps_run_only_actor_at_safe_service_boundary(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    async def hold(context: TaskNodeContext[None]) -> JsonValue:
        del context
        started.set()
        await asyncio.Event().wait()
        return {"finished": True}

    handler = Task("test.recovery-actor", hold, effect_policy="none")
    graph = TaskGraph(
        "recovery-actor",
        (
            TaskNode(
                "root",
                task=handler,
                input={"prompt": "private recovery input"},
            ),
        ),
    )
    execution_principal = Principal("executor", "default")
    actor = Principal("operator", "default")
    storage_root = tmp_path / "recovery-actor"

    async with Runtime.open(
        "recovery-actor",
        models=ModelRegistry.openai(model="gpt-test"),
        storage=RuntimeStorage.filesystem(storage_root),
    ) as runtime:
        await runtime.tasks.bind(handler).start(
            graph,
            idempotency_key="recovery-actor-run-0001",
            principal=execution_principal,
        )
        await asyncio.wait_for(started.wait(), 10)

    class RecoveryAuthorization:
        def __init__(self) -> None:
            self.calls: list[tuple[Principal, AuthorizationAction]] = []

        async def authorize(
            self,
            principal: Principal,
            action: AuthorizationAction,
            resource: ResourceRef,
        ) -> None:
            del resource
            self.calls.append((principal, action))
            if principal == actor and action is AuthorizationAction.TASK_RUN:
                return
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    authorization = RecoveryAuthorization()
    async with Runtime.open(
        "recovery-actor",
        models=ModelRegistry.openai(model="gpt-test"),
        storage=RuntimeStorage.filesystem(storage_root),
    ) as runtime:
        runtime._graph_service._authorization = authorization
        with pytest.raises(AIError) as read_denied:
            await runtime._graph_service.state(graph.graph_id, principal=actor)
        assert read_denied.value.code is ErrorCode.AUTHORIZATION_DENIED

        recovery_nodes = await runtime._graph_service.recovery_nodes(
            graph.graph_id,
            principal=actor,
        )
        assert len(recovery_nodes) == 1
        assert isinstance(recovery_nodes[0], TaskNodeInfo)
        recovery_fields = {item.name for item in fields(recovery_nodes[0])}
        assert "input" not in recovery_fields
        assert "status" not in recovery_fields
        assert "private recovery input" not in repr(recovery_nodes[0])

        page = await runtime.tasks.bind(handler).recover_pending(principal=actor)

    assert tuple(result.graph_id for result in page.items) == (graph.graph_id,)
    assert authorization.calls
    assert all(principal == actor for principal, _action in authorization.calls)
    assert all(
        action is AuthorizationAction.TASK_RUN
        for _principal, action in authorization.calls[1:]
    )
    assert authorization.calls[0] == (actor, AuthorizationAction.TASK_READ)
