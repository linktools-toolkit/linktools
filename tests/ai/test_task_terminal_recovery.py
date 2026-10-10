#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery settles against terminal graph truth after local binding cleanup."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

import pytest

from linktools.ai.core import (
    OperationStatus, TaskStatus, TenantAuthorizationPolicy, canonical_sha256, idempotency_key_digest,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.task import (
    DefaultTaskGraphService, RecoverGraphRequest, Task, TaskGraph, TaskNode, TaskNodeContext,
)

from .test_task_recovery_service import _Launcher


@asynccontextmanager
async def _running_graph(storage: RuntimeStorage) -> AsyncIterator[tuple]:
    async def target(context: TaskNodeContext[None]) -> str:
        return "result"

    task = Task("terminal-recovery.target", target, effect_policy="none")
    async with Runtime.open(
        "terminal-recovery", models=ModelRegistry(), storage=storage, auto_recover=False,
    ) as runtime:
        principal = runtime.default_principal
        submission = await runtime.tasks.bind(task).prepare_submission(
            TaskGraph("graph", (TaskNode("target", task=task),)),
            principal=principal, idempotency_key="submit",
        )
        await storage.task.admissions.admit_prepared(submission)
        runner = runtime._task_node_runtime
        await runner.activate_graph(submission.graph, (task,), ())
        lease = await storage.task.tasks.claim(
            "graph", "target", tenant_id=principal.tenant_id, owner="worker", lease_seconds=30,
        )
        launcher = _Launcher()
        service = DefaultTaskGraphService(
            storage.task, TenantAuthorizationPolicy(principal.tenant_id), launcher,
            preflight=runner, bound_execution_recovery=runner,
        )
        yield runtime, runner, service, launcher, lease


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("boundary", ("admission", "bound", "after_bound", "replay", "replay_retained_binding"))
async def test_recovery_settles_terminal_graph_without_rearming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, boundary: str,
) -> None:
    storage = (RuntimeStorage.in_memory() if backend == "memory"
               else RuntimeStorage.sqlite(tmp_path / "state.sqlite"))
    async with _running_graph(storage) as (runtime, runner, service, launcher, lease):
        principal = runtime.default_principal
        request = RecoverGraphRequest(principal, "recover")
        definitions = runner._active_definitions["graph"]
        if boundary == "replay_retained_binding":
            monkeypatch.setattr(runner, "_default_definitions", definitions)
        if boundary.startswith("replay"):
            assert (await service.recover("graph", request)).status is TaskStatus.RUNNING
            launcher.started.clear()
        entered, release = asyncio.Event(), asyncio.Event()
        in_bound_recovery = ContextVar("in_bound_recovery", default=False)
        load = runner.load_admission
        recover = runner.recover_bound_executions

        async def gated_load(admission):
            if not entered.is_set() and (
                ((boundary == "admission" or boundary.startswith("replay")) and not in_bound_recovery.get())
                or (boundary == "bound" and in_bound_recovery.get())
            ):
                entered.set()
                await asyncio.wait_for(release.wait(), 10)
            await load(admission)

        async def gated_recover(graph_id, recovery_request):
            token = in_bound_recovery.set(True)
            try:
                await recover(graph_id, recovery_request)
                if boundary == "after_bound":
                    entered.set()
                    await asyncio.wait_for(release.wait(), 10)
            finally:
                in_bound_recovery.reset(token)

        monkeypatch.setattr(runner, "load_admission", gated_load)
        monkeypatch.setattr(runner, "recover_bound_executions", gated_recover)
        pending = asyncio.create_task(service.recover("graph", request))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            await storage.task.tasks.complete(
                lease, tenant_id=principal.tenant_id, execution_id="execution",
                result_digest=canonical_sha256("result"),
            )
            completed = await storage.task.tasks.graph_state("graph", tenant_id=principal.tenant_id)
            assert completed.status is TaskStatus.SUCCEEDED
            await runner.release_graph_dependencies(completed, tenant_id=principal.tenant_id)
            release.set()
            result = await asyncio.wait_for(pending, 10)
            assert result.status is TaskStatus.SUCCEEDED
            assert result.node_results[0].execution_id == "execution"
            assert launcher.started == []
            operation = await storage.task.operations.get(
                idempotency_key_digest(request.idempotency_key), tenant_id=principal.tenant_id,
            )
            assert operation.status is OperationStatus.SUCCEEDED
            assert operation.result_ref == "graph"
            assert "graph" not in runner._admitted_binding_captures
            assert "graph" not in runner._active_definitions
            await runner.activate_graph(
                TaskGraph(completed.graph_id, completed.nodes), tuple(definitions[0].values()), (),
            )
            admission = await storage.task.admissions.get("graph", tenant_id=principal.tenant_id)
            await runner.load_admission(admission)
            assert "graph" in runner._active_definitions
            assert "graph" in runner._admitted_binding_captures
            assert (await service.recover("graph", request)).status is TaskStatus.SUCCEEDED
            assert launcher.started == []
            assert "graph" not in runner._active_definitions
            assert "graph" not in runner._admitted_binding_captures
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("effect_unknown", (False, True))
async def test_missing_binding_does_not_settle_nonterminal_recovery(effect_unknown: bool) -> None:
    storage = RuntimeStorage.in_memory()
    async with _running_graph(storage) as (runtime, runner, service, launcher, lease):
        principal = runtime.default_principal
        if effect_unknown:
            await storage.task.tasks.mark_recovery_required(
                lease, tenant_id=principal.tenant_id, execution_id="external-execution",
                error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                error_digest=canonical_sha256("unknown external outcome"),
            )
        runner._active_definitions.clear()
        with pytest.raises(AIError) as raised:
            await service.recover("graph", RecoverGraphRequest(principal, "missing-binding"))
        assert raised.value.code is ErrorCode.BINDING_NOT_REGISTERED
        state = await storage.task.tasks.graph_state("graph", tenant_id=principal.tenant_id)
        expected = TaskStatus.RECOVERY_REQUIRED if effect_unknown else TaskStatus.RUNNING
        assert state.status is expected
        assert state.node_states[0].status is expected
        assert launcher.started == []
        operation = await storage.task.operations.get(
            idempotency_key_digest("missing-binding"), tenant_id=principal.tenant_id,
        )
        assert operation.status is OperationStatus.RUNNING


@pytest.mark.asyncio
@pytest.mark.parametrize("code", (ErrorCode.STORAGE_INTEGRITY_ERROR, ErrorCode.TOOL_EFFECT_UNKNOWN))
async def test_terminal_graph_does_not_hide_recovery_integrity_or_effect_error(
    monkeypatch: pytest.MonkeyPatch, code: ErrorCode,
) -> None:
    storage = RuntimeStorage.in_memory()
    async with _running_graph(storage) as (runtime, runner, service, launcher, lease):
        principal = runtime.default_principal

        async def fail_after_completion(graph_id, request):
            await storage.task.tasks.complete(
                lease, tenant_id=principal.tenant_id, execution_id="execution",
                result_digest=canonical_sha256("result"),
            )
            raise AIError(code)

        monkeypatch.setattr(runner, "recover_bound_executions", fail_after_completion)
        with pytest.raises(AIError) as raised:
            await service.recover("graph", RecoverGraphRequest(principal, "failed-recovery"))
        assert raised.value.code is code
        assert launcher.started == []
        operation = await storage.task.operations.get(
            idempotency_key_digest("failed-recovery"), tenant_id=principal.tenant_id,
        )
        assert operation.status is OperationStatus.RUNNING
