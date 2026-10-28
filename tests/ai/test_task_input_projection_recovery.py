#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projection outages retain the original graph and successful predecessors."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import AgentTaskContextError, AgentTaskInputBuilder, AgentTaskInputContext
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine

from ._runtime_test_helpers import model_features
from .test_task_mixed_node_reliability import _TaskTestModels


@asynccontextmanager
async def _environment(
    path: Path, build: AgentTaskInputBuilder,
) -> AsyncIterator[tuple[Persistence, Runtime, TaskEngine, Task, Task]]:
    features = RuntimeFeatures("projection")
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with (
        Persistence.open("projection", storage=PersistenceStorage.sqlite(path)) as persistence,
        Runtime.attach(persistence=persistence, features=model_features(_TaskTestModels(), features)) as runtime,
        TaskEngine.open(persistence=persistence, context=runtime.context,
                        authorization=runtime._execution_service._authorization) as engine,
    ):
        a = Task.from_agent("example.a", runtime.agents.get())
        b = Task.from_agent("example.b", runtime.agents.get(), build_input=build)
        yield persistence, runtime, engine.with_definitions(a, b), a, b


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_source", ("callback", "declared_result", "recovery_commit_unknown"))
async def test_projection_outage_recovers_original_graph_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_source: str,
) -> None:
    requests = 0
    request = TestModel._request

    def count(*args: object, **kwargs: object) -> object:
        nonlocal requests
        requests += 1
        return request(*args, **kwargs)

    monkeypatch.setattr(TestModel, "_request", count)
    available = False
    builds = 0
    leases = []

    async def build(context: AgentTaskInputContext) -> str:
        nonlocal builds
        builds += 1
        leases.append(context._invocation.task_lease)
        result = await context.result("a")
        if not available and failure_source != "declared_result":
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE, "private diagnostic")
        return str(result)

    reader = AgentTaskInputContext.result

    async def read(context: AgentTaskInputContext, name: str) -> object:
        if not available and failure_source == "declared_result":
            cause = AIError(ErrorCode.STORAGE_UNAVAILABLE)
            raise AgentTaskContextError(cause.code) from cause
        return await reader(context, name)

    monkeypatch.setattr(AgentTaskInputContext, "result", read)

    path = tmp_path / "state.sqlite"
    async with _environment(path, build) as (persistence, runtime, engine, a, b):
        if failure_source == "recovery_commit_unknown":
            repository = persistence.task.tasks
            mark = repository.mark_recovery_required
            mutate = repository._mutate_with_event_retry

            async def commit_then_unknown(*args: object, **kwargs: object) -> object:
                await mutate(*args, **kwargs)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

            async def mark_with_unknown(*args: object, **kwargs: object) -> object:
                with monkeypatch.context() as fault:
                    fault.setattr(repository, "_mutate_with_event_retry", commit_then_unknown)
                    return await mark(*args, **kwargs)

            monkeypatch.setattr(repository, "mark_recovery_required", mark_with_unknown)
        run = await engine.start(TaskGraph("a-to-b", (
            TaskNode("a", task=a, input={"prompt": "A"}),
            TaskNode("b", ("a",), task=b),
        )), principal=runtime.default_principal, idempotency_key="start")
        failed = (await run.wait(timeout_seconds=10)).result
        assert failed.status is TaskStatus.RECOVERY_REQUIRED
        first_a, first_b = failed.node_states
        assert first_a.status is TaskStatus.SUCCEEDED
        assert first_b.status is TaskStatus.RECOVERY_REQUIRED
        assert first_b.execution_id is None
        assert first_b.error_code == ErrorCode.TASK_INPUT_PROJECTION_FAILED.value
        assert dict(first_b.safe_error_details) == {
            "phase": "task_input_projection", "cause_code": "STORAGE_UNAVAILABLE",
        }
        assert requests == 1
        assert await persistence.task.tasks.get_prepared_input(
            "a-to-b", "b", tenant_id=runtime.tenant_id,
        ) is None
        admitted = await persistence.task.admissions.get("a-to-b", tenant_id=runtime.tenant_id)
        assert admitted is not None

    async with _environment(path, build) as (persistence, runtime, engine, a, b):
        run = await engine.get("a-to-b", principal=runtime.default_principal)
        # A sustained outage remains recoverable after another explicit attempt.
        await run.recover(idempotency_key="outage")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.RECOVERY_REQUIRED
        assert requests == 1
        available = True
        await asyncio.gather(*(run.recover(idempotency_key="resume") for _ in range(2)))
        completed = (await run.wait(timeout_seconds=10)).result
        assert completed.status is TaskStatus.SUCCEEDED
        final_a, final_b = completed.node_states
        assert final_a.execution_id == first_a.execution_id
        assert final_a.result_digest == first_a.result_digest
        assert final_b.execution_id is not None
        assert final_b.fence > first_b.fence
        assert requests == 2
        assert builds == 3
        prepared = await persistence.task.tasks.get_prepared_input(
            "a-to-b", "b", tenant_id=runtime.tenant_id,
        )
        assert prepared is not None
        source_ref = prepared.source_refs[0][1]
        assert source_ref.graph_id == "a-to-b"
        assert source_ref.node_id == "a"
        assert source_ref.tenant_id == runtime.tenant_id
        assert source_ref.result_digest == first_a.result_digest
        assert await persistence.task.admissions.get("a-to-b", tenant_id=runtime.tenant_id) == admitted
        await run.recover(idempotency_key="resume")
        assert requests == 2
        with pytest.raises(AIError) as stale:
            await persistence.task.tasks.mark_recovery_required(
                leases[0], tenant_id=runtime.tenant_id,
                error_code=first_b.error_code, error_digest=first_b.error_digest,
                safe_error_details=first_b.safe_error_details,
                expected_fence=first_b.fence,
            )
        assert stale.value.code is ErrorCode.TASK_FENCE_STALE
        assert (await run.wait(timeout_seconds=10)).result.node_states == completed.node_states


@pytest.mark.asyncio
@pytest.mark.parametrize("code,retryable", (
    (ErrorCode.STORAGE_OWNER_MISMATCH, None),
    (ErrorCode.STORAGE_INTEGRITY_ERROR, True),
    (ErrorCode.REQUEST_FIELD_INVALID, True),
    (ErrorCode.AUTHORIZATION_DENIED, True),
    (ErrorCode.TASK_FENCE_STALE, True),
    (ErrorCode.STORAGE_UNAVAILABLE, False),
))
async def test_permanent_projection_failures_remain_terminal(
    tmp_path: Path, code: ErrorCode, retryable: bool | None,
) -> None:
    async def build(context: AgentTaskInputContext) -> str:
        raise AIError(code, retryable=retryable)

    async with _environment(tmp_path / "state.sqlite", build) as (_, runtime, engine, a, b):
        run = await engine.start(TaskGraph("permanent", (TaskNode("b", task=b),)),
                                 principal=runtime.default_principal, idempotency_key="start")
        failed = (await run.wait(timeout_seconds=10)).result
        assert failed.status is TaskStatus.FAILED
        assert failed.node_states[0].execution_id is None
        with pytest.raises(AIError) as caught:
            await run.recover(idempotency_key="recover")
        assert caught.value.code is ErrorCode.TASK_NOT_READY


@pytest.mark.asyncio
@pytest.mark.parametrize("published", (False, True))
async def test_projection_replays_only_before_prepared_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published: bool,
) -> None:
    builds = 0

    async def build(context: AgentTaskInputContext) -> str:
        nonlocal builds
        builds += 1
        return "fixed prompt"

    async with _environment(tmp_path / "state.sqlite", build) as (persistence, runtime, engine, a, b):
        publish = engine._task_node_runtime.publish_prepared_agent_input
        interrupted = False

        async def publish_then_interrupt(*args: object, **kwargs: object) -> object:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                if published:
                    await publish(*args, **kwargs)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return await publish(*args, **kwargs)

        monkeypatch.setattr(engine._task_node_runtime, "publish_prepared_agent_input", publish_then_interrupt)
        run = await engine.start(TaskGraph("prepared", (TaskNode("b", task=b),)),
                                 principal=runtime.default_principal, idempotency_key="start")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.RECOVERY_REQUIRED
        before = await persistence.task.tasks.get_prepared_input(
            "prepared", "b", tenant_id=runtime.tenant_id,
        )
        assert (before is not None) is published
        await run.recover(idempotency_key="recover")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert builds == (1 if published else 2)
        after = await persistence.task.tasks.get_prepared_input(
            "prepared", "b", tenant_id=runtime.tenant_id,
        )
        assert after is not None
        if published:
            assert after == before


@pytest.mark.asyncio
async def test_cancelled_projection_recovery_does_not_start_consumer(tmp_path: Path) -> None:
    builds = 0

    async def build(context: AgentTaskInputContext) -> str:
        nonlocal builds
        builds += 1
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    async with _environment(tmp_path / "state.sqlite", build) as (_, runtime, engine, a, b):
        run = await engine.start(TaskGraph("cancel", (TaskNode("b", task=b),)),
                                 principal=runtime.default_principal, idempotency_key="start")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.RECOVERY_REQUIRED
        await run.cancel(idempotency_key="cancel")
        # Accepted cancellation controls subsequent recovery of the same graph.
        try:
            await run.recover(idempotency_key="recover")
        except AIError as error:
            assert error.code is ErrorCode.TASK_NOT_READY
        completed = (await run.wait(timeout_seconds=10)).result
        assert completed.status is TaskStatus.CANCELLED
        assert completed.node_states[0].execution_id is None
        assert builds == 1


@pytest.mark.asyncio
async def test_uncertain_consumer_start_reuses_prepared_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds = 0

    async def build(context: AgentTaskInputContext) -> str:
        nonlocal builds
        builds += 1
        return "frozen prompt"

    async with _environment(tmp_path / "state.sqlite", build) as (_, runtime, engine, a, b):
        start = type(b.agent_execution).start
        accepted = None

        async def start_then_unknown(*args: object, **kwargs: object) -> object:
            nonlocal accepted
            execution = await start(*args, **kwargs)
            if accepted is None:
                accepted = execution.execution_id
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            return execution

        monkeypatch.setattr(type(b.agent_execution), "start", start_then_unknown)
        run = await engine.start(TaskGraph("uncertain", (TaskNode("b", task=b),)),
                                 principal=runtime.default_principal, idempotency_key="start")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.RECOVERY_REQUIRED
        assert accepted is not None
        await run.recover(idempotency_key="recover")
        completed = (await run.wait(timeout_seconds=10)).result
        assert completed.status is TaskStatus.SUCCEEDED
        assert completed.node_states[0].execution_id == accepted
        assert builds == 1
        interactions = await runtime.history.model_interactions(
            accepted, principal=runtime.default_principal,
        )
        assert len(interactions.items) == 1
