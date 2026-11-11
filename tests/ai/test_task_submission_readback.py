#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepared starts retain admission truth when their response cannot be read."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError, TimeoutError as SQLAlchemyTimeout

from linktools.ai.core import Principal, TaskStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import (
    DefaultTaskGraphService, Task, TaskGraph, TaskGraphHandle, TaskGraphLaunch,
    TaskGraphRequest, TaskGraphSubmission, TaskGraphView, TaskNode, TaskNodeContext,
)
from linktools.ai.task.engine import TaskEngine


class _Launcher:
    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle:
        return TaskGraphHandle(launch.graph_id)

    async def settle_cancel(
        self, launch: TaskGraphLaunch, *, invoke_effects: bool,
    ) -> TaskGraphView:
        return TaskGraphView(launch.graph_id, TaskStatus.CANCELLED, ())


def _request() -> TaskGraphRequest:
    return TaskGraphRequest(
        TaskGraph("graph", (TaskNode("node"),)), Principal("submitter", "tenant"), "start",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("failure", "resolution"), (("sql", "retry"), ("timeout", "retry"), ("sql", "cancel")))
async def test_armed_graph_readback_failure_preserves_execution_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, resolution: str,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    principals: list[Principal] = []

    async def execute(context: TaskNodeContext[None]) -> str:
        principals.append(context.principal)
        entered.set()
        await release.wait()
        return "done"

    definition = Task("execute", execute, effect_policy="none")
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("readback", storage=storage, tenant_id="tenant") as state, TaskEngine.open(
        definition, persistence=state,
    ) as engine:
        principal = Principal("submitter", "tenant")
        submission = await engine.prepare_submission(
            TaskGraph("graph", (TaskNode("node", task=definition),)), principal=principal, idempotency_key="start",
        )
        repository = state.task.tasks
        graph_state = repository.graph_state
        sql_engine = repository.state_store.storage_group._context.engine.sync_engine
        starter = asyncio.current_task()
        unavailable = False

        async def start_result(graph_id: str, *, tenant_id: str):
            nonlocal unavailable
            if asyncio.current_task() is starter:
                # Execution has entered through the real launcher before result readback.
                await asyncio.wait_for(entered.wait(), 5)
                unavailable = True
            return await graph_state(graph_id, tenant_id=tenant_id)

        def fail_read(connection, cursor, statement, parameters, context, executemany):
            if unavailable and asyncio.current_task() is starter:
                if failure == "timeout":
                    raise SQLAlchemyTimeout("unavailable read connection")
                # Exercise an actual DBAPI failure through the SQL StateStore reader.
                return "SELECT * FROM missing_admission_readback_table", ()
            return statement, parameters

        try:
            event.listen(sql_engine, "before_cursor_execute", fail_read, retval=True)
            with monkeypatch.context() as patch:
                patch.setattr(repository, "graph_state", start_result)
                with pytest.raises(AIError) as caught:
                    await engine.start_prepared(submission)
            error = caught.value
            assert error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            assert error.safe_details["durable_admitted"] is True
            assert error.safe_details["graph_id"] == submission.ref.graph_id
            assert isinstance(error.__cause__, OperationalError if failure == "sql" else SQLAlchemyTimeout)
        finally:
            event.remove(sql_engine, "before_cursor_execute", fail_read)

        first = await graph_state("graph", tenant_id="tenant")
        original = first.node_states[0]
        try:
            if resolution == "cancel":
                cancelled = await engine.cancel_submission(
                    submission.ref, principal=principal, idempotency_key="cancel",
                )
                assert cancelled.admitted and cancelled.status is TaskStatus.CANCELLED
                assert await engine.cancel_submission(
                    submission.ref, principal=principal, idempotency_key="cancel",
                ) == cancelled
                with pytest.raises(AIError) as caught:
                    await repository.claim("graph", "node", tenant_id="tenant", owner="late", lease_seconds=30)
                assert caught.value.code is ErrorCode.TASK_NOT_READY
            else:
                release.set()
            retry = await engine.start_prepared(submission)
            assert retry.admitted and retry.submission == submission.ref
            result = (await (await engine.get("graph", principal=principal)).wait(timeout_seconds=5)).result
            assert result.status is (TaskStatus.CANCELLED if resolution == "cancel" else TaskStatus.SUCCEEDED)
            assert result.node_states[0].execution_id == original.execution_id
            assert result.node_states[0].fence == original.fence
            assert principals == [principal]
            assert (await state.task.admissions.get("graph", tenant_id="tenant")).principal == principal
            assert (await engine.start_prepared(submission)).admitted
            assert principals == [principal]
        finally:
            release.set()



@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("admission", "load", "scheduler", "validate", "prepare", "result"))
async def test_post_admission_failures_keep_classification_and_confirmed_admission(
    monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    async with Persistence.open("classified", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        service = DefaultTaskGraphService(state.task, TenantAuthorizationPolicy(), _Launcher())
        submission = await service.prepare_submission(_request())
        diagnostics = ErrorDiagnostics.from_exception(RuntimeError("read failed"))
        source = AIError(
            ErrorCode.STORAGE_UNAVAILABLE, category="STORAGE", retryable=True,
            operation_id="read-operation", safe_details={"reader": "admission", "durable_admitted": False},
            diagnostics=diagnostics,
        )

        async def fail(*args, **kwargs):
            raise source

        async def ready(*args, **kwargs):
            return None

        def validate(*args):
            if boundary == "validate":
                raise source

        service._preflight = SimpleNamespace(
            load_admission=fail if boundary == "load" else ready,
            prepare_graph=fail if boundary == "prepare" else ready,
            validate_recovery=validate,
        )
        with monkeypatch.context() as patch:
            if boundary == "admission":
                patch.setattr(state.task.admissions, "get", fail)
            elif boundary == "scheduler":
                patch.setattr(state.task.tasks, "scheduler_state", fail)
            elif boundary == "result":
                patch.setattr(state.task.tasks, "graph_state", fail)
            with pytest.raises(AIError) as caught:
                await service.start_prepared(submission)
        error = caught.value
        assert error.code is source.code
        assert error.category == source.category
        assert error.retryable is source.retryable
        assert error.operation_id == source.operation_id
        assert error.diagnostics is diagnostics
        assert error.safe_details["reader"] == "admission"
        assert error.safe_details["durable_admitted"] is True
        assert await state.task.admissions.submission_status(submission.ref) == "admitted"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("authorization", "namespace", "admit", "unknown"))
async def test_unconfirmed_start_errors_are_not_reclassified(
    monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    async with Persistence.open("before-admit", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        authorization = TenantAuthorizationPolicy()
        service = DefaultTaskGraphService(state.task, authorization, _Launcher())
        submission = await service.prepare_submission(_request())
        source = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) if boundary == "unknown" else SQLAlchemyTimeout("before admit")

        async def fail(*args, **kwargs):
            raise source

        with monkeypatch.context() as patch:
            if boundary == "authorization":
                patch.setattr(authorization, "authorize", fail)
            elif boundary == "namespace":
                submission = replace(submission, namespace="foreign")
            else:
                patch.setattr(state.task.admissions, "admit_prepared", fail)
            with pytest.raises(AIError if boundary in {"namespace", "unknown"} else SQLAlchemyTimeout) as caught:
                await service.start_prepared(submission)
        if boundary == "namespace":
            assert caught.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        else:
            assert caught.value is source
        if isinstance(caught.value, AIError):
            assert "durable_admitted" not in caught.value.safe_details
        assert await state.task.tasks.get_graph("graph", tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", (False, True))
@pytest.mark.parametrize("boundary", ("admission", "result"))
async def test_cancelled_submission_readback_never_invents_admission(
    monkeypatch: pytest.MonkeyPatch, admitted: bool, boundary: str,
) -> None:
    async with Persistence.open("cancelled-readback", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        service = DefaultTaskGraphService(state.task, TenantAuthorizationPolicy(), _Launcher())
        submission = await service.prepare_submission(_request())
        if admitted:
            await state.task.admissions.admit_prepared(submission)
        cancelled = await service.cancel_submission(
            submission.ref, principal=submission.ref.principal, idempotency_key="cancel",
        )
        assert cancelled.admitted is admitted

        async def fail(*args, **kwargs):
            raise SQLAlchemyTimeout("cancelled read")

        with monkeypatch.context() as patch:
            if boundary == "admission":
                patch.setattr(state.task.admissions, "get", fail)
            elif admitted:
                patch.setattr(state.task.tasks, "scheduler_state", fail)
            else:
                patch.setattr(state.task.admissions, "get_submission_ref", fail)
            with pytest.raises(AIError) as caught:
                await service.start_prepared(submission)
        assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert (caught.value.safe_details.get("durable_admitted") is True) is (admitted and boundary == "result")
        assert await state.task.admissions.submission_status(submission.ref) == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_cancel", (False, True))
async def test_post_admission_read_cancellation_preserves_caller_semantics(
    monkeypatch: pytest.MonkeyPatch, caller_cancel: bool,
) -> None:
    async with Persistence.open("read-cancel", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        service = DefaultTaskGraphService(state.task, TenantAuthorizationPolicy(), _Launcher())
        submission = await service.prepare_submission(_request())
        entered = asyncio.Event()

        async def cancelled(*args, **kwargs):
            entered.set()
            if caller_cancel:
                await asyncio.Event().wait()
            raise asyncio.CancelledError

        with monkeypatch.context() as patch:
            patch.setattr(state.task.tasks, "graph_state", cancelled)
            starting = asyncio.create_task(service.start_prepared(submission))
            await asyncio.wait_for(entered.wait(), 5)
            if caller_cancel:
                starting.cancel()
            with pytest.raises(asyncio.CancelledError if caller_cancel else AIError) as caught:
                await starting
        if not caller_cancel:
            assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            assert caught.value.safe_details["durable_admitted"] is True
        assert (await service.start_prepared(submission)).admitted


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("unknown", "cancel"))
async def test_engine_preserves_unknown_or_cancel_when_cleanup_read_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    calls = 0
    release, entered = asyncio.Event(), asyncio.Event()

    async def execute(context: TaskNodeContext[None]) -> str:
        nonlocal calls
        calls += 1
        await release.wait()
        return "done"

    definition = Task("execute", execute, effect_policy="none")
    async with Persistence.open("unconfirmed", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(
        definition, persistence=state,
    ) as engine:
        submission = await engine.prepare_submission(
            TaskGraph("graph", (TaskNode("node", task=definition),)), idempotency_key="start",
        )
        store = state.task.admissions.state_store
        mutate, read = store.mutate, store.read
        admit = state.task.admissions.admit_prepared
        armed, unavailable = False, False
        read_error = SQLAlchemyTimeout("cannot confirm admission")
        starter = None

        async def admit_unknown(candidate):
            nonlocal armed
            armed = True
            return await admit(candidate)

        async def lose_response(operation):
            nonlocal unavailable
            value = await mutate(operation)
            if armed:
                unavailable = True
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return value

        async def read_unavailable(operation):
            if unavailable and (outcome == "unknown" or asyncio.current_task() is starter):
                raise read_error
            return await read(operation)

        async def result_cancelled(*args, **kwargs):
            nonlocal unavailable
            if asyncio.current_task() is starter:
                unavailable = True
                entered.set()
                await asyncio.Event().wait()
            return await graph_state(*args, **kwargs)

        graph_state = state.task.tasks.graph_state
        try:
            with monkeypatch.context() as patch:
                patch.setattr(store, "read", read_unavailable)
                if outcome == "unknown":
                    patch.setattr(state.task.admissions, "admit_prepared", admit_unknown)
                    patch.setattr(store, "mutate", lose_response)
                else:
                    patch.setattr(state.task.tasks, "graph_state", result_cancelled)
                starter = asyncio.create_task(engine.start_prepared(submission))
                if outcome == "cancel":
                    await asyncio.wait_for(entered.wait(), 5)
                    starter.cancel()
                with pytest.raises(AIError if outcome == "unknown" else asyncio.CancelledError) as caught:
                    await starter
            if outcome == "unknown":
                assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
                assert "durable_admitted" not in caught.value.safe_details
                assert caught.value.__cause__ is read_error
            assert any("execution resources retained" in note for note in caught.value.__notes__)
            assert await state.task.admissions.submission_status(submission.ref) == "admitted"
            release.set()
            result = await engine.start_prepared(submission)
            assert result.admitted and result.submission == submission.ref
            assert (await (await engine.get("graph")).wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
            assert calls == 1
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("prepare", "prepare_described", "start"))
async def test_preparation_unknown_survives_activation_readback_failure(
    monkeypatch: pytest.MonkeyPatch, entry: str,
) -> None:
    calls = 0

    async def execute(context: TaskNodeContext[None]) -> str:
        nonlocal calls
        calls += 1
        return "done"

    definition = Task("execute", execute, effect_policy="none")
    graph = TaskGraph("graph", (TaskNode("node", task=definition),))
    async with Persistence.open("prepare-unknown", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(
        definition, persistence=state,
    ) as engine:
        submission = await engine.describe_submission(graph, idempotency_key="start")
        store = state.task.admissions.state_store
        mutate, read = store.mutate, store.read
        unavailable = False
        read_error = SQLAlchemyTimeout("cannot confirm preparation")

        async def lose_response(operation):
            nonlocal unavailable
            result = await mutate(operation)
            # Capturing task bindings may write before the submission owner.
            # Lose the acknowledgment only after durable preparation itself.
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], TaskGraphSubmission):
                unavailable = True
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        async def read_unavailable(operation):
            if unavailable:
                raise read_error
            return await read(operation)

        with monkeypatch.context() as patch:
            patch.setattr(store, "mutate", lose_response)
            patch.setattr(store, "read", read_unavailable)
            with pytest.raises(AIError) as caught:
                if entry == "prepare":
                    await engine.prepare_submission(graph, idempotency_key="start")
                elif entry == "prepare_described":
                    await engine.prepare_described_submission(submission)
                else:
                    await engine.start_prepared(submission)
        assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert caught.value.__cause__ is read_error
        assert "durable_admitted" not in caught.value.safe_details
        assert any("execution resources retained" in note for note in caught.value.__notes__)
        assert await state.task.admissions.submission_status(submission.ref) == "prepared"
        assert calls == 0
        assert (await engine.start_prepared(submission)).admitted
        assert (await (await engine.get("graph")).wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
        assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("read_failure", (SQLAlchemyTimeout, asyncio.CancelledError))
async def test_engine_cancelled_preparation_retains_uncertain_response_classification(
    monkeypatch: pytest.MonkeyPatch, read_failure: type[BaseException],
) -> None:
    async def execute(context: TaskNodeContext[None]) -> str:
        pytest.fail("cancelled preparation must never run")

    definition = Task("execute", execute, effect_policy="none")
    async with Persistence.open("cancelled-response", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(
        definition, persistence=state,
    ) as engine:
        submission = await engine.prepare_submission(
            TaskGraph("graph", (TaskNode("node", task=definition),)), idempotency_key="start",
        )
        await engine.cancel_submission(submission.ref, idempotency_key="cancel")
        read_error = read_failure("cannot read cancelled admission")

        async def unavailable(*args, **kwargs):
            raise read_error

        with monkeypatch.context() as patch:
            patch.setattr(state.task.admissions, "get", unavailable)
            with pytest.raises(AIError) as caught:
                await engine.start_prepared(submission)
        assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert caught.value.__cause__ is read_error
        assert "durable_admitted" not in caught.value.safe_details
        assert any("execution resources retained" in note for note in caught.value.__notes__)
        result = await engine.start_prepared(submission)
        assert not result.admitted and result.result.status is TaskStatus.CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize("readback", ("available", "unavailable", "caller_cancel", "no_failure"))
async def test_activation_cleanup_preserves_failure_and_settles_resource_tracking(
    monkeypatch: pytest.MonkeyPatch, readback: str,
) -> None:
    async def execute(context: TaskNodeContext[None]) -> str:
        pytest.fail("denied start must never run")

    definition = Task("execute", execute, effect_policy="none")
    async with Persistence.open("activation-cleanup", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(
        definition, persistence=state,
    ) as engine:
        submission = await engine.prepare_submission(
            TaskGraph("graph", (TaskNode("node", task=definition),)), idempotency_key="start",
        )
        source = AIError(ErrorCode.AUTHORIZATION_DENIED, safe_details={"reason": "policy"})
        entered = asyncio.Event()

        async def denied(*args, **kwargs):
            raise source

        async def unavailable(*args, **kwargs):
            entered.set()
            if readback == "caller_cancel":
                await asyncio.Event().wait()
            raise SQLAlchemyTimeout("private SQL/parameters must not enter diagnostics")

        with monkeypatch.context() as patch:
            if readback != "no_failure":
                patch.setattr(engine._authorization, "authorize", denied)
            if readback != "available":
                patch.setattr(state.task.admissions, "get", unavailable)
            starting = asyncio.create_task(
                engine.prepare_described_submission(submission) if readback == "no_failure"
                else engine.start_prepared(submission)
            )
            if readback == "caller_cancel":
                await asyncio.wait_for(entered.wait(), 5)
                starting.cancel()
            error_type = {
                "caller_cancel": asyncio.CancelledError, "no_failure": SQLAlchemyTimeout,
            }.get(readback, AIError)
            with pytest.raises(error_type) as caught:
                await starting
        if readback not in {"caller_cancel", "no_failure"}:
            assert caught.value is source
            assert caught.value.safe_details == {"reason": "policy"}
        if readback == "unavailable":
            assert any("execution resources retained" in note for note in source.__notes__)
            assert all("private SQL" not in note for note in source.__notes__)
        # These owner bookkeeping invariants have no public query surface.
        runtime = engine._task_node_runtime
        assert "graph" not in runtime._pending_definition_activations
        assert ("graph" in runtime._active_definitions) is (readback != "available")
        assert await state.task.admissions.submission_status(submission.ref) == "prepared"
