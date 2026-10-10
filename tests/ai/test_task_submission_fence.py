#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepared TaskGraph cancellation fences survive admission races and reopen."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import (
    AuthorizationAction, Principal, ResourceKind, ResourceRef, RunBudget, TaskStatus, TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain, RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain, _encode_persisted_domain, encode_envelope,
)
from linktools.ai.runtime.state._store import OperationQuery, RecordQuery, StateTransaction, stream_digest
from linktools.ai.task import (
    DefaultTaskGraphService,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphSubmission,
    TaskGraphView,
    TaskNode,
)


class _Authorization:
    async def authorize(
        self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
    ) -> None:
        assert principal.tenant_id == resource.tenant_id
        if action is AuthorizationAction.TASK_CANCEL:
            assert resource.owner_principal_id == "submitter"


class _Launcher:
    def __init__(self) -> None:
        self.started: list[TaskGraphLaunch] = []
        self.cancelled: list[TaskGraphLaunch] = []

    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle:
        self.started.append(launch)
        return TaskGraphHandle(launch.graph_id)

    async def settle_cancel(
        self, launch: TaskGraphLaunch, *, invoke_effects: bool,
    ) -> TaskGraphView:
        assert invoke_effects
        self.cancelled.append(launch)
        return TaskGraphView(launch.graph_id, TaskStatus.CANCELLED, ())


def _storage(backend: str, path: Path) -> RuntimeStorage:
    if backend == "memory":
        return RuntimeStorage.in_memory()
    if backend == "filesystem":
        return RuntimeStorage.filesystem(path)
    if backend == "sqlite":
        return RuntimeStorage.sqlite(path / "state.db")
    return RuntimeStorage(RuntimeStoragePlan(
        task=RuntimeStorageRoute.filesystem(path / "task"),
        evaluation=RuntimeStorageRoute.sqlite(path / "evaluation.db"),
        execution=RuntimeStorageRoute.sqlite(path / "execution.db"),
    ))


def _request(graph_id: str = "trial") -> TaskGraphRequest:
    return TaskGraphRequest(
        TaskGraph(graph_id, (TaskNode("target", input={"secret": "prepared input"}),)),
        Principal("submitter", "tenant"),
        f"submit:{graph_id}",
    )


def _service(state: RuntimeStorage, launcher: _Launcher) -> DefaultTaskGraphService:
    return DefaultTaskGraphService(state.task, _Authorization(), launcher)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "mixed"))
async def test_cancelled_preparation_never_admits_and_releases_sensitive_payload(
    backend: str, tmp_path: Path,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="submission", tenant_id="tenant")
    try:
        launcher = _Launcher()
        service = _service(state, launcher)
        submission = await service.prepare_submission(_request())
        assert await state.task.tasks.get_graph("trial", tenant_id="tenant") is None
        assert launcher.started == []
        assert _decode_enveloped_domain(
            encode_envelope({"type": "task_graph_submission", "payload": _encode_persisted_domain(submission)}), TaskGraphSubmission,
        ) == submission

        actor = Principal("canceller", "tenant")
        cancelled = await service.cancel_submission(
            submission.ref, principal=actor, idempotency_key="cancel",
        )
        assert not cancelled.admitted
        assert cancelled.status is TaskStatus.CANCELLED
        assert await service.cancel_submission(
            submission.ref, principal=actor, idempotency_key="cancel",
        ) == cancelled

        if backend != "memory":
            await state.close()
            state = _storage(backend, tmp_path)
            await state.initialize(namespace="submission", tenant_id="tenant")
            service = _service(state, launcher)

        # A stale caller holding every input cannot upload and start it again.
        assert await service.prepare_submission(_request()) == submission
        started = await service.start_prepared(submission)
        assert not started.admitted
        assert started.result.status is TaskStatus.CANCELLED
        assert (await service.run(_request())).status is TaskStatus.CANCELLED
        assert launcher.started == []
        assert await state.task.tasks.get_graph("trial", tenant_id="tenant") is None
        assert await state.task.admissions.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="task_submission_payload"))
        ) == ()
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "mixed"))
@pytest.mark.parametrize("admit_first", (False, True))
async def test_submission_cancel_serializes_with_admission_and_blocks_node_claims(
    backend: str, tmp_path: Path, admit_first: bool,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="submission-race", tenant_id="tenant")
    try:
        launcher = _Launcher()
        service = _service(state, launcher)
        submission = await service.prepare_submission(_request())
        admission = state.task.admissions.admit_prepared(submission)
        cancellation = service.cancel_submission(
            submission.ref,
            principal=Principal("canceller", "tenant"),
            idempotency_key="cancel",
        )
        if admit_first:
            admitted, cancelled = await asyncio.gather(admission, cancellation)
        else:
            cancelled, admitted = await asyncio.gather(cancellation, admission)
        assert cancelled.status is TaskStatus.CANCELLED
        assert admitted.graph_id == "trial"
        with pytest.raises(AIError) as caught:
            await state.task.tasks.claim(
                "trial", "target", tenant_id="tenant", owner="late-worker", lease_seconds=30,
            )
        assert caught.value.code is ErrorCode.TASK_NOT_READY
        replay = await service.start_prepared(submission)
        assert replay.result.status is TaskStatus.CANCELLED
        assert replay.admitted == cancelled.admitted
        assert launcher.started == []
        if cancelled.admitted:
            durable = await state.task.admissions.get("trial", tenant_id="tenant")
            assert durable.principal == submission.admission.principal
            assert launcher.cancelled[0].principal == submission.admission.principal
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_submission_identity_is_registered_and_cannot_be_reassigned() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="submission-identity", tenant_id="tenant")
    try:
        service = _service(state, _Launcher())
        request = _request()
        unregistered = TaskGraphSubmission(
            "submission-identity", TaskGraphAdmission.from_request(request), request.graph,
        )
        with pytest.raises(AIError) as caught:
            await service.start_prepared(unregistered)
        assert caught.value.code is ErrorCode.STORAGE_NOT_FOUND
        cancelled = await service.cancel_submission(
            unregistered.ref, principal=request.principal, idempotency_key="cancel",
        )
        assert not cancelled.admitted
        assert cancelled.status is TaskStatus.CANCELLED

        submission = await service.prepare_submission(request)
        with pytest.raises(AIError) as caught:
            await service.cancel_submission(
                replace(submission.ref, request_digest="f" * 64),
                principal=request.principal,
                idempotency_key="cancel",
            )
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        await service.cancel_submission(
            submission.ref, principal=request.principal, idempotency_key="cancel",
        )
        with pytest.raises(AIError) as caught:
            await service.prepare_submission(replace(request, idempotency_key="other"))
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepare", "admit", "cancel"))
async def test_submission_commit_readback_resolves_lost_response(
    phase: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="submission-unknown", tenant_id="tenant")
    try:
        service = _service(state, _Launcher())
        submission = None if phase == "prepare" else await service.prepare_submission(_request())
        store = state.task.admissions.state_store
        mutate = store.mutate
        failed = False

        async def lose_response(
            operation: Callable[[StateTransaction], Awaitable[object]],
        ) -> object:
            nonlocal failed
            value = await mutate(operation)
            if not failed:
                failed = True
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return value

        monkeypatch.setattr(store, "mutate", lose_response)
        if phase == "prepare":
            submission = await service.prepare_submission(_request())
            assert await service.prepare_submission(_request()) == submission
        elif phase == "admit":
            result = await service.start_prepared(submission)
            assert result.admitted
            assert (await service.start_prepared(submission)).admitted
        else:
            result = await service.cancel_submission(
                submission.ref, principal=submission.ref.principal, idempotency_key="cancel",
            )
            assert result.status is TaskStatus.CANCELLED
            assert not (await service.start_prepared(submission)).admitted
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_late_scheduler_arm_cannot_claim_after_submission_cancel() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="late-arm", tenant_id="tenant")
    arming = asyncio.Event()
    release = asyncio.Event()

    class Launcher(_Launcher):
        async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle:
            arming.set()
            await release.wait()
            with pytest.raises(AIError) as caught:
                await state.task.tasks.claim(
                    launch.graph_id, "target", tenant_id="tenant", owner="late", lease_seconds=30,
                )
            assert caught.value.code is ErrorCode.TASK_NOT_READY
            return TaskGraphHandle(launch.graph_id)

    try:
        service = _service(state, Launcher())
        submission = await service.prepare_submission(_request())
        starting = asyncio.create_task(service.start_prepared(submission))
        await arming.wait()
        cancelled = await service.cancel_submission(
            submission.ref, principal=submission.ref.principal, idempotency_key="cancel",
        )
        assert cancelled.admitted
        release.set()
        result = await starting
        assert result.admitted
        assert result.result.status is TaskStatus.CANCELLED
    finally:
        release.set()
        await state.close()


@pytest.mark.asyncio
async def test_submission_cancellation_authorizes_registered_owner() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="submission-owner", tenant_id="tenant")
    try:
        service = DefaultTaskGraphService(state.task, TenantAuthorizationPolicy(), _Launcher())
        submission = await service.prepare_submission(_request())
        with pytest.raises(AIError) as caught:
            await service.cancel_submission(
                submission.ref, principal=Principal("other", "tenant"), idempotency_key="cancel",
            )
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(AIError) as caught:
            await service.cancel_submission(
                replace(submission.ref, principal=Principal("other", "tenant")),
                principal=Principal("other", "tenant"), idempotency_key="forged",
            )
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert (await service.start_prepared(submission)).admitted
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepared", "admitted", "cancelled"))
async def test_submission_identity_and_fence_survive_snapshot_restore(
    phase: str, tmp_path: Path,
) -> None:
    from linktools.ai.runtime import Runtime
    from linktools.ai.runtime.state import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore
    from linktools.ai.task import Task, TaskNodeContext
    from ._runtime_test_helpers import RuntimeUsageModels

    calls: list[str] = []

    async def execute(context: TaskNodeContext[None]) -> str:
        calls.append(context.graph_id)
        return "done"

    definition = Task("snapshot-target", execute, effect_policy="none")
    graph = TaskGraph("trial", (TaskNode("target", task=definition),))
    root = tmp_path / "source"
    async with Runtime.open(
        "submission-snapshot", models=RuntimeUsageModels(),
        storage=RuntimeStorage.from_root(root),
    ) as runtime:
        engine = runtime.tasks.bind(definition)
        submission = await engine.prepare_submission(graph, idempotency_key="prepare")
        assert calls == []
        if phase == "admitted":
            await engine.start_prepared(submission)
            assert (await (await engine.get("trial")).wait()).result.wait_status is TaskStatus.SUCCEEDED
        elif phase == "cancelled":
            await engine.cancel_submission(submission.ref, idempotency_key="cancel")

    state = RuntimeStorage.from_root(root)
    await state.initialize(namespace="submission-snapshot", tenant_id="default", read_only=True)
    objects = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(10000, 10 * 1024 * 1024)
    try:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    finally:
        await state.close()
    restored_root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(snapshot, object_store=objects, root=restored_root, limits=limits)
    async with Runtime.open(
        "submission-snapshot", models=RuntimeUsageModels(),
        storage=RuntimeStorage.from_root(restored_root),
    ) as runtime:
        engine = runtime.tasks.bind(definition)
        result = await engine.start_prepared(submission)
        assert result.admitted is (phase != "cancelled")
        if result.admitted:
            assert (await (await engine.get("trial")).wait()).result.wait_status is TaskStatus.SUCCEEDED
        else:
            assert result.result.status is TaskStatus.CANCELLED
        assert len(calls) == (0 if phase == "cancelled" else 1)


@pytest.mark.asyncio
async def test_independent_task_owners_share_the_submission_fence(tmp_path: Path) -> None:
    first = _storage("sqlite", tmp_path)
    second = _storage("sqlite", tmp_path)
    await first.initialize(namespace="independent", tenant_id="tenant")
    await second.initialize(namespace="independent", tenant_id="tenant")
    try:
        service = _service(first, _Launcher())
        other = _service(second, _Launcher())
        submission = await service.prepare_submission(_request())
        admitted, cancelled = await asyncio.gather(
            first.task.admissions.admit_prepared(submission),
            other.cancel_submission(
                submission.ref, principal=submission.ref.principal, idempotency_key="cancel",
            ),
        )
        assert admitted.graph_id == "trial"
        assert cancelled.status is TaskStatus.CANCELLED
        result = await service.start_prepared(submission)
        assert result.admitted == cancelled.admitted
        assert result.result.status is TaskStatus.CANCELLED
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_prepare_disposition_has_one_creator_across_independent_sqlite_owners(
    tmp_path: Path,
) -> None:
    first = _storage("sqlite", tmp_path)
    second = _storage("sqlite", tmp_path)
    await first.initialize(namespace="prepare-disposition", tenant_id="tenant")
    await second.initialize(namespace="prepare-disposition", tenant_id="tenant")
    try:
        launchers = (_Launcher(), _Launcher())
        services = (_service(first, launchers[0]), _service(second, launchers[1]))
        submission = await services[0].describe_submission(_request())
        results = await asyncio.gather(*(
            service.prepare_described_with_disposition(submission) for service in services
        ))
        assert sorted(created for _, created in results) == [False, True]
        assert all(prepared == submission for prepared, _ in results)
        for state, service, launcher in zip((first, second), services, launchers):
            assert await state.task.admissions.prepare_with_disposition(submission) == (submission, False)
            assert await state.task.admissions.prepare(submission) == submission
            assert await service.prepare_described(submission) == submission
            assert await service.prepare_submission(_request()) == submission
            assert await state.task.admissions.submission_status(submission.ref) == "prepared"
            assert await state.task.tasks.get_graph("trial", tenant_id="tenant") is None
            assert launcher.started == []
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepared", "admitted", "cancelled"))
async def test_prepare_disposition_existing_submission_never_claims_creation(phase: str) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="existing-preparation", tenant_id="tenant")
    try:
        launcher = _Launcher()
        service = _service(state, launcher)
        submission = await service.describe_submission(_request())
        assert await service.prepare_described_with_disposition(submission) == (submission, True)
        if phase == "admitted":
            await state.task.admissions.admit_prepared(submission)
        elif phase == "cancelled":
            await service.cancel_submission(
                submission.ref, principal=submission.ref.principal, idempotency_key="cancel",
            )
        assert await state.task.admissions.submission_status(submission.ref) == phase
        assert await service.prepare_described_with_disposition(submission) == (submission, False)
        assert await state.task.admissions.submission_status(submission.ref) == phase
        assert launcher.started == []
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_prepare_disposition_commit_readback_cannot_prove_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _storage("sqlite", tmp_path)
    await state.initialize(namespace="unknown-preparation", tenant_id="tenant")
    try:
        launcher = _Launcher()
        service = _service(state, launcher)
        submission = await service.describe_submission(_request())
        store = state.task.admissions.state_store
        mutate = store.mutate

        async def lose_response(
            operation: Callable[[StateTransaction], Awaitable[object]],
        ) -> object:
            await mutate(operation)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(store, "mutate", lose_response)
            assert await service.prepare_described_with_disposition(submission) == (submission, False)
        assert await state.task.admissions.submission_status(submission.ref) == "prepared"
        assert await service.prepare_described_with_disposition(submission) == (submission, False)
        assert await state.task.tasks.get_graph("trial", tenant_id="tenant") is None
        assert launcher.started == []
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_prepare_disposition_preserves_single_authorization_and_capture() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="captured-preparation", tenant_id="tenant")
    events: list[str] = []

    class Authorization(_Authorization):
        async def authorize(
            self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
        ) -> None:
            events.append("authorize")
            await super().authorize(principal, action, resource)

    class Preflight:
        async def capture_admission(
            self, admission: TaskGraphAdmission, graph: TaskGraph,
        ) -> TaskGraph:
            events.append("capture")
            return graph

    try:
        launcher = _Launcher()
        service = DefaultTaskGraphService(state.task, Authorization(), launcher, preflight=Preflight())
        request = _request()
        submission = TaskGraphSubmission(
            "captured-preparation", TaskGraphAdmission.from_request(request), request.graph,
        )
        assert await service.prepare_described_with_disposition(submission) == (submission, True)
        assert events == ["authorize", "capture"]
        events.clear()
        assert await service.prepare_described(submission) == submission
        assert events == ["authorize"]
        assert await state.task.tasks.get_graph("trial", tenant_id="tenant") is None
        assert launcher.started == []
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ("namespace", "authorization", "budget"))
async def test_prepare_disposition_preserves_guards_before_persistence(guard: str) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="guarded-preparation", tenant_id="tenant")

    class Authorization(_Authorization):
        async def authorize(
            self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
        ) -> None:
            if guard == "authorization":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await super().authorize(principal, action, resource)

    try:
        launcher = _Launcher()
        service = DefaultTaskGraphService(state.task, Authorization(), launcher)
        request = _request()
        if guard == "budget":
            request = replace(request, budget=RunBudget(model_requests=1))
        submission = TaskGraphSubmission(
            "foreign" if guard == "namespace" else "guarded-preparation",
            TaskGraphAdmission.from_request(request), request.graph,
        )
        with pytest.raises(AIError) as caught:
            await service.prepare_described_with_disposition(submission)
        assert caught.value.code is {
            "namespace": ErrorCode.STORAGE_OWNER_MISMATCH,
            "authorization": ErrorCode.AUTHORIZATION_DENIED,
            "budget": ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
        }[guard]
        assert await state.task.admissions.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="task_submission"))
        ) == ()
        assert launcher.started == []
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", (
    "task_graph", "task_admission", "task_submission_payload", "admit_operation",
))
async def test_prepare_disposition_missing_head_never_recreates_existing_resources(remaining: str) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="broken-preparation", tenant_id="tenant")
    try:
        launcher = _Launcher()
        service = _service(state, launcher)
        submission = await service.prepare_submission(_request())
        if remaining != "task_submission_payload":
            await state.task.admissions.admit_prepared(submission)

        async def remove_head(transaction: StateTransaction) -> None:
            for kind in (
                "task_submission", "task_graph", "task_admission", "task_submission_payload",
                "task_node_definition", "task_node_state",
            ):
                if kind != remaining:
                    records = await transaction.list_records(RecordQuery(kind=kind))
                    await transaction.delete_records(tuple(record.key_digest for record in records))
            if remaining != "admit_operation":
                await transaction.delete_operations(OperationQuery(stream_digest=stream_digest(
                    submission.namespace, submission.ref.tenant_id, RuntimeDomain.TASK.value,
                    "operation", [ResourceKind.TASK_GRAPH.value, submission.graph.graph_id],
                )))

        await state.task.admissions.state_store.mutate(remove_head)
        with pytest.raises(AIError) as caught:
            await service.prepare_described_with_disposition(submission)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await state.task.admissions.submission_status(submission.ref) is None
        assert launcher.started == []
    finally:
        await state.close()
