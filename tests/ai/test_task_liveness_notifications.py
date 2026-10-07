#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for TaskGraph activity notifications."""

import asyncio
from types import SimpleNamespace

import pytest
from ._task_test_helpers import admit_graph
import linktools.ai.task._local as task_local
from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.task import (
    DefaultTaskGraphService,
    LocalTaskGraphLauncher,
    TaskGraph,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskNode,
    TaskNodeInvocation,
    TaskNodeRunControl,
    TaskNodeRunError,
    TaskNodeRunResult,
    RecoverGraphRequest,
)
from linktools.ai.core import Principal, PrincipalKind


class _AllowAuthorization:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _BlockingRunner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        node = invocation.node
        graph_id = invocation.graph_id
        principal = invocation.principal
        correlation = invocation.correlation
        dependency_results = invocation.dependency_results
        del node, graph_id, principal, correlation, dependency_results, control
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return TaskNodeRunResult("c" * 64, execution_id="execution-local")

    async def cancel(
        self,
        invocation: TaskNodeInvocation,
    ) -> None:
        node = invocation.node
        graph_id = invocation.graph_id
        principal = invocation.principal
        correlation = invocation.correlation
        dependency_results = invocation.dependency_results
        del node, graph_id, principal, correlation, dependency_results

    async def inspect_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult | None:
        del invocation, execution_id
        return None


class _DrainBarrierRunner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.run_cancelled = asyncio.Event()
        self.release_run = asyncio.Event()
        self.cancel_entered = asyncio.Event()
        self.release_cancel = asyncio.Event()
        self.cancel_calls = 0

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del invocation
        await control.bind_execution("execution-quiesce")
        await control.handoff_execution("execution-quiesce")
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.run_cancelled.set()
            await self.release_run.wait()
            raise
        raise AssertionError("blocked task unexpectedly completed")

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        del invocation
        self.cancel_calls += 1
        self.cancel_entered.set()
        await self.release_cancel.wait()

    async def inspect_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult | None:
        del invocation, execution_id
        return None


class _ExecutionConflictRunner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del invocation
        await control.bind_execution("execution-race")
        await control.handoff_execution("execution-race")
        self.entered.set()
        await self.release.wait()
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult:
        del invocation
        raise TaskNodeRunError(ErrorCode.EXECUTION_CANCELLED, execution_id)

    async def inspect_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult | None:
        del invocation
        raise TaskNodeRunError(ErrorCode.EXECUTION_CANCELLED, execution_id)

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        del invocation


class _FenceRaceRunner:
    def __init__(self) -> None:
        self.old_entered = asyncio.Event()
        self.release_old = asyncio.Event()
        self.new_wait_entered = asyncio.Event()
        self.release_new = asyncio.Event()

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del invocation
        await control.bind_execution("execution-fence")
        await control.handoff_execution("execution-fence")
        self.old_entered.set()
        await self.release_old.wait()
        return TaskNodeRunResult("a" * 64, execution_id="execution-fence")

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult:
        del invocation
        assert execution_id == "execution-fence"
        self.new_wait_entered.set()
        await self.release_new.wait()
        return TaskNodeRunResult("b" * 64, execution_id=execution_id)

    async def inspect_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult | None:
        del invocation, execution_id
        return None

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        del invocation


@pytest.mark.asyncio
async def test_handoff_conflict_preserves_recovery_without_false_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-handoff-conflict-readback", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None

    class _HandoffConflictRunner:
        def __init__(self) -> None:
            self.failed = asyncio.Event()
            self.release = asyncio.Event()
            self.error: AIError | None = None
            self.control: TaskNodeRunControl | None = None

        async def run(
            self,
            invocation: TaskNodeInvocation,
            *,
            control: TaskNodeRunControl,
        ) -> TaskNodeRunResult:
            del invocation
            self.control = control
            await control.bind_execution("execution-handoff-conflict")
            try:
                await control.handoff_execution("execution-handoff-conflict")
            except AIError as error:
                self.error = error
                self.failed.set()
                await self.release.wait()
                raise
            raise AssertionError("injected handoff conflict unexpectedly succeeded")

        async def wait_bound(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
        ) -> TaskNodeRunResult:
            del invocation, execution_id
            raise AssertionError("a conflicted handoff must remain recovery required")

        async def inspect_bound(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
        ) -> TaskNodeRunResult | None:
            del invocation, execution_id
            return None

        async def cancel(self, invocation: TaskNodeInvocation) -> None:
            del invocation

    try:
        repository = state.task.tasks
        graph = TaskGraph("handoff-conflict-readback", (TaskNode("node"),))
        await admit_graph(state, graph)
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        runner = _HandoffConflictRunner()
        heartbeat_renewed = asyncio.Event()
        original_renew = repository.renew

        async def observe_renew(
            lease: object,
            *,
            tenant_id: str,
            lease_seconds: int,
        ) -> object:
            result = await original_renew(
                lease,  # type: ignore[arg-type]
                tenant_id=tenant_id,
                lease_seconds=lease_seconds,
            )
            heartbeat_renewed.set()
            return result

        async def reject_handoff(
            lease: object,
            *,
            tenant_id: str,
            execution_id: str,
            occupies_concurrency: bool = True,
        ) -> object:
            del lease, tenant_id, execution_id, occupies_concurrency
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        monkeypatch.setattr(task_local, "_HEARTBEAT_SECONDS", 0.01)
        monkeypatch.setattr(repository, "renew", observe_renew)
        monkeypatch.setattr(repository, "handoff_execution", reject_handoff)
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )
        await asyncio.wait_for(runner.failed.wait(), 1)
        await asyncio.wait_for(heartbeat_renewed.wait(), 1)
        before_recovery = await repository.graph_state(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert before_recovery is not None
        assert before_recovery.node_states[0].status is TaskStatus.RUNNING
        assert before_recovery.node_states[0].execution_id == "execution-handoff-conflict"
        assert runner.control is not None
        assert runner.control.handed_off_execution_id is None

        async def recovery_state():
            while True:
                snapshot = await repository.graph_state(
                    graph.graph_id,
                    tenant_id="tenant",
                )
                assert snapshot is not None
                if snapshot.node_states[0].status is TaskStatus.RECOVERY_REQUIRED:
                    return snapshot
                await asyncio.sleep(0)

        runner.release.set()
        snapshot = await asyncio.wait_for(recovery_state(), 2)
        node_state = snapshot.node_states[0]
        assert runner.error is not None
        assert runner.error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert runner.control is not None
        assert runner.control.handed_off_execution_id is None
        assert node_state.execution_id == "execution-handoff-conflict"
        assert node_state.error_code == ErrorCode.STORAGE_RECOVERY_REQUIRED.value
        assert node_state.status is not TaskStatus.WAITING
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_handoff_commit_unknown_reads_back_waiting_state_before_acknowledging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-handoff-commit-unknown", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None

    class _CommitUnknownHandoffRunner:
        def __init__(self) -> None:
            self.handoff_returned = asyncio.Event()
            self.release = asyncio.Event()
            self.control: TaskNodeRunControl | None = None
            self.injected = False
            self.error: BaseException | None = None
            self.calls = 0

        async def run(
            self,
            invocation: TaskNodeInvocation,
            *,
            control: TaskNodeRunControl,
        ) -> TaskNodeRunResult:
            del invocation
            self.calls += 1
            self.control = control
            await control.bind_execution("execution-handoff-commit-unknown")
            store = state.task.tasks.state_store
            original_mutate = store.mutate

            async def commit_then_unknown(operation: object) -> object:
                result = await original_mutate(operation)  # type: ignore[arg-type]
                if not self.injected:
                    self.injected = True
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return result

            monkeypatch.setattr(store, "mutate", commit_then_unknown)
            try:
                await control.handoff_execution("execution-handoff-commit-unknown")
            except BaseException as error:
                self.error = error
                raise
            finally:
                monkeypatch.setattr(store, "mutate", original_mutate)
            self.handoff_returned.set()
            await self.release.wait()
            return TaskNodeRunResult(
                "f" * 64,
                execution_id="execution-handoff-commit-unknown",
            )

        async def wait_bound(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
        ) -> TaskNodeRunResult:
            del invocation, execution_id
            raise AssertionError("successful handoff must not wait on a second run")

        async def inspect_bound(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
        ) -> TaskNodeRunResult | None:
            del invocation, execution_id
            return None

        async def cancel(self, invocation: TaskNodeInvocation) -> None:
            del invocation

    try:
        repository = state.task.tasks
        graph = TaskGraph("handoff-commit-unknown", (TaskNode("node"),))
        await admit_graph(state, graph)
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        runner = _CommitUnknownHandoffRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )
        await asyncio.wait_for(runner.handoff_returned.wait(), 1)

        handed_off = await repository.graph_state(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert handed_off is not None
        assert handed_off.node_states[0].status is TaskStatus.WAITING
        assert handed_off.node_states[0].execution_id == "execution-handoff-commit-unknown"
        assert runner.injected
        assert runner.error is None
        assert runner.control is not None
        assert runner.control.handed_off_execution_id == "execution-handoff-commit-unknown"

        runner.release.set()

        async def completed_state():
            while True:
                snapshot = await repository.graph_state(
                    graph.graph_id,
                    tenant_id="tenant",
                )
                assert snapshot is not None
                if snapshot.status is TaskStatus.SUCCEEDED:
                    return snapshot
                await asyncio.sleep(0)

        completed = await asyncio.wait_for(completed_state(), 2)
        assert completed.node_states[0].result_digest == "f" * 64
        assert runner.calls == 1
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_recover_waits_for_closing_graph_run_to_drain_before_rearming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-recover-drain-race", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    release_drain = asyncio.Event()
    drain_entered = asyncio.Event()
    rearm_called = asyncio.Event()
    recovery: asyncio.Task | None = None

    class _RecoverAfterBoundaryRunner:
        def __init__(self) -> None:
            self.calls = 0
            self.completed = asyncio.Event()

        async def run(
            self,
            invocation: TaskNodeInvocation,
            *,
            control: TaskNodeRunControl,
        ) -> TaskNodeRunResult:
            del invocation, control
            self.calls += 1
            if self.calls == 1:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            self.completed.set()
            return TaskNodeRunResult(
                "d" * 64,
                execution_id="execution-recovered",
            )

        async def wait_bound(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
        ) -> TaskNodeRunResult:
            del invocation, execution_id
            raise AssertionError("unbound recovery must re-enter run")

        async def inspect_bound(
            self,
            invocation: TaskNodeInvocation,
            execution_id: str,
        ) -> TaskNodeRunResult | None:
            del invocation, execution_id
            return None

        async def cancel(self, invocation: TaskNodeInvocation) -> None:
            del invocation

    try:
        graph = TaskGraph("recover-drain-race", (TaskNode("node"),))
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        await admit_graph(state, graph)
        runner = _RecoverAfterBoundaryRunner()
        launcher = LocalTaskGraphLauncher(
            state.task.tasks,
            runner,
            owner="local-worker",
        )
        service = DefaultTaskGraphService(
            state.task,
            _AllowAuthorization(),
            launcher,
        )
        launch = TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        await launcher.start(launch)

        original_drain = launcher._drain_inflight
        drains = 0

        async def pause_first_drain(run: object) -> None:
            nonlocal drains
            drains += 1
            if drains == 1:
                drain_entered.set()
                await release_drain.wait()
            await original_drain(run)  # type: ignore[arg-type]

        original_start = launcher.start

        async def observe_rearm(request: TaskGraphLaunch):
            rearm_called.set()
            return await original_start(request)

        monkeypatch.setattr(launcher, "_drain_inflight", pause_first_drain)
        await asyncio.wait_for(drain_entered.wait(), 2)
        monkeypatch.setattr(launcher, "start", observe_rearm)
        recovery = asyncio.create_task(
            service.recover(
                graph.graph_id,
                RecoverGraphRequest(principal, "recover:drain-race"),
            )
        )
        await asyncio.wait_for(rearm_called.wait(), 2)

        assert not recovery.done()
        assert runner.calls == 1
        release_drain.set()
        rearmed = await asyncio.wait_for(recovery, 3)
        final = await service.wait(
            graph.graph_id,
            principal=principal,
            timeout_seconds=3,
        )

        assert rearmed.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
        assert final.status is TaskStatus.SUCCEEDED
        assert runner.calls == 2
        assert runner.completed.is_set()
        assert drains == 2
    finally:
        release_drain.set()
        if recovery is not None:
            await asyncio.gather(recovery, return_exceptions=True)
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_quiesce_keeps_inflight_owned_through_caller_cancellation() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-quiesce-drain", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("quiesce-drain", (TaskNode("node"),))
        await admit_graph(state, graph)
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        runner = _DrainBarrierRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )
        await asyncio.wait_for(runner.entered.wait(), 1)
        graph_state = await repository.graph_state(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert graph_state is not None
        run = launcher._graphs[("tenant", graph.graph_id)]
        inflight = run.inflight["node"]

        interrupted_cleaner = asyncio.create_task(
            launcher._quiesce_node(
                run,
                graph_state,
                "node",
                invoke_cancel=False,
            )
        )
        await asyncio.wait_for(runner.run_cancelled.wait(), 1)
        surviving_cleaner = asyncio.create_task(
            launcher._quiesce_node(
                run,
                graph_state,
                "node",
                invoke_cancel=True,
            )
        )
        interrupted_cleaner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interrupted_cleaner

        assert run.inflight.get("node") is inflight
        runner.release_run.set()
        await asyncio.wait_for(runner.cancel_entered.wait(), 1)
        assert run.inflight.get("node") is inflight
        assert runner.cancel_calls == 1

        runner.release_cancel.set()
        await asyncio.wait_for(surviving_cleaner, 1)
        assert run.inflight.get("node") is not inflight
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_execution_result_conflict_reads_cancel_fact_instead_of_failing_node() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-execution-conflict", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    runner = _ExecutionConflictRunner()
    try:
        graph = TaskGraph("execution-conflict", (TaskNode("node"),))
        await admit_graph(state, graph)
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        launcher = LocalTaskGraphLauncher(
            state.task.tasks,
            runner,
            owner="local-worker",
        )
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )
        await asyncio.wait_for(runner.entered.wait(), 1)
        runner.release.set()

        async def wait_for_cancelled() -> None:
            while True:
                snapshot = await state.task.tasks.graph_state(
                    graph.graph_id,
                    tenant_id="tenant",
                )
                assert snapshot is not None
                if snapshot.status is TaskStatus.CANCELLED:
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_cancelled(), 2)
        snapshot = await state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert snapshot is not None
        assert snapshot.node_states[0].status is TaskStatus.CANCELLED
        assert snapshot.node_states[0].error_code is None
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_old_handoff_result_cannot_commit_after_new_fence_handoff() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-handoff-fence-race", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    runner = _FenceRaceRunner()
    try:
        graph = TaskGraph("handoff-fence-race", (TaskNode("node"),))
        await admit_graph(state, graph)
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        repository = state.task.tasks
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )
        await asyncio.wait_for(runner.old_entered.wait(), 1)

        original_state = await repository.graph_state(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert original_state is not None
        old_fence = original_state.node_states[0].fence
        await repository.mark_recovery_required(
            None,
            tenant_id="tenant",
            graph_id=graph.graph_id,
            node_id="node",
            execution_id="execution-fence",
            expected_fence=old_fence,
            error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value,
            error_digest="c" * 64,
        )
        await repository.recover_graph(graph.graph_id, tenant_id="tenant")
        replacement = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="replacement-worker",
            lease_seconds=30,
        )
        await repository.handoff_execution(
            replacement,
            tenant_id="tenant",
            execution_id="execution-fence",
        )
        runner.release_old.set()
        await asyncio.wait_for(runner.new_wait_entered.wait(), 2)

        current = await repository.graph_state(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert current is not None
        assert current.node_states[0].status is TaskStatus.WAITING
        assert current.node_states[0].fence == replacement.fence
        assert current.node_states[0].result_digest is None

        runner.release_new.set()
        async def wait_for_success() -> None:
            while True:
                snapshot = await repository.graph_state(
                    graph.graph_id,
                    tenant_id="tenant",
                )
                assert snapshot is not None
                if snapshot.status is TaskStatus.SUCCEEDED:
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_success(), 2)
        completed = await repository.graph_state(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert completed is not None
        assert completed.node_states[0].result_digest == "b" * 64
    finally:
        runner.release_old.set()
        runner.release_new.set()
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_scheduler_retries_transient_reconcile_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-reconcile-retry", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("reconcile-retry", (TaskNode("node"),))
        await admit_graph(state, graph)
        original_reconcile = repository.scheduler_state
        attempts = 0

        async def reconcile(graph_id: str, *, tenant_id: str):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await original_reconcile(graph_id, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "scheduler_state", reconcile)
        runner = _BlockingRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph.graph_id,
                Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value),
                TaskGraphLimits(),
            )
        )

        await asyncio.wait_for(runner.entered.wait(), 1)

        assert attempts >= 2
        assert launcher.owns_graph(graph.graph_id, tenant_id="tenant")
    finally:
        if launcher is not None:
            await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_scheduler_timeout_boundary_does_not_lose_completion_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-boundary-activity", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "boundary-activity",
            (TaskNode("local"), TaskNode("foreign")),
        )
        await admit_graph(state, graph)
        await repository.claim(
            graph.graph_id,
            "foreign",
            tenant_id="tenant",
            owner="remote-worker",
            lease_seconds=30,
        )
        runner = _BlockingRunner()
        scheduler_wait_entered = asyncio.Event()
        boundary_returned = asyncio.Event()
        original_wait = asyncio.wait

        async def boundary_wait(
            tasks: object,
            *,
            timeout: float | None = None,
            return_when: str = asyncio.ALL_COMPLETED,
        ) -> tuple[set[asyncio.Task[object]], set[asyncio.Task[object]]]:
            selected = tuple(tasks)  # type: ignore[arg-type]
            scheduler_wait = any(
                task.get_name().startswith("task-node-") for task in selected
            )
            if scheduler_wait and not boundary_returned.is_set():
                scheduler_wait_entered.set()
                while True:
                    states = await repository.list_nodes(
                        graph.graph_id,
                        tenant_id="tenant",
                    )
                    local = next(node for node in states if node.node_id == "local")
                    if local.status is TaskStatus.SUCCEEDED:
                        boundary_returned.set()
                        return set(), set(selected)
                    await asyncio.sleep(0)
            return await original_wait(
                selected,
                timeout=timeout,
                return_when=return_when,
            )

        monkeypatch.setattr(task_local.asyncio, "wait", boundary_wait)
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph.graph_id,
                Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value),
                TaskGraphLimits(max_concurrency=2),
            )
        )

        await asyncio.wait_for(runner.entered.wait(), 1)
        await asyncio.wait_for(scheduler_wait_entered.wait(), 1)
        generation = launcher.graph_activity_generation(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert generation is not None
        activity = asyncio.create_task(
            launcher.wait_graph_activity(
                graph.graph_id,
                tenant_id="tenant",
                after_generation=generation,
            )
        )
        runner.release.set()

        await asyncio.wait_for(boundary_returned.wait(), 1)
        await asyncio.wait_for(activity, 1)
        states = await repository.list_nodes(graph.graph_id, tenant_id="tenant")
        by_id = {node.node_id: node for node in states}
        assert by_id["local"].status is TaskStatus.SUCCEEDED
        assert by_id["foreign"].status is TaskStatus.RUNNING
    finally:
        if launcher is not None:
            await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_local_event_stream_observers_do_not_poll_durable_snapshots_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-observer-idle", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    streams = []
    pending: list[asyncio.Task[object]] = []
    try:
        repository = state.task.tasks
        graph = TaskGraph("observer-idle", (TaskNode("local"),))
        await admit_graph(state, graph)
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        runner = _BlockingRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )
        await asyncio.wait_for(runner.entered.wait(), 1)
        await asyncio.sleep(0.05)

        service = DefaultTaskGraphService(
            SimpleNamespace(tasks=repository),
            _AllowAuthorization(),
            local_waiter=launcher,
        )
        history = await service.list_events(
            graph.graph_id,
            principal=principal,
            limit=100,
        )
        assert history.items
        after_event_seq = history.items[-1].event_seq

        snapshot_calls = 0
        original_snapshot = repository.graph_state

        async def counting_snapshot(
            graph_id: str,
            *,
            tenant_id: str,
        ):
            nonlocal snapshot_calls
            snapshot_calls += 1
            return await original_snapshot(graph_id, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "graph_state", counting_snapshot)
        streams = [
            service.stream_events(
                graph.graph_id,
                principal=principal,
                after_event_seq=after_event_seq,
            ),
            service.stream_events(
                graph.graph_id,
                principal=principal,
                after_event_seq=after_event_seq,
            ),
        ]
        pending = [asyncio.create_task(anext(stream)) for stream in streams]
        await asyncio.sleep(0.05)

        assert snapshot_calls == 0
        assert all(not task.done() for task in pending)
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for stream in streams:
            await stream.aclose()
        if launcher is not None:
            await launcher.shutdown()
            assert runner.cancelled.is_set()
        await state.close()


@pytest.mark.asyncio
async def test_local_event_stream_observes_foreign_update_via_scheduler_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(
        namespace="task-observer-scheduler-notify", tenant_id="tenant"
    )
    launcher: LocalTaskGraphLauncher | None = None
    stream = None
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "observer-scheduler-notify",
            (TaskNode("local"), TaskNode("foreign")),
        )
        await admit_graph(state, graph)
        foreign_lease = await repository.claim(
            graph.graph_id,
            "foreign",
            tenant_id="tenant",
            owner="remote-worker",
            lease_seconds=30,
        )
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        runner = _BlockingRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph.graph_id,
                principal,
                TaskGraphLimits(max_concurrency=2),
            )
        )
        await asyncio.wait_for(runner.entered.wait(), 1)

        service = DefaultTaskGraphService(
            SimpleNamespace(tasks=repository),
            _AllowAuthorization(),
            local_waiter=launcher,
        )
        history = await service.list_events(
            graph.graph_id,
            principal=principal,
            limit=100,
        )
        assert history.items
        stream = service.stream_events(
            graph.graph_id,
            principal=principal,
            after_event_seq=history.items[-1].event_seq,
        )
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)

        await repository.complete(
            foreign_lease,
            tenant_id="tenant",
            execution_id="execution-foreign",
            result_digest="b" * 64,
        )

        event = await asyncio.wait_for(pending, 2)
        assert event.node_id == "foreign"
        assert event.previous_status is TaskStatus.RUNNING
        assert event.status is TaskStatus.SUCCEEDED
        assert event.result_digest == "b" * 64
    finally:
        if stream is not None:
            await stream.aclose()
        if launcher is not None:
            await launcher.shutdown()
            assert runner.cancelled.is_set()
        await state.close()
