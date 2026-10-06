#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task wait returns the consistent durable read that satisfied its boundary."""

import asyncio
from types import SimpleNamespace

import pytest

from linktools.ai.core import AuthorizationAction, Principal, ResourceKind, ResourceRef, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.task import (
    DefaultTaskGraphService,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphState,
    TaskGraphSubmission,
    TaskNode,
    TaskNodeView,
    TaskRef,
    TaskSubmissionResult,
)


_PRINCIPAL = Principal("reader", "tenant")
_TERMINAL = (
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.BLOCKED,
)


def _state(
    status: TaskStatus,
    *,
    node_status: TaskStatus | None = None,
    nodes: tuple[TaskNode, ...] | None = None,
    event_sequence: int = 7,
) -> TaskGraphState:
    nodes = (TaskNode("node"),) if nodes is None else nodes
    node_status = status if node_status is None else node_status
    recovery = node_status is TaskStatus.RECOVERY_REQUIRED
    states = tuple(
        TaskNodeView(
            graph_id="graph",
            node_id=node.node_id,
            dependencies=node.dependencies,
            status=node_status,
            owner=None,
            fence=1,
            lease_expires_at=None,
            result_digest="a" * 64 if node_status is TaskStatus.SUCCEEDED else None,
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value if recovery else None,
            error_digest="b" * 64 if recovery else None,
            execution_id=None if node_status is TaskStatus.PENDING else f"execution-{node.node_id}",
        )
        for node in nodes
    )
    return TaskGraphState("graph", status, nodes, states, event_sequence)


class _Repository:
    def __init__(self, state: TaskGraphState) -> None:
        self.current = state

    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef:
        return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def graph_state(self, graph_id: str, *, tenant_id: str) -> TaskGraphState:
        assert (graph_id, tenant_id) == ("graph", _PRINCIPAL.tenant_id)
        return self.current


class _Authorization:
    async def authorize(
        self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
    ) -> None:
        assert principal == _PRINCIPAL
        assert action is AuthorizationAction.TASK_READ
        assert resource == ResourceRef(ResourceKind.TASK_GRAPH, "graph", principal.tenant_id)


class _Waiter:
    def __init__(
        self,
        repository: _Repository,
        *,
        owned: bool = True,
        next_state: TaskGraphState | None = None,
    ) -> None:
        self.repository = repository
        self.owned = owned
        self.next_state = next_state
        self.started = asyncio.Event()
        self.interrupted = asyncio.Event()

    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:
        del graph_id, tenant_id
        return self.owned

    def graph_activity_generation(self, graph_id: str, *, tenant_id: str) -> int:
        del graph_id, tenant_id
        return 0

    def graph_failure(self, graph_id: str, *, tenant_id: str) -> AIError | None:
        del graph_id, tenant_id
        return None

    async def wait_graph_activity(
        self, graph_id: str, *, tenant_id: str, after_generation: int | None = None,
    ) -> None:
        del graph_id, tenant_id, after_generation
        assert self.owned
        self.started.set()
        if self.next_state is not None:
            self.repository.current = self.next_state
            return
        try:
            await asyncio.Event().wait()
        finally:
            self.interrupted.set()


def _service(repository: _Repository, waiter: _Waiter | None = None) -> DefaultTaskGraphService:
    return DefaultTaskGraphService(
        SimpleNamespace(tasks=repository), _Authorization(), local_waiter=waiter,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", _TERMINAL + (TaskStatus.WAITING, TaskStatus.RECOVERY_REQUIRED))
async def test_wait_returns_durable_stop_state(status: TaskStatus) -> None:
    state = _state(status)
    service = _service(_Repository(state))

    assert await service.wait("graph", principal=_PRINCIPAL) == state


@pytest.mark.asyncio
async def test_run_projects_the_wait_state_as_a_submission_result() -> None:
    state = _state(TaskStatus.SUCCEEDED)

    class SubmittedService(DefaultTaskGraphService):
        async def prepare_submission(self, request: TaskGraphRequest) -> TaskGraphSubmission:
            return TaskGraphSubmission("test", TaskGraphAdmission.from_request(request), request.graph)

        async def start_prepared(self, submission: TaskGraphSubmission) -> TaskSubmissionResult:
            return TaskSubmissionResult(
                submission.ref, True, TaskGraphResult("graph", TaskStatus.RUNNING),
            )

    service = SubmittedService(SimpleNamespace(tasks=_Repository(state)), _Authorization())
    result = await service.run(TaskGraphRequest(TaskGraph("graph", state.nodes), _PRINCIPAL, "run"))

    assert isinstance(result, TaskGraphResult)
    assert result.status is state.status
    assert tuple(node.node_id for node in result.node_results) == ("node",)
    assert result.node_results[0].result_digest == state.node_states[0].result_digest
    assert result.node_results[0].execution_id == state.node_states[0].execution_id


@pytest.mark.asyncio
async def test_wait_retains_raw_running_status_at_stable_input_boundary() -> None:
    state = _state(TaskStatus.RUNNING, node_status=TaskStatus.WAITING)
    service = _service(_Repository(state))

    observed = await service.wait("graph", principal=_PRINCIPAL)
    assert observed == state
    assert observed.status is TaskStatus.RUNNING
    assert observed.node_states[0].status is TaskStatus.WAITING


@pytest.mark.asyncio
async def test_wait_keeps_stop_read_when_graph_resumes_before_return() -> None:
    stopped = _state(TaskStatus.RUNNING, node_status=TaskStatus.WAITING, event_sequence=11)
    resumed = _state(
        TaskStatus.RUNNING,
        nodes=(TaskNode("node"), TaskNode("expanded", dependencies=("node",))),
        event_sequence=13,
    )

    class ResumingRepository(_Repository):
        async def graph_state(self, graph_id: str, *, tenant_id: str) -> TaskGraphState:
            state = await super().graph_state(graph_id, tenant_id=tenant_id)
            self.current = resumed
            return state

    service = _service(ResumingRepository(stopped))
    observed = await service.wait("graph", principal=_PRINCIPAL, timeout_seconds=0.1)

    assert observed == stopped
    assert observed.event_sequence == 11
    assert tuple(node.node_id for node in observed.nodes) == ("node",)
    assert await service.state("graph", principal=_PRINCIPAL) == resumed


@pytest.mark.asyncio
async def test_wait_includes_expanded_nodes_from_the_stopping_read() -> None:
    repository = _Repository(_state(TaskStatus.PENDING))
    expanded = _state(
        TaskStatus.SUCCEEDED,
        nodes=(TaskNode("node"), TaskNode("expanded", dependencies=("node",))),
        event_sequence=19,
    )
    waiter = _Waiter(repository, next_state=expanded)

    observed = await _service(repository, waiter).wait("graph", principal=_PRINCIPAL)

    assert waiter.started.is_set()
    assert observed == expanded
    assert tuple(node.node_id for node in observed.nodes) == tuple(
        node.node_id for node in observed.node_states
    ) == ("node", "expanded")


@pytest.mark.asyncio
@pytest.mark.parametrize("owned,deferred_input", [(False, False), (True, True), (True, False)])
async def test_wait_keeps_local_wait_ownership_and_deferred_input_semantics(
    owned: bool, deferred_input: bool,
) -> None:
    node = TaskNode("node", task=TaskRef.deferred_input() if deferred_input else None)
    waiting = _state(TaskStatus.RUNNING, node_status=TaskStatus.WAITING, nodes=(node,))
    succeeded = _state(TaskStatus.SUCCEEDED, nodes=(node,), event_sequence=9)
    repository = _Repository(waiting)
    waiter = _Waiter(repository, owned=owned, next_state=succeeded)

    observed = await _service(repository, waiter).wait("graph", principal=_PRINCIPAL)

    keep_waiting = owned and not deferred_input
    assert observed == (succeeded if keep_waiting else waiting)
    assert waiter.started.is_set() is keep_waiting


@pytest.mark.asyncio
async def test_wait_releases_dependencies_from_the_same_terminal_read() -> None:
    state = _state(TaskStatus.SUCCEEDED)
    released: list[TaskGraphState] = []

    class Preflight:
        async def release_graph_dependencies(self, graph: TaskGraphState, *, tenant_id: str) -> None:
            assert tenant_id == _PRINCIPAL.tenant_id
            released.append(graph)

    service = DefaultTaskGraphService(
        SimpleNamespace(tasks=_Repository(state)), _Authorization(), preflight=Preflight(),
    )

    assert await service.wait("graph", principal=_PRINCIPAL) == state
    assert released == [state]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [True, -1, float("nan"), float("inf"), "1"])
async def test_wait_timeout_validation_precedes_authorization(timeout: object) -> None:
    service = DefaultTaskGraphService(SimpleNamespace(), _Authorization())

    with pytest.raises(AIError) as caught:
        await service.wait("graph", principal=_PRINCIPAL, timeout_seconds=timeout)

    assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, 0.01])
async def test_wait_timeout_preserves_running_graph(timeout: float) -> None:
    running = _state(TaskStatus.RUNNING)
    repository = _Repository(running)
    waiter = _Waiter(repository)

    with pytest.raises(AIError) as caught:
        await _service(repository, waiter).wait(
            "graph", principal=_PRINCIPAL, timeout_seconds=timeout,
        )

    assert caught.value.code is ErrorCode.TASK_WAIT_TIMEOUT
    assert caught.value.safe_details["graph_id"] == "graph"
    assert repository.current == running
    if timeout > 0:
        assert waiter.interrupted.is_set()


@pytest.mark.asyncio
async def test_wait_caller_cancellation_only_interrupts_local_waiting() -> None:
    running = _state(TaskStatus.RUNNING)
    repository = _Repository(running)
    waiter = _Waiter(repository)
    task = asyncio.create_task(_service(repository, waiter).wait("graph", principal=_PRINCIPAL))
    await waiter.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert waiter.interrupted.is_set()
    assert repository.current == running


@pytest.mark.asyncio
async def test_wait_propagates_authoritative_read_failure_unchanged() -> None:
    error = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    class BrokenRepository(_Repository):
        async def graph_state(self, graph_id: str, *, tenant_id: str) -> TaskGraphState:
            raise error

    with pytest.raises(AIError) as caught:
        await _service(BrokenRepository(_state(TaskStatus.RUNNING))).wait(
            "graph", principal=_PRINCIPAL,
        )

    assert caught.value is error
