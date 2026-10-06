#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode, TaskObservationError
from linktools.ai.runtime import Runtime, RuntimeContext, TaskGraphRun, TaskGraphWaitResult
from linktools.ai.runtime.service_api import _ExecutionStreamFailure
from linktools.ai.task import TaskEvent, TaskEventType, TaskGraphInfo, TaskGraphState


class Graph:
    def __init__(self, status=TaskStatus.SUCCEEDED):
        self.value = TaskGraphState("graph", status, (), (), 1)
        self.read = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.stream_release = asyncio.Event()
        self.error = None
        self.stream_error = None
        self.eof = False
        self.emit = False
        self.wait_calls = 0

    async def state(self, graph_id, *, principal):
        return TaskGraphState("graph", TaskStatus.RUNNING, (), (), 0)

    async def wait(self, graph_id, *, principal, timeout_seconds=None):
        assert timeout_seconds is None
        self.wait_calls += 1
        self.read.set()
        await self.release.wait()
        if self.error:
            raise self.error
        return self.value

    async def stream_events(self, graph_id, *, principal, after_sequence=0):
        if self.emit:
            for sequence in (1, 2):
                yield TaskEvent(1, "graph", sequence, TaskEventType.GRAPH_CHANGED,
                                datetime.now(timezone.utc), TaskStatus.RUNNING, TaskStatus.PENDING)
        if self.stream_error:
            self.release.set()
            raise self.stream_error
        if not self.eof:
            await self.stream_release.wait()
        if False:
            yield None


def make_run(graph, close_callback=None):
    stub = object()
    runtime = Runtime(stub, stub, stub, stub, graph, stub, stub, stub, stub, stub,
                      None, namespace="observed-test", context=RuntimeContext(None, tenant_id="tenant"),
                      close_callback=close_callback)
    async def tree(*args, **kwargs):
        if False:
            yield None
    return runtime, TaskGraphRun(runtime, graph, "graph", Principal("owner", "tenant"), tree)


async def ignore(event):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED,
                                    TaskStatus.BLOCKED, TaskStatus.WAITING, TaskStatus.RECOVERY_REQUIRED])
@pytest.mark.parametrize("include_content", [False, True])
async def test_wait_observed_returns_authoritative_same_read(status, include_content):
    graph = Graph(status)
    runtime, run = make_run(graph)
    outcome = await run.wait_observed(ignore, include_content=include_content)
    assert isinstance(outcome, TaskGraphWaitResult)
    assert outcome.status is status
    assert outcome.graph.event_sequence == 1
    assert isinstance(outcome.graph, TaskGraphState if include_content else TaskGraphInfo)
    if include_content:
        assert outcome.graph is graph.value
    assert not runtime._observation_sessions
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_eof_still_waits_for_authority():
    graph = Graph()
    graph.eof = True
    graph.release.clear()
    runtime, run = make_run(graph)
    task = asyncio.create_task(run.wait_observed(ignore))
    await graph.read.wait()
    await asyncio.sleep(0)
    assert not task.done()
    graph.release.set()
    assert (await task).status is TaskStatus.SUCCEEDED
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [AIError(ErrorCode.STORAGE_UNAVAILABLE), ValueError("decode")])
async def test_wait_observed_preserves_authoritative_stream_errors(error):
    graph = Graph()
    graph.stream_error = error
    graph.release.clear()
    runtime, run = make_run(graph)
    with pytest.raises(type(error)) as raised:
        await run.wait_observed(ignore)
    assert raised.value is error
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_only_downgrades_optional_stream_failure():
    graph = Graph()
    graph.stream_error = _ExecutionStreamFailure(RuntimeError("broker"))
    graph.release.clear()
    runtime, run = make_run(graph)
    outcome = await run.wait_observed(ignore)
    assert outcome.status is TaskStatus.SUCCEEDED
    assert outcome.observation_error.origin == "stream"
    assert isinstance(outcome.observation_error.__cause__, RuntimeError)
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_callback_cause_and_ack_survive_simultaneous_success():
    graph = Graph()
    graph.emit = True
    graph.release.clear()
    runtime, run = make_run(graph)
    cause = ValueError("callback")
    cursors = []
    async def callback(event):
        if cursors:
            graph.release.set()
            raise cause
        cursors.append(event.cursor)
    with pytest.raises(TaskObservationError) as raised:
        await run.wait_observed(callback)
    assert raised.value.origin == "callback"
    assert raised.value.__cause__ is cause
    assert raised.value.cursor == cursors[0]
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_authority_failure_wins_callback_failure():
    graph = Graph()
    graph.emit = True
    graph.release.clear()
    graph.error = AIError(ErrorCode.AUTHORIZATION_DENIED)
    runtime, run = make_run(graph)
    async def callback(event):
        graph.release.set()
        raise ValueError("callback")
    with pytest.raises(AIError) as raised:
        await run.wait_observed(callback)
    assert raised.value is graph.error
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_callback_cancelled_error_is_not_success():
    graph = Graph()
    graph.emit = True
    graph.release.clear()
    runtime, run = make_run(graph)
    async def callback(event):
        graph.release.set()
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await run.wait_observed(callback)
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("timeout_seconds", True), ("timeout_seconds", float("nan")),
    ("timeout_seconds", float("inf")), ("timeout_seconds", -1), ("close_timeout_seconds", 0),
    ("close_timeout_seconds", float("inf")), ("close_timeout_seconds", None), ("include_content", 1),
    ("cursor", "invalid")])
async def test_wait_observed_invalid_arguments_start_nothing(field, value):
    graph = Graph()
    runtime, run = make_run(graph)
    with pytest.raises(AIError):
        await run.wait_observed(ignore, **{field: value})
    assert graph.wait_calls == 0
    assert not runtime._observation_sessions
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_timeout_keeps_noncooperative_wait_owned_until_close_retry():
    graph = Graph()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    async def wait(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        raise ValueError("late read failure")
    graph.wait = wait
    closed = []
    async def close_storage():
        closed.append(True)
    runtime, run = make_run(graph, close_storage)
    start = asyncio.get_running_loop().time()
    with pytest.raises(AIError) as raised:
        await run.wait_observed(ignore, timeout_seconds=0.01, close_timeout_seconds=0.02)
    assert raised.value.code is ErrorCode.TASK_WAIT_TIMEOUT
    assert asyncio.get_running_loop().time() - start < 1
    assert cancelled.is_set()
    with pytest.raises(TaskObservationError) as raised:
        await runtime.close()
    assert raised.value.safe_details["cleanup_pending"] is True
    assert not closed
    with pytest.raises(AIError) as raised:
        await run.wait_observed(ignore)
    assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    release.set()
    await runtime.close()
    assert closed == [True]
    assert not runtime._observation_sessions


@pytest.mark.asyncio
async def test_wait_observed_cleanup_failure_does_not_report_success_or_deliver_again():
    graph = Graph()
    graph.emit = True
    graph.release.clear()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    async def callback(event):
        calls.append(event)
        entered.set()
        graph.release.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
    runtime, run = make_run(graph)
    with pytest.raises(TaskObservationError) as raised:
        await run.wait_observed(callback, close_timeout_seconds=0.01)
    assert raised.value.safe_details["phase"] == "cleanup"
    assert entered.is_set()
    release.set()
    await runtime.close()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_wait_observed_caller_cancel_does_not_control_graph():
    graph = Graph()
    graph.release.clear()
    runtime, run = make_run(graph)
    task = asyncio.create_task(run.wait_observed(ignore))
    await graph.read.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert graph.value.status is TaskStatus.SUCCEEDED
    assert not runtime._observation_sessions
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_raw_waiting_projection_keeps_dynamic_nodes_and_inputs():
    from linktools.ai.task import TaskNode, TaskNodeView
    graph = Graph()
    nodes = (TaskNode("original"), TaskNode("expanded", input={"secret": "input"}))
    states = tuple(TaskNodeView("graph", node.node_id, (), status, None, 1, None, None, None, None, execution_id="execution")
                   for node, status in zip(nodes, (TaskStatus.SUCCEEDED, TaskStatus.WAITING)))
    graph.value = TaskGraphState("graph", TaskStatus.RUNNING, nodes, states, 7)
    runtime, run = make_run(graph)
    hidden = await run.wait_observed(ignore)
    full = await run.wait_observed(ignore, include_content=True)
    assert hidden.status is full.status is TaskStatus.WAITING
    assert hidden.graph.status is full.graph.status is TaskStatus.RUNNING
    assert [node.node_id for node in hidden.graph.nodes] == ["original", "expanded"]
    assert full.graph is graph.value
    assert full.graph.nodes[1].input == {"secret": "input"}
    assert (await run.wait()).status is TaskStatus.WAITING
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_close_reclaims_simultaneous_observers_before_storage():
    graph = Graph()
    graph.release.clear()
    callbacks = []
    async def close_storage():
        callbacks.append("storage")
    runtime, run = make_run(graph, close_storage)
    calls = [asyncio.create_task(run.wait_observed(ignore)) for _ in range(3)]
    await graph.read.wait()
    await runtime.close()
    outcomes = await asyncio.gather(*calls, return_exceptions=True)
    assert all(isinstance(error, asyncio.CancelledError) for error in outcomes)
    assert callbacks == ["storage"]
    assert not runtime._observation_sessions


@pytest.mark.asyncio
async def test_wait_observed_owns_nested_noncooperative_stream_cleanup():
    graph = Graph()
    graph.release.clear()
    started = asyncio.Event()
    finish = asyncio.Event()
    async def stream_events(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await finish.wait()
        if False:
            yield None
    graph.stream_events = stream_events
    runtime, run = make_run(graph)
    task = asyncio.create_task(run.wait_observed(ignore, close_timeout_seconds=0.01))
    await started.wait()
    graph.release.set()
    with pytest.raises(TaskObservationError) as raised:
        await task
    assert raised.value.safe_details["cleanup_pending"]
    assert runtime._observation_sessions
    finish.set()
    await runtime.close()
    assert not runtime._observation_sessions


@pytest.mark.asyncio
async def test_wait_observed_authoritative_nested_cleanup_overrides_wait_success():
    graph = Graph()
    graph.release.clear()
    cause = ValueError("durable cleanup failure")
    async def stream_events(*args, **kwargs):
        graph.release.set()
        try:
            await asyncio.Event().wait()
        finally:
            raise cause
        yield
    graph.stream_events = stream_events
    runtime, run = make_run(graph)
    with pytest.raises(ValueError) as raised:
        await run.wait_observed(ignore)
    assert raised.value is cause
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_cancel", [False, True])
async def test_wait_observed_callback_cleanup_keeps_authority_and_caller_cancel_priority(caller_cancel):
    graph = Graph()
    graph.emit = True
    graph.release.clear()
    entered = asyncio.Event()
    cause = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    async def stream_events(*args, **kwargs):
        try:
            yield TaskEvent(1, "graph", 1, TaskEventType.GRAPH_CHANGED,
                            datetime.now(timezone.utc), TaskStatus.RUNNING, TaskStatus.PENDING)
            await asyncio.Event().wait()
        finally:
            raise cause
    graph.stream_events = stream_events
    async def callback(event):
        entered.set()
        await asyncio.Event().wait()
    runtime, run = make_run(graph)
    task = asyncio.create_task(run.wait_observed(callback))
    await entered.wait()
    if caller_cancel:
        task.cancel()
    else:
        graph.release.set()
    with pytest.raises(asyncio.CancelledError if caller_cancel else AIError) as raised:
        await task
    if not caller_cancel:
        assert raised.value is cause
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_timeout_precedes_optional_cleanup_failure():
    graph = Graph()
    graph.release.clear()
    async def stream_events(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            raise _ExecutionStreamFailure(RuntimeError("close failed"))
        yield
    graph.stream_events = stream_events
    runtime, run = make_run(graph)
    with pytest.raises(AIError) as raised:
        await run.wait_observed(ignore, timeout_seconds=0.01)
    assert raised.value.code is ErrorCode.TASK_WAIT_TIMEOUT
    await runtime.close()


def test_wait_observed_synchronous_callback_requires_schedulable_loop():
    import subprocess
    import sys
    import textwrap
    script = textwrap.dedent('''
        import asyncio
        import time
        from tests.ai.test_task_wait_observed import Graph, make_run
        from linktools.ai.errors import AIError, ErrorCode
        async def main():
            graph = Graph()
            graph.emit = True
            graph.release.clear()
            runtime, run = make_run(graph)
            elapsed = []
            async def callback(event):
                start = time.monotonic()
                time.sleep(0.05)
                elapsed.append(time.monotonic() - start)
            start = time.monotonic()
            try:
                await run.wait_observed(callback, timeout_seconds=0.001)
            except AIError as error:
                assert error.code is ErrorCode.TASK_WAIT_TIMEOUT
            else:
                raise AssertionError("wait should time out")
            assert elapsed and time.monotonic() - start >= 0.05
            await runtime.close()
        asyncio.run(main())
    ''')
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
