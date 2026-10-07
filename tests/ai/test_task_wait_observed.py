#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.runtime import Runtime, RuntimeContext, TaskGraphRun, WaitResult
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
    from types import SimpleNamespace
    stub = SimpleNamespace(_bind_observation=lambda *args: None)
    runtime = Runtime(stub, stub, stub, stub, graph, stub, stub, stub, stub, stub,
                      None, namespace="observed-test", context=RuntimeContext(None, tenant_id="tenant"),
                      close_callback=close_callback)
    async def tree(*args, **kwargs):
        if kwargs.get("ready") is not None:
            kwargs["ready"].set()
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
    outcome = await run.wait(on_event=ignore, include_content=include_content)
    assert isinstance(outcome, WaitResult)
    assert outcome.result.wait_status is status
    assert outcome.result.event_sequence == 1
    assert isinstance(outcome.result, TaskGraphState if include_content else TaskGraphInfo)
    if include_content:
        assert outcome.result is graph.value
    assert not runtime._observation_sessions
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_eof_still_waits_for_authority():
    graph = Graph()
    graph.eof = True
    graph.release.clear()
    runtime, run = make_run(graph)
    task = asyncio.create_task(run.wait(on_event=ignore))
    await graph.read.wait()
    await asyncio.sleep(0)
    assert not task.done()
    graph.release.set()
    assert (await task).result.wait_status is TaskStatus.SUCCEEDED
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [AIError(ErrorCode.STORAGE_UNAVAILABLE), ValueError("decode")])
async def test_wait_observed_preserves_authoritative_stream_errors(error):
    graph = Graph()
    graph.stream_error = error
    graph.release.clear()
    runtime, run = make_run(graph)
    with pytest.raises(type(error)) as raised:
        await run.wait(on_event=ignore)
    assert raised.value is error
    await runtime.close()


@pytest.mark.asyncio
async def test_wait_observed_only_downgrades_optional_stream_failure():
    graph = Graph()
    graph.stream_error = _ExecutionStreamFailure(RuntimeError("broker"))
    graph.release.clear()
    runtime, run = make_run(graph)
    outcome = await run.wait(on_event=ignore)
    assert outcome.result.wait_status is TaskStatus.SUCCEEDED
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
    with pytest.raises(ObservationError) as raised:
        await run.wait(on_event=callback)
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
        await run.wait(on_event=callback)
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
        await run.wait(on_event=callback)
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
        await run.wait(on_event=ignore, **{field: value})
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
        await run.wait(on_event=ignore, timeout_seconds=0.01, close_timeout_seconds=0.02)
    assert raised.value.code is ErrorCode.WAIT_TIMEOUT
    assert asyncio.get_running_loop().time() - start < 1
    assert cancelled.is_set()
    with pytest.raises(ObservationError) as raised:
        await runtime.close()
    assert raised.value.safe_details["cleanup_pending"] is True
    assert not closed
    with pytest.raises(AIError) as raised:
        await run.wait(on_event=ignore)
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
    with pytest.raises(ObservationError) as raised:
        await run.wait(on_event=callback, close_timeout_seconds=0.01)
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
    task = asyncio.create_task(run.wait(on_event=ignore))
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
    hidden = await run.wait(on_event=ignore)
    full = await run.wait(on_event=ignore, include_content=True)
    assert hidden.result.wait_status is full.result.wait_status is TaskStatus.WAITING
    assert hidden.result.status is full.result.status is TaskStatus.RUNNING
    assert [node.node_id for node in hidden.result.nodes] == ["original", "expanded"]
    assert full.result is graph.value
    assert full.result.nodes[1].input == {"secret": "input"}
    assert (await run.wait()).result.wait_status is TaskStatus.WAITING
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_close_reclaims_simultaneous_observers_before_storage():
    graph = Graph()
    graph.release.clear()
    callbacks = []
    async def close_storage():
        callbacks.append("storage")
    runtime, run = make_run(graph, close_storage)
    calls = [asyncio.create_task(run.wait(on_event=ignore)) for _ in range(3)]
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
    task = asyncio.create_task(run.wait(on_event=ignore, close_timeout_seconds=0.01))
    await started.wait()
    graph.release.set()
    with pytest.raises(ObservationError) as raised:
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
        await run.wait(on_event=ignore)
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
    task = asyncio.create_task(run.wait(on_event=callback))
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
        await run.wait(on_event=ignore, timeout_seconds=0.01)
    assert raised.value.code is ErrorCode.WAIT_TIMEOUT
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
                await run.wait(on_event=callback, timeout_seconds=0.001)
            except AIError as error:
                assert error.code is ErrorCode.WAIT_TIMEOUT
            else:
                raise AssertionError("wait should time out")
            assert elapsed and time.monotonic() - start >= 0.05
            await runtime.close()
        asyncio.run(main())
    ''')
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_runtime_close_cancels_all_sessions_before_draining_resistant_callback():
    graph = Graph()
    graph.emit = True
    graph.release.clear()
    cancelled_waiters = []
    async def wait(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_waiters.append(asyncio.current_task())
            raise
    graph.wait = wait
    storage_closed = []
    async def close_storage():
        storage_closed.append(True)
    runtime, run = make_run(graph, close_storage)
    gates = [asyncio.Event(), asyncio.Event()]
    entered = {}
    cancelled = set()
    effects = set()
    resistant = None
    async def callback(label, event):
        entered[label] = asyncio.current_task()
        try:
            await gates[label].wait()
        except asyncio.CancelledError:
            cancelled.add(label)
            if label != resistant:
                raise
            await gates[label].wait()
        effects.add(label)
    calls = [asyncio.create_task(run.wait(on_event=
        lambda event, label=label: callback(label, event), close_timeout_seconds=0.02,
    )) for label in range(2)]
    while len(entered) < 2:
        await asyncio.sleep(0)
    # Select the first drained session to exercise the early-budget-exhaustion case.
    first = next(iter(runtime._observation_sessions))
    resistant = next(label for label, task in entered.items() if task in first.tasks)
    cooperative = 1 - resistant
    try:
        with pytest.raises(ObservationError) as raised:
            await runtime.close()
        assert raised.value.safe_details["cleanup_pending"]
        assert not storage_closed
        assert not runtime._closed
        gates[cooperative].set()
        for _ in range(20):
            await asyncio.sleep(0)
        assert cancelled == {0, 1}
        assert len(cancelled_waiters) == 2
        assert cooperative not in effects
        assert calls[cooperative].done()
    finally:
        gates[resistant].set()
        await runtime.close()
        await asyncio.gather(*calls, return_exceptions=True)
    assert storage_closed == [True]
    assert not runtime._observation_sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["pending", "success", "failure", "timeout"])
@pytest.mark.parametrize("cancel_callback", [False, True])
async def test_observer_failure_starts_bounded_cleanup_before_stream_closes(authority, cancel_callback):
    graph = Graph()
    graph.release.clear()
    if authority == "failure":
        graph.error = AIError(ErrorCode.AUTHORIZATION_DENIED)
    close_release = asyncio.Event()
    failed = asyncio.Event()
    close_started = asyncio.Event()
    async def events(*args, **kwargs):
        try:
            for sequence in (1, 2):
                yield TaskEvent(1, "graph", sequence, TaskEventType.GRAPH_CHANGED,
                                datetime.now(timezone.utc), TaskStatus.RUNNING, TaskStatus.PENDING)
        finally:
            close_started.set()
            await close_release.wait()
    graph.stream_events = events
    runtime, run = make_run(graph)
    cause = asyncio.CancelledError("callback cancelled") if cancel_callback else ValueError("callback failed")
    acknowledged = []
    async def callback(event):
        if not acknowledged:
            acknowledged.append(event.cursor)
            return
        if authority in {"success", "failure"}:
            graph.release.set()
        failed.set()
        raise cause
    async def outcome():
        try:
            return await run.wait(on_event=
                callback, timeout_seconds=0.01 if authority == "timeout" else None,
                close_timeout_seconds=0.01,
            )
        except BaseException as error:
            return error
    task = asyncio.create_task(outcome())
    try:
        await failed.wait()
        await close_started.wait()
        done, _ = await asyncio.wait({task}, timeout=0.15)
        assert done, "observer failure must start the cleanup budget before aclose finishes"
        expected = asyncio.CancelledError if cancel_callback else (AIError if authority == "failure" else ObservationError)
        error = task.result()
        assert isinstance(error, expected)
        if cancel_callback:
            assert error is cause
        elif authority == "failure":
            assert error is graph.error
        else:
            assert error.origin == "callback"
            assert error.__cause__ is cause
            assert error.cursor == acknowledged[0]
        assert runtime._observation_sessions
        with pytest.raises(ObservationError):
            await runtime.close()
        assert not runtime._closed
    finally:
        close_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()
    assert not runtime._observation_sessions


@pytest.mark.asyncio
async def test_callback_background_observation_does_not_report_to_parent_session():
    outer = Graph()
    outer.emit = True
    outer.release.clear()
    inner = Graph()
    inner.emit = True
    inner.release.clear()
    outer_runtime, outer_run = make_run(outer)
    inner_runtime, inner_run = make_run(inner)
    release_child = asyncio.Event()
    callback_returned = asyncio.Event()
    spawned = []
    cause = ValueError("background observation")
    async def child_callback(event):
        await release_child.wait()
        raise cause
    async def parent_callback(event):
        if not spawned:
            spawned.append(asyncio.create_task(inner_run.wait(on_event=child_callback)))
            callback_returned.set()
    task = asyncio.create_task(outer_run.wait(on_event=parent_callback))
    try:
        await callback_returned.wait()
        release_child.set()
        with pytest.raises(ObservationError) as raised:
            await spawned[0]
        assert raised.value.__cause__ is cause
        assert not task.done()
        outer.release.set()
        assert (await task).observation_error is None
    finally:
        await inner_runtime.close()
        await outer_runtime.close()
        await asyncio.gather(task, *spawned, return_exceptions=True)


@pytest.mark.asyncio
async def test_nested_observed_wait_keeps_its_error_in_its_own_session():
    outer = Graph()
    outer.emit = True
    outer.release.clear()
    inner = Graph()
    inner.emit = True
    inner.release.clear()
    outer_runtime, outer_run = make_run(outer)
    inner_runtime, inner_run = make_run(inner)
    cause = ValueError("inner observer")
    async def inner_callback(event):
        raise cause
    async def outer_callback(event):
        with pytest.raises(ObservationError) as raised:
            await inner_run.wait(on_event=inner_callback)
        assert raised.value.__cause__ is cause
        outer.release.set()
    try:
        assert (await outer_run.wait(on_event=outer_callback)).observation_error is None
    finally:
        await inner_runtime.close()
        await outer_runtime.close()


@pytest.mark.asyncio
async def test_parallel_observed_wait_failure_does_not_interrupt_other_session():
    failing = Graph()
    failing.emit = True
    failing.release.clear()
    healthy = Graph()
    healthy.release.clear()
    bad_runtime, bad_run = make_run(failing)
    good_runtime, good_run = make_run(healthy)
    async def callback(event):
        raise ValueError("one observer")
    bad = asyncio.create_task(bad_run.wait(on_event=callback))
    good = asyncio.create_task(good_run.wait(on_event=ignore))
    try:
        with pytest.raises(ObservationError):
            await bad
        assert not good.done()
        healthy.release.set()
        assert (await good).result.wait_status is TaskStatus.SUCCEEDED
    finally:
        await bad_runtime.close()
        await good_runtime.close()
        await asyncio.gather(bad, good, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["authority", "cancellation", "optional"])
async def test_nested_stream_failure_is_reported_before_sibling_cleanup_finishes(kind):
    from .test_runtime_watch_api import _ExecutionService, _TaskGraphService
    graph = Graph()
    graph.release.clear()
    graph.state = _TaskGraphService().state
    stream_started = asyncio.Event()
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    cause = (AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) if kind == "authority" else
             asyncio.CancelledError("nested cancellation") if kind == "cancellation" else
             _ExecutionStreamFailure(RuntimeError("optional stream")))
    async def events(*args, **kwargs):
        try:
            yield TaskEvent(1, "graph", 1, TaskEventType.NODE_CHANGED, datetime.now(timezone.utc),
                            TaskStatus.RUNNING, TaskStatus.READY, "node", "worker", 1, "execution")
            stream_started.set()
            await asyncio.Event().wait()
        finally:
            close_started.set()
            await close_release.wait()
    async def tree(*args, **kwargs):
        if kwargs.get("ready") is not None:
            kwargs["ready"].set()
        await stream_started.wait()
        raise cause
        yield
    graph.stream_events = events
    from types import SimpleNamespace
    stub = SimpleNamespace(_bind_observation=lambda *args: None)
    runtime = Runtime(stub, stub, _ExecutionService(), stub, graph, stub, stub, stub, stub, stub,
                      None, namespace="nested-errors", context=RuntimeContext(None, tenant_id="tenant"))
    run = TaskGraphRun(runtime, graph, "graph", Principal("owner", "tenant"), tree)
    async def outcome():
        try:
            return await run.wait(on_event=ignore, close_timeout_seconds=0.01)
        except BaseException as error:
            return error
    task = asyncio.create_task(outcome())
    try:
        await close_started.wait()
        done, _ = await asyncio.wait({task}, timeout=0.1)
        if kind == "optional":
            assert not done
            close_release.set()
            graph.release.set()
            outcome = await task
            assert outcome.observation_error.__cause__ is cause.cause
        else:
            assert done
            if kind == "cancellation":
                # Python 3.10 normalizes cancellation payloads at the source task boundary.
                assert isinstance(task.result(), asyncio.CancelledError)
            else:
                assert task.result() is cause
            assert runtime._observation_sessions
    finally:
        close_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()
    assert not runtime._observation_sessions


@pytest.mark.asyncio
async def test_completed_stream_cleanup_error_does_not_wait_for_resistant_sibling():
    from .test_runtime_watch_api import _ExecutionService, _TaskGraphService
    graph = Graph()
    graph.release.clear()
    graph.state = _TaskGraphService().state
    release = asyncio.Event()
    execution_started = asyncio.Event()
    graph_started = asyncio.Event()
    cause = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    class GraphStream:
        first = True
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self.first:
                self.first = False
                return TaskEvent(1, "graph", 1, TaskEventType.NODE_CHANGED, datetime.now(timezone.utc),
                                 TaskStatus.RUNNING, TaskStatus.READY, "node", "worker", 1, "execution")
            graph_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise cause
        async def aclose(self):
            pass
    class ExecutionStream:
        def __aiter__(self):
            return self
        async def __anext__(self):
            execution_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            raise StopAsyncIteration
        async def aclose(self):
            await release.wait()
    graph.stream_events = lambda *args, **kwargs: GraphStream()
    from types import SimpleNamespace
    stub = SimpleNamespace(_bind_observation=lambda *args: None)
    runtime = Runtime(stub, stub, _ExecutionService(), stub, graph, stub, stub, stub, stub, stub,
                      None, namespace="cleanup-errors", context=RuntimeContext(None, tenant_id="tenant"))
    def tree(*args, **kwargs):
        if kwargs.get("ready") is not None:
            kwargs["ready"].set()
        return ExecutionStream()
    run = TaskGraphRun(runtime, graph, "graph", Principal("owner", "tenant"), tree)
    task = asyncio.create_task(run.wait(on_event=ignore, close_timeout_seconds=0.01))
    try:
        await execution_started.wait()
        await graph_started.wait()
        graph.release.set()
        done, _ = await asyncio.wait({task}, timeout=0.15)
        assert done
        with pytest.raises(AIError) as raised:
            task.result()
        assert raised.value is cause
        assert runtime._observation_sessions
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()
    assert not runtime._observation_sessions
