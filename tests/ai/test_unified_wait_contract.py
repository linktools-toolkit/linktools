#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared SDK wait contracts across independently owned run handles."""

import ast
import asyncio
import inspect
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal, TaskStatus, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.evaluation import DatasetRef, EvaluationProgress, EvaluationView
from linktools.ai.runtime import Agent, Execution, Runtime, RuntimeContext, Session, TaskGraphRun, WaitResult
from linktools.ai.runtime._domains import RuntimeExecutions
from linktools.ai.runtime._evaluation import EvaluationRun
from linktools.ai.runtime._execution_tree import ExecutionTreeBroker, ExecutionTreeStreamer
from linktools.ai.runtime._watch_cursor import (
    decode_execution_watch_cursor,
    encode_evaluation_watch_cursor,
    encode_execution_watch_cursor,
    encode_graph_watch_cursor,
)
from linktools.ai.runtime.service_api import ExecutionResult, ExecutionStreamEvent, ExecutionTreeEvent, ExecutionView
from linktools.ai.task import TaskGraphInfo, TaskGraphState


_NAMESPACE = "unified-wait-test"
_PRINCIPAL = Principal("owner", "tenant")


class _ExecutionService:
    def __init__(self) -> None:
        self.result = ExecutionResult("execution", ExecutionStatus.SUCCEEDED, {"answer": 42}, UsageMetrics())
        self.wait_calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.error: BaseException | None = None

    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds=None) -> ExecutionResult:
        assert execution_id == "execution" and principal == _PRINCIPAL
        assert timeout_seconds is None
        self.wait_calls += 1
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result


class _Tree:
    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.closed = asyncio.Event()
        self.release = asyncio.Event()
        self.emit = False

    def stream(
        self, execution_id: str, *, principal: Principal, after_sequences=None,
        include_content: bool = False, ready: asyncio.Event | None = None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        self.calls += 1

        async def values() -> AsyncIterator[ExecutionTreeEvent]:
            self.entered.set()
            if ready is not None:
                ready.set()
            try:
                if self.emit:
                    yield _execution_event(1)
                await self.release.wait()
            finally:
                self.closed.set()

        return values()


def _execution_event(sequence: int | None) -> ExecutionTreeEvent:
    return ExecutionTreeEvent(
        "execution", "agent", ExecutionLineageKind.RUN, None, "execution", None, 0,
        ExecutionStreamEvent("execution", sequence, "PROGRESS", {}),
    )


class _GraphService:
    def __init__(self) -> None:
        self.result = TaskGraphState("graph", TaskStatus.SUCCEEDED, (), (), 0)
        self.state_calls = 0
        self.wait_calls = 0
        self.stream_calls = 0
        self.started = asyncio.Event()
        self.stream_closed = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.error: BaseException | None = None

    async def state(self, graph_id: str, *, principal: Principal) -> TaskGraphState:
        self.state_calls += 1
        return self.result

    async def wait(self, graph_id: str, *, principal: Principal, timeout_seconds=None) -> TaskGraphState:
        assert timeout_seconds is None
        self.wait_calls += 1
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result

    async def stream_events(self, graph_id: str, *, principal: Principal, after_sequence=0):
        self.stream_calls += 1
        try:
            await asyncio.Event().wait()
            yield
        finally:
            self.stream_closed.set()


class _Evaluations:
    _namespace = _NAMESPACE

    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.result = EvaluationView(
            "evaluation", "experiment", None, DatasetRef("dataset", 1),
            EvaluationProgress(), "complete", (), now, now,
        )
        self.record_calls = 0
        self.inspect_calls = 0
        self.inspected = asyncio.Event()

    def _bind_observation(self, watch_graph, replay_graph, register, release) -> None:
        self._watch_graph = watch_graph
        self._replay_graph = replay_graph
        self._register_observation = register
        self._release_observation = release

    async def _record(self, experiment_id: str, principal: Principal):
        self.record_calls += 1
        return SimpleNamespace(intents=())

    async def _inspect(self, experiment_id: str, principal: Principal) -> EvaluationView:
        self.inspect_calls += 1
        self.inspected.set()
        return self.result


class _Bundle:
    def __init__(self) -> None:
        self.execution = _ExecutionService()
        self.graph = _GraphService()
        self.evaluations = _Evaluations()
        self.tree = _Tree()
        self.storage_closed = False
        self.streams_closed_at_storage_close = None
        stub = object()

        async def close() -> None:
            self.streams_closed_at_storage_close = (
                self.tree.closed.is_set(), self.graph.stream_closed.is_set(),
            )
            self.storage_closed = True

        self.runtime = Runtime(
            stub, stub, self.execution, stub, self.graph, self.evaluations,
            stub, stub, stub, stub, None,
            namespace=_NAMESPACE, context=RuntimeContext(None, tenant_id="tenant"),
            close_callback=close,
        )
        self.execution_run = Execution(self.runtime, "execution", _PRINCIPAL, self.tree.stream)
        self.graph_run = TaskGraphRun(self.runtime, self.graph, "graph", _PRINCIPAL, self.tree.stream)
        self.evaluation_run = EvaluationRun(self.evaluations, "evaluation", _PRINCIPAL)
        self.get_calls = 0

        async def get_execution(execution_id: str, principal: Principal) -> Execution:
            assert execution_id == "execution" and principal == _PRINCIPAL
            self.get_calls += 1
            return self.execution_run

        self.facade = RuntimeExecutions(self.execution, get_execution)

    def wait(self, owner: str, **kwargs):
        if owner == "facade":
            return self.facade.wait("execution", principal=_PRINCIPAL, **kwargs)
        return getattr(self, f"{owner}_run").wait(**kwargs)

    def cursor(self, owner: str, *, identity: str | None = None, include_content=False) -> str:
        if owner in {"execution", "facade"}:
            return encode_execution_watch_cursor(
                _NAMESPACE, "tenant", identity or "execution", include_content=include_content, sequences={},
            )
        if owner == "graph":
            return encode_graph_watch_cursor(
                _NAMESPACE, "tenant", identity or "graph", include_content=include_content,
                graph_sequence=0, execution_sequences={},
            )
        return encode_evaluation_watch_cursor(
            _NAMESPACE, "tenant", identity or "evaluation", include_content=include_content, graph_cursors={},
        )


async def _ignore(event) -> None:
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["execution", "graph", "evaluation", "facade"])
@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("include_content", [False, True])
async def test_each_owner_returns_wait_result_independent_of_callback(owner, observe, include_content) -> None:
    bundle = _Bundle()
    try:
        outcome = await bundle.wait(owner, on_event=_ignore if observe else None, include_content=include_content)
        assert type(outcome) is WaitResult
        assert outcome.cursor is None
        assert outcome.observation_error is None
        if owner in {"execution", "facade"}:
            assert outcome.result is bundle.execution.result
            assert bundle.execution.wait_calls == 1
            assert bundle.tree.calls == int(observe)
            assert bundle.get_calls == int(owner == "facade")
        elif owner == "graph":
            assert type(outcome.result) is (TaskGraphState if include_content else TaskGraphInfo)
            assert outcome.result.wait_status is TaskStatus.SUCCEEDED
            assert bundle.graph.wait_calls == 1
            if include_content:
                assert outcome.result is bundle.graph.result
            assert bundle.graph.state_calls == int(observe)
            assert bundle.graph.stream_calls == int(observe)
        else:
            assert outcome.result is bundle.evaluations.result
            assert bool(bundle.evaluations.record_calls) is observe
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["execution", "graph", "evaluation", "facade"])
async def test_eventless_wait_preserves_input_ack_and_rejects_cursor_without_callback(owner) -> None:
    bundle = _Bundle()
    cursor = bundle.cursor(owner)
    try:
        with pytest.raises(AIError) as raised:
            await bundle.wait(owner, cursor=cursor)
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
        outcome = await bundle.wait(owner, on_event=_ignore, cursor=cursor)
        assert outcome.cursor == cursor
        assert outcome.observation_error is None
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["execution", "graph", "evaluation", "facade"])
@pytest.mark.parametrize("wrong", ["owner", "content", "malformed"])
async def test_terminal_owner_cannot_short_circuit_invalid_cursor(owner, wrong) -> None:
    bundle = _Bundle()
    cursor = "not-a-cursor" if wrong == "malformed" else bundle.cursor(
        owner, identity="different" if wrong == "owner" else None, include_content=wrong == "content",
    )
    try:
        with pytest.raises(AIError) as raised:
            await bundle.wait(owner, on_event=_ignore, cursor=cursor)
        assert raised.value.code is ErrorCode.CURSOR_INVALID
        assert bundle.execution.wait_calls == bundle.graph.wait_calls == bundle.evaluations.inspect_calls == 0
        assert bundle.tree.calls == bundle.graph.stream_calls == bundle.evaluations.record_calls == 0
    finally:
        await bundle.runtime.close()


def test_graph_wait_overloads_preserve_content_selected_result_type() -> None:
    declaration = ast.parse(inspect.getsource(TaskGraphRun)).body[0]
    overloads = [
        node for node in declaration.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "wait"
        and any(isinstance(decorator, ast.Name) and decorator.id == "overload" for decorator in node.decorator_list)
    ]
    assert len(overloads) == 3
    assert {
        ast.unparse(next(arg.annotation for arg in node.args.kwonlyargs if arg.arg == "include_content")):
        ast.unparse(node.returns)
        for node in overloads
    } == {
        "Literal[False]": "WaitResult[TaskGraphInfo]",
        "Literal[True]": "WaitResult[TaskGraphState]",
        "bool": "WaitResult[TaskGraphInfo | TaskGraphState]",
    }
    expected = set(inspect.signature(TaskGraphRun.wait).parameters) - {"self"}
    assert all({arg.arg for arg in node.args.kwonlyargs} == expected for node in overloads)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["agent", "session"])
@pytest.mark.parametrize("method", ["run", "plan"])
@pytest.mark.parametrize("invalid", [
    {"timeout_seconds": True}, {"timeout_seconds": -1}, {"timeout_seconds": float("nan")},
    {"timeout_seconds": float("inf")}, {"close_timeout_seconds": 0},
    {"close_timeout_seconds": True}, {"include_content": 1}, {"on_event": object()},
])
async def test_convenience_wait_arguments_are_validated_before_start(owner, method, invalid) -> None:
    starts = []

    async def start(*args, **kwargs):
        starts.append((args, kwargs))
        raise AssertionError("invalid wait options must not start an execution")

    runtime = SimpleNamespace(_start_for_agent=start)
    handle = Agent(runtime, "agent", 1) if owner == "agent" else Session(runtime, "agent", 1, "session")
    with pytest.raises(AIError) as raised:
        await getattr(handle, method)("prompt", **invalid)
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert starts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["agent", "session"])
@pytest.mark.parametrize("method", ["run", "plan"])
@pytest.mark.parametrize("observe", [False, True])
async def test_convenience_wait_preserves_container_and_all_options(owner, method, observe) -> None:
    starts = []
    waits = []
    expected = WaitResult(object(), "ack", ObservationError("stream", cursor="ack"))

    async def wait(**kwargs):
        waits.append(kwargs)
        return expected

    async def start(*args, **kwargs):
        starts.append(kwargs)
        return SimpleNamespace(wait=wait)

    runtime = SimpleNamespace(_start_for_agent=start)
    handle = Agent(runtime, "agent", 1) if owner == "agent" else Session(runtime, "agent", 1, "session")
    callback = _ignore if observe else None
    result = await getattr(handle, method)(
        "prompt", on_event=callback, include_content=True, timeout_seconds=0.25, close_timeout_seconds=0.5,
    )
    assert result is expected
    assert starts[0]["mode"] == method
    assert waits == [{
        "on_event": callback, "include_content": True, "timeout_seconds": 0.25, "close_timeout_seconds": 0.5,
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["execution", "graph", "evaluation"])
async def test_terminal_owner_validates_cursor_membership_before_authoritative_wait(owner) -> None:
    bundle = _Bundle()
    broker = ExecutionTreeBroker()

    class Reader:
        async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
            return ExecutionView(
                execution_id, "agent", ExecutionStatus.SUCCEEDED,
                ExecutionLineageKind.RUN, None, execution_id, None,
            )

        async def list_children(self, execution_id: str, *, principal: Principal):
            return ()

    class Events:
        def stream(self, *args, **kwargs):
            raise AssertionError("invalid membership must fail before opening event streams")

    if owner == "execution":
        tree = ExecutionTreeStreamer(Reader(), Events(), broker)
        bundle.execution_run = Execution(bundle.runtime, "execution", _PRINCIPAL, tree.stream)
        cursor = encode_execution_watch_cursor(
            _NAMESPACE, "tenant", "execution", include_content=False, sequences={"outside": 1},
        )
    elif owner == "graph":
        cursor = encode_graph_watch_cursor(
            _NAMESPACE, "tenant", "graph", include_content=False,
            graph_sequence=0, execution_sequences={"outside": {"outside-execution": 1}},
        )
    else:
        cursor = encode_evaluation_watch_cursor(
            _NAMESPACE, "tenant", "evaluation", include_content=False, graph_cursors={"outside": "inner-cursor"},
        )
    try:
        with pytest.raises(AIError) as raised:
            await bundle.wait(owner, on_event=_ignore, cursor=cursor)
        assert raised.value.code in {ErrorCode.REQUEST_FIELD_INVALID, ErrorCode.CURSOR_INVALID}
        assert bundle.execution.wait_calls == bundle.graph.wait_calls == bundle.evaluations.inspect_calls == 0
        assert not broker._subscriptions
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["execution", "graph", "evaluation", "facade"])
@pytest.mark.parametrize("options", [
    {"timeout_seconds": True}, {"timeout_seconds": float("nan")},
    {"close_timeout_seconds": 0}, {"include_content": 1}, {"on_event": 1},
])
async def test_invalid_wait_options_start_no_work_for_any_owner(owner, options) -> None:
    bundle = _Bundle()
    try:
        with pytest.raises(AIError) as raised:
            await bundle.wait(owner, **options)
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert bundle.execution.wait_calls == bundle.graph.wait_calls == bundle.evaluations.inspect_calls == 0
        assert bundle.tree.calls == bundle.graph.stream_calls == bundle.evaluations.record_calls == 0
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["manual_close", "protocol_failure"])
async def test_execution_watch_closes_owned_iterator_on_every_exit(mode) -> None:
    bundle = _Bundle()
    closed = asyncio.Event()

    async def tree(*args, **kwargs):
        try:
            yield _execution_event(1)
            yield _execution_event(1)
        finally:
            closed.set()

    run = Execution(bundle.runtime, "execution", _PRINCIPAL, tree)
    stream = run.watch()
    try:
        first = await anext(stream)
        assert first.cursor is not None
        if mode == "manual_close":
            await stream.aclose()
        else:
            with pytest.raises(AIError) as raised:
                await anext(stream)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert closed.is_set()
    finally:
        await stream.aclose()
        await bundle.runtime.close()


@pytest.mark.asyncio
async def test_execution_wait_ack_advances_only_after_success_and_live_delta_keeps_watermark() -> None:
    bundle = _Bundle()
    bundle.execution.release.clear()

    async def tree(*args, ready=None, **kwargs):
        if ready is not None:
            ready.set()
        for sequence in (1, None, 2):
            yield _execution_event(sequence)

    run = Execution(bundle.runtime, "execution", _PRINCIPAL, tree)
    committed = []
    cause = TimeoutError("callback-owned timeout")

    async def callback(event) -> None:
        if event.event.durable_sequence == 2:
            raise cause
        committed.append(event)

    try:
        with pytest.raises(ObservationError) as raised:
            await run.wait(on_event=callback, timeout_seconds=1)
        assert raised.value.origin == "callback"
        assert raised.value.__cause__ is cause
        assert raised.value.cursor == committed[-1].cursor
        assert [item.event.durable_sequence for item in committed] == [1, None]
        assert [decode_execution_watch_cursor(
            _NAMESPACE, "tenant", "execution", item.cursor, include_content=False,
        ) for item in committed] == [{"execution": 1}, {"execution": 1}]
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("observe", [False, True])
async def test_authoritative_timeout_error_is_never_reclassified_as_sdk_deadline(observe) -> None:
    bundle = _Bundle()
    cause = TimeoutError("storage-owned timeout")
    bundle.execution.error = cause
    try:
        with pytest.raises(TimeoutError) as raised:
            await bundle.wait("execution", on_event=_ignore if observe else None, timeout_seconds=1)
        assert raised.value is cause
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["authority", "contract", "callback"])
async def test_failure_revealed_during_deadline_cleanup_keeps_its_source_priority(failure) -> None:
    bundle = _Bundle()
    bundle.execution.release.clear()
    cause = TimeoutError("callback timeout") if failure == "callback" else AIError(ErrorCode.STORAGE_UNAVAILABLE)

    async def authority(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise cause

    async def tree(*args, ready=None, **kwargs):
        if ready is not None:
            ready.set()
        try:
            if failure == "callback":
                yield _execution_event(1)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if failure == "contract":
                raise cause
            raise

    async def callback(event) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise cause

    if failure == "authority":
        bundle.execution.wait = authority
    run = Execution(bundle.runtime, "execution", _PRINCIPAL, tree)
    try:
        with pytest.raises(ObservationError if failure == "callback" else AIError) as raised:
            await run.wait(on_event=callback, timeout_seconds=0.01, close_timeout_seconds=0.2)
        if failure == "callback":
            assert raised.value.origin == "callback"
            assert raised.value.__cause__ is cause
        else:
            assert raised.value is cause
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["authority", "contract"])
async def test_execution_failure_outranks_simultaneous_callback_failure(winner) -> None:
    bundle = _Bundle()
    bundle.execution.release.clear()
    cause = AIError(ErrorCode.AUTHORIZATION_DENIED)
    if winner == "authority":
        bundle.execution.error = cause

    async def tree(*args, ready=None, **kwargs):
        if ready is not None:
            ready.set()
        try:
            yield _execution_event(1)
            await asyncio.Event().wait()
        finally:
            if winner == "contract":
                raise cause

    async def callback(event) -> None:
        bundle.execution.release.set()
        raise ValueError("callback failure")

    run = Execution(bundle.runtime, "execution", _PRINCIPAL, tree)
    try:
        with pytest.raises(AIError) as raised:
            await run.wait(on_event=callback)
        assert raised.value is cause
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
async def test_execution_watch_preparation_is_inside_wait_deadline() -> None:
    bundle = _Bundle()
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def tree(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
            yield
        finally:
            closed.set()

    run = Execution(bundle.runtime, "execution", _PRINCIPAL, tree)
    try:
        with pytest.raises(AIError) as raised:
            await run.wait(on_event=_ignore, timeout_seconds=0.01, close_timeout_seconds=0.2)
        assert raised.value.code is ErrorCode.WAIT_TIMEOUT
        assert raised.value.safe_details == {"scope": "execution", "resource_id": "execution", "cursor": None}
        assert entered.is_set() and closed.is_set()
        assert bundle.execution.wait_calls == 0
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["execution", "graph", "evaluation"])
@pytest.mark.parametrize("observe", [False, True])
async def test_runtime_close_reclaims_each_wait_owner_before_storage(owner, observe) -> None:
    from dataclasses import replace

    bundle = _Bundle()
    bundle.execution.release.clear()
    bundle.graph.release.clear()
    bundle.evaluations.result = replace(bundle.evaluations.result, completion="running")
    waiting = asyncio.create_task(bundle.wait(owner, on_event=_ignore if observe else None, close_timeout_seconds=0.2))
    entered = {
        "execution": bundle.execution.started,
        "graph": bundle.graph.started,
        "evaluation": bundle.evaluations.inspected,
    }[owner]
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await bundle.runtime.close()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert bundle.storage_closed
        if owner == "execution":
            assert bundle.streams_closed_at_storage_close[0] is observe
        elif owner == "graph":
            assert bundle.streams_closed_at_storage_close[1] is observe
    finally:
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await bundle.runtime.close()


@pytest.mark.asyncio
async def test_plain_wait_bounds_noncooperative_waiter_cleanup_without_opening_watch() -> None:
    bundle = _Bundle()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def wait(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return bundle.execution.result

    bundle.execution.wait = wait
    waiting = asyncio.create_task(bundle.execution_run.wait(timeout_seconds=0.01, close_timeout_seconds=0.01))
    try:
        done, _ = await asyncio.wait({waiting}, timeout=0.5)
        assert done, "the close budget must also bound plain authoritative waits"
        with pytest.raises(AIError) as raised:
            await waiting
        assert raised.value.code is ErrorCode.WAIT_TIMEOUT
        assert cancelled.is_set()
        assert bundle.tree.calls == 0
        with pytest.raises(ObservationError) as cleanup:
            await bundle.runtime.close()
        assert cleanup.value.safe_details["cleanup_pending"] is True
        assert not bundle.storage_closed
    finally:
        release.set()
        await asyncio.gather(waiting, return_exceptions=True)
        await bundle.runtime.close()
    assert bundle.storage_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["complete", "cancelled", "needs_attention"])
@pytest.mark.parametrize("observe", [False, True])
async def test_evaluation_wait_preserves_all_authoritative_stopping_phases(completion, observe) -> None:
    from dataclasses import replace

    bundle = _Bundle()
    bundle.evaluations.result = replace(bundle.evaluations.result, completion=completion)
    try:
        outcome = await bundle.evaluation_run.wait(on_event=_ignore if observe else None)
        assert type(outcome) is WaitResult
        assert outcome.result is bundle.evaluations.result
        assert outcome.result.completion == completion
    finally:
        await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("observe", [False, True])
async def test_execution_wait_keeps_task_bound_authority_instead_of_using_facade(observe) -> None:
    bundle = _Bundle()
    calls = []

    async def bound_wait(timeout_seconds):
        calls.append(timeout_seconds)
        return bundle.execution.result

    run = Execution(bundle.runtime, "execution", _PRINCIPAL, bundle.tree.stream, bound_wait)
    try:
        outcome = await run.wait(on_event=_ignore if observe else None, timeout_seconds=1)
        assert outcome.result is bundle.execution.result
        assert calls == [None]
        assert bundle.execution.wait_calls == 0
    finally:
        await bundle.runtime.close()


def test_unified_observation_surface_has_no_redundant_methods_or_result_state() -> None:
    from dataclasses import fields

    for owner in (Execution, TaskGraphRun, EvaluationRun):
        assert callable(owner.watch)
        assert callable(owner.wait)
        assert not hasattr(owner, "observe")
        assert not hasattr(owner, "wait_observed")
    assert [field.name for field in fields(WaitResult)] == ["result", "cursor", "observation_error"]
