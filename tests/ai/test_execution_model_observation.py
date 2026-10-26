#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Opt-in execution observations retain independent cursors and finite coverage."""
from ._runtime_test_helpers import model_features

from linktools.ai.persistence import Persistence
import asyncio
import time
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.execution import ExecutionObservationEvent, ExecutionTreeEvent, TaskModelProjection
from linktools.ai.core import _signed_cursor as runtime_cursor
from linktools.ai.observe.cursors import (
    decode_execution_observation_cursor,
    decode_execution_watch_cursor,
    encode_execution_observation_cursor,
)
from linktools.ai.execution import ModelInteractionItem, ModelInteractionReadBoundary, UsageReadCutoff

from .test_unified_wait_contract import _Bundle, _NAMESPACE, _PRINCIPAL


class _Subscription:
    def __init__(self, history):
        self.history = history
        self.generation = history.generation
        self.closed = False

    async def wait(self, generation):
        while self.history.generation == generation:
            await self.history.changed.wait()
            self.history.changed.clear()
        return self.history.generation

    async def close(self):
        self.closed = True


class _History:
    def __init__(self):
        self.rows = [ModelInteractionItem(
            "execution", 1, 0, 1, "response", 0, None, {}, {}, None,
            "SUCCEEDED", None, None, None, content_included=False,
        )]
        self.subscriptions = []
        self.generation = 0
        self.changed = asyncio.Event()
        self.available = True
        self.durable = True

    async def subscribe_model_interactions(self, execution_id, *, principal):
        assert principal == _PRINCIPAL
        subscription = _Subscription(self)
        self.subscriptions.append(subscription)
        return subscription

    async def capture_model_interaction_cutoffs(self, execution_id, *, principal):
        assert self.subscriptions
        cutoff = UsageReadCutoff(execution_id, 1, len(self.rows))
        return ModelInteractionReadBoundary(
            (cutoff,), (cutoff,) if self.durable else (), self.available, self.durable,
        )

    async def read_model_interaction_metadata(self, execution_id, *, principal,
            agent_run_seq, after_model_request_seq, through_model_request_seq, limit):
        return tuple(self.rows[after_model_request_seq:through_model_request_seq])[:limit]

    async def read_model_interaction_updates(self, execution_id, *, principal, after_cutoffs=(), limit=200):
        from linktools.ai.execution.ranges import model_interaction_update_ranges
        boundary = await self.capture_model_interaction_cutoffs(execution_id, principal=principal)
        rows = []
        for run, after, through in model_interaction_update_ranges(execution_id, boundary, after_cutoffs, limit, allow_unavailable=not self.durable):
            rows.extend(await self.read_model_interaction_metadata(
                execution_id, principal=principal, agent_run_seq=run,
                after_model_request_seq=after, through_model_request_seq=through, limit=through-after,
            ))
        return boundary, tuple(rows)

    def update(self, item):
        self.rows[item.model_request_seq - 1] = item
        self.generation += 1
        self.changed.set()


async def _bundle():
    bundle = await _Bundle.create()
    bundle.runtime._history = _History()
    binding = replace(bundle.runtime._execution_binding, history=bundle.runtime.history)
    bundle.runtime._execution_binding = binding
    bundle.execution_run = replace(bundle.execution_run, _binding=binding)
    return bundle


def _positions(cursor):
    return decode_execution_observation_cursor(
        _NAMESPACE, "tenant", "execution", cursor, include_content=False,
    )


@pytest.mark.asyncio
async def test_default_tree_mode_and_authority_only_wait_do_not_touch_metadata() -> None:
    bundle = await _bundle()
    try:
        bundle.tree.emit = True
        events = [event async for event in bundle.execution_run.watch()]
        assert all(isinstance(event, ExecutionTreeEvent) for event in events)
        result = await bundle.execution_run.wait(include_model_interactions=True)
        assert result.result is bundle.execution.result
        assert not bundle.runtime.history.subscriptions
        assert result.cursor is None
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_metadata_envelope_keeps_inner_tree_cursor_and_recompensates_on_reconnect() -> None:
    bundle = await _bundle()
    try:
        bundle.tree.emit = True
        events = [event async for event in bundle.execution_run.watch(include_model_interactions=True)]
        assert all(isinstance(event, ExecutionObservationEvent) for event in events)
        tree = next(event for event in events if isinstance(event.event, ExecutionTreeEvent))
        assert _positions(tree.cursor) == {"execution": 1}
        assert decode_execution_watch_cursor(
            _NAMESPACE, "tenant", "execution", tree.event.cursor, include_content=False,
        ) == {"execution": 1}
        for cursor, metadata in [(tree.cursor, False), (tree.event.cursor, True)]:
            with pytest.raises(AIError) as caught:
                bundle.execution_run.watch(cursor=cursor, include_model_interactions=metadata)
            assert caught.value.code is ErrorCode.CURSOR_INVALID
        metadata = [event for event in events if isinstance(event.event, TaskModelProjection)]
        assert _positions(metadata[0].cursor) == {}
        assert all(event.event.item.request == {} and event.event.item.response is None
                   and not event.event.item.content_included for event in metadata)
        bundle.tree.emit = False
        seen = []

        async def observe(event):
            seen.append(event)

        result = await bundle.execution_run.wait(
            include_model_interactions=True, cursor=events[-1].cursor, on_event=observe,
        )
        assert result.result is bundle.execution.result
        assert result.observation_error is None
        assert any(isinstance(event.event, TaskModelProjection) for event in seen)
        assert _positions(result.cursor) == {"execution": 1}
        assert all(subscription.closed for subscription in bundle.runtime.history.subscriptions)
        assert not bundle.runtime._observation_sessions
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_model_notification_rereads_active_identity_without_new_event_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [time.time()]
    monkeypatch.setattr(runtime_cursor, "time", SimpleNamespace(time=lambda: clock[0]))
    bundle = await _bundle()
    try:
        history = bundle.runtime.history
        initial = replace(history.rows[0], status="RUNNING", started_at=datetime.now(timezone.utc))
        history.rows[0] = initial
        history.durable = False
        bundle.tree.release.clear()
        stream = bundle.execution_run.watch(include_model_interactions=True)
        try:
            first = await anext(stream)
            clock[0] += 2
            history.update(replace(initial, status="SUCCEEDED"))
            second = await asyncio.wait_for(anext(stream), 1)
            assert first.event.item.status == "RUNNING"
            assert second.event.item.status == "SUCCEEDED"
            assert _positions(first.cursor) == _positions(second.cursor) == {}
        finally:
            await stream.aclose()
        assert all(subscription.closed for subscription in history.subscriptions)
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_runtime_close_owns_standalone_watch_while_consumer_is_paused() -> None:
    bundle = await _bundle()
    try:
        bundle.tree.release.clear()
        stream = bundle.execution_run.watch(include_model_interactions=True)
        await anext(stream)
        await bundle.close()
        assert bundle.storage_closed
        assert bundle.tree.closed.is_set()
        assert all(subscription.closed for subscription in bundle.runtime.history.subscriptions)
        await stream.aclose()
    finally:
        await bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["watch", "wait"])
async def test_missing_required_metadata_coverage_is_not_silent_success(mode) -> None:
    bundle = await _bundle()
    try:
        bundle.runtime.history.durable = False
        bundle.runtime.history.available = False
        if mode == "watch":
            with pytest.raises(ObservationError) as caught:
                _ = [event async for event in bundle.execution_run.watch(include_model_interactions=True)]
            error = caught.value
        else:
            async def observe(event):
                return None
            result = await bundle.execution_run.wait(include_model_interactions=True, on_event=observe)
            assert result.result is bundle.execution.result
            error = result.observation_error
        assert error.origin == "stream"
        assert error.safe_details["phase"] == "drain"
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_callback_failure_keeps_input_ack_and_original_cause() -> None:
    bundle = await _bundle()
    try:
        cursor = encode_execution_observation_cursor(
            _NAMESPACE, "tenant", "execution", include_content=False, event_seqs={},
        )
        cause = ValueError("host callback")

        async def observe(event):
            raise cause

        with pytest.raises(ObservationError) as caught:
            await bundle.execution_run.wait(include_model_interactions=True, cursor=cursor, on_event=observe)
        assert caught.value.origin == "callback"
        assert caught.value.cursor == cursor
        assert caught.value.__cause__ is cause
        assert all(subscription.closed for subscription in bundle.runtime.history.subscriptions)
    finally:
        await bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("include_model_interactions", 1), ("include_event_content", 1)])
async def test_invalid_flags_precede_authority_and_observation(field, value) -> None:
    bundle = await _bundle()
    try:
        with pytest.raises(AIError) as caught:
            await bundle.execution_run.wait(**{field: value})
        assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert bundle.execution.wait_calls == 0
        assert not bundle.runtime.history.subscriptions
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_observation_cursor_scope_is_checked_before_terminal_fast_path() -> None:
    bundle = await _bundle()
    try:
        async def observe(event):
            return None
        for namespace, tenant, root, content in [
            ("other", "tenant", "execution", False),
            (_NAMESPACE, "other", "execution", False),
            (_NAMESPACE, "tenant", "other", False),
            (_NAMESPACE, "tenant", "execution", True),
        ]:
            cursor = encode_execution_observation_cursor(
                namespace, tenant, root, include_content=content, event_seqs={},
            )
            with pytest.raises(AIError) as caught:
                await bundle.execution_run.wait(include_model_interactions=True, cursor=cursor, on_event=observe)
            assert caught.value.code is ErrorCode.CURSOR_INVALID
        assert bundle.execution.wait_calls == 0
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_facade_forwards_metadata_mode() -> None:
    bundle = await _bundle()
    try:
        seen = []
        async def observe(event):
            seen.append(event)
        await bundle.facade.wait("execution", principal=_PRINCIPAL, include_model_interactions=True, on_event=observe)
        assert seen and all(isinstance(event, ExecutionObservationEvent) for event in seen)
    finally:
        await bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "filesystem", "sqlite", "transient"])
@pytest.mark.parametrize("durable_trace", [False, True])
async def test_real_execution_metadata_live_and_final_handoff(backend, durable_trace, tmp_path) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.capability import TraceCapability
    from linktools.ai.runtime import Runtime
    from linktools.ai.persistence import PersistenceStorage, PersistencePlan
    from linktools.ai.storage import StorageRoute
    from .test_live_history_readback_integration import _Models
    from ._runtime_test_helpers import _UsageFunctionModel

    release = asyncio.Event()
    async def model(messages, info):
        await release.wait()
        return ModelResponse(parts=[TextPart("private response")])

    storage = {
        "memory": lambda: PersistenceStorage.in_memory(),
        "filesystem": lambda: PersistenceStorage.filesystem(tmp_path / "state"),
        "sqlite": lambda: PersistenceStorage.sqlite(tmp_path / "state.db"),
        "transient": lambda: PersistenceStorage(PersistencePlan(execution=StorageRoute.transient())),
    }[backend]()
    group = RuntimeFeatures[None]("execution-observation")
    if durable_trace:
        group.capability(TraceCapability(), scope="runtime")
    group.agent("default", model="default", allow_tools=())
    async with Persistence.open('execution-observation', storage=storage) as persistence_284, Runtime.attach(persistence=persistence_284, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        execution = await runtime.agents.get().start("private prompt")
        rows = []
        async def observe(event):
            if isinstance(event.event, TaskModelProjection):
                rows.append(event.event)
                if event.event.item.status == "RUNNING":
                    release.set()
        try:
            result = await execution.wait(
                include_model_interactions=True, include_event_content=True,
                on_event=observe, timeout_seconds=5,
            )
        finally:
            release.set()
        assert result.result.output == {"text": "private response"}
        assert rows[0].item.status == "RUNNING"
        if durable_trace and backend != "transient":
            assert result.observation_error is None
            assert rows[-1].item.status == "SUCCEEDED"
            assert rows[-1].visibility == "durable_history"
        else:
            assert result.observation_error is not None
            assert result.observation_error.origin == "stream"
            assert result.observation_error.safe_details == {
                "phase": "drain", "reason": "model_state_coverage_unavailable",
            }
            assert all(row.visibility == "local_staging" for row in rows)
            assert (await execution.wait()).result.output == {"text": "private response"}
        assert all(not row.item.content_included and not row.item.request
                   and row.item.response is None for row in rows)
        assert all(row.root_execution_id == execution.execution_id for row in rows)
        assert not runtime._observation_sessions


@pytest.mark.asyncio
async def test_standalone_runtime_close_propagates_owned_subscription_failure() -> None:
    bundle = await _bundle()
    bundle.tree.release.clear()
    stream = bundle.execution_run.watch(include_model_interactions=True)
    await anext(stream)
    failure = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    async def fail_close():
        raise failure
    bundle.runtime.history.subscriptions[0].close = fail_close
    with pytest.raises(AIError) as caught:
        await bundle.close()
    assert caught.value is failure
    assert not bundle.storage_closed
    with pytest.raises(AIError) as retry:
        await bundle.close()
    assert retry.value is failure
    with pytest.raises(AIError):
        await stream.aclose()


@pytest.mark.asyncio
async def test_metadata_read_failure_keeps_authoritative_error_channel() -> None:
    bundle = await _bundle()
    try:
        failure = AIError(ErrorCode.AUTHORIZATION_DENIED)
        async def fail_read(*args, **kwargs):
            raise failure
        bundle.runtime.history.read_model_interaction_metadata = fail_read
        async def observe(event):
            return None
        with pytest.raises(AIError) as caught:
            await bundle.execution_run.wait(include_model_interactions=True, on_event=observe)
        assert caught.value is failure
        assert all(subscription.closed for subscription in bundle.runtime.history.subscriptions)
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_standalone_watch_does_not_retain_previously_delivered_event_tasks() -> None:
    from .test_unified_wait_contract import _execution_event

    bundle = await _bundle()
    try:
        async def tree(*args, ready=None, **kwargs):
            if ready is not None:
                ready.set()
            for sequence in range(1, 501):
                yield _execution_event(sequence)
            await asyncio.Event().wait()

        run = replace(bundle.execution_run, _watch_tree=tree)
        stream = run.watch(include_model_interactions=True)
        for _ in range(501):
            await anext(stream)
        assert bundle.runtime._observation_sessions
        assert all(len(session.tasks) < 10 for session in bundle.runtime._observation_sessions)
        await stream.aclose()
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_final_capture_authorization_error_wins_over_subscription_cleanup_failure() -> None:
    bundle = await _bundle()
    original = bundle.runtime.history.capture_model_interaction_cutoffs
    calls = 0
    denied = AIError(ErrorCode.AUTHORIZATION_DENIED)
    async def close_failed():
        raise RuntimeError("subscription cleanup")
    async def capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            bundle.runtime.history.subscriptions[-1].close = close_failed
            raise denied
        return await original(*args, **kwargs)
    bundle.runtime.history.capture_model_interaction_cutoffs = capture
    async def observe(event):
        return None
    with pytest.raises(AIError) as caught:
        await bundle.execution_run.wait(include_model_interactions=True, on_event=observe)
    assert caught.value is denied


@pytest.mark.asyncio
async def test_standalone_final_drain_and_close_budgets_retain_noncooperative_capture(monkeypatch) -> None:
    from linktools.ai.observe import stream as _observation

    monkeypatch.setattr(_observation, "_FINAL_DRAIN_TIMEOUT", 0.01)
    bundle = await _bundle()
    try:
        register = bundle.runtime._register_observation

        def register_short_cleanup(session: _observation.ObservationSession) -> None:
            session.close_timeout = 0.1
            register(session)

        monkeypatch.setattr(bundle.runtime, "_register_observation", register_short_cleanup)
        original = bundle.runtime.history.capture_model_interaction_cutoffs
        release = asyncio.Event()
        entered = asyncio.Event()
        calls = 0
        async def capture(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                entered.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        pass
            return await original(*args, **kwargs)
        bundle.runtime.history.capture_model_interaction_cutoffs = capture
        stream = bundle.execution_run.watch(include_model_interactions=True)
        async def consume():
            return [event async for event in stream]
        pending = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 1)
        try:
            done, _ = await asyncio.wait({pending}, timeout=6)
            assert pending in done
            with pytest.raises(ObservationError) as caught:
                pending.result()
            assert caught.value.safe_details["phase"] == "cleanup"
            assert caught.value.safe_details["cleanup_pending"] is True
            assert bundle.runtime._observation_sessions
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await bundle.close()
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_standalone_drain_does_not_start_another_read_after_deadline(monkeypatch) -> None:
    from linktools.ai.observe import stream as _observation

    monkeypatch.setattr(_observation, "_FINAL_DRAIN_TIMEOUT", 0.01)
    bundle = await _bundle()
    try:
        bundle.runtime.history.rows.append(replace(bundle.runtime.history.rows[0], model_request_seq=2))
        stream = bundle.execution_run.watch(include_model_interactions=True)
        for _ in range(3):
            await anext(stream)
        await asyncio.sleep(0.03)
        with pytest.raises(ObservationError) as caught:
            await anext(stream)
        assert caught.value.safe_details["phase"] == "drain"
        assert not bundle.runtime._observation_sessions
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_execution_final_metadata_remains_available_without_live_notifications() -> None:
    bundle = await _bundle()
    try:
        stream = bundle.execution_run.watch(include_model_interactions=True)
        first = await anext(stream)
        assert isinstance(first.event, TaskModelProjection)

        async def unavailable(*args, **kwargs):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

        bundle.runtime.history.subscribe_model_interactions = unavailable
        try:
            remaining = [event async for event in stream]
        finally:
            await stream.aclose()
        metadata = [event.event for event in remaining if isinstance(event.event, TaskModelProjection)]
        assert metadata and metadata[-1].item == first.event.item
        assert metadata[-1].visibility == "durable_history"
        assert all(subscription.closed for subscription in bundle.runtime.history.subscriptions)
        assert not bundle.runtime._observation_sessions
    finally:
        await bundle.close()


@pytest.mark.parametrize("local,durable", [(True, False), (False, True), (False, False)])
def test_declared_model_coverage_cannot_lose_observed_cutoffs(local, durable) -> None:
    from linktools.ai.history._projection import model_interaction_update_ranges

    boundary = ModelInteractionReadBoundary((), (), local, durable)
    with pytest.raises(AIError) as caught:
        model_interaction_update_ranges(
            "execution", boundary, (UsageReadCutoff("execution", 1, 1),), 200,
        )
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("composed", [False, True])
async def test_missing_model_coverage_preserves_previously_observed_durable_facts(
    durable: bool, composed: bool,
) -> None:
    from linktools.ai.execution.projection import ModelProjection

    bundle = await _bundle()
    try:
        history = bundle.runtime.history
        history.durable = durable
        projection = ModelProjection(history, _PRINCIPAL)
        view, depth, _ = (await bundle.tree.capture("execution", principal=_PRINCIPAL))[0]
        await projection.add("node", view, depth)
        try:
            first = [row async for _, row in projection.refresh("execution")]
            assert len(first) == 1
            assert first[0].visibility == ("durable_history" if durable else "local_staging")
            unavailable = ModelInteractionReadBoundary((), (), False, False)

            async def capture(*args, **kwargs):
                return unavailable

            async def updates(*args, **kwargs):
                return unavailable, ()

            history.capture_model_interaction_cutoffs = capture
            if composed:
                history.read_model_interaction_updates = updates
            if durable:
                with pytest.raises(AIError) as caught:
                    _ = [row async for row in projection.refresh("execution")]
                assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            else:
                assert [row async for row in projection.refresh("execution")] == []
                assert projection.high_waters[("execution", 1)] == 1
        finally:
            await projection.close()
    finally:
        await bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("composed", [False, True])
async def test_model_cutoff_without_retained_rows_cannot_claim_local_only_coverage(composed: bool) -> None:
    from linktools.ai.execution.projection import ModelProjection

    bundle = await _bundle()
    try:
        projection = ModelProjection(bundle.runtime.history, _PRINCIPAL)
        view, depth, _ = (await bundle.tree.capture("execution", principal=_PRINCIPAL))[0]
        await projection.add("node", view, depth)
        projection.high_waters[("execution", 1)] = 1
        unavailable = ModelInteractionReadBoundary((), (), False, False)

        async def capture(*args, **kwargs):
            return unavailable

        async def updates(*args, **kwargs):
            return unavailable, ()

        bundle.runtime.history.capture_model_interaction_cutoffs = capture
        if composed:
            bundle.runtime.history.read_model_interaction_updates = updates
        try:
            with pytest.raises(AIError) as caught:
                _ = [row async for row in projection.refresh("execution")]
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        finally:
            await projection.close()
    finally:
        await bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("durable_trace,durable_route,separate_owner", [
    (True, True, False), (False, True, False), (True, False, False), (False, True, True),
])
async def test_reopened_model_update_cutoff_uses_admitted_and_routed_durability(
    tmp_path, durable_trace: bool, durable_route: bool, separate_owner: bool,
) -> None:
    from linktools.ai.history._history import History
    from .test_model_capture_reads import _NAMESPACE as namespace, _publish, _reader, _service
    from .test_incremental_projection_boundaries import _storage
    from .test_step_projection import _execution

    state = _storage("memory", tmp_path)
    state = Persistence(state)
    await state.initialize(namespace=namespace, tenant_id="tenant")
    other = None
    try:
        await state.execution.executions.create_with_history_head(replace(_execution(), durable_trace=durable_trace))
        if separate_owner:
            other = _storage("memory", tmp_path)
            other = Persistence(other)
            await other.initialize(namespace=namespace, tenant_id="tenant")
            await other.execution.executions.create_with_history_head(_execution())
        reader = _reader(state if other is None else other)
        reader._durable_history_available = durable_route
        history = History(_service(state, reader=reader), tenant_id="tenant")
        after_cutoffs = (UsageReadCutoff("execution", 1, 1),)
        if durable_trace and durable_route:
            from linktools.ai.persistence import PersistenceDomain

            run = await _publish(state)
            before, _ = await history.read_model_interaction_updates("execution", principal=_PRINCIPAL)
            assert before.cutoffs == after_cutoffs and before.durable_history_available
            await state.run_store.read_store(PersistenceDomain.TRACE).release_agent_run(
                run.agent_run_id, execution_id="execution",
            )
            history = History(_service(state, reader=_reader(state)), tenant_id="tenant")
        request = history.read_model_interaction_updates(
            "execution", principal=_PRINCIPAL, after_cutoffs=after_cutoffs,
        )
        if separate_owner or durable_trace and durable_route:
            with pytest.raises(AIError) as caught:
                await request
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            boundary, rows = await request
            assert boundary == ModelInteractionReadBoundary((), (), False, False)
            assert rows == ()
    finally:
        if other is not None:
            await other.close()
        await state.close()


@pytest.mark.asyncio
async def test_transient_task_model_observation_keeps_canonical_success_after_staging_release() -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.core import TaskStatus
    from linktools.ai.inputs import AgentTaskInput
    from linktools.ai.runtime import Runtime
    from linktools.ai.persistence import PersistenceStorage, PersistencePlan
    from linktools.ai.storage import StorageRoute
    from linktools.ai.task import Task, TaskGraph, TaskNode
    from linktools.ai.task_engine import TaskEngine
    from ._runtime_test_helpers import _UsageFunctionModel
    from .test_live_history_readback_integration import _Models

    release = asyncio.Event()

    async def model(messages, info):
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    group = RuntimeFeatures[None]("transient-task-observation")
    group.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage(PersistencePlan(execution=StorageRoute.transient()))
    async with Persistence.open('transient-task-observation', storage=storage) as persistence_644, Runtime.attach(persistence=persistence_644, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        task = Task.from_agent("answer", runtime.agents.get())
        async with TaskEngine.open(task, persistence=persistence_644) as engine:
            graph = await engine.start(TaskGraph(
                "answer", (TaskNode("answer", task=task, input=AgentTaskInput("question")),),
            ), idempotency_key="once")
            rows = []

            async def observe(event):
                if isinstance(event.event, TaskModelProjection):
                    rows.append(event.event)
                    if event.event.item.status == "RUNNING":
                        release.set()

            try:
                result = await graph.wait(on_event=observe, timeout_seconds=5)
            finally:
                release.set()
            assert rows and all(row.visibility == "local_staging" for row in rows)
            assert result.result.status is TaskStatus.SUCCEEDED
            assert result.observation_error is not None
            assert result.observation_error.origin == "stream"
            assert (await graph.wait()).result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_read", ["tree_capture", "metadata_read"])
@pytest.mark.parametrize("outcome", ["complete", "close", "error"])
async def test_live_events_do_not_wait_for_metadata_reconciliation(
    blocked_read: str, outcome: str,
) -> None:
    from .test_unified_wait_contract import _execution_event

    bundle = await _bundle()
    try:
        history = bundle.runtime.history
        initial = replace(history.rows[0], status="RUNNING", started_at=datetime.now(timezone.utc))
        history.rows[0] = initial
        history.durable = False
        events = asyncio.Queue()
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        failure = AIError(ErrorCode.AUTHORIZATION_DENIED)
        captured_positions = []

        async def tree(*args, ready=None, **kwargs):
            if ready is not None:
                ready.set()
            while True:
                yield await events.get()

        run = replace(bundle.execution_run, _watch_tree=tree)
        stream = run.watch(include_model_interactions=True)
        pending = None
        try:
            first = await anext(stream)
            assert isinstance(first.event, TaskModelProjection)
            if blocked_read == "tree_capture":
                original = bundle.tree.capture

                async def capture(*args, **kwargs):
                    positions = kwargs["after_event_seqs"]
                    entered.set()
                    try:
                        await release.wait()
                        captured_positions.append(dict(positions))
                        if outcome == "error":
                            raise failure
                        return await original(*args, **kwargs)
                    finally:
                        finished.set()

                bundle.tree.capture = capture
            else:
                original = history.read_model_interaction_metadata

                async def read(*args, **kwargs):
                    entered.set()
                    try:
                        await release.wait()
                        if outcome == "error":
                            raise failure
                        return await original(*args, **kwargs)
                    finally:
                        finished.set()

                history.read_model_interaction_metadata = read
            history.update(replace(initial, status="SUCCEEDED"))
            pending = asyncio.create_task(anext(stream))
            if blocked_read == "tree_capture":
                # A notification may be delivered before the periodic tree capture.
                update = await asyncio.wait_for(pending, 1)
                assert update.event.item.status == "SUCCEEDED"
                pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(entered.wait(), 2)
            events.put_nowait(_execution_event(1))
            delivered = await asyncio.wait_for(pending, 0.5)
            pending = None
            assert isinstance(delivered.event, ExecutionTreeEvent)
            assert _positions(delivered.cursor) == {"execution": 1}
            assert not release.is_set() and not finished.is_set()
            if outcome == "close":
                await stream.aclose()
                assert finished.is_set()
            else:
                release.set()
                pending = asyncio.create_task(anext(stream))
                if outcome == "error":
                    with pytest.raises(AIError) as caught:
                        await asyncio.wait_for(pending, 1)
                    assert caught.value is failure
                elif blocked_read == "metadata_read":
                    update = await asyncio.wait_for(pending, 1)
                    assert update.event.item.status == "SUCCEEDED"
                    assert _positions(update.cursor) == {"execution": 1}
                else:
                    await asyncio.wait_for(finished.wait(), 1)
                    assert captured_positions == [{}]
                    events.put_nowait(_execution_event(2))
                    delivered = await asyncio.wait_for(pending, 1)
                    assert _positions(delivered.cursor) == {"execution": 2}
                pending = None
        finally:
            release.set()
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
        assert all(subscription.closed for subscription in history.subscriptions)
        assert not bundle.runtime._observation_sessions
    finally:
        await bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_tree_completion_drains_metadata_reader_before_final_coverage(cleanup_error: bool) -> None:
    from .test_unified_wait_contract import _execution_event

    bundle = await _bundle()
    try:
        history = bundle.runtime.history
        initial = replace(history.rows[0], status="RUNNING", started_at=datetime.now(timezone.utc))
        history.rows[0] = initial
        events = asyncio.Queue()
        entered, cancelled = asyncio.Event(), asyncio.Event()
        failure = AIError(ErrorCode.AUTHORIZATION_DENIED)

        async def tree(*args, ready=None, **kwargs):
            if ready is not None:
                ready.set()
            while True:
                event = await events.get()
                if event is None:
                    return
                yield event

        run = replace(bundle.execution_run, _watch_tree=tree)
        stream = run.watch(include_model_interactions=True)
        pending = None
        try:
            await anext(stream)
            original = history.read_model_interaction_metadata

            async def read(*args, **kwargs):
                if not entered.is_set():
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
                        if cleanup_error:
                            raise failure
                assert cancelled.is_set()
                return await original(*args, **kwargs)

            history.read_model_interaction_metadata = read
            history.update(replace(initial, status="SUCCEEDED"))
            pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(entered.wait(), 1)
            events.put_nowait(_execution_event(1))
            delivered = await asyncio.wait_for(pending, 1)
            pending = None
            assert _positions(delivered.cursor) == {"execution": 1}
            assert not cancelled.is_set()
            events.put_nowait(None)
            if cleanup_error:
                with pytest.raises(AIError) as caught:
                    await asyncio.wait_for(anext(stream), 1)
                assert caught.value is failure
            else:
                final = await asyncio.wait_for(anext(stream), 1)
                assert isinstance(final.event, TaskModelProjection)
                assert final.event.item.status == "SUCCEEDED"
                assert final.event.visibility == "durable_history"
                assert _positions(final.cursor) == {"execution": 1}
                with pytest.raises(StopAsyncIteration):
                    await asyncio.wait_for(anext(stream), 1)
            assert cancelled.is_set()
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
        assert all(subscription.closed for subscription in history.subscriptions)
        assert not bundle.runtime._observation_sessions
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_new_child_text_triggers_metadata_discovery_and_final_compensation() -> None:
    from linktools.ai.core import ExecutionLineageKind
    from .test_unified_wait_contract import _execution_event

    bundle = await _bundle()
    try:
        history = bundle.runtime.history
        original = history.read_model_interaction_metadata
        events = asyncio.Queue()

        async def read(execution_id, **kwargs):
            rows = await original(execution_id, **kwargs)
            return tuple(replace(row, execution_id=execution_id) for row in rows)

        history.read_model_interaction_metadata = read

        async def tree(*args, ready=None, **kwargs):
            if ready is not None:
                ready.set()
            while True:
                event = await events.get()
                if event is None:
                    return
                yield event

        run = replace(bundle.execution_run, _watch_tree=tree)
        stream = run.watch(include_model_interactions=True)
        try:
            assert (await anext(stream)).event.item.execution_id == "execution"
            root = _execution_event(1)
            child = replace(
                root, execution_id="child", lineage_kind=ExecutionLineageKind.SUBAGENT,
                parent_execution_id="execution", parent_invocation_id="invocation", depth=1,
                event=replace(root.event, execution_id="child"),
            )
            events.put_nowait(child)
            text = await asyncio.wait_for(anext(stream), 1)
            assert isinstance(text.event, ExecutionTreeEvent)
            assert text.event.execution_id == "child"
            assert _positions(text.cursor) == {"child": 1}
            metadata = await asyncio.wait_for(anext(stream), 1)
            assert isinstance(metadata.event, TaskModelProjection)
            assert metadata.event.item.execution_id == "child"
            assert metadata.event.item.depth == 1
            assert metadata.event.parent_execution_id == "execution"
            assert _positions(metadata.cursor) == {"child": 1}
            events.put_nowait(None)
            remaining = [value async for value in stream]
            assert {value.event.item.execution_id for value in remaining} == {"execution", "child"}
            assert all(value.event.visibility == "durable_history" for value in remaining)
            assert all(_positions(value.cursor) == {"child": 1} for value in remaining)
        finally:
            await stream.aclose()
        assert all(subscription.closed for subscription in history.subscriptions)
        assert not bundle.runtime._observation_sessions
    finally:
        await bundle.close()
