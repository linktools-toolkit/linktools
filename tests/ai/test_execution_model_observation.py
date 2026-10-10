#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in execution observations retain independent cursors and finite coverage."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.runtime import ExecutionObservationEvent, ExecutionTreeEvent, TaskModelProjection
from linktools.ai.runtime._watch_cursor import (
    decode_execution_observation_cursor, decode_execution_watch_cursor,
    encode_execution_observation_cursor,
)
from linktools.ai.runtime.service_api import ModelInteractionItem, ModelInteractionReadBoundary, UsageReadCutoff

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

    def update(self, item):
        self.rows[item.model_request_seq - 1] = item
        self.generation += 1
        self.changed.set()


def _bundle():
    bundle = _Bundle()
    bundle.runtime.history = _History()
    return bundle


def _positions(cursor):
    return decode_execution_observation_cursor(
        _NAMESPACE, "tenant", "execution", cursor, include_content=False,
    )


@pytest.mark.asyncio
async def test_default_tree_mode_and_authority_only_wait_do_not_touch_metadata() -> None:
    bundle = _bundle()
    bundle.tree.emit = True
    events = [event async for event in bundle.execution_run.watch()]
    assert all(isinstance(event, ExecutionTreeEvent) for event in events)
    result = await bundle.execution_run.wait(include_model_interactions=True)
    assert result.result is bundle.execution.result
    assert not bundle.runtime.history.subscriptions
    assert result.cursor is None


@pytest.mark.asyncio
async def test_metadata_envelope_keeps_inner_tree_cursor_and_recompensates_on_reconnect() -> None:
    bundle = _bundle()
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


@pytest.mark.asyncio
async def test_model_notification_rereads_active_identity_without_new_event_position() -> None:
    bundle = _bundle()
    history = bundle.runtime.history
    initial = replace(history.rows[0], status="RUNNING", started_at=datetime.now(timezone.utc))
    history.rows[0] = initial
    history.durable = False
    bundle.tree.release.clear()
    stream = bundle.execution_run.watch(include_model_interactions=True)
    first = await anext(stream)
    history.update(replace(initial, status="SUCCEEDED"))
    second = await asyncio.wait_for(anext(stream), 1)
    assert first.event.item.status == "RUNNING"
    assert second.event.item.status == "SUCCEEDED"
    assert first.cursor == second.cursor
    await stream.aclose()
    assert all(subscription.closed for subscription in history.subscriptions)


@pytest.mark.asyncio
async def test_runtime_close_owns_standalone_watch_while_consumer_is_paused() -> None:
    bundle = _bundle()
    bundle.tree.release.clear()
    stream = bundle.execution_run.watch(include_model_interactions=True)
    await anext(stream)
    await bundle.runtime.close()
    assert bundle.storage_closed
    assert bundle.tree.closed.is_set()
    assert all(subscription.closed for subscription in bundle.runtime.history.subscriptions)
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["watch", "wait"])
async def test_missing_required_metadata_coverage_is_not_silent_success(mode) -> None:
    bundle = _bundle()
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


@pytest.mark.asyncio
async def test_callback_failure_keeps_input_ack_and_original_cause() -> None:
    bundle = _bundle()
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


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("include_model_interactions", 1), ("include_event_content", 1)])
async def test_invalid_flags_precede_authority_and_observation(field, value) -> None:
    bundle = _bundle()
    with pytest.raises(AIError) as caught:
        await bundle.execution_run.wait(**{field: value})
    assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert bundle.execution.wait_calls == 0
    assert not bundle.runtime.history.subscriptions


@pytest.mark.asyncio
async def test_observation_cursor_scope_is_checked_before_terminal_fast_path() -> None:
    bundle = _bundle()
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


@pytest.mark.asyncio
async def test_facade_forwards_metadata_mode() -> None:
    bundle = _bundle()
    seen = []
    async def observe(event):
        seen.append(event)
    await bundle.facade.wait("execution", principal=_PRINCIPAL, include_model_interactions=True, on_event=observe)
    assert seen and all(isinstance(event, ExecutionObservationEvent) for event in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "filesystem", "sqlite"])
async def test_real_execution_metadata_live_and_final_handoff(backend, tmp_path) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.capability import CapabilityGroup
    from linktools.ai.runtime import Runtime, RuntimeStorage
    from .test_live_history_readback_integration import _Models
    from ._runtime_test_helpers import _UsageFunctionModel

    release = asyncio.Event()
    async def model(messages, info):
        await release.wait()
        return ModelResponse(parts=[TextPart("private response")])

    storage = {
        "memory": lambda: RuntimeStorage.in_memory(),
        "filesystem": lambda: RuntimeStorage.filesystem(tmp_path / "state"),
        "sqlite": lambda: RuntimeStorage.sqlite(tmp_path / "state.db"),
    }[backend]()
    group = CapabilityGroup[None]("execution-observation")
    group.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        "execution-observation", models=_Models(_UsageFunctionModel(model)),
        storage=storage, capabilities=(group,),
    ) as runtime:
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
        assert result.observation_error is None
        assert rows[0].item.status == "RUNNING"
        assert rows[-1].item.status == "SUCCEEDED"
        assert rows[-1].visibility == "durable_history"
        assert all(not row.item.content_included and not row.item.request
                   and row.item.response is None for row in rows)
        assert all(row.root_execution_id == execution.execution_id for row in rows)
        assert not runtime._observation_sessions


@pytest.mark.asyncio
async def test_standalone_runtime_close_propagates_owned_subscription_failure() -> None:
    bundle = _bundle()
    bundle.tree.release.clear()
    stream = bundle.execution_run.watch(include_model_interactions=True)
    await anext(stream)
    failure = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    async def fail_close():
        raise failure
    bundle.runtime.history.subscriptions[0].close = fail_close
    with pytest.raises(AIError) as caught:
        await bundle.runtime.close()
    assert caught.value is failure
    assert not bundle.storage_closed
    with pytest.raises(AIError) as retry:
        await bundle.runtime.close()
    assert retry.value is failure
    with pytest.raises(AIError):
        await stream.aclose()


@pytest.mark.asyncio
async def test_metadata_read_failure_keeps_authoritative_error_channel() -> None:
    bundle = _bundle()
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


@pytest.mark.asyncio
async def test_standalone_watch_does_not_retain_previously_delivered_event_tasks() -> None:
    from .test_unified_wait_contract import _execution_event

    bundle = _bundle()
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


@pytest.mark.asyncio
async def test_final_capture_authorization_error_wins_over_subscription_cleanup_failure() -> None:
    bundle = _bundle()
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
    from linktools.ai.runtime import _observation

    monkeypatch.setattr(_observation, "_FINAL_DRAIN_TIMEOUT", 0.01)
    bundle = _bundle()
    register = bundle.runtime._register_observation

    def register_short_cleanup(session: _observation._ObservationSession) -> None:
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
        await bundle.runtime.close()


@pytest.mark.asyncio
async def test_standalone_drain_does_not_start_another_read_after_deadline(monkeypatch) -> None:
    from linktools.ai.runtime import _observation

    monkeypatch.setattr(_observation, "_FINAL_DRAIN_TIMEOUT", 0.01)
    bundle = _bundle()
    bundle.runtime.history.rows.append(replace(bundle.runtime.history.rows[0], model_request_seq=2))
    stream = bundle.execution_run.watch(include_model_interactions=True)
    for _ in range(3):
        await anext(stream)
    await asyncio.sleep(0.03)
    with pytest.raises(ObservationError) as caught:
        await anext(stream)
    assert caught.value.safe_details["phase"] == "drain"
    assert not bundle.runtime._observation_sessions
