#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from linktools.ai.agent import AgentBindingContract
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._execution_tree import (
    ExecutionTreeBroker,
    ExecutionTreeStreamer,
)
from linktools.ai.runtime.service_api import (
    ExecutionStreamEvent,
    ExecutionTreeEvent,
    ExecutionView,
    ForkExecutionRequest,
    RetryExecutionRequest,
)
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.spec import AgentSpec
from ._runtime_test_helpers import execution_owner_fields


def _binding(agent_id: str = "agent") -> AgentBindingContract:
    spec = AgentSpec(agent_id)
    return AgentBindingContract(
        agent_spec=spec,
        model_contract={},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={},
    )


def _record(*, subagent: bool, parent_invocation_id: str | None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="child" if subagent else "root",
        session_id=None,
        parent_execution_id="root" if subagent else None,
        root_execution_id="root",
        previous_execution_id=None,
        fork_base_execution_id=None,
        lineage_kind=(
            ExecutionLineageKind.SUBAGENT
            if subagent
            else ExecutionLineageKind.RUN
        ),
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
        parent_invocation_id=parent_invocation_id,
        **execution_owner_fields(),
    )


def test_subagent_lineage_requires_parent_invocation() -> None:
    with pytest.raises(ValueError):
        _record(subagent=True, parent_invocation_id=None)
    child = _record(subagent=True, parent_invocation_id="tool-call")
    assert child.parent_invocation_id == "tool-call"
    with pytest.raises(ValueError):
        _record(subagent=False, parent_invocation_id="tool-call")


class _ExecutionReader:
    def __init__(self) -> None:
        self.root = ExecutionView(
            "root",
            "root-agent",
            ExecutionStatus.STARTED,
            ExecutionLineageKind.RUN,
            None,
            "root",
            None,
        )
        self.child = ExecutionView(
            "child",
            "child-agent",
            ExecutionStatus.SUCCEEDED,
            ExecutionLineageKind.SUBAGENT,
            "root",
            "root",
            "delegate-call",
        )

    async def inspect(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView:
        del principal
        return self.root if execution_id == "root" else self.child

    async def list_children(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionView, ...]:
        del principal
        if execution_id != "root":
            raise AssertionError("execution tree must only list direct children")
        return (self.child,)


class _RootOnlyExecutionReader(_ExecutionReader):
    async def list_children(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionView, ...]:
        del principal
        if execution_id != "root":
            raise AssertionError("execution tree must only list direct children")
        return ()


class _EventStreamer:
    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ):
        del principal

        async def events():
            sequence = after_sequence + 1
            yield ExecutionStreamEvent(
                execution_id,
                sequence,
                ExecutionEventType.EXECUTION_SUCCEEDED.value,
                {},
            )

        return events()


@pytest.mark.asyncio
async def test_tree_stream_projects_root_and_child_without_global_sequence() -> None:
    broker = ExecutionTreeBroker()
    streamer = ExecutionTreeStreamer(
        _ExecutionReader(),
        _EventStreamer(),
        broker,
    )
    values = [
        item
        async for item in streamer.stream(
            "root",
            principal=Principal("owner", "tenant", "service"),
            after_sequences={"child": 4},
        )
    ]
    assert {(item.execution_id, item.depth) for item in values} == {
        ("root", 0),
        ("child", 1),
    }
    child = next(
        item for item in values if item.execution_id == "child"
    )
    assert child.parent_invocation_id == "delegate-call"
    assert child.event.durable_sequence == 5


def test_execution_tree_event_rejects_nested_depth() -> None:
    event = ExecutionStreamEvent(
        "child",
        1,
        ExecutionEventType.EXECUTION_SUCCEEDED.value,
        {},
    )
    with pytest.raises(ValueError):
        ExecutionTreeEvent(
            "child",
            "child-agent",
            ExecutionLineageKind.SUBAGENT,
            "root",
            "root",
            "delegate-call",
            2,
            event,
        )


@pytest.mark.asyncio
async def test_tree_stream_validates_cursors_before_starting_event_sources() -> None:
    class RecordingEventStreamer(_EventStreamer):
        def __init__(self) -> None:
            self.started: list[str] = []

        def stream(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
        ):
            self.started.append(execution_id)
            return super().stream(
                execution_id,
                principal=principal,
                after_sequence=after_sequence,
            )

    events = RecordingEventStreamer()
    streamer = ExecutionTreeStreamer(
        _ExecutionReader(),
        events,
        ExecutionTreeBroker(),
    )
    stream = streamer.stream(
        "root",
        principal=Principal("owner", "tenant", "service"),
        after_sequences={"unknown": 1},
    )

    with pytest.raises(AIError) as error:
        await anext(stream)

    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert events.started == []


@pytest.mark.asyncio
async def test_tree_stream_keeps_at_most_one_prefetched_event_per_execution() -> None:
    class BurstEventStreamer:
        def __init__(self) -> None:
            self.produced: dict[str, int] = {}

        def stream(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
        ):
            del principal

            async def events():
                for offset in range(1, 101):
                    self.produced[execution_id] = (
                        self.produced.get(execution_id, 0) + 1
                    )
                    yield ExecutionStreamEvent(
                        execution_id,
                        after_sequence + offset,
                        ExecutionEventType.EXECUTION_STARTED.value,
                        {},
                    )

            return events()

    events = BurstEventStreamer()
    streamer = ExecutionTreeStreamer(
        _ExecutionReader(),
        events,
        ExecutionTreeBroker(),
    )
    stream = streamer.stream(
        "root",
        principal=Principal("owner", "tenant", "service"),
    )

    await anext(stream)

    assert events.produced
    assert max(events.produced.values()) == 1
    await stream.aclose()


@pytest.mark.asyncio
async def test_tree_stream_does_not_close_terminal_source_before_delivery() -> None:
    class TerminalAwareEventStreamer:
        def __init__(self) -> None:
            self.closed = False

        def stream(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
        ):
            del principal

            async def events():
                try:
                    yield ExecutionStreamEvent(
                        execution_id,
                        after_sequence + 1,
                        ExecutionEventType.EXECUTION_SUCCEEDED.value,
                        {},
                    )
                finally:
                    self.closed = True

            return events()

    events = TerminalAwareEventStreamer()
    streamer = ExecutionTreeStreamer(
        _RootOnlyExecutionReader(),
        events,
        ExecutionTreeBroker(),
    )
    stream = streamer.stream(
        "root",
        principal=Principal("owner", "tenant", "service"),
    )

    event = await anext(stream)

    assert event.event.event_type == ExecutionEventType.EXECUTION_SUCCEEDED.value
    assert events.closed is False

    await stream.aclose()
    assert events.closed is True


@pytest.mark.asyncio
async def test_tree_stream_discovers_persisted_child_without_local_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "linktools.ai.runtime._execution_tree._DISCOVERY_BACKOFF_INITIAL",
        0.01,
    )
    monkeypatch.setattr(
        "linktools.ai.runtime._execution_tree._DISCOVERY_BACKOFF_MAX",
        0.01,
    )

    class PersistedExecutionReader(_RootOnlyExecutionReader):
        def __init__(self) -> None:
            super().__init__()
            self.child_visible = False

        async def list_children(
            self,
            execution_id: str,
            *,
            principal: Principal,
        ) -> tuple[ExecutionView, ...]:
            del principal
            if execution_id != "root":
                raise AssertionError("execution tree must only list direct children")
            return (self.child,) if self.child_visible else ()

    class PersistedEventStreamer:
        def __init__(self) -> None:
            self.release_root = asyncio.Event()

        def stream(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
        ):
            del principal

            async def events():
                if execution_id == "root":
                    await self.release_root.wait()
                yield ExecutionStreamEvent(
                    execution_id,
                    after_sequence + 1,
                    ExecutionEventType.EXECUTION_SUCCEEDED.value,
                    {},
                )

            return events()

    reader = PersistedExecutionReader()
    events = PersistedEventStreamer()
    streamer = ExecutionTreeStreamer(
        reader,
        events,
        ExecutionTreeBroker(),
    )
    stream = streamer.stream(
        "root",
        principal=Principal("owner", "tenant", "service"),
    )
    first = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    reader.child_visible = True

    child = await asyncio.wait_for(first, timeout=1)
    assert child.execution_id == "child"
    assert child.depth == 1

    events.release_root.set()
    remaining = [item async for item in stream]
    assert [item.execution_id for item in remaining] == ["root"]


@pytest.mark.asyncio
async def test_tree_stream_adds_dynamic_direct_child_once() -> None:
    class DynamicExecutionReader(_RootOnlyExecutionReader):
        async def list_children(
            self,
            execution_id: str,
            *,
            principal: Principal,
        ) -> tuple[ExecutionView, ...]:
            del principal
            if execution_id != "root":
                raise AssertionError("execution tree must only list direct children")
            return ()

    class DynamicEventStreamer:
        def __init__(self) -> None:
            self.release_root = asyncio.Event()

        def stream(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
        ):
            del principal

            async def events():
                if execution_id == "root":
                    await self.release_root.wait()
                yield ExecutionStreamEvent(
                    execution_id,
                    after_sequence + 1,
                    ExecutionEventType.EXECUTION_SUCCEEDED.value,
                    {},
                )

            return events()

    broker = ExecutionTreeBroker()
    events = DynamicEventStreamer()
    streamer = ExecutionTreeStreamer(
        DynamicExecutionReader(),
        events,
        broker,
    )
    stream = streamer.stream(
        "root",
        principal=Principal("owner", "tenant", "service"),
    )
    first = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    broker.publish("root", "child")
    broker.publish("root", "child")
    child = await first

    assert child.execution_id == "child"
    assert child.depth == 1

    events.release_root.set()
    remaining = [item async for item in stream]
    assert [item.execution_id for item in remaining] == ["root"]


def test_non_subagent_lineage_rejects_parent_execution() -> None:
    with pytest.raises(ValueError):
        replace(
            _record(subagent=False, parent_invocation_id=None),
            parent_execution_id="parent",
        )


class _RetryExecutionReader:
    def __init__(self) -> None:
        self.root = ExecutionView(
            "retry",
            "root-agent",
            ExecutionStatus.STARTED,
            ExecutionLineageKind.RETRY,
            None,
            "original-root",
            None,
        )
        self.child = ExecutionView(
            "retry-child",
            "child-agent",
            ExecutionStatus.SUCCEEDED,
            ExecutionLineageKind.SUBAGENT,
            "retry",
            "original-root",
            "retry-delegate",
        )

    async def inspect(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView:
        del principal
        return self.root if execution_id == "retry" else self.child

    async def list_children(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionView, ...]:
        del principal
        if execution_id != "retry":
            raise AssertionError("execution tree must only list direct children")
        return (self.child,)


@pytest.mark.asyncio
async def test_tree_stream_accepts_retry_lineage_root() -> None:
    streamer = ExecutionTreeStreamer(
        _RetryExecutionReader(),
        _EventStreamer(),
        ExecutionTreeBroker(),
    )
    values = [
        item
        async for item in streamer.stream(
            "retry",
            principal=Principal("owner", "tenant", "service"),
        )
    ]
    assert {(item.execution_id, item.root_execution_id) for item in values} == {
        ("retry", "original-root"),
        ("retry-child", "original-root"),
    }


@pytest.mark.asyncio
async def test_tree_broker_keys_notifications_by_direct_parent() -> None:
    broker = ExecutionTreeBroker()
    parent = broker.subscribe("retry")
    historical_root = broker.subscribe("original-root")
    broker.publish("retry", "child")
    assert parent.drain() == ("child",)
    assert historical_root.drain() == ()
    await parent.close()
    await historical_root.close()


@pytest.mark.asyncio
async def test_subagent_execution_cannot_be_retried_or_forked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(DefaultExecutionService)
    child = _record(subagent=True, parent_invocation_id="delegate-call")
    monkeypatch.setattr(
        service,
        "_load_authorized",
        AsyncMock(return_value=child),
    )
    principal = Principal("owner", "tenant", "service")

    with pytest.raises(AIError) as retry_error:
        await service.retry(
            "child",
            RetryExecutionRequest("retry", principal, "retry-key"),
        )
    assert retry_error.value.code is ErrorCode.REQUEST_FIELD_INVALID

    with pytest.raises(AIError) as fork_error:
        await service.fork(
            "child",
            ForkExecutionRequest("fork", principal, "fork-key"),
        )
    assert fork_error.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["root", "child", "event_identity", "model_payload"])
async def test_tree_stream_protocol_failures_remain_authoritative(invalid: str) -> None:
    reader = _ExecutionReader()
    if invalid == "root":
        reader.root = replace(reader.root, execution_id="other-root")
    elif invalid == "child":
        reader.child = replace(reader.child, parent_execution_id="other-root")

    class InvalidEvents(_EventStreamer):
        def stream(self, execution_id: str, **kwargs):
            if invalid not in {"event_identity", "model_payload"}:
                return super().stream(execution_id, **kwargs)

            async def values():
                yield ExecutionStreamEvent(
                    "other" if invalid == "event_identity" else execution_id,
                    1,
                    ExecutionEventType.MODEL_REQUEST_STARTED.value,
                    [] if invalid == "model_payload" else {},
                )
            return values()

    streamer = ExecutionTreeStreamer(reader, InvalidEvents(), ExecutionTreeBroker())
    with pytest.raises(ValueError if invalid == "event_identity" else AIError) as raised:
        await anext(streamer.stream("root", principal=Principal("owner", "tenant")))

    if invalid != "event_identity":
        assert raised.value.code is (
            ErrorCode.REQUEST_FIELD_INVALID if invalid == "root" else ErrorCode.STORAGE_INTEGRITY_ERROR
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["subscribe", "wait", "drain", "close"])
@pytest.mark.parametrize("typed", [False, True])
async def test_tree_broker_wraps_only_untyped_optional_failures(
    operation: str,
    typed: bool,
) -> None:
    from linktools.ai.runtime.service_api import _ExecutionStreamFailure

    cause = AIError(ErrorCode.STORAGE_UNAVAILABLE) if typed else OSError("broker failed")

    class Subscription:
        async def wait(self):
            if operation == "wait":
                raise cause
            await asyncio.Event().wait()

        def drain(self):
            if operation == "drain":
                raise cause
            return ()

        async def close(self):
            if operation == "close":
                raise cause

    class Broker:
        def subscribe(self, _execution_id):
            if operation == "subscribe":
                raise cause
            return Subscription()

    streamer = ExecutionTreeStreamer(_RootOnlyExecutionReader(), _EventStreamer(), Broker())
    with pytest.raises(AIError if typed else _ExecutionStreamFailure) as raised:
        async for _event in streamer.stream("root", principal=Principal("owner", "tenant")):
            pass

    assert (raised.value if typed else raised.value.cause) is cause


@pytest.mark.asyncio
async def test_tree_stream_durable_error_wins_over_same_round_broker_failure() -> None:
    cause = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    class BrokenEvents:
        def stream(self, *args, **kwargs):
            async def values():
                raise cause
                yield
            return values()

    class Subscription:
        async def wait(self):
            raise OSError("broker failed")

        async def close(self):
            pass

    class Broker:
        def subscribe(self, _execution_id):
            return Subscription()

    streamer = ExecutionTreeStreamer(_RootOnlyExecutionReader(), BrokenEvents(), Broker())
    with pytest.raises(AIError) as raised:
        await anext(streamer.stream("root", principal=Principal("owner", "tenant")))

    assert raised.value is cause


@pytest.mark.asyncio
async def test_tree_stream_preserves_authoritative_failure_during_cleanup() -> None:
    cause = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    started = asyncio.Event()

    class BrokenEvents:
        def stream(self, *args, **kwargs):
            async def values():
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise cause
                yield
            return values()

    class Subscription:
        async def wait(self):
            await started.wait()
            raise OSError("broker failed")

        async def close(self):
            pass

    class Broker:
        def subscribe(self, _execution_id):
            return Subscription()

    streamer = ExecutionTreeStreamer(_RootOnlyExecutionReader(), BrokenEvents(), Broker())
    with pytest.raises(AIError) as raised:
        await anext(streamer.stream("root", principal=Principal("owner", "tenant")))

    assert raised.value is cause


@pytest.mark.asyncio
async def test_tree_close_keeps_child_live_cleanup_authoritative_failure():
    from linktools.ai.core import ExecutionDeltaType
    from linktools.ai.runtime._event import DefaultEventService, ExecutionDelta
    cause = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    class Subscription:
        def __init__(self, key):
            self.key = key
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self.key == "root":
                return ExecutionDelta("root", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "x")
            await asyncio.Event().wait()
        async def close(self):
            if self.key == "child":
                raise cause
    class Live:
        def claim_local_producer(self, key):
            return Subscription(key)
        def is_local_producer(self, key):
            return True
        def base_sequence(self, key):
            return 0
    class Events(DefaultEventService):
        async def _authorize_read(self, *args):
            pass
    tree = ExecutionTreeStreamer(_ExecutionReader(), Events(None, None, None, None, Live()),
                                 ExecutionTreeBroker())
    stream = tree.stream("root", principal=Principal("owner", "tenant"))
    await anext(stream)
    with pytest.raises(AIError) as raised:
        await stream.aclose()
    assert raised.value is cause
