#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from linktools.ai.agent import AgentBindingSnapshot
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


def _binding(agent_id: str = "agent") -> AgentBindingSnapshot:
    spec = AgentSpec(agent_id)
    return AgentBindingSnapshot(
        version=1,
        agent_spec=spec,
        base_model={},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={},
    )


def _record(*, subagent: bool, parent_invocation_id: str | None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="child" if subagent else "root",
        tenant_id="tenant",
        session_id=None,
        parent_execution_id="root" if subagent else None,
        root_execution_id="root",
        source_execution_id=None,
        base_execution_id=None,
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
                ExecutionEventType.EXECUTION_SUCCEEDED,
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
        ExecutionEventType.EXECUTION_SUCCEEDED,
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
                        ExecutionEventType.EXECUTION_STARTED,
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
                        ExecutionEventType.EXECUTION_SUCCEEDED,
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

    assert event.event.event_type is ExecutionEventType.EXECUTION_SUCCEEDED
    assert events.closed is False

    await stream.aclose()
    assert events.closed is True


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
                    ExecutionEventType.EXECUTION_SUCCEEDED,
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
            "a" * 64,
            "child",
            RetryExecutionRequest("retry", principal, "retry-key"),
        )
    assert retry_error.value.code is ErrorCode.REQUEST_FIELD_INVALID

    with pytest.raises(AIError) as fork_error:
        await service.fork(
            "a" * 64,
            "child",
            ForkExecutionRequest("fork", principal, "fork-key"),
        )
    assert fork_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
