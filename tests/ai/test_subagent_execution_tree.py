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
        event_seq=0,
        agent_run_seq=0,
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
        return (self.child,) if execution_id == "root" else ()


class _RootOnlyExecutionReader(_ExecutionReader):
    async def list_children(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionView, ...]:
        del principal
        return ()


class _EventStreamer:
    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
    ):
        del principal

        async def events():
            sequence = after_event_seq + 1
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
            after_event_seqs={"child": 4},
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
    assert child.event.durable_seq == 5


@pytest.mark.parametrize("depth", [-1, True, 1.5])
def test_execution_tree_event_rejects_invalid_relative_depth(depth) -> None:
    event = ExecutionStreamEvent(
        "child", 1, ExecutionEventType.EXECUTION_SUCCEEDED.value, {},
    )
    with pytest.raises(ValueError):
        ExecutionTreeEvent(
            "child", "child-agent", ExecutionLineageKind.SUBAGENT,
            "root", "root", "delegate-call", depth, event,
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
            after_event_seq: int = 0,
        ):
            self.started.append(execution_id)
            return super().stream(
                execution_id,
                principal=principal,
                after_event_seq=after_event_seq,
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
        after_event_seqs={"unknown": 1},
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
            after_event_seq: int = 0,
        ):
            del principal

            async def events():
                for offset in range(1, 101):
                    self.produced[execution_id] = (
                        self.produced.get(execution_id, 0) + 1
                    )
                    yield ExecutionStreamEvent(
                        execution_id,
                        after_event_seq + offset,
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
            after_event_seq: int = 0,
        ):
            del principal

            async def events():
                try:
                    yield ExecutionStreamEvent(
                        execution_id,
                        after_event_seq + 1,
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
            return (self.child,) if execution_id == "root" and self.child_visible else ()

    class PersistedEventStreamer:
        def __init__(self) -> None:
            self.release_root = asyncio.Event()

        def stream(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_event_seq: int = 0,
        ):
            del principal

            async def events():
                if execution_id == "root":
                    await self.release_root.wait()
                yield ExecutionStreamEvent(
                    execution_id,
                    after_event_seq + 1,
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
            return ()

    class DynamicEventStreamer:
        def __init__(self) -> None:
            self.release_root = asyncio.Event()

        def stream(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_event_seq: int = 0,
        ):
            del principal

            async def events():
                if execution_id == "root":
                    await self.release_root.wait()
                yield ExecutionStreamEvent(
                    execution_id,
                    after_event_seq + 1,
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
        return (self.child,) if execution_id == "retry" else ()


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
        def base_event_seq(self, key):
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


class _RecursiveExecutionReader:
    def __init__(self) -> None:
        self.views = {
            "root": ExecutionView(
                "root", "root-agent", ExecutionStatus.SUCCEEDED,
                ExecutionLineageKind.RUN, None, "root", None,
            ),
        }
        parent = "root"
        for key in ("child", "grandchild", "leaf"):
            self.views[key] = ExecutionView(
                key, f"{key}-agent", ExecutionStatus.SUCCEEDED,
                ExecutionLineageKind.SUBAGENT, parent, "root", f"{key}-invocation",
            )
            parent = key
        self.visible = set(self.views)

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        return self.views[execution_id]

    async def list_children(
        self, execution_id: str, *, principal: Principal,
    ) -> tuple[ExecutionView, ...]:
        return tuple(
            view for key, view in self.views.items()
            if key in self.visible and view.parent_execution_id == execution_id
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("root_id", ["root", "child"])
async def test_tree_stream_recurses_with_relative_depth_and_real_lineage(root_id: str) -> None:
    reader = _RecursiveExecutionReader()
    ready = asyncio.Event()
    stream = ExecutionTreeStreamer(reader, _EventStreamer(), ExecutionTreeBroker()).stream(
        root_id, principal=Principal("owner", "tenant"),
        after_event_seqs={"leaf": 8}, ready=ready,
    )
    values = [item async for item in stream]
    offset = 0 if root_id == "root" else 1
    assert {item.execution_id: item.depth for item in values} == {
        key: depth - offset
        for depth, key in enumerate(("root", "child", "grandchild", "leaf"))
        if depth >= offset
    }
    assert ready.is_set()
    for item in values:
        view = reader.views[item.execution_id]
        assert item.parent_execution_id == view.parent_execution_id
        assert item.root_execution_id == "root"
        assert item.lineage_kind is view.lineage_kind
    assert next(item for item in values if item.execution_id == "leaf").event.durable_seq == 9


@pytest.mark.asyncio
async def test_tree_prepare_validates_cursor_ancestor_chain_before_ready() -> None:
    class Reader(_RecursiveExecutionReader):
        async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
            if execution_id == "grandchild":
                assert not ready.is_set()
                raise denied
            return await super().inspect(execution_id, principal=principal)

    ready = asyncio.Event()
    denied = AIError(ErrorCode.AUTHORIZATION_DENIED)
    reader = Reader()
    reader.visible = {"root"}
    stream = ExecutionTreeStreamer(reader, _EventStreamer(), ExecutionTreeBroker()).stream(
        "root", principal=Principal("owner", "tenant"),
        after_event_seqs={"leaf": 1}, ready=ready,
    )
    with pytest.raises(AIError) as raised:
        await anext(stream)
    assert raised.value is denied
    assert not ready.is_set()


@pytest.mark.asyncio
async def test_tree_cursor_resumes_descendants_missing_from_child_index() -> None:
    reader = _RecursiveExecutionReader()
    reader.visible = {"root"}
    values = [
        item async for item in ExecutionTreeStreamer(
            reader, _EventStreamer(), ExecutionTreeBroker(),
        ).stream(
            "root", principal=Principal("owner", "tenant"), after_event_seqs={"leaf": 4},
        )
    ]
    assert {item.execution_id for item in values} == set(reader.views)
    assert next(item for item in values if item.execution_id == "leaf").event.durable_seq == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["outside", "cycle", "parent", "root"])
async def test_tree_prepare_rejects_invalid_deep_membership(invalid: str) -> None:
    reader = _RecursiveExecutionReader()
    ready = asyncio.Event()
    if invalid == "outside":
        reader.visible = {"root"}
        reader.views["leaf"] = replace(reader.views["leaf"], parent_execution_id="foreign")
        reader.views["foreign"] = replace(reader.views["root"], execution_id="foreign")
    elif invalid == "cycle":
        reader.visible = {"root"}
        reader.views["grandchild"] = replace(reader.views["grandchild"], parent_execution_id="leaf")
    elif invalid == "parent":
        reader.views["leaf"] = replace(reader.views["leaf"], parent_invocation_id=None)
    else:
        reader.views["leaf"] = replace(reader.views["leaf"], root_execution_id="foreign")
    stream = ExecutionTreeStreamer(reader, _EventStreamer(), ExecutionTreeBroker()).stream(
        "root", principal=Principal("owner", "tenant"),
        after_event_seqs={"leaf": 2}, ready=ready,
    )
    with pytest.raises(AIError) as raised:
        await anext(stream)
    assert raised.value.code is (
        ErrorCode.REQUEST_FIELD_INVALID if invalid == "outside"
        else ErrorCode.STORAGE_INTEGRITY_ERROR
    )
    assert not ready.is_set()


@pytest.mark.asyncio
async def test_tree_subscribes_each_parent_before_listing_its_children() -> None:
    broker = ExecutionTreeBroker()

    class Reader(_RecursiveExecutionReader):
        async def list_children(
            self, execution_id: str, *, principal: Principal,
        ) -> tuple[ExecutionView, ...]:
            if execution_id == "child":
                broker.publish("child", "grandchild")
                return ()
            return await super().list_children(execution_id, principal=principal)

    values = [
        item async for item in ExecutionTreeStreamer(Reader(), _EventStreamer(), broker).stream(
            "root", principal=Principal("owner", "tenant"),
        )
    ]
    assert sorted(item.execution_id for item in values) == ["child", "grandchild", "leaf", "root"]
    assert not broker._subscriptions


@pytest.mark.asyncio
async def test_tree_discovers_cross_process_descendants_below_existing_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("linktools.ai.runtime._execution_tree._DISCOVERY_BACKOFF_INITIAL", 0.01)
    monkeypatch.setattr("linktools.ai.runtime._execution_tree._DISCOVERY_BACKOFF_MAX", 0.01)
    reader = _RecursiveExecutionReader()
    reader.visible = {"root", "child"}
    release = asyncio.Event()
    ready = asyncio.Event()

    class Events(_EventStreamer):
        def stream(self, execution_id: str, **kwargs):
            async def values():
                if execution_id in {"root", "child"}:
                    await release.wait()
                async for event in super(Events, self).stream(execution_id, **kwargs):
                    yield event
            return values()

    stream = ExecutionTreeStreamer(reader, Events(), ExecutionTreeBroker()).stream(
        "root", principal=Principal("owner", "tenant"), ready=ready,
    )
    first = asyncio.create_task(anext(stream))
    await asyncio.wait_for(ready.wait(), 1)
    reader.visible.update(("grandchild", "leaf"))
    first_value = await asyncio.wait_for(first, 1)
    release.set()
    values = [first_value, *[item async for item in stream]]
    assert {item.execution_id: item.depth for item in values} == {
        "root": 0, "child": 1, "grandchild": 2, "leaf": 3,
    }


@pytest.mark.asyncio
async def test_tree_final_discovery_drains_once_and_leaves_later_admissions() -> None:
    class Reader(_RecursiveExecutionReader):
        def __init__(self) -> None:
            super().__init__()
            self.visible = {"root"}
            self.final_scan = False

        async def list_children(
            self, execution_id: str, *, principal: Principal,
        ) -> tuple[ExecutionView, ...]:
            values = await super().list_children(execution_id, principal=principal)
            if execution_id == "root" and self.final_scan:
                # This child is admitted after the final root membership read.
                self.visible.add("child")
            return values

    reader = Reader()

    class Events(_EventStreamer):
        def stream(self, execution_id: str, **kwargs):
            async def values():
                async for event in super(Events, self).stream(execution_id, **kwargs):
                    yield event
                reader.final_scan = True
            return values()

    streamer = ExecutionTreeStreamer(reader, Events(), ExecutionTreeBroker())
    first = [item async for item in streamer.stream("root", principal=Principal("owner", "tenant"))]
    assert [item.execution_id for item in first] == ["root"]
    second = [item async for item in streamer.stream("root", principal=Principal("owner", "tenant"))]
    assert {item.execution_id for item in second} == {"root", "child"}


@pytest.mark.asyncio
async def test_tree_final_discovery_drains_new_recursive_members_once() -> None:
    class Reader(_RecursiveExecutionReader):
        def __init__(self) -> None:
            super().__init__()
            self.visible = {"root", "child"}
            self.scans: dict[str, int] = {}

        async def list_children(
            self, execution_id: str, *, principal: Principal,
        ) -> tuple[ExecutionView, ...]:
            self.scans[execution_id] = self.scans.get(execution_id, 0) + 1
            return await super().list_children(execution_id, principal=principal)

    reader = Reader()

    class Events(_EventStreamer):
        def stream(self, execution_id: str, **kwargs):
            async def values():
                async for event in super(Events, self).stream(execution_id, **kwargs):
                    yield event
                if execution_id == "child":
                    reader.visible.update(("grandchild", "leaf"))
            return values()

    broker = ExecutionTreeBroker()
    values = [
        item async for item in ExecutionTreeStreamer(reader, Events(), broker).stream(
            "root", principal=Principal("owner", "tenant"),
        )
    ]
    assert {item.execution_id: item.depth for item in values} == {
        "root": 0, "child": 1, "grandchild": 2, "leaf": 3,
    }
    assert reader.scans == {"root": 2, "child": 2, "grandchild": 1, "leaf": 1}
    assert not broker._subscriptions


@pytest.mark.asyncio
async def test_tree_broker_discovers_descendants_after_parent_stream_ends() -> None:
    reader = _RecursiveExecutionReader()
    reader.visible = {"root", "child"}
    release_root = asyncio.Event()

    class Events(_EventStreamer):
        def stream(self, execution_id: str, **kwargs):
            async def values():
                if execution_id == "root":
                    await release_root.wait()
                async for event in super(Events, self).stream(execution_id, **kwargs):
                    yield event
            return values()

    broker = ExecutionTreeBroker()
    stream = ExecutionTreeStreamer(reader, Events(), broker).stream(
        "root", principal=Principal("owner", "tenant"),
    )
    try:
        values = [await asyncio.wait_for(anext(stream), 1)]
        assert values[0].execution_id == "child"
        next_item = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        reader.visible.update(("grandchild", "leaf"))
        broker.publish("child", "grandchild")
        broker.publish("child", "grandchild")
        values.append(await asyncio.wait_for(next_item, 1))
        values.append(await asyncio.wait_for(anext(stream), 1))
        release_root.set()
        values.extend([item async for item in stream])
    finally:
        release_root.set()
        await stream.aclose()
    assert {item.execution_id: item.depth for item in values} == {
        "root": 0, "child": 1, "grandchild": 2, "leaf": 3,
    }
    assert len(values) == 4
    assert not broker._subscriptions
