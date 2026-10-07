#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derived live event projection over one execution and all its descendants."""

import asyncio
import sys
from collections import deque
from collections.abc import AsyncIterator, Mapping
from typing import Protocol

from ..core import ExecutionEventType, ExecutionLineageKind, Principal
from ..errors import AIError, ErrorCode
from ._task_observation import _is_observation_cleanup, _cancel_stream_task, _await_stream_cleanup, _report_observation_error
from .service_api import (
    ExecutionStreamEvent,
    ExecutionTreeEvent,
    ExecutionView,
    _ExecutionStreamFailure,
)

_DISCOVERY_BACKOFF_INITIAL = 1.0
_DISCOVERY_BACKOFF_MAX = 30.0


class _ExecutionTreeReader(Protocol):
    async def inspect(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView: ...

    async def list_children(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionView, ...]: ...


class _ExecutionEventStreamer(Protocol):
    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]: ...


class _ExecutionTreeSubscription:
    def __init__(
        self,
        owner: "ExecutionTreeBroker",
        parent_execution_id: str,
    ) -> None:
        self._owner = owner
        self._parent_execution_id = parent_execution_id
        self._pending: set[str] = set()
        self._event = asyncio.Event()
        self._closed = False

    def publish(self, child_execution_id: str) -> None:
        if self._closed:
            return
        self._pending.add(child_execution_id)
        self._event.set()

    def drain(self) -> tuple[str, ...]:
        values = tuple(sorted(self._pending))
        self._pending.clear()
        self._event.clear()
        return values

    async def wait(self) -> tuple[str, ...]:
        while not self._pending and not self._closed:
            self._event.clear()
            await self._event.wait()
        return self.drain()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._owner._remove(self._parent_execution_id, self)
        self._event.set()


class ExecutionTreeBroker:
    """Notify active tree readers about newly established direct children."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, set[_ExecutionTreeSubscription]] = {}

    def subscribe(self, parent_execution_id: str) -> _ExecutionTreeSubscription:
        subscription = _ExecutionTreeSubscription(self, parent_execution_id)
        self._subscriptions.setdefault(parent_execution_id, set()).add(subscription)
        return subscription

    def publish(self, parent_execution_id: str, child_execution_id: str) -> None:
        for subscription in tuple(
            self._subscriptions.get(parent_execution_id, ())
        ):
            subscription.publish(child_execution_id)

    def _remove(
        self,
        parent_execution_id: str,
        subscription: _ExecutionTreeSubscription,
    ) -> None:
        values = self._subscriptions.get(parent_execution_id)
        if values is None:
            return
        values.discard(subscription)
        if not values:
            self._subscriptions.pop(parent_execution_id, None)


class ExecutionTreeStreamer:
    """Merge an execution subtree with relative depths and per-stream ordering."""

    def __init__(
        self,
        executions: _ExecutionTreeReader,
        events: _ExecutionEventStreamer,
        broker: ExecutionTreeBroker,
    ) -> None:
        self._executions = executions
        self._events = events
        self._broker = broker

    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences: Mapping[str, int] | None = None,
        include_content: bool = False,
        ready: asyncio.Event | None = None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return self._stream(
            execution_id,
            principal=principal,
            after_sequences=_normalize_after_sequences(after_sequences),
            include_content=include_content,
            ready=ready,
        )

    async def _stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences: Mapping[str, int],
        include_content: bool,
        ready: asyncio.Event | None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        root = await self._executions.inspect(execution_id, principal=principal)
        _validate_root(root, execution_id)

        views: dict[str, ExecutionView] = {}
        depths: dict[str, int] = {}
        subscriptions: dict[str, _ExecutionTreeSubscription] = {}
        child_waits: dict[str, asyncio.Task[tuple[str, ...]]] = {}
        streams: dict[str, AsyncIterator[ExecutionStreamEvent]] = {}
        pending: dict[str, asyncio.Task[ExecutionStreamEvent]] = {}
        discovery_wait: asyncio.Task[None] | None = None
        observing = False

        def start(view: ExecutionView) -> None:
            stream = self._events.stream(
                view.execution_id,
                principal=principal,
                after_sequence=after_sequences.get(view.execution_id, 0),
            )
            streams[view.execution_id] = stream
            pending[view.execution_id] = _next_event_task(stream, view.execution_id)

        def wait_for_children(parent_id: str) -> None:
            child_waits[parent_id] = asyncio.create_task(
                subscriptions[parent_id].wait(),
                name=f"execution-tree-children-{parent_id}",
            )

        def add_view(view: ExecutionView, depth: int) -> bool:
            existing = views.get(view.execution_id)
            if existing is not None:
                if (
                    existing.parent_execution_id != view.parent_execution_id
                    or existing.root_execution_id != view.root_execution_id
                    or existing.lineage_kind != view.lineage_kind
                    or existing.parent_invocation_id != view.parent_invocation_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return False
            try:
                subscription = self._broker.subscribe(view.execution_id)
            except AIError:
                raise
            except Exception as error:
                raise _ExecutionStreamFailure(error) from error
            # Subscribe before reading children, including at every new depth.
            subscriptions[view.execution_id] = subscription
            views[view.execution_id] = view
            depths[view.execution_id] = depth
            if observing:
                start(view)
                wait_for_children(view.execution_id)
            return True

        def add_child(parent_id: str, child: ExecutionView) -> bool:
            _validate_child(views[parent_id], child)
            return add_view(child, depths[parent_id] + 1)

        async def discover_persisted_children(parent_ids: tuple[str, ...]) -> bool:
            added = False
            remaining = deque(parent_ids)
            scanned: set[str] = set()
            while remaining:
                parent_id = remaining.popleft()
                if parent_id in scanned:
                    continue
                scanned.add(parent_id)
                for child in await self._executions.list_children(
                    parent_id, principal=principal,
                ):
                    if add_child(parent_id, child):
                        added = True
                        remaining.append(child.execution_id)
            return added

        async def discover_child(parent_id: str, child_id: str) -> None:
            if child_id in views:
                _validate_child(views[parent_id], views[child_id])
                return
            child = await self._executions.inspect(child_id, principal=principal)
            if child.execution_id != child_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if add_child(parent_id, child):
                await discover_persisted_children((child_id,))

        async def validate_cursor_member(member_id: str) -> None:
            # A durable cursor may name a member whose index is not visible yet.
            # Read back its parent chain instead of rejecting an authorized resume.
            chain: list[ExecutionView] = []
            seen: set[str] = set()
            current_id = member_id
            while current_id not in views:
                if current_id in seen:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                seen.add(current_id)
                view = await self._executions.inspect(current_id, principal=principal)
                if view.execution_id != current_id:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                if (
                    view.lineage_kind is not ExecutionLineageKind.SUBAGENT
                    or not view.parent_execution_id
                    or view.root_execution_id != root.root_execution_id
                ):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                chain.append(view)
                current_id = view.parent_execution_id
            for view in reversed(chain):
                add_child(current_id, view)
                current_id = view.execution_id
            if chain:
                await discover_persisted_children(tuple(view.execution_id for view in chain))

        def broker_result(task: asyncio.Task[tuple[str, ...]]) -> tuple[str, ...]:
            try:
                return task.result()
            except AIError:
                raise
            except Exception as error:
                raise _ExecutionStreamFailure(error) from error

        try:
            add_view(root, 0)
            await discover_persisted_children((execution_id,))
            for member_id in after_sequences:
                await validate_cursor_member(member_id)

            for view in tuple(views.values()):
                start(view)
                wait_for_children(view.execution_id)
            observing = True
            if ready is not None:
                ready.set()

            discovery_backoff = _DISCOVERY_BACKOFF_INITIAL
            discovery_wait = asyncio.create_task(
                asyncio.sleep(discovery_backoff),
                name=f"execution-tree-discovery-{execution_id}",
            )
            final_discovery = False
            while True:
                if not pending:
                    if final_discovery:
                        return
                    # This is a finite observation phase, not permanent subtree
                    # closure. Later admissions belong to the next watch call.
                    final_discovery = True
                    for parent_id, subscription in tuple(subscriptions.items()):
                        task = child_waits.get(parent_id)
                        child_ids = broker_result(task) if task is not None and task.done() else ()
                        try:
                            child_ids += subscription.drain()
                        except AIError:
                            raise
                        except Exception as error:
                            raise _ExecutionStreamFailure(error) from error
                        for child_id in child_ids:
                            await discover_child(parent_id, child_id)
                    await discover_persisted_children(tuple(views))
                    if not pending:
                        return

                wait_tasks = tuple(pending.values())
                if not final_discovery:
                    wait_tasks += tuple(child_waits.values())
                    if discovery_wait is not None:
                        wait_tasks += (discovery_wait,)
                done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

                # Contract failures and cancellation outrank optional transport
                # failures from any parent notification in the same round.
                for task in done:
                    if task.cancelled():
                        task.result()
                broker_tasks = set(child_waits.values())
                for task in (*pending.values(), *child_waits.values()):
                    if task not in done:
                        continue
                    error = task.exception()
                    if error is not None and not isinstance(
                        error, (StopAsyncIteration, _ExecutionStreamFailure)
                    ) and (task not in broker_tasks or isinstance(error, AIError)):
                        raise error
                if not final_discovery:
                    for parent_id, task in tuple(child_waits.items()):
                        if task in done:
                            for child_id in broker_result(task):
                                await discover_child(parent_id, child_id)
                            wait_for_children(parent_id)

                    if discovery_wait in done:
                        added = await discover_persisted_children(tuple(views))
                        discovery_backoff = (
                            _DISCOVERY_BACKOFF_INITIAL if added
                            else min(_DISCOVERY_BACKOFF_MAX, discovery_backoff * 2)
                        )
                        discovery_wait = asyncio.create_task(
                            asyncio.sleep(discovery_backoff),
                            name=f"execution-tree-discovery-{execution_id}",
                        )

                completed = sorted(
                    (key, task) for key, task in pending.items() if task in done
                )
                for key, task in completed:
                    pending.pop(key, None)
                    try:
                        event = task.result()
                    except StopAsyncIteration:
                        streams.pop(key, None)
                        continue
                    view = views[key]
                    yield ExecutionTreeEvent(
                        view.execution_id,
                        view.agent_id,
                        view.lineage_kind,
                        view.parent_execution_id,
                        view.root_execution_id,
                        view.parent_invocation_id,
                        depths[key],
                        _project_stream_event(event, include_content),
                    )
                    pending[key] = _next_event_task(streams[key], key)
        finally:
            active_error = sys.exc_info()[1]

            async def cleanup() -> None:
                errors: list[BaseException] = []
                wait_tasks = tuple(child_waits.values()) + tuple(pending.values())
                if discovery_wait is not None:
                    wait_tasks += (discovery_wait,)
                for task in wait_tasks:
                    if task.cancelled():
                        try:
                            task.result()
                        except asyncio.CancelledError as error:
                            _report_observation_error(error)
                            errors.append(error)
                    elif not task.done():
                        _cancel_stream_task(task)
                pending_tasks = set(wait_tasks)
                while pending_tasks:
                    done, pending_tasks = await asyncio.wait(
                        pending_tasks, return_when=asyncio.FIRST_COMPLETED,
                    )
                    completed_errors = _completed_errors(tuple(done), set(child_waits.values()))
                    for error in completed_errors:
                        _report_observation_error(error)
                    errors.extend(completed_errors)

                async def close_stream(stream: AsyncIterator[ExecutionStreamEvent]) -> None:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        try:
                            await close()
                        except BaseException as error:
                            _report_observation_error(error)
                            errors.append(error)

                async def close_subscription(subscription: _ExecutionTreeSubscription) -> None:
                    try:
                        await subscription.close()
                    except AIError as error:
                        _report_observation_error(error)
                        errors.append(error)
                    except Exception as error:
                        errors.append(_ExecutionStreamFailure(error))

                await asyncio.gather(
                    *(close_stream(stream) for stream in tuple(streams.values())),
                    *(close_subscription(subscription) for subscription in tuple(subscriptions.values())),
                )
                failure = _cleanup_failure(active_error, errors)
                if failure is not None:
                    raise failure

            await _await_stream_cleanup(cleanup(), active_error)


def _completed_errors(
    tasks: tuple[asyncio.Task, ...],
    broker_tasks: set[asyncio.Task],
) -> list[BaseException]:
    errors: list[BaseException] = []
    for task in tasks:
        if task.cancelled():
            continue
        error = task.exception()
        if error is None or isinstance(error, StopAsyncIteration):
            continue
        if task in broker_tasks and isinstance(error, Exception) and not isinstance(
            error, (AIError, _ExecutionStreamFailure)
        ):
            error = _ExecutionStreamFailure(error)
        errors.append(error)
    return errors


def _cleanup_failure(
    active_error: BaseException | None,
    errors: list[BaseException],
) -> BaseException | None:
    if (active_error is not None
            and not isinstance(active_error, (_ExecutionStreamFailure, GeneratorExit))
            and not _is_observation_cleanup(active_error)):
        return None
    return next(
        (error for error in errors if isinstance(error, asyncio.CancelledError)),
        next(
            (error for error in errors if not isinstance(error, _ExecutionStreamFailure)),
            errors[0] if errors and (active_error is None or isinstance(active_error, GeneratorExit)
                                     or _is_observation_cleanup(active_error)) else None,
        ),
    )


def _next_event_task(
    stream: AsyncIterator[ExecutionStreamEvent],
    execution_id: str,
) -> asyncio.Task[ExecutionStreamEvent]:
    return asyncio.create_task(
        stream.__anext__(),
        name=f"execution-tree-stream-{execution_id}",
    )


def _validate_root(view: ExecutionView, execution_id: str) -> None:
    if view.execution_id != execution_id:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if view.lineage_kind is ExecutionLineageKind.SUBAGENT:
        if (
            not view.parent_execution_id
            or view.parent_execution_id == execution_id
            or not view.parent_invocation_id
            or view.root_execution_id == execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif view.parent_execution_id is not None or view.parent_invocation_id is not None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _validate_child(root: ExecutionView, child: ExecutionView) -> None:
    if (
        child.execution_id == root.execution_id
        or child.lineage_kind is not ExecutionLineageKind.SUBAGENT
        or child.parent_execution_id != root.execution_id
        or child.root_execution_id != root.root_execution_id
        or not child.parent_invocation_id
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _project_stream_event(
    event: ExecutionStreamEvent,
    include_content: bool,
) -> ExecutionStreamEvent:
    if include_content:
        return event
    if event.event_type in {
        ExecutionEventType.MODEL_REQUEST_STARTED.value,
        ExecutionEventType.MODEL_REQUEST_FINISHED.value,
    }:
        if not isinstance(event.payload, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return ExecutionStreamEvent(
            event.execution_id,
            event.durable_sequence,
            event.event_type,
            dict(event.payload),
        )
    return ExecutionStreamEvent(
        event.execution_id,
        event.durable_sequence,
        event.event_type,
        {},
    )


def _normalize_after_sequences(
    value: Mapping[str, int] | None,
) -> Mapping[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result: dict[str, int] = {}
    for execution_id, sequence in value.items():
        if (
            not isinstance(execution_id, str)
            or not execution_id
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        result[execution_id] = sequence
    return result


__all__ = ["ExecutionTreeBroker", "ExecutionTreeStreamer"]
