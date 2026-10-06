#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derived live event projection over one root execution and its direct children."""

import asyncio
import sys
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
    """Merge one root stream with its direct subagent streams on demand."""

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
    ) -> AsyncIterator[ExecutionTreeEvent]:
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return self._stream(
            execution_id,
            principal=principal,
            after_sequences=_normalize_after_sequences(after_sequences),
            include_content=include_content,
        )

    async def _stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences: Mapping[str, int],
        include_content: bool,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        root = await self._executions.inspect(
            execution_id,
            principal=principal,
        )
        _validate_root(root, execution_id)

        try:
            subscription = self._broker.subscribe(execution_id)
        except AIError:
            raise
        except Exception as error:
            raise _ExecutionStreamFailure(error) from error
        views: dict[str, ExecutionView] = {execution_id: root}
        streams: dict[str, AsyncIterator[ExecutionStreamEvent]] = {}
        pending: dict[str, asyncio.Task[ExecutionStreamEvent]] = {}

        def add_child(child: ExecutionView) -> bool:
            if child.execution_id in views:
                return False
            _validate_child(root, child)
            views[child.execution_id] = child
            return True

        def start(view: ExecutionView) -> None:
            stream = self._events.stream(
                view.execution_id,
                principal=principal,
                after_sequence=after_sequences.get(view.execution_id, 0),
            )
            streams[view.execution_id] = stream
            pending[view.execution_id] = _next_event_task(stream, view.execution_id)

        async def discover_child(child_id: str) -> None:
            if child_id in views:
                return
            child = await self._executions.inspect(
                child_id,
                principal=principal,
            )
            if add_child(child):
                start(child)

        async def discover_persisted_children() -> bool:
            added = False
            for child in await self._executions.list_children(
                execution_id,
                principal=principal,
            ):
                if add_child(child):
                    start(child)
                    added = True
            return added

        child_wait: asyncio.Task[tuple[str, ...]] | None = None
        discovery_wait: asyncio.Task[None] | None = None
        try:
            for child in await self._executions.list_children(
                execution_id,
                principal=principal,
            ):
                add_child(child)

            if set(after_sequences) - set(views):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

            for view in tuple(views.values()):
                start(view)

            child_wait = asyncio.create_task(
                subscription.wait(),
                name=f"execution-tree-children-{execution_id}",
            )
            discovery_backoff = _DISCOVERY_BACKOFF_INITIAL
            discovery_wait = asyncio.create_task(
                asyncio.sleep(discovery_backoff),
                name=f"execution-tree-discovery-{execution_id}",
            )
            while True:
                if not pending:
                    try:
                        child_ids = subscription.drain()
                    except AIError:
                        raise
                    except Exception as error:
                        raise _ExecutionStreamFailure(error) from error
                    for child_id in child_ids:
                        await discover_child(child_id)
                    if pending:
                        continue
                    if await discover_persisted_children():
                        continue
                    return

                if child_wait is None or discovery_wait is None:
                    raise RuntimeError("execution tree waiters are unavailable")
                done, _ = await asyncio.wait(
                    (*pending.values(), child_wait, discovery_wait),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Durable stream failures and cancellation take priority
                # over optional broker notifications from the same round.
                for task in done:
                    if task.cancelled():
                        task.result()
                for task in (*pending.values(), child_wait):
                    if task not in done:
                        continue
                    error = task.exception()
                    if error is not None and not isinstance(
                        error, (StopAsyncIteration, _ExecutionStreamFailure)
                    ) and (task is not child_wait or isinstance(error, AIError)):
                        raise error
                if child_wait in done:
                    try:
                        child_ids = child_wait.result()
                    except AIError:
                        raise
                    except Exception as error:
                        raise _ExecutionStreamFailure(error) from error
                    for child_id in child_ids:
                        await discover_child(child_id)
                    child_wait = asyncio.create_task(
                        subscription.wait(),
                        name=f"execution-tree-children-{execution_id}",
                    )

                if discovery_wait in done:
                    added = await discover_persisted_children()
                    discovery_backoff = (
                        _DISCOVERY_BACKOFF_INITIAL
                        if added
                        else min(
                            _DISCOVERY_BACKOFF_MAX,
                            discovery_backoff * 2,
                        )
                    )
                    discovery_wait = asyncio.create_task(
                        asyncio.sleep(discovery_backoff),
                        name=f"execution-tree-discovery-{execution_id}",
                    )

                ready = sorted(
                    (
                        execution_key,
                        task,
                    )
                    for execution_key, task in pending.items()
                    if task in done
                )
                for execution_key, task in ready:
                    pending.pop(execution_key, None)
                    try:
                        event = task.result()
                    except StopAsyncIteration:
                        streams.pop(execution_key, None)
                        continue

                    view = views[execution_key]
                    projected_event = _project_stream_event(
                        event,
                        include_content,
                    )
                    tree_event = ExecutionTreeEvent(
                        view.execution_id,
                        view.agent_id,
                        view.lineage_kind,
                        view.parent_execution_id,
                        view.root_execution_id,
                        view.parent_invocation_id,
                        0 if execution_key == execution_id else 1,
                        projected_event,
                    )
                    yield tree_event
                    pending[execution_key] = _next_event_task(
                        streams[execution_key],
                        execution_key,
                    )
        finally:
            active_error = sys.exc_info()[1]

            async def cleanup() -> None:
                errors: list[BaseException] = []
                wait_tasks = tuple(
                    task for task in (child_wait, discovery_wait, *pending.values())
                    if task is not None
                )
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
                    completed_errors = _completed_errors(tuple(done), child_wait)
                    for error in completed_errors:
                        _report_observation_error(error)
                    errors.extend(completed_errors)
                for stream in tuple(streams.values()):
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        try:
                            await close()
                        except BaseException as error:
                            _report_observation_error(error)
                            errors.append(error)
                try:
                    await subscription.close()
                except AIError as error:
                    _report_observation_error(error)
                    errors.append(error)
                except Exception as error:
                    errors.append(_ExecutionStreamFailure(error))
                failure = _cleanup_failure(active_error, errors)
                if failure is not None:
                    raise failure

            await _await_stream_cleanup(cleanup(), active_error)



def _completed_errors(
    tasks: tuple[asyncio.Task, ...],
    broker_task: asyncio.Task | None,
) -> list[BaseException]:
    errors: list[BaseException] = []
    for task in tasks:
        if task.cancelled():
            continue
        error = task.exception()
        if error is None or isinstance(error, StopAsyncIteration):
            continue
        if task is broker_task and isinstance(error, Exception) and not isinstance(
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
    if (
        view.execution_id != execution_id
        or view.parent_execution_id is not None
        or view.parent_invocation_id is not None
        or view.lineage_kind is ExecutionLineageKind.SUBAGENT
    ):
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
