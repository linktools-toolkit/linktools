#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derived live event projection over one root execution and its direct children."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Protocol

from ..core import ExecutionLineageKind, Principal
from ..errors import AIError, ErrorCode
from .service_api import ExecutionStreamEvent, ExecutionTreeEvent, ExecutionView


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
    ) -> AsyncIterator[ExecutionTreeEvent]:
        return self._stream(
            execution_id,
            principal=principal,
            after_sequences=_normalize_after_sequences(after_sequences),
        )

    async def _stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences: Mapping[str, int],
    ) -> AsyncIterator[ExecutionTreeEvent]:
        root = await self._executions.inspect(
            execution_id,
            principal=principal,
        )
        _validate_root(root, execution_id)

        subscription = self._broker.subscribe(execution_id)
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
            try:
                while True:
                    if not pending:
                        child_ids = subscription.drain()
                        if not child_ids:
                            return
                        for child_id in child_ids:
                            await discover_child(child_id)
                        continue

                    done, _ = await asyncio.wait(
                        (*pending.values(), child_wait),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if child_wait in done:
                        child_ids = child_wait.result()
                        for child_id in child_ids:
                            await discover_child(child_id)
                        child_wait = asyncio.create_task(
                            subscription.wait(),
                            name=f"execution-tree-children-{execution_id}",
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
                        yield ExecutionTreeEvent(
                            view.execution_id,
                            view.agent_id,
                            view.lineage_kind,
                            view.parent_execution_id,
                            view.root_execution_id,
                            view.parent_invocation_id,
                            0 if execution_key == execution_id else 1,
                            event,
                        )
                        pending[execution_key] = _next_event_task(
                            streams[execution_key],
                            execution_key,
                        )
            finally:
                if not child_wait.done():
                    child_wait.cancel()
                for task in pending.values():
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    *pending.values(),
                    child_wait,
                    return_exceptions=True,
                )
                for stream in tuple(streams.values()):
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()
        finally:
            await subscription.close()


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
