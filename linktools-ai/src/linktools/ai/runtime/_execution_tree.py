#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derived live event projection over one durable execution tree."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from ..core import ExecutionLineageKind, Principal
from ..errors import AIError, ErrorCode
from .service_api import ExecutionStreamEvent, ExecutionTreeEvent, ExecutionView

_MAX_TREE_DEPTH = 8


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
        root_execution_id: str,
    ) -> None:
        self._owner = owner
        self._root_execution_id = root_execution_id
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
        if self._pending:
            self._event.set()
        return values

    async def wait(self) -> tuple[str, ...]:
        while not self._pending:
            if self._closed:
                return ()
            self._event.clear()
            if self._pending:
                break
            await self._event.wait()
        return self.drain()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._owner._remove(self._root_execution_id, self)
        self._event.set()


class ExecutionTreeBroker:
    """Notify active tree readers about newly established children."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, set[_ExecutionTreeSubscription]] = {}

    def subscribe(self, root_execution_id: str) -> _ExecutionTreeSubscription:
        subscription = _ExecutionTreeSubscription(self, root_execution_id)
        self._subscriptions.setdefault(root_execution_id, set()).add(subscription)
        return subscription

    def publish(self, root_execution_id: str, child_execution_id: str) -> None:
        for subscription in tuple(
            self._subscriptions.get(root_execution_id, ())
        ):
            subscription.publish(child_execution_id)

    def _remove(
        self,
        root_execution_id: str,
        subscription: _ExecutionTreeSubscription,
    ) -> None:
        values = self._subscriptions.get(root_execution_id)
        if values is None:
            return
        values.discard(subscription)
        if not values:
            self._subscriptions.pop(root_execution_id, None)


@dataclass(frozen=True, slots=True)
class _EventItem:
    view: ExecutionView
    event: ExecutionStreamEvent


@dataclass(frozen=True, slots=True)
class _DoneItem:
    execution_id: str


@dataclass(frozen=True, slots=True)
class _ErrorItem:
    error: BaseException


@dataclass(frozen=True, slots=True)
class _ChildItem:
    execution_id: str


_TreeItem = _EventItem | _DoneItem | _ErrorItem | _ChildItem


class ExecutionTreeStreamer:
    """Merge per-execution streams without creating a durable tree log."""

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
        subscription = self._broker.subscribe(execution_id)
        queue: asyncio.Queue[_TreeItem] = asyncio.Queue()
        views: dict[str, ExecutionView] = {}
        depths: dict[str, int] = {}
        pumps: dict[str, asyncio.Task[None]] = {}
        completed: set[str] = set()

        async def pump(view: ExecutionView) -> None:
            try:
                async for event in self._events.stream(
                    view.execution_id,
                    principal=principal,
                    after_sequence=after_sequences.get(view.execution_id, 0),
                ):
                    await queue.put(_EventItem(view, event))
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001
                await queue.put(_ErrorItem(error))
            finally:
                await queue.put(_DoneItem(view.execution_id))

        async def pump_children() -> None:
            while True:
                child_ids = await subscription.wait()
                if not child_ids:
                    return
                for child_id in child_ids:
                    await queue.put(_ChildItem(child_id))

        def start(view: ExecutionView, depth: int) -> bool:
            current = views.get(view.execution_id)
            if current is not None:
                if (
                    not _same_lineage(current, view)
                    or depths[view.execution_id] != depth
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return False
            views[view.execution_id] = view
            depths[view.execution_id] = depth
            pumps[view.execution_id] = asyncio.create_task(
                pump(view),
                name=f"execution-tree-stream-{view.execution_id}",
            )
            return True

        async def discover(
            view: ExecutionView,
            depth: int,
            path: frozenset[str],
        ) -> bool:
            if depth > _MAX_TREE_DEPTH or view.execution_id in path:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if depth == 0:
                if (
                    view.execution_id != execution_id
                    or view.parent_execution_id is not None
                    or view.parent_invocation_id is not None
                    or view.root_execution_id != execution_id
                    or view.lineage_kind is ExecutionLineageKind.SUBAGENT
                ):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            elif (
                view.lineage_kind is not ExecutionLineageKind.SUBAGENT
                or view.root_execution_id != execution_id
                or not view.parent_execution_id
                or not view.parent_invocation_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            added = start(view, depth)
            next_path = path | {view.execution_id}
            children = await self._executions.list_children(
                view.execution_id,
                principal=principal,
            )
            for child in children:
                if child.parent_execution_id != view.execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                added = await discover(
                    child,
                    depth + 1,
                    next_path,
                ) or added
            return added

        async def discover_child(child_id: str) -> bool:
            child = await self._executions.inspect(
                child_id,
                principal=principal,
            )
            parent_id = child.parent_execution_id
            parent = views.get(parent_id) if parent_id is not None else None
            if parent is None:
                fresh_root = await self._executions.inspect(
                    execution_id,
                    principal=principal,
                )
                await discover(fresh_root, 0, frozenset())
                parent = views.get(parent_id) if parent_id is not None else None
            if parent is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await discover(
                child,
                depths[parent.execution_id] + 1,
                frozenset(),
            )

        child_pump: asyncio.Task[None] | None = None
        try:
            root = await self._executions.inspect(
                execution_id,
                principal=principal,
            )
            await discover(root, 0, frozenset())
            if set(after_sequences) - set(views):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            child_pump = asyncio.create_task(
                pump_children(),
                name=f"execution-tree-children-{execution_id}",
            )

            while True:
                item = await queue.get()
                try:
                    if isinstance(item, _EventItem):
                        view = item.view
                        yield ExecutionTreeEvent(
                            view.execution_id,
                            view.agent_id,
                            view.lineage_kind,
                            view.parent_execution_id,
                            view.root_execution_id,
                            view.parent_invocation_id,
                            depths[view.execution_id],
                            item.event,
                        )
                    elif isinstance(item, _DoneItem):
                        completed.add(item.execution_id)
                    elif isinstance(item, _ErrorItem):
                        raise item.error
                    else:
                        await discover_child(item.execution_id)
                finally:
                    queue.task_done()

                if (
                    execution_id in completed
                    and all(task.done() for task in pumps.values())
                    and queue.empty()
                ):
                    added = False
                    fresh_root = await self._executions.inspect(
                        execution_id,
                        principal=principal,
                    )
                    added = await discover(
                        fresh_root,
                        0,
                        frozenset(),
                    ) or added
                    for child_id in subscription.drain():
                        added = await discover_child(child_id) or added
                    if not added and queue.empty():
                        return
        finally:
            if child_pump is not None and not child_pump.done():
                child_pump.cancel()
            for task in pumps.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                *(
                    task
                    for task in (child_pump, *pumps.values())
                    if task is not None
                ),
                return_exceptions=True,
            )
            await subscription.close()


def _same_lineage(left: ExecutionView, right: ExecutionView) -> bool:
    return (
        left.execution_id == right.execution_id
        and left.agent_id == right.agent_id
        and left.lineage_kind is right.lineage_kind
        and left.parent_execution_id == right.parent_execution_id
        and left.root_execution_id == right.root_execution_id
        and left.parent_invocation_id == right.parent_invocation_id
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
