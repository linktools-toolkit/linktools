#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-bound TaskGraph behavior and complete observation projection."""

import asyncio
import secrets
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from ..core import Principal
from ..errors import AIError, ErrorCode
from ..task import (
    CancelGraphRequest,
    TaskEvent,
    TaskGraphResult,
    TaskGraphSnapshot,
    TaskGraphView,
)
from .service_api import ExecutionTreeEvent, TaskGraphRunEvent

if TYPE_CHECKING:
    from ._runtime_service import Runtime

AppT = TypeVar("AppT")


class _ExecutionTreeWatcher(Protocol):
    def __call__(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences: "Mapping[str, int] | None" = None,
    ) -> AsyncIterator[ExecutionTreeEvent]: ...


@dataclass(frozen=True, slots=True)
class TaskGraphRun(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    graph_id: str
    _principal: Principal
    _watch_tree: _ExecutionTreeWatcher

    async def wait(
        self,
        *,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult:
        return await self._runtime.graph.wait(
            self.graph_id,
            principal=self._principal,
            timeout_seconds=timeout_seconds,
        )

    async def inspect(self) -> TaskGraphView:
        return await self._runtime.graph.inspect(
            self.graph_id,
            principal=self._principal,
        )

    async def snapshot(self) -> TaskGraphSnapshot:
        return await self._runtime.graph.snapshot(
            self.graph_id,
            principal=self._principal,
        )

    async def cancel(
        self,
        *,
        idempotency_key: "str | None" = None,
        force: bool = False,
    ) -> TaskGraphView:
        return await self._runtime.graph.cancel(
            self.graph_id,
            CancelGraphRequest(
                self._principal,
                idempotency_key or secrets.token_urlsafe(32),
                force,
            ),
        )

    def watch(
        self,
        *,
        after_graph_sequence: int = 0,
        after_execution_sequences: (
            "Mapping[str, Mapping[str, int]] | None"
        ) = None,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        if (
            isinstance(after_graph_sequence, bool)
            or not isinstance(after_graph_sequence, int)
            or after_graph_sequence < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return self._watch(
            after_graph_sequence=after_graph_sequence,
            after_execution_sequences=_normalize_execution_sequences(
                after_execution_sequences
            ),
        )

    async def _watch(
        self,
        *,
        after_graph_sequence: int,
        after_execution_sequences: Mapping[str, Mapping[str, int]],
    ) -> AsyncIterator[TaskGraphRunEvent]:
        snapshot = await self.snapshot()
        states = {state.node_id: state for state in snapshot.node_states}
        if len(states) != len(snapshot.node_states):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if set(after_execution_sequences) - set(states):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        graph_stream = self._runtime.graph.stream_events(
            self.graph_id,
            principal=self._principal,
            after_sequence=after_graph_sequence,
        )
        graph_task: "asyncio.Task[TaskEvent] | None" = asyncio.create_task(
            graph_stream.__anext__(),
            name=f"task-run-graph-{self.graph_id}",
        )
        execution_ids: dict[str, str] = {}
        execution_streams: dict[str, AsyncIterator[ExecutionTreeEvent]] = {}
        execution_tasks: dict[str, asyncio.Task[ExecutionTreeEvent]] = {}

        def start_execution(node_id: str, execution_id: str) -> None:
            current = execution_ids.get(node_id)
            if current is not None:
                if current != execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return
            execution_ids[node_id] = execution_id
            stream = self._watch_tree(
                execution_id,
                principal=self._principal,
                after_sequences=after_execution_sequences.get(node_id),
            )
            execution_streams[node_id] = stream
            execution_tasks[node_id] = asyncio.create_task(
                stream.__anext__(),
                name=f"task-run-execution-{self.graph_id}-{node_id}",
            )

        if after_graph_sequence > 0 or after_execution_sequences:
            for node_id, state in states.items():
                sequences = after_execution_sequences.get(node_id)
                if sequences and state.execution_id is None:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                if state.execution_id is not None:
                    start_execution(node_id, state.execution_id)

        try:
            while graph_task is not None or execution_tasks:
                waiters = list(execution_tasks.values())
                if graph_task is not None:
                    waiters.append(graph_task)
                done, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if graph_task is not None and graph_task in done:
                    task = graph_task
                    graph_task = None
                    try:
                        event = task.result()
                    except StopAsyncIteration:
                        pass
                    else:
                        yield TaskGraphRunEvent(self.graph_id, event.node_id, event)
                        if event.execution_id is not None:
                            if event.node_id is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            start_execution(event.node_id, event.execution_id)
                        graph_task = asyncio.create_task(
                            graph_stream.__anext__(),
                            name=f"task-run-graph-{self.graph_id}",
                        )

                ready = sorted(
                    (node_id, task)
                    for node_id, task in execution_tasks.items()
                    if task in done
                )
                for node_id, task in ready:
                    execution_tasks.pop(node_id, None)
                    try:
                        event = task.result()
                    except StopAsyncIteration:
                        execution_streams.pop(node_id, None)
                        continue
                    yield TaskGraphRunEvent(self.graph_id, node_id, event)
                    stream = execution_streams[node_id]
                    execution_tasks[node_id] = asyncio.create_task(
                        stream.__anext__(),
                        name=f"task-run-execution-{self.graph_id}-{node_id}",
                    )
        finally:
            tasks = list(execution_tasks.values())
            if graph_task is not None:
                tasks.append(graph_task)
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            close = getattr(graph_stream, "aclose", None)
            if close is not None:
                await close()
            for stream in tuple(execution_streams.values()):
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()


def _normalize_execution_sequences(
    value: "Mapping[str, Mapping[str, int]] | None",
) -> Mapping[str, Mapping[str, int]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result: dict[str, Mapping[str, int]] = {}
    for node_id, sequences in value.items():
        if (
            not isinstance(node_id, str)
            or not node_id
            or not isinstance(sequences, Mapping)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        normalized: dict[str, int] = {}
        for execution_id, sequence in sequences.items():
            if (
                not isinstance(execution_id, str)
                or not execution_id
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            normalized[execution_id] = sequence
        result[node_id] = normalized
    return result


__all__ = ["TaskGraphRun"]
