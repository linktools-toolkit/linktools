#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-bound TaskGraph behavior and complete observation projection."""

import asyncio
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from pydantic import BaseModel

from ..core import JsonValue, Principal, TaskStatus
from ..errors import AIError, ErrorCode
from ..task import (
    CancelGraphRequest,
    TaskEvent,
    TaskGraphInfo,
    TaskGraphResult,
    TaskGraphSnapshot,
    TaskGraphView,
    TaskEffectResolution,
    TaskEffectResolutionRequest,
    RecoverGraphRequest,
    TaskNodeResult,
    TaskResultRef,
    TaskInputSupplyRequest,
)
from ._watch_cursor import (
    decode_graph_watch_cursor,
    encode_graph_watch_cursor,
)
from .service_api import (
    ExecutionStreamEvent,
    ExecutionTreeEvent,
    ExecutionView,
    TaskGraphRunEvent,
)

if TYPE_CHECKING:
    from ._agent import Execution
    from ._runtime_service import Runtime

AppT = TypeVar("AppT")


class _ExecutionTreeWatcher(Protocol):
    def __call__(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences: "Mapping[str, int] | None" = None,
        include_content: bool = False,
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
        observer: "Callable[[TaskGraphRunEvent], Awaitable[None]] | None" = None,
    ) -> TaskGraphResult:
        wait_task = asyncio.create_task(
            self._runtime.graph.wait(
                self.graph_id,
                principal=self._principal,
                timeout_seconds=timeout_seconds,
            ),
            name=f"task-graph-wait-{self.graph_id}",
        )
        if observer is None:
            return _public_task_result(await wait_task)
        observer_task = asyncio.create_task(
            self._observe(observer),
            name=f"task-graph-observer-{self.graph_id}",
        )
        try:
            done, _ = await asyncio.wait(
                (wait_task, observer_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_task in done:
                return _public_task_result(wait_task.result())
            error = observer_task.exception()
            if error is not None:
                raise error
            return _public_task_result(await wait_task)
        finally:
            for task in (observer_task, wait_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                observer_task,
                wait_task,
                return_exceptions=True,
            )

    async def recover(self, *, idempotency_key: str | None = None) -> TaskGraphResult:
        return _public_task_result(await self._runtime.graph.recover(
            self.graph_id,
            RecoverGraphRequest(
                self._principal,
                idempotency_key or secrets.token_urlsafe(32),
            ),
        ))

    async def resume(
        self,
        node_id: str,
        request: "TaskInputSupplyRequest",
    ) -> TaskGraphResult:
        if request.principal != self._principal:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        snapshot = await self._snapshot()
        node = next(
            (item for item in snapshot.nodes if item.node_id == node_id),
            None,
        )
        if node is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return _public_task_result(await self._runtime.graph.resume(
            self.graph_id,
            node_id,
            request,
        ))

    async def resolve_effect(
        self,
        node_id: str,
        expected_fence: int,
        resolution: TaskEffectResolution,
        *,
        idempotency_key: str,
    ) -> TaskGraphResult:
        if not isinstance(resolution, TaskEffectResolution):
            raise TypeError("resolution must be TaskEffectResolution")
        snapshot = await self._snapshot()
        node = next(
            (item for item in snapshot.nodes if item.node_id == node_id),
            None,
        )
        if node is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return _public_task_result(
            await self._runtime.graph.resolve_effect(
                self.graph_id,
                node_id,
                TaskEffectResolutionRequest(
                    self._principal,
                    expected_fence,
                    resolution,
                    idempotency_key,
                ),
            )
        )

    async def result(self, node_id: str) -> JsonValue:
        return await self._runtime.read_task_result(
            self.graph_id,
            node_id,
            principal=self._principal,
        )

    async def result_ref(self, node_id: str) -> TaskResultRef:
        snapshot = await self._snapshot()
        state = next(
            (item for item in snapshot.node_states if item.node_id == node_id),
            None,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if state.status is not TaskStatus.SUCCEEDED or state.result_digest is None:
            raise AIError(ErrorCode.TASK_NOT_READY)
        return TaskResultRef(
            self._runtime.namespace,
            self._principal.tenant_id,
            self.graph_id,
            node_id,
            state.result_digest,
        )

    async def execution(self, node_id: str) -> "Execution[AppT]":
        snapshot = await self._snapshot()
        state = next(
            (item for item in snapshot.node_states if item.node_id == node_id),
            None,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if state.execution_id is None:
            raise AIError(ErrorCode.EXECUTION_NOT_READY)
        return self._runtime._task_execution(
            self.graph_id,
            node_id,
            state.execution_id,
            self._principal,
        )

    async def replay(
        self,
        observer: "Callable[[TaskGraphRunEvent], Awaitable[None]] | None" = None,
    ) -> TaskGraphResult:
        snapshot = await self._snapshot()
        result = TaskGraphResult(
            snapshot.graph_id,
            _public_task_status(snapshot.status, snapshot.node_states),
            tuple(
                state.execution_id
                for state in snapshot.node_states
                if state.execution_id is not None
            ),
            tuple(
                TaskNodeResult(
                    state.node_id,
                    state.status,
                    state.result_digest,
                    state.execution_id,
                    state.error_code,
                    state.error_digest,
                )
                for state in snapshot.node_states
            ),
        )
        if observer is not None:
            events = await self._capture_replay_events(snapshot)
            for event in events:
                await observer(event)
        return result

    async def inspect(self) -> TaskGraphView:
        return await self._runtime.graph.inspect(
            self.graph_id,
            principal=self._principal,
        )

    async def snapshot(
        self,
        *,
        include_content: bool = False,
    ) -> "TaskGraphInfo | TaskGraphSnapshot":
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        snapshot = await self._snapshot()
        if include_content:
            return snapshot
        return TaskGraphInfo.from_snapshot(snapshot)

    async def _snapshot(self) -> TaskGraphSnapshot:
        return await self._runtime.graph.snapshot(
            self.graph_id,
            principal=self._principal,
        )

    async def cancel(
        self,
        *,
        idempotency_key: "str | None" = None,
        force: bool = False,
    ) -> TaskGraphResult:
        await self._runtime.graph.cancel(
            self.graph_id,
            CancelGraphRequest(
                self._principal,
                idempotency_key or secrets.token_urlsafe(32),
                force,
            ),
        )
        return _snapshot_result(await self._snapshot())

    async def _observe(
        self,
        observer: "Callable[[TaskGraphRunEvent], Awaitable[None]]",
    ) -> None:
        async for event in self.watch():
            await observer(event)

    def watch(
        self,
        *,
        cursor: "str | None" = None,
        include_content: bool = False,
        after_graph_sequence: int = 0,
        after_execution_sequences: (
            "Mapping[str, Mapping[str, int]] | None"
        ) = None,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if cursor is not None:
            if after_graph_sequence != 0 or after_execution_sequences:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            (
                after_graph_sequence,
                after_execution_sequences,
            ) = decode_graph_watch_cursor(
                self._runtime.namespace,
                self._principal.tenant_id,
                self.graph_id,
                cursor,
                include_content=include_content,
            )
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
            include_content=include_content,
        )

    async def _watch(
        self,
        *,
        after_graph_sequence: int,
        after_execution_sequences: Mapping[str, Mapping[str, int]],
        include_content: bool,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        snapshot = await self._snapshot()
        cursor_graph_sequence = after_graph_sequence
        cursor_execution_sequences = {
            node_id: dict(sequences)
            for node_id, sequences in after_execution_sequences.items()
        }
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
            if _is_task_execution_id(execution_id):
                return
            stream = self._watch_tree(
                execution_id,
                principal=self._principal,
                after_sequences=after_execution_sequences.get(node_id),
                include_content=include_content,
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
                        if event.sequence <= cursor_graph_sequence:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        cursor_graph_sequence = event.sequence
                        yield TaskGraphRunEvent(
                            self.graph_id,
                            event.node_id,
                            event,
                            encode_graph_watch_cursor(
                                self._runtime.namespace,
                                self._principal.tenant_id,
                                self.graph_id,
                                include_content=include_content,
                                graph_sequence=cursor_graph_sequence,
                                execution_sequences=cursor_execution_sequences,
                            ),
                        )
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
                    durable_sequence = event.event.durable_sequence
                    if durable_sequence is not None:
                        node_sequences = cursor_execution_sequences.setdefault(
                            node_id,
                            {},
                        )
                        previous = node_sequences.get(event.execution_id, 0)
                        if durable_sequence <= previous:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        node_sequences[event.execution_id] = durable_sequence
                    yield TaskGraphRunEvent(
                        self.graph_id,
                        node_id,
                        event,
                        encode_graph_watch_cursor(
                            self._runtime.namespace,
                            self._principal.tenant_id,
                            self.graph_id,
                            include_content=include_content,
                            graph_sequence=cursor_graph_sequence,
                            execution_sequences=cursor_execution_sequences,
                        ),
                    )
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

    async def _capture_replay_events(
        self,
        snapshot: TaskGraphSnapshot,
    ) -> tuple[TaskGraphRunEvent, ...]:
        graph_cutoff = snapshot.event_sequence
        if graph_cutoff < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        captured: dict[str, tuple[str, ExecutionView, int]] = {}
        for state in snapshot.node_states:
            execution_id = state.execution_id
            if execution_id is None:
                continue
            try:
                root = await self._runtime.execution.inspect(
                    execution_id,
                    principal=self._principal,
                )
            except AIError as error:
                if (
                    execution_id.startswith("wait:")
                    or execution_id.startswith("task-execution-")
                ) and error.code in {
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.STORAGE_NOT_FOUND,
                }:
                    continue
                raise
            captured[root.execution_id] = (
                state.node_id,
                root,
                root.event_sequence,
            )
            for child in await self._runtime.execution.list_children(
                root.execution_id,
                principal=self._principal,
            ):
                if (
                    child.parent_execution_id != root.execution_id
                    or child.root_execution_id != root.root_execution_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                captured[child.execution_id] = (
                    state.node_id,
                    child,
                    child.event_sequence,
                )

        events: list[TaskGraphRunEvent] = []
        after_sequence = 0
        while after_sequence < graph_cutoff:
            page = await self._runtime.graph.list_events(
                self.graph_id,
                principal=self._principal,
                after_sequence=after_sequence,
                limit=min(200, graph_cutoff - after_sequence),
            )
            if not page.items:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for event in page.items:
                if event.sequence > graph_cutoff:
                    break
                events.append(
                    TaskGraphRunEvent(
                        self.graph_id,
                        event.node_id,
                        event,
                    )
                )
                after_sequence = event.sequence

        for execution_id in sorted(
            captured,
            key=lambda value: (
                captured[value][0],
                0
                if captured[value][1].parent_execution_id is None
                else 1,
                value,
            ),
        ):
            node_id, view, cutoff = captured[execution_id]
            sequence = 0
            depth = 0 if view.parent_execution_id is None else 1
            while sequence < cutoff:
                page = await self._runtime.event.list(
                    execution_id,
                    principal=self._principal,
                    after_sequence=sequence,
                    limit=min(200, cutoff - sequence),
                )
                if not page.items:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for event in page.items:
                    if event.sequence > cutoff:
                        break
                    stream = ExecutionStreamEvent(
                        execution_id,
                        event.sequence,
                        event.event_type,
                        {},
                    )
                    tree = ExecutionTreeEvent(
                        execution_id,
                        view.agent_id,
                        view.lineage_kind,
                        view.parent_execution_id,
                        view.root_execution_id,
                        view.parent_invocation_id,
                        depth,
                        stream,
                    )
                    events.append(
                        TaskGraphRunEvent(
                            self.graph_id,
                            node_id,
                            tree,
                        )
                    )
                    sequence = event.sequence
        return tuple(events)


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


def _is_task_execution_id(value: str) -> bool:
    return value.startswith("task-execution-") or value.startswith("wait:")


__all__ = ["TaskGraphRun"]


def _public_task_status(
    status: TaskStatus,
    states: object = (),
) -> TaskStatus:
    if status is not TaskStatus.RUNNING or not isinstance(states, tuple):
        return status
    unfinished = tuple(
        state
        for state in states
        if getattr(state, "status", None)
        not in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }
    )
    if unfinished and any(
        getattr(state, "status", None) is TaskStatus.WAITING
        for state in unfinished
    ) and all(
        getattr(state, "status", None)
        not in {TaskStatus.READY, TaskStatus.RUNNING}
        for state in unfinished
    ):
        return TaskStatus.WAITING
    return status


def _public_task_result(result: TaskGraphResult) -> TaskGraphResult:
    if result.status is not TaskStatus.RUNNING:
        return result
    if not result.node_results:
        return result
    unfinished = tuple(
        node
        for node in result.node_results
        if node.status
        not in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }
    )
    if unfinished and any(
        node.status is TaskStatus.WAITING for node in unfinished
    ) and all(
        node.status not in {TaskStatus.READY, TaskStatus.RUNNING}
        for node in unfinished
    ):
        return TaskGraphResult(
            result.graph_id,
            TaskStatus.WAITING,
            result.execution_ids,
            result.node_results,
        )
    return result


def _snapshot_result(snapshot: TaskGraphSnapshot) -> TaskGraphResult:
    return TaskGraphResult(
        snapshot.graph_id,
        _public_task_status(snapshot.status, snapshot.node_states),
        tuple(
            state.execution_id
            for state in snapshot.node_states
            if state.execution_id is not None
        ),
        tuple(
            TaskNodeResult(
                state.node_id,
                state.status,
                state.result_digest,
                state.execution_id,
                state.error_code,
                state.error_digest,
            )
            for state in snapshot.node_states
        ),
    )


def _detach_task(task: "asyncio.Task[object]") -> None:
    def consume(done: "asyncio.Task[object]") -> None:
        try:
            done.exception()
        except (asyncio.CancelledError, Exception):
            pass

    task.add_done_callback(consume)
