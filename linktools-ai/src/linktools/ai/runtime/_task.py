#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-bound TaskGraph behavior and complete observation projection."""

import asyncio
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from linktools.core import environ

from ..core import JsonValue, Page, Principal, TaskStatus, canonical_json_bytes
from ..errors import AIError, ErrorCode, TaskObservationError
from ..task import (
    CancelGraphRequest,
    TaskEvent,
    TaskGraphInfo,
    TaskGraphResult,
    TaskGraphService,
    TaskGraphState,
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
    decode_task_results_cursor,
    encode_execution_watch_cursor,
    encode_graph_watch_cursor,
    encode_task_results_cursor,
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
    from ._tasks import TaskEngine

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.runtime.task")


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
    _graph: TaskGraphService
    graph_id: str
    _principal: Principal
    _watch_tree: _ExecutionTreeWatcher
    _engine: "TaskEngine[AppT] | None" = field(default=None, repr=False, compare=False)

    async def wait(
        self,
        *,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult:
        return _public_task_result(
            await self._graph.wait(
                self.graph_id,
                principal=self._principal,
                timeout_seconds=timeout_seconds,
            )
        )

    async def recover(self, *, idempotency_key: str | None = None) -> TaskGraphResult:
        await self._activate_for_control()
        return _public_task_result(await self._graph.recover(
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
        await self._activate_for_control()
        if request.principal != self._principal:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        state = await self._state()
        node = next(
            (item for item in state.nodes if item.node_id == node_id),
            None,
        )
        if node is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return _public_task_result(await self._graph.resume(
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
        await self._activate_for_control()
        if not isinstance(resolution, TaskEffectResolution):
            raise TypeError("resolution must be TaskEffectResolution")
        state = await self._state()
        node = next(
            (item for item in state.nodes if item.node_id == node_id),
            None,
        )
        if node is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return _public_task_result(
            await self._graph.resolve_effect(
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
        graph_state = await self._state()
        state = next(
            (item for item in graph_state.node_states if item.node_id == node_id),
            None,
        )
        if state is None:
            raise AIError(
                ErrorCode.STORAGE_NOT_FOUND,
                safe_details={"graph_id": self.graph_id, "node_id": node_id},
            )
        if state.status in {
            TaskStatus.PENDING,
            TaskStatus.READY,
            TaskStatus.RUNNING,
            TaskStatus.WAITING,
            TaskStatus.RECOVERY_REQUIRED,
        }:
            raise AIError(
                ErrorCode.TASK_NOT_READY,
                safe_details={"graph_id": self.graph_id, "node_id": node_id},
            )
        if state.status in {
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }:
            details: dict[str, JsonValue] = {
                "graph_id": self.graph_id,
                "node_id": node_id,
                "status": state.status.value,
            }
            if state.error_code is not None:
                details["error_code"] = state.error_code
            raise AIError(ErrorCode.TASK_NODE_FAILED, safe_details=details)
        if state.status is not TaskStatus.SUCCEEDED or state.result_digest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        task_runtime = self._runtime._require_task_node_runtime()
        record = await task_runtime.get_result_record(
            self.graph_id,
            node_id,
            tenant_id=self._principal.tenant_id,
        )
        if (
            record is None
            or record.result_digest != state.result_digest
            or record.execution_id != state.execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await task_runtime.read_result_record(
            record,
            principal=self._principal,
        )

    async def result_ref(self, node_id: str) -> TaskResultRef:
        graph_state = await self._state()
        state = next(
            (item for item in graph_state.node_states if item.node_id == node_id),
            None,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if state.status is not TaskStatus.SUCCEEDED or state.result_digest is None:
            raise AIError(ErrorCode.TASK_NOT_READY)
        if state.execution_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        record = await self._runtime._require_task_node_runtime().get_result_record(
            self.graph_id,
            node_id,
            tenant_id=self._principal.tenant_id,
        )
        if (
            record is None
            or record.result_digest != state.result_digest
            or record.execution_id != state.execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return TaskResultRef(
            self._runtime.namespace,
            self._principal.tenant_id,
            self.graph_id,
            node_id,
            state.result_digest,
        )

    async def results(
        self,
        *,
        cursor: str | None = None,
        limit: int = 100,
        include_content: bool = False,
        max_content_bytes: int = 1_048_576,
    ) -> Page[TaskNodeResult]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
            or isinstance(max_content_bytes, bool)
            or not isinstance(max_content_bytes, int)
            or max_content_bytes < 1
            or not isinstance(include_content, bool)
            or cursor is not None
            and (not isinstance(cursor, str) or not cursor)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        expected_sequence: int | None = None
        last_node_id = ""
        if cursor is not None:
            expected_sequence, last_node_id = decode_task_results_cursor(
                self._runtime.namespace,
                self._principal.tenant_id,
                self.graph_id,
                cursor,
                include_content=include_content,
                max_content_bytes=max_content_bytes,
            )

        graph_state = await self._state()
        if expected_sequence is not None and graph_state.event_sequence != expected_sequence:
            raise AIError(ErrorCode.CURSOR_INVALID)
        if graph_state.graph_id != self.graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state_by_id = {
            item.node_id: item for item in graph_state.node_states
        }
        if len(state_by_id) != len(graph_state.node_states):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        ordered_ids = sorted(
            node.node_id
            for node in graph_state.nodes
            if node.node_id > last_node_id
        )
        page_ids = tuple(ordered_ids[:limit])
        has_more = len(ordered_ids) > len(page_ids)
        task_runtime = self._runtime._require_task_node_runtime()
        records = await task_runtime.get_result_records(
            self.graph_id,
            tuple(
                node_id
                for node_id in page_ids
                if state_by_id[node_id].status is TaskStatus.SUCCEEDED
            ),
            tenant_id=self._principal.tenant_id,
        )

        remaining_bytes = max_content_bytes
        items: list[TaskNodeResult] = []
        for node_id in page_ids:
            node_state = state_by_id[node_id]
            result_ref: TaskResultRef | None = None
            output: JsonValue | None = None
            content_included = False
            error_details = (
                dict(node_state.safe_error_details)
                if node_state.error_origin == "node"
                else {}
            )
            error_diagnostics = None
            if node_state.status is TaskStatus.SUCCEEDED:
                record = records.get(node_id)
                if (
                    node_state.result_digest is None
                    or node_state.execution_id is None
                    or record is None
                    or record.result_digest != node_state.result_digest
                    or record.execution_id != node_state.execution_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                result_ref = TaskResultRef(
                    self._runtime.namespace,
                    self._principal.tenant_id,
                    self.graph_id,
                    node_id,
                    node_state.result_digest,
                )
                if include_content:
                    declared_size = await task_runtime.result_payload_size(
                        record,
                        principal=self._principal,
                    )
                    if (
                        isinstance(declared_size, bool)
                        or not isinstance(declared_size, int)
                        or declared_size < 0
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if declared_size <= remaining_bytes:
                        output = await task_runtime.read_result_record(
                            record,
                            principal=self._principal,
                        )
                        actual_size = len(canonical_json_bytes(output))
                        if actual_size != declared_size:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        remaining_bytes -= actual_size
                        content_included = True
            elif (
                node_state.error_origin == "execution"
                and node_state.status
                in {
                    TaskStatus.FAILED,
                    TaskStatus.BLOCKED,
                    TaskStatus.CANCELLED,
                }
            ):
                if node_state.execution_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                execution_result = await task_runtime.read_execution_failure(
                    node_state.execution_id,
                    principal=self._principal,
                )
                if execution_result is not None:
                    error_details = dict(execution_result.safe_error_details)
                    error_diagnostics = execution_result.error_diagnostics
            items.append(
                TaskNodeResult(
                    node_id,
                    node_state.status,
                    node_state.result_digest,
                    node_state.execution_id,
                    node_state.error_code,
                    node_state.error_digest,
                    result_ref=result_ref,
                    safe_error_details=error_details,
                    error_diagnostics=error_diagnostics,
                    output=output,
                    content_included=content_included,
                )
            )

        final_state = await self._state()
        if final_state.event_sequence != graph_state.event_sequence:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_cursor = None
        if has_more and page_ids:
            next_cursor = encode_task_results_cursor(
                self._runtime.namespace,
                self._principal.tenant_id,
                self.graph_id,
                include_content=include_content,
                max_content_bytes=max_content_bytes,
                graph_sequence=graph_state.event_sequence,
                last_node_id=page_ids[-1],
            )
        return Page(tuple(items), next_cursor)

    async def execution(self, node_id: str) -> "Execution[AppT]":
        graph_state = await self._state()
        state = next(
            (item for item in graph_state.node_states if item.node_id == node_id),
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
        graph_state = await self._state()
        result = _state_result(graph_state)
        if observer is not None:
            last_cursor: str | None = None
            async for event in self._replay_events(graph_state):
                await _call_observer(observer, event, cursor=last_cursor)
                last_cursor = event.cursor
        return result

    async def inspect(self) -> TaskGraphView:
        return await self._graph.inspect(
            self.graph_id,
            principal=self._principal,
        )

    async def state(
        self,
        *,
        include_content: bool = False,
    ) -> "TaskGraphInfo | TaskGraphState":
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        graph_state = await self._state()
        if include_content:
            return graph_state
        return TaskGraphInfo.from_state(graph_state)

    async def _state(self) -> TaskGraphState:
        return await self._graph.state(
            self.graph_id,
            principal=self._principal,
        )

    async def cancel(
        self,
        *,
        idempotency_key: "str | None" = None,
        force: bool = False,
    ) -> TaskGraphResult:
        await self._activate_for_control()
        await self._graph.cancel(
            self.graph_id,
            CancelGraphRequest(
                self._principal,
                idempotency_key or secrets.token_urlsafe(32),
                force,
            ),
        )
        return _state_result(await self._state())

    async def _activate_for_control(self) -> None:
        engine = self._engine
        if engine is not None:
            await engine._activate_graph(self.graph_id, self._principal)

    async def observe(
        self,
        observer: "Callable[[TaskGraphRunEvent], Awaitable[None]]",
        *,
        cursor: str | None = None,
        include_content: bool = False,
    ) -> None:
        if not callable(observer):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        events = self.watch(cursor=cursor, include_content=include_content)
        last_cursor = cursor
        try:
            async for event in events:
                await _call_observer(observer, event, cursor=last_cursor)
                last_cursor = event.cursor
        finally:
            await events.aclose()

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
        graph_state = await self._state()
        cursor_graph_sequence = after_graph_sequence
        cursor_execution_sequences = {
            node_id: dict(sequences)
            for node_id, sequences in after_execution_sequences.items()
        }
        states = {node_state.node_id: node_state for node_state in graph_state.node_states}
        nodes = {node.node_id: node for node in graph_state.nodes}
        if len(states) != len(graph_state.node_states):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if set(after_execution_sequences) - set(states):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        if after_graph_sequence > 0 or after_execution_sequences:
            for node_id, sequences in after_execution_sequences.items():
                if sequences and states[node_id].execution_id is None:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        graph_stream: "AsyncIterator[TaskEvent] | None" = None
        graph_task: "asyncio.Task[TaskEvent] | None" = None
        execution_ids: dict[str, str] = {}
        execution_streams: dict[str, AsyncIterator[ExecutionTreeEvent]] = {}
        execution_tasks: dict[str, asyncio.Task[ExecutionTreeEvent]] = {}

        async def start_execution(node_id: str, execution_id: str) -> None:
            current = execution_ids.get(node_id)
            if current is not None:
                if current != execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return
            execution_ids[node_id] = execution_id
            node = nodes.get(node_id)
            if node is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                node.task is not None
                and node.task.id == "linktools.ai.input"
                and node.task.revision == 1
            ):
                return
            execution = await self._runtime.executions.inspect(
                execution_id,
                principal=self._principal,
            )
            if execution.binding_kind == "task":
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

        try:
            graph_stream = self._graph.stream_events(
                self.graph_id,
                principal=self._principal,
                after_sequence=after_graph_sequence,
            )
            graph_task = asyncio.create_task(
                graph_stream.__anext__(),
                name=f"task-run-graph-{self.graph_id}",
            )
            if after_graph_sequence > 0 or after_execution_sequences:
                for node_id, state in states.items():
                    if state.execution_id is not None:
                        await start_execution(node_id, state.execution_id)

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
                        if event.execution_id is not None:
                            if event.node_id is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            await start_execution(event.node_id, event.execution_id)
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
                    node_sequences = cursor_execution_sequences.get(
                        node_id,
                        {},
                    )
                    event_with_cursor = replace(
                        event,
                        cursor=encode_execution_watch_cursor(
                            self._runtime.namespace,
                            self._principal.tenant_id,
                            event.root_execution_id,
                            include_content=include_content,
                            sequences=node_sequences,
                        ),
                    )
                    yield TaskGraphRunEvent(
                        self.graph_id,
                        node_id,
                        event_with_cursor,
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
            if graph_stream is not None:
                close = getattr(graph_stream, "aclose", None)
                if close is not None:
                    await close()
            for stream in tuple(execution_streams.values()):
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()

    async def _replay_events(
        self,
        graph_state: TaskGraphState,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        graph_cutoff = graph_state.event_sequence
        if graph_cutoff < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        captured: dict[str, tuple[str, ExecutionView, int, int]] = {}
        nodes = {node.node_id: node for node in graph_state.nodes}
        for node_state in graph_state.node_states:
            execution_id = node_state.execution_id
            if execution_id is None:
                continue
            node = nodes.get(node_state.node_id)
            if node is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                node.task is not None
                and node.task.id == "linktools.ai.input"
                and node.task.revision == 1
            ):
                continue
            try:
                root = await self._runtime.executions.inspect(
                    execution_id,
                    principal=self._principal,
                )
            except AIError:
                raise
            if root.binding_kind == "task":
                continue
            captured[root.execution_id] = (
                node_state.node_id,
                root,
                root.event_sequence,
                0,
            )
            for child in await self._runtime.executions.list_children(
                root.execution_id,
                principal=self._principal,
            ):
                if (
                    child.parent_execution_id != root.execution_id
                    or child.root_execution_id != root.root_execution_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if child.execution_id in captured:
                    continue
                captured[child.execution_id] = (
                    node_state.node_id,
                    child,
                    child.event_sequence,
                    1,
                )

        replay_execution_sequences: dict[str, dict[str, int]] = {}
        after_sequence = 0
        while after_sequence < graph_cutoff:
            page = await self._graph.list_events(
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
                after_sequence = event.sequence
                yield TaskGraphRunEvent(
                    self.graph_id,
                    event.node_id,
                    event,
                    encode_graph_watch_cursor(
                        self._runtime.namespace,
                        self._principal.tenant_id,
                        self.graph_id,
                        include_content=False,
                        graph_sequence=after_sequence,
                        execution_sequences=replay_execution_sequences,
                    ),
                )

        for execution_id in sorted(
            captured,
            key=lambda value: (
                captured[value][0],
                0
                if captured[value][1].parent_execution_id is None
                else 1,
                captured[value][3],
                value,
            ),
        ):
            node_id, view, cutoff, depth = captured[execution_id]
            sequence = 0
            while sequence < cutoff:
                page = await self._runtime.events.list(
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
                    sequence = event.sequence
                    node_sequences = replay_execution_sequences.setdefault(
                        node_id,
                        {},
                    )
                    node_sequences[execution_id] = sequence
                    tree = ExecutionTreeEvent(
                        execution_id,
                        view.agent_id,
                        view.lineage_kind,
                        view.parent_execution_id,
                        view.root_execution_id,
                        view.parent_invocation_id,
                        depth,
                        stream,
                        encode_execution_watch_cursor(
                            self._runtime.namespace,
                            self._principal.tenant_id,
                            view.root_execution_id,
                            include_content=False,
                            sequences=node_sequences,
                        ),
                    )
                    yield TaskGraphRunEvent(
                        self.graph_id,
                        node_id,
                        tree,
                        encode_graph_watch_cursor(
                            self._runtime.namespace,
                            self._principal.tenant_id,
                            self.graph_id,
                            include_content=False,
                            graph_sequence=graph_cutoff,
                            execution_sequences=replay_execution_sequences,
                        ),
                    )


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


async def _call_observer(
    observer: "Callable[[TaskGraphRunEvent], Awaitable[None]]",
    event: TaskGraphRunEvent,
    *,
    cursor: str | None,
) -> None:
    try:
        await observer(event)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        cause_code = error.code.value if isinstance(error, AIError) else None
        raise TaskObservationError(
            "callback",
            cursor=cursor,
            cause_code=cause_code,
            safe_details=(
                error.safe_details if isinstance(error, AIError) else None
            ),
            diagnostics=(error.diagnostics if isinstance(error, AIError) else None),
        ) from error


__all__ = ["TaskGraphRun"]


_OBSERVER_DRAIN_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
        TaskStatus.RECOVERY_REQUIRED,
    }
)


def _must_drain_observer(status: TaskStatus) -> bool:
    """Return whether wait() must observe the durable graph boundary before returning."""
    return status in _OBSERVER_DRAIN_STATUSES


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
            result.node_results,
        )
    return result


def _state_result(state: TaskGraphState) -> TaskGraphResult:
    return TaskGraphResult(
        state.graph_id,
        _public_task_status(state.status, state.node_states),
        tuple(
            TaskNodeResult(
                state.node_id,
                state.status,
                state.result_digest,
                state.execution_id,
                state.error_code,
                state.error_digest,
            )
            for state in state.node_states
        ),
    )


def _detach_task(task: "asyncio.Task[object]") -> None:
    def consume(done: "asyncio.Task[object]") -> None:
        try:
            done.exception()
        except (asyncio.CancelledError, Exception):
            pass

    task.add_done_callback(consume)
