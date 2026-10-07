#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-bound TaskGraph behavior and complete observation projection."""

import asyncio
import secrets
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, overload

from linktools.core import environ

from ..core import ExecutionStatus, JsonValue, Page, Principal, TaskStatus, canonical_json_bytes
from ..errors import AIError, ErrorCode, ObservationError
from ..task import (
    CancelGraphRequest,
    TaskEvent,
    TaskEventType,
    TaskGraphInfo,
    TaskGraphResult,
    TaskGraphService,
    TaskGraphState,
    TaskEffectResolution,
    TaskEffectResolutionRequest,
    RecoverGraphRequest,
    TaskNodeResult,
    TaskResultRef,
    TaskResultRecord,
    TaskInputSupplyRequest,
)
from ._graph_projection import _GraphModelProjection
from ._observation import (
    _wait, _validate_wait, _call_observer,
    _is_observation_cleanup, _drain_stream_tasks, _await_stream_cleanup,
    _report_observation_error,
)
from ._wait import WaitResult
from ._watch_cursor import (
    decode_graph_watch_cursor,
    decode_task_results_cursor,
    encode_execution_watch_cursor,
    encode_graph_watch_cursor,
    encode_task_results_cursor,
)
from .service_api import (
    ExecutionTreeEvent,
    ExecutionView,
    TaskGraphRunEvent,
    TaskGraphProjection,
    TaskGraphProjectionCoverage,
    _ExecutionStreamFailure,
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
        after_event_seqs: "Mapping[str, int] | None" = None,
        include_content: bool = False,
        ready: asyncio.Event | None = None,
    ) -> AsyncIterator[ExecutionTreeEvent]: ...


@dataclass(frozen=True, slots=True)
class TaskGraphRun(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    _graph: TaskGraphService
    graph_id: str
    _principal: Principal
    _watch_tree: _ExecutionTreeWatcher
    _engine: "TaskEngine[AppT] | None" = field(default=None, repr=False, compare=False)

    @overload
    async def wait(
        self, *, on_event: Callable[[TaskGraphRunEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_content: Literal[False] = False,
        include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
    ) -> WaitResult[TaskGraphInfo]: ...

    @overload
    async def wait(
        self, *, on_event: Callable[[TaskGraphRunEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_content: Literal[True],
        include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
    ) -> WaitResult[TaskGraphState]: ...

    @overload
    async def wait(
        self, *, on_event: Callable[[TaskGraphRunEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_content: bool,
        include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
    ) -> WaitResult[TaskGraphInfo | TaskGraphState]: ...

    async def wait(
        self, *, on_event: Callable[[TaskGraphRunEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_content: bool = False,
        include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
    ) -> WaitResult[TaskGraphInfo | TaskGraphState]:
        """Wait for an authoritative graph snapshot, optionally observing events.

        include_content selects raw node inputs in the result; include_event_content
        controls callback payloads only. Neither option reads node outputs.
        """
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _validate_wait(on_event, cursor, include_event_content, timeout_seconds, close_timeout_seconds)
        initial_ready = asyncio.Event()
        outcome = await _wait(
            scope="task_graph", resource_id=self.graph_id,
            waiter=lambda: self._graph.wait(self.graph_id, principal=self._principal),
            watch=lambda ready: self._watch_prepared(cursor, include_event_content, ready, initial_ready),
            on_event=on_event, cursor=cursor, timeout_seconds=timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            register=self._runtime._register_observation, release=self._runtime._release_observation,
            finalize=lambda result, acknowledged: self._final_projection(acknowledged, include_event_content),
            handover_ready=initial_ready,
        )
        result = outcome.result if include_content else TaskGraphInfo.from_state(outcome.result)
        return WaitResult(result, outcome.cursor, outcome.observation_error)

    async def recover(self, *, idempotency_key: str | None = None) -> TaskGraphResult:
        request = RecoverGraphRequest(
            self._principal,
            secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
        )
        await self._activate_for_control()
        return _public_task_result(await self._graph.recover(self.graph_id, request))

    async def resume(
        self,
        node_id: str,
        request: "TaskInputSupplyRequest",
    ) -> TaskGraphResult:
        if request.principal != self._principal:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._activate_for_control()
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
        request = TaskEffectResolutionRequest(
            self._principal, expected_fence, resolution, idempotency_key,
        )
        await self._activate_for_control()
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
                request,
            )
        )

    async def result(self, node_id: str) -> JsonValue:
        record = await self._result_record(node_id)
        return await self._runtime._require_task_node_runtime().read_result_record(
            record,
            principal=self._principal,
        )

    async def result_ref(self, node_id: str) -> TaskResultRef:
        record = await self._result_record(node_id)
        return TaskResultRef(
            self._runtime.namespace,
            self._principal.tenant_id,
            self.graph_id,
            node_id,
            record.result_digest,
        )

    async def _result_record(self, node_id: str) -> TaskResultRecord:
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
        return record

    async def results(
        self,
        *,
        cursor: str | None = None,
        limit: int = 100,
        include_content: bool = False,
        max_content_bytes: int = 1_048_576,
    ) -> Page[TaskNodeResult]:
        """Page results at one graph revision; discard partial pages and restart if it changes."""
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

        expected_event_seq: int | None = None
        last_node_id = ""
        if cursor is not None:
            expected_event_seq, last_node_id = decode_task_results_cursor(
                self._runtime.namespace,
                self._principal.tenant_id,
                self.graph_id,
                cursor,
                include_content=include_content,
                max_content_bytes=max_content_bytes,
            )

        graph, graph_event_seq = await self._graph.result_header(
            self.graph_id,
            principal=self._principal,
        )
        if (
            expected_event_seq is not None
            and graph_event_seq != expected_event_seq
        ):
            raise AIError(ErrorCode.CURSOR_INVALID)
        if graph.graph_id != self.graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        ordered_ids = sorted(
            node.node_id
            for node in graph.nodes
            if node.node_id > last_node_id
        )
        page_ids = tuple(ordered_ids[:limit])
        has_more = len(ordered_ids) > len(page_ids)
        page_states = await self._graph.result_node_states(
            self.graph_id,
            page_ids,
            principal=self._principal,
        )
        if tuple(state.node_id for state in page_states) != page_ids:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state_by_id = {state.node_id: state for state in page_states}
        if len(state_by_id) != len(page_states):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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

        final_header = await self._graph.result_header(
            self.graph_id,
            principal=self._principal,
        )
        if final_header[1] != graph_event_seq:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_cursor = None
        if has_more and page_ids:
            next_cursor = encode_task_results_cursor(
                self._runtime.namespace,
                self._principal.tenant_id,
                self.graph_id,
                include_content=include_content,
                max_content_bytes=max_content_bytes,
                graph_event_seq=graph_event_seq,
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

    @overload
    async def state(self, *, include_content: Literal[False] = False) -> TaskGraphInfo: ...

    @overload
    async def state(self, *, include_content: Literal[True]) -> TaskGraphState: ...

    @overload
    async def state(self, *, include_content: bool) -> TaskGraphInfo | TaskGraphState: ...

    async def state(
        self,
        *,
        include_content: bool = False,
    ) -> "TaskGraphInfo | TaskGraphState":
        """Read current expanded nodes and state; include_content opts into raw invocation inputs."""
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
        request = CancelGraphRequest(
            self._principal,
            secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
            force,
        )
        await self._activate_for_control()
        await self._graph.cancel(self.graph_id, request)
        return _state_result(await self._state())

    async def _activate_for_control(self) -> None:
        self._runtime._ensure_open()
        engine = self._engine
        if engine is not None:
            await engine._activate_graph(self.graph_id, self._principal)

    def watch(
        self, *, cursor: str | None = None, include_content: bool = False,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        return self._watch_prepared(cursor, include_content, None)

    def _watch_prepared(
        self, cursor: str | None, include_content: bool, ready: asyncio.Event | None,
        initial_ready: asyncio.Event | None = None,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        self._runtime._ensure_open()
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        graph_event_seq, sequences = (0, {}) if cursor is None else decode_graph_watch_cursor(
            self._runtime.namespace, self._principal.tenant_id, self.graph_id,
            cursor, include_content=include_content,
        )
        return self._watch(
            after_graph_event_seq=graph_event_seq, after_execution_event_seqs=sequences,
            include_content=include_content, ready=ready, initial_ready=initial_ready,
        )

    async def _watch(
        self,
        *,
        after_graph_event_seq: int,
        after_execution_event_seqs: Mapping[str, Mapping[str, int]],
        include_content: bool,
        ready: asyncio.Event | None = None,
        initial_ready: asyncio.Event | None = None,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        standalone = ready is None
        graph_state = await self._state()
        cursor_graph_event_seq = after_graph_event_seq
        cursor_execution_event_seqs = {
            node_id: dict(sequences)
            for node_id, sequences in after_execution_event_seqs.items()
        }
        if after_graph_event_seq > graph_state.event_seq:
            raise AIError(ErrorCode.CURSOR_INVALID)
        states = {node_state.node_id: node_state for node_state in graph_state.node_states}
        nodes = {node.node_id: node for node in graph_state.nodes}
        if len(states) != len(graph_state.node_states):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if set(after_execution_event_seqs) - set(states):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        if after_graph_event_seq > 0 or after_execution_event_seqs:
            for node_id, sequences in after_execution_event_seqs.items():
                if sequences and states[node_id].execution_id is None:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        graph_stream: "AsyncIterator[TaskEvent] | None" = None
        graph_task: "asyncio.Task[TaskEvent] | None" = None
        execution_ids: dict[str, str] = {}
        execution_streams: dict[str, AsyncIterator[ExecutionTreeEvent]] = {}
        execution_tasks: dict[str, asyncio.Task[ExecutionTreeEvent]] = {}
        models = _GraphModelProjection(self._runtime.history, self._principal)
        reconciliation: asyncio.Task[None] | None = None

        async def start_execution(node_id: str, execution_id: str) -> None:
            current = execution_ids.get(node_id)
            if current is not None:
                if current != execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return
            execution_ids[node_id] = execution_id
            node = nodes.get(node_id)
            if node is None:
                snapshot = await self._state()
                refreshed_states = {
                    item.node_id: item for item in snapshot.node_states
                }
                refreshed_nodes = {
                    item.node_id: item for item in snapshot.nodes
                }
                if (
                    len(refreshed_states) != len(snapshot.node_states)
                    or len(refreshed_nodes) != len(snapshot.nodes)
                    or set(refreshed_states) != set(refreshed_nodes)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                states.clear()
                states.update(refreshed_states)
                nodes.clear()
                nodes.update(refreshed_nodes)
                node = nodes.get(node_id)
            if node is None:
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": self.graph_id,
                        "node_id": node_id,
                    },
                )
            if (
                node.task is not None
                and node.task.id == "linktools.ai.input"
                and node.task.revision == 1
            ):
                if after_execution_event_seqs.get(node_id):
                    raise AIError(ErrorCode.CURSOR_INVALID)
                return
            execution = await self._runtime.executions.inspect(
                execution_id,
                principal=self._principal,
            )
            if execution.binding_kind == "task":
                if after_execution_event_seqs.get(node_id):
                    raise AIError(ErrorCode.CURSOR_INVALID)
                return
            tree_ready = asyncio.Event()
            stream = self._watch_tree(
                execution_id,
                principal=self._principal,
                after_event_seqs=after_execution_event_seqs.get(node_id),
                include_content=include_content, ready=tree_ready,
            )
            execution_streams[node_id] = stream
            execution_tasks[node_id] = asyncio.create_task(
                stream.__anext__(),
                name=f"task-run-execution-{self.graph_id}-{node_id}",
            )
            prepared = asyncio.create_task(tree_ready.wait())
            try:
                await asyncio.wait({prepared, execution_tasks[node_id]}, return_when=asyncio.FIRST_COMPLETED)
                if execution_tasks[node_id].done():
                    try:
                        execution_tasks[node_id].result()
                    except StopAsyncIteration:
                        pass
            finally:
                prepared.cancel()
                await asyncio.gather(prepared, return_exceptions=True)

        def last_delivered_cursor() -> str | None:
            return encode_graph_watch_cursor(
                self._runtime.namespace,
                self._principal.tenant_id,
                self.graph_id,
                include_content=include_content,
                graph_event_seq=cursor_graph_event_seq,
                execution_event_seqs=cursor_execution_event_seqs,
            )

        try:
            graph_stream = self._graph.stream_events(
                self.graph_id,
                principal=self._principal,
                after_event_seq=after_graph_event_seq,
            )
            graph_task = asyncio.create_task(
                graph_stream.__anext__(),
                name=f"task-run-graph-{self.graph_id}",
            )
            preparation_failure: _ExecutionStreamFailure | None = None
            for node_id, state in tuple(states.items()):
                if state.execution_id is not None:
                    try:
                        await start_execution(node_id, state.execution_id)
                    except _ExecutionStreamFailure as error:
                        if preparation_failure is None:
                            preparation_failure = error
            if ready is not None:
                ready.set()
            if preparation_failure is not None:
                raise preparation_failure

            # The streams are attached before this owner-backed state capture.
            # Replay retains the caller's cursor even if the view is newer.
            graph_state = await self._state()
            states = {item.node_id: item for item in graph_state.node_states}
            nodes = {item.node_id: item for item in graph_state.nodes}
            yield TaskGraphRunEvent(
                self.graph_id, None,
                TaskGraphProjection(TaskGraphInfo.from_state(graph_state), "initial", datetime.now(timezone.utc)),
                last_delivered_cursor(),
            )
            for node_id, state in tuple(states.items()):
                if state.execution_id is not None:
                    await start_execution(node_id, state.execution_id)
            captured = await self._capture_replay(graph_state, cursor=last_delivered_cursor(), include_content=include_content)
            for execution_id, (node_id, view, _, depth) in captured.items():
                await models.add(node_id, view, depth)
            for execution_id in tuple(models.views):
                async for node_id, projection in models.refresh(execution_id):
                    yield TaskGraphRunEvent(self.graph_id, node_id, projection, last_delivered_cursor())
            yield self._projection_checkpoint(graph_state, "initial", models, captured, last_delivered_cursor(), False)
            if initial_ready is not None:
                initial_ready.set()
            reconciliation = asyncio.create_task(asyncio.sleep(1.0))

            while graph_task is not None or execution_tasks:
                waiters = list(execution_tasks.values())
                if graph_task is not None:
                    waiters.append(graph_task)
                waiters.extend(models.waits.values())
                if reconciliation is not None:
                    waiters.append(reconciliation)
                done, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # An optional stream failure must not hide a durable failure
                # that has already completed in the same observation round.
                for task in done:
                    if task.cancelled():
                        task.result()
                for task in waiters:
                    if task not in done:
                        continue
                    error = task.exception()
                    if error is not None and not isinstance(
                        error, (StopAsyncIteration, _ExecutionStreamFailure)
                    ):
                        raise error
                dirty = models.changed(done)
                if reconciliation in done:
                    dirty.update(models.views)
                    reconciliation = asyncio.create_task(asyncio.sleep(1.0))
                for execution_id in sorted(dirty):
                    async for node_id, projection in models.refresh(execution_id):
                        yield TaskGraphRunEvent(self.graph_id, node_id, projection, last_delivered_cursor())
                if graph_task is not None and graph_task in done:
                    task = graph_task
                    graph_task = None
                    try:
                        event = task.result()
                    except StopAsyncIteration:
                        pass
                    else:
                        if event.event_seq <= cursor_graph_event_seq:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if (event.event_type is TaskEventType.GRAPH_EXPANDED
                                and event.event_seq > graph_state.event_seq
                                or event.execution_id is not None
                                and execution_ids.get(event.node_id) != event.execution_id):
                            refreshed = await self._state()
                            if refreshed.event_seq < event.event_seq:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            graph_state = refreshed
                            states = {item.node_id: item for item in refreshed.node_states}
                            nodes = {item.node_id: item for item in refreshed.nodes}
                            yield TaskGraphRunEvent(
                                self.graph_id, None,
                                TaskGraphProjection(TaskGraphInfo.from_state(refreshed), "update", datetime.now(timezone.utc)),
                                last_delivered_cursor(),
                            )
                        if event.execution_id is not None:
                            if event.node_id is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            await start_execution(event.node_id, event.execution_id)
                        projected_event = TaskGraphRunEvent(
                            self.graph_id,
                            event.node_id,
                            event,
                            encode_graph_watch_cursor(
                                self._runtime.namespace,
                                self._principal.tenant_id,
                                self.graph_id,
                                include_content=include_content,
                                graph_event_seq=event.event_seq,
                                execution_event_seqs=cursor_execution_event_seqs,
                            ),
                        )
                        cursor_graph_event_seq = event.event_seq
                        yield projected_event
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
                    durable_seq = event.event.durable_seq
                    if event.execution_id not in models.views:
                        view = await self._runtime.executions.inspect(event.execution_id, principal=self._principal)
                        await models.add(node_id, view, event.depth)
                        async for projected_node, projection in models.refresh(event.execution_id):
                            yield TaskGraphRunEvent(self.graph_id, projected_node, projection, last_delivered_cursor())
                    next_execution_event_seqs = {
                        name: dict(sequences)
                        for name, sequences in cursor_execution_event_seqs.items()
                    }
                    if durable_seq is not None:
                        node_sequences = next_execution_event_seqs.setdefault(
                            node_id,
                            {},
                        )
                        previous = node_sequences.get(event.execution_id, 0)
                        if durable_seq <= previous:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        node_sequences[event.execution_id] = durable_seq
                    node_sequences = next_execution_event_seqs.get(node_id, {})
                    event_with_cursor = replace(
                        event,
                        cursor=encode_execution_watch_cursor(
                            self._runtime.namespace,
                            self._principal.tenant_id,
                            execution_ids[node_id],
                            include_content=include_content,
                            event_seqs=node_sequences,
                        ),
                    )
                    projected_event = TaskGraphRunEvent(
                        self.graph_id,
                        node_id,
                        event_with_cursor,
                        encode_graph_watch_cursor(
                            self._runtime.namespace,
                            self._principal.tenant_id,
                            self.graph_id,
                            include_content=include_content,
                            graph_event_seq=cursor_graph_event_seq,
                            execution_event_seqs=next_execution_event_seqs,
                        ),
                    )
                    cursor_execution_event_seqs = next_execution_event_seqs
                    yield projected_event
                    stream = execution_streams[node_id]
                    execution_tasks[node_id] = asyncio.create_task(
                        stream.__anext__(),
                        name=f"task-run-execution-{self.graph_id}-{node_id}",
                    )
            if standalone:
                final_events = await self._final_projection(last_delivered_cursor(), include_content)
                async for event in final_events:
                    yield event
        except _ExecutionStreamFailure as failure:
            raise _task_stream_observation_error(
                failure,
                last_delivered_cursor(),
            ) from failure.cause
        finally:
            active_error = sys.exc_info()[1]

            async def cleanup() -> None:
                tasks = list(execution_tasks.values())
                if reconciliation is not None:
                    tasks.append(reconciliation)
                if graph_task is not None:
                    tasks.append(graph_task)
                pending_stream_failures = {
                    id(task.exception()) for task in tasks
                    if task.done() and not task.cancelled()
                    and isinstance(task.exception(), _ExecutionStreamFailure)
                }
                cleanup_errors = await _drain_stream_tasks(tasks)
                await models.close()
                if graph_stream is not None:
                    close = getattr(graph_stream, "aclose", None)
                    if close is not None:
                        try:
                            await close()
                        except BaseException as error:
                            _report_observation_error(error)
                            cleanup_errors.append(error)
                for stream in tuple(execution_streams.values()):
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        try:
                            await close()
                        except BaseException as error:
                            _report_observation_error(error)
                            cleanup_errors.append(error)
                if cleanup_errors and (
                    active_error is None
                    or isinstance(active_error, GeneratorExit)
                    or _is_observation_cleanup(active_error)
                    or isinstance(active_error, ObservationError)
                    and active_error.origin == "stream"
                ):
                    error = next(
                        (error for error in cleanup_errors if isinstance(error, asyncio.CancelledError)),
                        next(
                            (error for error in cleanup_errors if not isinstance(error, _ExecutionStreamFailure)),
                            cleanup_errors[0] if (active_error is None or isinstance(active_error, GeneratorExit)
                                                 or _is_observation_cleanup(active_error)) else None,
                        ),
                    )
                    if isinstance(error, _ExecutionStreamFailure):
                        raise _task_stream_observation_error(
                            error,
                            last_delivered_cursor(),
                            cleanup=id(error) not in pending_stream_failures,
                        ) from error.cause
                    if error is not None:
                        raise error

            await _await_stream_cleanup(cleanup(), active_error)

    async def _capture_replay(
        self, graph_state: TaskGraphState, *, cursor: str | None = None,
        include_content: bool = False,
    ) -> dict[str, tuple[str, ExecutionView, int, int]]:
        captured: dict[str, tuple[str, ExecutionView, int, int]] = {}
        _, prior = (0, {}) if cursor is None else decode_graph_watch_cursor(
            self._runtime.namespace, self._principal.tenant_id, self.graph_id,
            cursor, include_content=include_content,
        )
        nodes = {node.node_id: node for node in graph_state.nodes}
        if set(prior) - set(nodes):
            raise AIError(ErrorCode.CURSOR_INVALID)
        for node_state in graph_state.node_states:
            execution_id = node_state.execution_id
            node = nodes.get(node_state.node_id)
            if node is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            sequences = prior.get(node_state.node_id, {})
            if execution_id is None:
                if sequences:
                    raise AIError(ErrorCode.CURSOR_INVALID)
                continue
            if node.task is not None and node.task.id == "linktools.ai.input" and node.task.revision == 1:
                if sequences:
                    raise AIError(ErrorCode.CURSOR_INVALID)
                continue
            root = await self._runtime.executions.inspect(execution_id, principal=self._principal)
            if root.binding_kind == "task":
                if sequences:
                    raise AIError(ErrorCode.CURSOR_INVALID)
                continue
            tree = await self._runtime._capture_execution_tree(
                execution_id, principal=self._principal, after_event_seqs=sequences,
            )
            for view, depth, cutoff in tree:
                if view.execution_id in captured:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                captured[view.execution_id] = (node_state.node_id, view, cutoff, depth)
        return captured

    def _projection_checkpoint(
        self, state: TaskGraphState, phase: Literal["initial", "final"],
        models: _GraphModelProjection,
        captured: dict[str, tuple[str, ExecutionView, int, int]],
        cursor: str | None, durable_complete: bool,
    ) -> TaskGraphRunEvent:
        positions: dict[str, dict[str, int]] = {}
        for execution_id, (node_id, _, cutoff, _) in captured.items():
            positions.setdefault(node_id, {})[execution_id] = cutoff
        unavailable_active = tuple(sorted(
            key for key, value in models.boundaries.items()
            if not value.local_staging_available and captured[key][1].status not in {
                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
            }
        ))
        coverage = TaskGraphProjectionCoverage(
            positions, models.cutoffs(),
            tuple(value for key in sorted(models.boundaries) for value in models.boundaries[key].durable_cutoffs),
            tuple(sorted(key for key, value in models.boundaries.items() if value.local_staging_available)),
            unavailable_active,
            tuple(sorted(key for key, value in models.boundaries.items() if not value.durable_history_available)),
            durable_complete,
            not unavailable_active and all(value.durable_history_available for value in models.boundaries.values()),
        )
        return TaskGraphRunEvent(
            self.graph_id, None,
            TaskGraphProjection(TaskGraphInfo.from_state(state), phase, datetime.now(timezone.utc), coverage),
            cursor,
        )

    async def _final_projection(
        self, cursor: str | None, include_content: bool,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        # The authoritative result is retained by wait. This later source read
        # fixes a separate, finite observation boundary that cannot rewind ACK.
        state = await self._state()
        captured = await self._capture_replay(state, cursor=cursor, include_content=include_content)
        models = _GraphModelProjection(self._runtime.history, self._principal)
        try:
            for execution_id, (node_id, view, _, depth) in captured.items():
                await models.add(node_id, view, depth)
                await models.capture(execution_id)
        except BaseException:
            await models.close()
            raise

        async def deliver() -> AsyncIterator[TaskGraphRunEvent]:
            acknowledged = cursor
            try:
                yield TaskGraphRunEvent(
                    self.graph_id, None,
                    TaskGraphProjection(TaskGraphInfo.from_state(state), "final", datetime.now(timezone.utc)),
                    acknowledged,
                )
                async for event in self._replay_events(
                    state, cursor=acknowledged, include_content=include_content, captured=captured,
                ):
                    acknowledged = event.cursor
                    yield event
                for execution_id in tuple(models.views):
                    async for node_id, projection in models.refresh(
                        execution_id, boundary=models.boundaries[execution_id],
                    ):
                        yield TaskGraphRunEvent(self.graph_id, node_id, projection, acknowledged)
                checkpoint = self._projection_checkpoint(state, "final", models, captured, acknowledged, True)
                yield checkpoint
                if checkpoint.event.coverage is not None and not checkpoint.event.coverage.state_complete:
                    raise ObservationError(
                        "stream", cursor=acknowledged,
                        safe_details={"phase": "drain", "coverage_complete": False,
                                      "reason": "model_state_coverage_unavailable", "graph_id": self.graph_id},
                    )
            finally:
                await models.close()
        return deliver()

    async def _finite_events(
        self, cursor: str | None, include_content: bool,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        state = await self._state()
        captured = await self._capture_replay(state, cursor=cursor, include_content=include_content)
        return self._replay_events(state, cursor=cursor, include_content=include_content, captured=captured)

    async def _replay_events(
        self, graph_state: TaskGraphState, *, cursor: str | None = None,
        include_content: bool = False,
        captured: dict[str, tuple[str, ExecutionView, int, int]] | None = None,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        graph_cutoff = graph_state.event_seq
        if graph_cutoff < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if captured is None:
            captured = await self._capture_replay(graph_state, cursor=cursor, include_content=include_content)
        after_event_seq, prior = (0, {}) if cursor is None else decode_graph_watch_cursor(
            self._runtime.namespace, self._principal.tenant_id, self.graph_id,
            cursor, include_content=include_content,
        )
        if after_event_seq > graph_cutoff:
            raise AIError(ErrorCode.CURSOR_INVALID)
        for node_id, sequences in prior.items():
            for execution_id, sequence in sequences.items():
                if (execution_id not in captured or captured[execution_id][0] != node_id
                        or sequence > captured[execution_id][2]):
                    raise AIError(ErrorCode.CURSOR_INVALID)
        replay_execution_event_seqs = {node_id: dict(values) for node_id, values in prior.items()}
        while after_event_seq < graph_cutoff:
            page = await self._graph.list_events(
                self.graph_id,
                principal=self._principal,
                after_event_seq=after_event_seq,
                limit=min(200, graph_cutoff - after_event_seq),
            )
            if not page.items:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for event in page.items:
                if event.graph_id != self.graph_id or event.event_seq != after_event_seq + 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if event.event_seq > graph_cutoff:
                    break
                after_event_seq = event.event_seq
                yield TaskGraphRunEvent(
                    self.graph_id,
                    event.node_id,
                    event,
                    encode_graph_watch_cursor(
                        self._runtime.namespace,
                        self._principal.tenant_id,
                        self.graph_id,
                        include_content=include_content,
                        graph_event_seq=after_event_seq,
                        execution_event_seqs=replay_execution_event_seqs,
                    ),
                )

        by_node: dict[str, list[tuple[ExecutionView, int, int]]] = {}
        for node_id, view, cutoff, depth in captured.values():
            by_node.setdefault(node_id, []).append((view, depth, cutoff))
        roots = {state.node_id: state.execution_id for state in graph_state.node_states}
        for node_id in sorted(by_node):
            tree = tuple(sorted(by_node[node_id], key=lambda value: (value[1], value[0].execution_id)))
            node_sequences = replay_execution_event_seqs.setdefault(node_id, {})
            async for event in self._runtime._replay_execution_tree(
                tree, principal=self._principal, after_event_seqs=node_sequences,
                include_content=include_content,
            ):
                sequence = event.event.durable_seq
                if sequence is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                node_sequences[event.execution_id] = sequence
                projected = replace(event, cursor=encode_execution_watch_cursor(
                    self._runtime.namespace, self._principal.tenant_id, roots[node_id],
                    include_content=include_content, event_seqs=node_sequences,
                ))
                yield TaskGraphRunEvent(
                    self.graph_id, node_id, projected,
                    encode_graph_watch_cursor(
                        self._runtime.namespace, self._principal.tenant_id, self.graph_id,
                        include_content=include_content, graph_event_seq=graph_cutoff,
                        execution_event_seqs=replay_execution_event_seqs,
                    ),
                )


def _task_stream_observation_error(
    failure: _ExecutionStreamFailure,
    cursor: str | None,
    *,
    cleanup: bool = False,
) -> ObservationError:
    cause = failure.cause
    details = dict(cause.safe_details) if isinstance(cause, AIError) else {"cause_type": type(cause).__name__}
    if cleanup:
        details["phase"] = "cleanup"
    return ObservationError(
        "stream",
        cursor=cursor,
        cause_code=cause.code.value if isinstance(cause, AIError) else None,
        safe_details=details,
        diagnostics=cause.diagnostics if isinstance(cause, AIError) else None,
    )


__all__ = ["TaskGraphRun"]


def _public_task_result(result: TaskGraphResult) -> TaskGraphResult:
    status = result.wait_status
    return result if status is result.status else replace(result, status=status)


def _state_result(state: TaskGraphState) -> TaskGraphResult:
    return TaskGraphResult(
        state.graph_id,
        state.wait_status,
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
