#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-backed Task API independent of Runtime composition."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol, cast

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    canonical_sha256,
    idempotency_key_digest,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ._event import TaskEvent
from ._graph import (
    CancelGraphRequest,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphSnapshot,
    TaskGraphView,
    TaskNodeResult,
    TaskNodeView,
)
from ._service import TaskApi, TaskGraphLauncher

_logger = environ.get_logger("ai.task.service")
_GRAPH_OBSERVATION_RECHECK_SECONDS = 1.0
_TASK_EVENT_READ_LIMIT = 200


class _TaskReleaseCallback(Protocol):
    async def __call__(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> None: ...


class _LocalTaskWaiter(Protocol):
    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool: ...

    def graph_activity_generation(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> int | None: ...

    async def wait_graph_activity(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_generation: "int | None" = None,
    ) -> None: ...


class _TaskGraphPreflight(Protocol):
    def validate_request(self, graph: TaskGraph) -> None: ...

    def validate_recovery(self, graph: TaskGraph) -> None: ...


async def _no_release_terminal(graph_id: str, *, tenant_id: str) -> None:
    del graph_id, tenant_id


@dataclass
class _TaskHandoffState:
    active_consumers: int = 0
    release_requested: bool = False
    release_in_progress: bool = False


class _TaskRepository(Protocol):
    async def get_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None: ...

    async def create_graph(
        self,
        graph: TaskGraph,
        *,
        tenant_id: str,
    ) -> TaskGraphView: ...

    async def get_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView | None: ...

    async def snapshot_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphSnapshot | None: ...

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]: ...

    async def reconcile_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView: ...

    async def cancel_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView: ...

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskNodeView, ...]: ...


class _OperationRepository(Protocol):
    async def append(
        self,
        record: OperationLedgerInput,
    ) -> OperationLedgerRecord: ...

    async def get(
        self,
        operation_id: str,
        *,
        tenant_id: str,
    ) -> OperationLedgerRecord | None: ...

    async def compare_and_swap(
        self,
        operation_id: str,
        *,
        tenant_id: str,
        expected_status: OperationStatus,
        next_record: OperationLedgerRecord,
    ) -> OperationLedgerRecord: ...


class _TaskAdmissionPersistence(Protocol):
    async def admit(self, admission: TaskGraphAdmission, graph: TaskGraph) -> TaskGraphView: ...

    async def list_recoverable_page(
        self,
        *,
        cursor: "str | None",
        limit: int,
    ) -> Page[TaskGraphLaunch]: ...


class TaskPersistence(Protocol):
    tasks: _TaskRepository
    operations: _OperationRepository
    admissions: _TaskAdmissionPersistence


class DefaultTaskService(TaskApi):
    """Own durable Task submission, observation and cancellation semantics."""

    def __init__(
        self,
        persistence: TaskPersistence,
        authorization: AuthorizationPolicy,
        launcher: TaskGraphLauncher | None = None,
        *,
        release_terminal: _TaskReleaseCallback | None = None,
        local_waiter: "_LocalTaskWaiter | None" = None,
        preflight: "_TaskGraphPreflight | None" = None,
    ) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._launcher = launcher
        self._release_terminal = release_terminal or _no_release_terminal
        self._local_waiter = local_waiter
        self._preflight = preflight
        self._handoff_states: dict[tuple[str, str], _TaskHandoffState] = {}
        self._handoff_condition = asyncio.Condition()
        self._detached_finalizers: set[asyncio.Task[object]] = set()
        self._detached_finalizer_failure: AIError | None = None

    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult:
        return await self._run_graph(request, release_terminal=True)

    async def _run_graph(
        self,
        request: TaskGraphRequest,
        *,
        release_terminal: bool,
    ) -> TaskGraphResult:
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        graph_id = request.graph.graph_id
        tenant_id = request.principal.tenant_id
        async with self._graph_consumer(graph_id, tenant_id):
            await self._authorization.authorize(
                request.principal,
                AuthorizationAction.TASK_RUN,
                ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id),
            )
            if self._preflight is not None:
                self._preflight.validate_request(request.graph)
            admission = TaskGraphAdmission.from_request(request)
            view = await self._persistence.admissions.admit(admission, request.graph)
            if not _terminal(view.status):
                await self._arm_graph(admission.bind(request.graph))
            result = await self._result(view, tenant_id)
            if release_terminal and _terminal(result.status):
                await self._request_graph_release(graph_id, tenant_id)
            return result

    async def _arm_graph(self, launch: TaskGraphLaunch) -> None:
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        task = asyncio.create_task(
            self._launcher.start(launch),
            name=f"task-scheduler-arm-{launch.principal.tenant_id}-{launch.graph.graph_id}",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                self._detach_finalizer(
                    cast("asyncio.Task[object]", task),
                    launch.graph.graph_id,
                    label="task scheduler arm",
                )
                raise
            try:
                task.result()
            except asyncio.CancelledError as error:
                raise _task_service_failure(
                    error,
                    phase="task_scheduler_arm",
                    graph_id=launch.graph.graph_id,
                    durable_admitted=True,
                ) from error
            except BaseException as error:  # noqa: BLE001
                raise _task_service_failure(
                    error,
                    phase="task_scheduler_arm",
                    graph_id=launch.graph.graph_id,
                    durable_admitted=True,
                ) from error
            raise
        except BaseException as error:  # noqa: BLE001
            raise _task_service_failure(
                error,
                phase="task_scheduler_arm",
                graph_id=launch.graph.graph_id,
                durable_admitted=True,
            ) from error

    async def recover_pending(self) -> None:
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if self._preflight is not None:
            cursor: str | None = None
            while True:
                page = await self._persistence.admissions.list_recoverable_page(
                    cursor=cursor,
                    limit=128,
                )
                for launch in page.items:
                    self._preflight.validate_recovery(launch.graph)
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
        cursor = None
        recovered = 0
        while True:
            page = await self._persistence.admissions.list_recoverable_page(
                cursor=cursor,
                limit=128,
            )
            for launch in page.items:
                if self._preflight is not None:
                    self._preflight.validate_recovery(launch.graph)
                view = await self._persistence.tasks.reconcile_graph(
                    launch.graph.graph_id,
                    tenant_id=launch.principal.tenant_id,
                )
                if _terminal(view.status):
                    continue
                await self._arm_graph(launch)
                recovered += 1
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        _logger.info("task graph recovery scan completed: graphs=%s", recovered)

    async def run_graph_and_wait(
        self,
        request: TaskGraphRequest,
        *,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult:
        submitted = await self._run_graph(request, release_terminal=False)
        return await self.wait_graph(
            submitted.graph_id,
            principal=request.principal,
            timeout_seconds=timeout_seconds,
        )

    async def inspect_graph(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> TaskGraphView:
        async with self._graph_consumer(graph_id, principal.tenant_id):
            header = await self._persistence.tasks.get_header(
                graph_id,
                tenant_id=principal.tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                principal,
                AuthorizationAction.TASK_READ,
                header,
            )
            view = await self._persistence.tasks.get_graph(
                graph_id,
                tenant_id=principal.tenant_id,
            )
            if view is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return view

    async def inspect_graph_state(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> TaskGraphSnapshot:
        async with self._graph_consumer(graph_id, principal.tenant_id):
            header = await self._persistence.tasks.get_header(
                graph_id,
                tenant_id=principal.tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                principal,
                AuthorizationAction.TASK_READ,
                header,
            )
            snapshot = await self._persistence.tasks.snapshot_graph(
                graph_id,
                tenant_id=principal.tenant_id,
            )
            if snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return snapshot

    async def list_graph_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> Page[TaskEvent]:
        _validate_event_window(after_sequence, limit)
        tenant_id = principal.tenant_id
        async with self._graph_consumer(graph_id, tenant_id):
            header = await self._persistence.tasks.get_header(
                graph_id,
                tenant_id=tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                principal,
                AuthorizationAction.TASK_READ,
                header,
            )
            page = await self._persistence.tasks.list_events(
                graph_id,
                tenant_id=tenant_id,
                after_sequence=after_sequence,
                limit=limit,
            )
            _validate_event_page(graph_id, after_sequence, page)
            return page

    def stream_graph_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[TaskEvent]:
        return self._observe_graph_events(
            graph_id,
            principal=principal,
            after_sequence=after_sequence,
        )

    async def _observe_graph_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int,
    ) -> AsyncIterator[TaskEvent]:
        _validate_event_window(after_sequence, _TASK_EVENT_READ_LIMIT)
        tenant_id = principal.tenant_id
        cursor = after_sequence
        async with self._graph_consumer(graph_id, tenant_id):
            header = await self._persistence.tasks.get_header(
                graph_id,
                tenant_id=tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                principal,
                AuthorizationAction.TASK_READ,
                header,
            )
            while True:
                generation = self._local_activity_generation(
                    graph_id,
                    tenant_id=tenant_id,
                )
                page = await self._persistence.tasks.list_events(
                    graph_id,
                    tenant_id=tenant_id,
                    after_sequence=cursor,
                    limit=_TASK_EVENT_READ_LIMIT,
                )
                _validate_event_page(graph_id, cursor, page)
                if page.items:
                    for event in page.items:
                        cursor = event.sequence
                        yield event
                    continue
                snapshot = await self._persistence.tasks.snapshot_graph(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if snapshot is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if _terminal(snapshot.status):
                    snapshot, generation = await self._stabilize_local_terminal(
                        snapshot,
                        tenant_id=tenant_id,
                        generation=generation,
                    )
                    trailing = await self._persistence.tasks.list_events(
                        graph_id,
                        tenant_id=tenant_id,
                        after_sequence=cursor,
                        limit=_TASK_EVENT_READ_LIMIT,
                    )
                    _validate_event_page(graph_id, cursor, trailing)
                    if trailing.items:
                        for event in trailing.items:
                            cursor = event.sequence
                            yield event
                        continue
                    await self._request_graph_release(graph_id, tenant_id)
                    return
                await self._wait_observation_opportunity(
                    snapshot,
                    tenant_id=tenant_id,
                    after_generation=generation,
                )

    async def _observe_graph(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[TaskGraphSnapshot]:
        tenant_id = principal.tenant_id
        generation = self._local_activity_generation(graph_id, tenant_id=tenant_id)
        async with self._graph_consumer(graph_id, tenant_id):
            header = await self._persistence.tasks.get_header(
                graph_id,
                tenant_id=tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                principal,
                AuthorizationAction.TASK_READ,
                header,
            )
            snapshot = await self._persistence.tasks.snapshot_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            if snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if _terminal(snapshot.status):
                await self._request_graph_release(graph_id, tenant_id)
        fingerprint = _observation_fingerprint(snapshot)
        yield snapshot
        if _terminal(snapshot.status):
            return
        while True:
            try:
                await self._wait_observation_opportunity(
                    snapshot,
                    tenant_id=tenant_id,
                    after_generation=generation,
                )
            except AIError:
                latest = await self._snapshot_with_terminal_release(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if _terminal(latest.status):
                    latest_fingerprint = _observation_fingerprint(latest)
                    if latest_fingerprint != fingerprint or not _terminal(snapshot.status):
                        yield latest
                    return
                raise
            generation = self._local_activity_generation(
                graph_id,
                tenant_id=tenant_id,
            )
            latest = await self._snapshot_with_terminal_release(
                graph_id,
                tenant_id=tenant_id,
            )
            latest, generation = await self._stabilize_local_terminal(
                latest,
                tenant_id=tenant_id,
                generation=generation,
            )
            latest_fingerprint = _observation_fingerprint(latest)
            if latest_fingerprint != fingerprint or _terminal(latest.status):
                yield latest
                fingerprint = latest_fingerprint
            if _terminal(latest.status):
                return
            snapshot = latest

    async def _snapshot_with_terminal_release(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphSnapshot:
        async with self._graph_consumer(graph_id, tenant_id):
            snapshot = await self._persistence.tasks.snapshot_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            if snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if _terminal(snapshot.status) and (
                self._local_waiter is None
                or not self._local_waiter.owns_graph(graph_id, tenant_id=tenant_id)
            ):
                await self._request_graph_release(graph_id, tenant_id)
            return snapshot

    async def _stabilize_local_terminal(
        self,
        snapshot: TaskGraphSnapshot,
        *,
        tenant_id: str,
        generation: int | None,
    ) -> tuple[TaskGraphSnapshot, int | None]:
        waiter = self._local_waiter
        while (
            _terminal(snapshot.status)
            and waiter is not None
            and waiter.owns_graph(snapshot.graph_id, tenant_id=tenant_id)
        ):
            try:
                await self._wait_observation_opportunity(
                    snapshot,
                    tenant_id=tenant_id,
                    after_generation=generation,
                )
            except AIError:
                latest = await self._snapshot_with_terminal_release(
                    snapshot.graph_id,
                    tenant_id=tenant_id,
                )
                if _terminal(latest.status):
                    return (
                        latest,
                        self._local_activity_generation(
                            snapshot.graph_id,
                            tenant_id=tenant_id,
                        ),
                    )
                raise
            generation = self._local_activity_generation(
                snapshot.graph_id,
                tenant_id=tenant_id,
            )
            snapshot = await self._snapshot_with_terminal_release(
                snapshot.graph_id,
                tenant_id=tenant_id,
            )
        return snapshot, generation

    def _local_activity_generation(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> int | None:
        waiter = self._local_waiter
        if waiter is None or not waiter.owns_graph(graph_id, tenant_id=tenant_id):
            return None
        return waiter.graph_activity_generation(graph_id, tenant_id=tenant_id)

    async def _wait_observation_opportunity(
        self,
        snapshot: TaskGraphSnapshot,
        *,
        tenant_id: str,
        after_generation: int | None,
    ) -> None:
        waiter = self._local_waiter
        if waiter is None or not waiter.owns_graph(snapshot.graph_id, tenant_id=tenant_id):
            await asyncio.sleep(_GRAPH_OBSERVATION_RECHECK_SECONDS)
            return
        if after_generation is None:
            await waiter.wait_graph_activity(
                snapshot.graph_id,
                tenant_id=tenant_id,
            )
            return
        timeout = _nearest_live_lease_delay(snapshot)
        if timeout is None:
            await waiter.wait_graph_activity(
                snapshot.graph_id,
                tenant_id=tenant_id,
                after_generation=after_generation,
            )
            return
        if timeout <= 0:
            return
        try:
            await asyncio.wait_for(
                waiter.wait_graph_activity(
                    snapshot.graph_id,
                    tenant_id=tenant_id,
                    after_generation=after_generation,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return

    async def wait_graph(
        self,
        graph_id: str,
        *,
        principal: Principal,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def consume() -> TaskGraphResult:
            terminal: TaskGraphSnapshot | None = None
            async for snapshot in self._observe_graph(graph_id, principal=principal):
                if _terminal(snapshot.status):
                    terminal = snapshot
                    break
            if terminal is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return _snapshot_result(terminal)

        try:
            if timeout_seconds is None:
                return await consume()
            return await asyncio.wait_for(consume(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise AIError(
                ErrorCode.TASK_WAIT_TIMEOUT,
                safe_details={"graph_id": graph_id},
            ) from error

    async def cancel_graph(
        self,
        graph_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        async with self._graph_consumer(graph_id, tenant_id):
            header = await self._persistence.tasks.get_header(
                graph_id,
                tenant_id=tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                request.principal,
                AuthorizationAction.TASK_CANCEL,
                header,
            )
            request_digest = canonical_sha256(
                {
                    "action": "task.cancel",
                    "principal": principal_identity_payload(request.principal),
                    "graph_id": graph_id,
                    "force": request.force,
                }
            )
            finalizer = asyncio.create_task(
                self._cancel_finalizer_with_handoff(
                    graph_id,
                    request,
                    idempotency_key_digest(request.idempotency_key),
                    request_digest,
                ),
                name=f"task-cancel-finalizer-{tenant_id}-{graph_id}",
            )
            try:
                return await asyncio.shield(finalizer)
            except asyncio.CancelledError:
                if finalizer.done():
                    try:
                        return finalizer.result()
                    except asyncio.CancelledError:
                        raise
                    except BaseException:  # noqa: BLE001
                        raise
                self._detach_finalizer(
                    cast("asyncio.Task[object]", finalizer),
                    graph_id,
                )
                raise

    async def _cancel_finalizer_with_handoff(
        self,
        graph_id: str,
        request: CancelGraphRequest,
        operation_id: str,
        request_digest: str,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        async with self._graph_consumer(graph_id, tenant_id):
            view = await self._cancel_finalizer(
                graph_id,
                request,
                operation_id,
                request_digest,
            )
            await self._request_graph_release(graph_id, tenant_id)
            return view

    @asynccontextmanager
    async def _graph_consumer(self, graph_id: str, tenant_id: str):
        key = (tenant_id, graph_id)
        async with self._handoff_condition:
            while True:
                state = self._handoff_states.get(key)
                if state is None:
                    state = _TaskHandoffState()
                    self._handoff_states[key] = state
                if not state.release_in_progress:
                    state.active_consumers += 1
                    break
                await self._handoff_condition.wait()
        cleanup_owner = False
        try:
            yield
        finally:
            async with self._handoff_condition:
                state.active_consumers -= 1
                if state.active_consumers < 0:
                    raise RuntimeError("task graph consumer count became negative")
                if state.active_consumers == 0:
                    if state.release_requested and not state.release_in_progress:
                        state.release_in_progress = True
                        cleanup_owner = True
                    elif (
                        not state.release_requested
                        and self._handoff_states.get(key) is state
                    ):
                        self._handoff_states.pop(key, None)
                self._handoff_condition.notify_all()
            if cleanup_owner:
                cleanup_succeeded = False
                cleanup_error: BaseException | None = None
                try:
                    await self._release_terminal(
                        graph_id,
                        tenant_id=tenant_id,
                    )
                    cleanup_succeeded = True
                except BaseException as error:  # noqa: BLE001
                    cleanup_error = error
                    if isinstance(error, Exception):
                        _logger.error(
                            "task graph transient handoff cleanup failed: graph=%s",
                            graph_id,
                            exc_info=environ.debug,
                        )
                async with self._handoff_condition:
                    if self._handoff_states.get(key) is state:
                        if cleanup_succeeded and state.active_consumers == 0:
                            self._handoff_states.pop(key, None)
                        else:
                            state.release_in_progress = False
                            state.release_requested = True
                    self._handoff_condition.notify_all()
                if cleanup_error is not None and not isinstance(
                    cleanup_error,
                    Exception,
                ):
                    raise cleanup_error

    async def _request_graph_release(
        self,
        graph_id: str,
        tenant_id: str,
    ) -> None:
        async with self._handoff_condition:
            state = self._handoff_states.get((tenant_id, graph_id))
            if state is None:
                raise RuntimeError("task graph release requested without consumer")
            state.release_requested = True
            self._handoff_condition.notify_all()

    async def _claim_cancel_operation(
        self,
        operation_id: str,
        tenant_id: str,
        graph_id: str,
        request_digest: str,
    ) -> tuple[bool, OperationLedgerRecord]:
        operation = await self._persistence.operations.get(
            operation_id,
            tenant_id=tenant_id,
        )
        if operation is None:
            now = datetime.now(timezone.utc)
            pending = OperationLedgerInput(
                operation_id,
                tenant_id,
                ResourceKind.TASK_GRAPH,
                graph_id,
                None,
                OperationKind.TASK_CANCEL,
                OperationStatus.PENDING,
                request_digest,
                None,
                None,
                None,
                True,
                now,
                now,
            )
            try:
                operation = await self._persistence.operations.append(pending)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                operation = await self._persistence.operations.get(
                    operation_id,
                    tenant_id=tenant_id,
                )
                if operation is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        if operation.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if operation.status is OperationStatus.PENDING:
            running = replace(
                operation,
                status=OperationStatus.RUNNING,
                updated_at=datetime.now(timezone.utc),
            )
            try:
                claimed = await self._persistence.operations.compare_and_swap(
                    operation_id,
                    tenant_id=tenant_id,
                    expected_status=OperationStatus.PENDING,
                    next_record=running,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                current = await self._persistence.operations.get(
                    operation_id,
                    tenant_id=tenant_id,
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if current.request_digest != request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                if current.status is OperationStatus.PENDING:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
                operation = current
            else:
                return True, claimed
        if operation.status is OperationStatus.SUCCEEDED:
            return False, operation
        if operation.status is OperationStatus.FAILED:
            raise _stable_operation_error(operation.error_code)
        if operation.status in {
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
        }:
            return False, operation
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _cancel_finalizer(
        self,
        graph_id: str,
        request: CancelGraphRequest,
        operation_id: str,
        request_digest: str,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        claimed, operation = await self._claim_cancel_operation(
            operation_id,
            tenant_id,
            graph_id,
            request_digest,
        )
        try:
            view = await self._persistence.tasks.get_graph(
                graph_id,
                tenant_id=tenant_id,
            )
        except BaseException as error:  # noqa: BLE001
            if claimed or operation.status in {
                OperationStatus.RUNNING,
                OperationStatus.EFFECT_UNKNOWN,
            }:
                return await self._settle_cancel_error(
                    operation,
                    graph_id,
                    request,
                    error,
                )
            raise
        if view is None:
            if claimed or operation.status in {
                OperationStatus.RUNNING,
                OperationStatus.EFFECT_UNKNOWN,
            }:
                return await self._settle_cancel_error(
                    operation,
                    graph_id,
                    request,
                    AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if claimed:
            try:
                view = await self._persistence.tasks.cancel_graph(
                    graph_id,
                    tenant_id=tenant_id,
                )
                view = await self._persistence.tasks.get_graph(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if view is None or not _terminal(view.status):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            except BaseException as error:  # noqa: BLE001
                return await self._settle_cancel_error(
                    operation,
                    graph_id,
                    request,
                    error,
                )
        elif operation.status in {
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
        }:
            if not _terminal(view.status):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        elif not _terminal(view.status):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not _terminal(view.status):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            claimed
            or operation.status
            in {OperationStatus.RUNNING, OperationStatus.EFFECT_UNKNOWN}
        ) and view.status is TaskStatus.CANCELLED:
            try:
                await self._cleanup_cancelled_graph(graph_id, request)
            except BaseException as error:  # noqa: BLE001
                await self._raise_cancel_cleanup_error(
                    operation,
                    tenant_id,
                    graph_id,
                    error,
                )
        if claimed or operation.status in {
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
        }:
            current = await self._record_success(
                operation,
                tenant_id,
                view,
                expected_status=operation.status,
            )
            if current.status is not OperationStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.info(
            "task graph cancel settled: tenant=%s graph=%s status=%s",
            tenant_id,
            graph_id,
            view.status.value,
        )
        return view

    async def _settle_cancel_error(
        self,
        operation: OperationLedgerRecord,
        graph_id: str,
        request: CancelGraphRequest,
        error: BaseException,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        try:
            view = await self._persistence.tasks.get_graph(
                graph_id,
                tenant_id=tenant_id,
            )
        except Exception as reload_error:
            await self._record_effect_unknown(operation, tenant_id)
            if isinstance(error, AIError):
                raise error
            raise AIError(
                ErrorCode.INTERNAL_ERROR,
                safe_details={"phase": "task_cancel_readback"},
            ) from reload_error
        if view is None:
            await self._record_failure(
                operation,
                tenant_id,
                ErrorCode.STORAGE_INTEGRITY_ERROR.value,
            )
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if _terminal(view.status):
            if view.status is TaskStatus.CANCELLED:
                try:
                    await self._cleanup_cancelled_graph(graph_id, request)
                except BaseException as cleanup_error:  # noqa: BLE001
                    await self._raise_cancel_cleanup_error(
                        operation,
                        tenant_id,
                        graph_id,
                        cleanup_error,
                    )
            current = await self._record_success(
                operation,
                tenant_id,
                view,
                expected_status=operation.status,
            )
            if current.status is not OperationStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return view
        if isinstance(error, AIError):
            await self._record_failure(
                operation,
                tenant_id,
                error.code.value,
            )
            raise error
        await self._record_effect_unknown(operation, tenant_id)
        raise AIError(
            ErrorCode.INTERNAL_ERROR,
            safe_details={"phase": "task_cancel"},
        ) from error

    async def _raise_cancel_cleanup_error(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
        graph_id: str,
        error: BaseException,
    ) -> None:
        await self._record_effect_unknown(operation, tenant_id)
        if isinstance(error, asyncio.CancelledError):
            raise error
        if isinstance(error, AIError):
            raise error
        raise AIError(
            ErrorCode.STORAGE_RECOVERY_REQUIRED,
            safe_details={"phase": "task_cancel_cleanup", "graph_id": graph_id},
        ) from error

    async def _cleanup_cancelled_graph(
        self,
        graph_id: str,
        request: CancelGraphRequest,
    ) -> None:
        if self._launcher is None:
            return
        try:
            await self._launcher.cancel(graph_id, request)
        except asyncio.CancelledError:
            raise
        except AIError:
            raise
        except BaseException as error:  # noqa: BLE001
            raise AIError(
                ErrorCode.INTERNAL_ERROR,
                safe_details={"phase": "task_cancel_cleanup", "graph_id": graph_id},
            ) from error

    async def _record_success(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
        view: TaskGraphView,
        *,
        expected_status: OperationStatus = OperationStatus.RUNNING,
    ) -> OperationLedgerRecord:
        completed = replace(
            operation,
            status=OperationStatus.SUCCEEDED,
            result_ref=view.graph_id,
            result_digest=canonical_sha256(
                {"graph_id": view.graph_id, "status": view.status.value}
            ),
            error_code=None,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=expected_status,
                next_record=completed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(
                operation.operation_id,
                tenant_id=tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status is OperationStatus.SUCCEEDED:
                return current
            if current.status is OperationStatus.FAILED:
                raise _stable_operation_error(current.error_code)
            raise

    async def _record_failure(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
        error_code: str,
    ) -> OperationLedgerRecord:
        failed = replace(
            operation,
            status=OperationStatus.FAILED,
            error_code=error_code,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=operation.status,
                next_record=failed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(
                operation.operation_id,
                tenant_id=tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status is OperationStatus.FAILED:
                return current
            raise

    async def _record_effect_unknown(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
    ) -> OperationLedgerRecord:
        if operation.status is OperationStatus.EFFECT_UNKNOWN:
            return operation
        unknown = replace(
            operation,
            status=OperationStatus.EFFECT_UNKNOWN,
            error_code=None,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=operation.status,
                next_record=unknown,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(
                operation.operation_id,
                tenant_id=tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in {
                OperationStatus.EFFECT_UNKNOWN,
                OperationStatus.SUCCEEDED,
            }:
                return current
            raise

    async def _result(
        self,
        view: TaskGraphView,
        tenant_id: str,
    ) -> TaskGraphResult:
        nodes = await self._persistence.tasks.list_nodes(
            view.graph_id,
            tenant_id=tenant_id,
        )
        return _node_result(view.graph_id, view.status, nodes)

    def _detach_finalizer(
        self,
        task: "asyncio.Task[object]",
        graph_id: str,
        *,
        label: str = "task graph cancel finalizer",
    ) -> None:
        if task in self._detached_finalizers:
            return
        self._detached_finalizers.add(task)

        def consume(done: "asyncio.Task[object]") -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except BaseException as error:  # noqa: BLE001
                _logger.warning(
                    "%s failed after caller cancellation: graph=%s error=%s",
                    label,
                    graph_id,
                    type(error).__name__,
                )
                if self._detached_finalizer_failure is None:
                    self._detached_finalizer_failure = _task_service_failure(
                        error,
                        phase="task_service_finalizer",
                        graph_id=graph_id,
                    )
            finally:
                self._detached_finalizers.discard(done)

        task.add_done_callback(consume)

    async def drain_owned_finalizers(self) -> None:
        while True:
            pending = tuple(
                task for task in self._detached_finalizers if not task.done()
            )
            if not pending:
                await asyncio.sleep(0)
                return
            await asyncio.gather(
                *(asyncio.shield(task) for task in pending),
                return_exceptions=True,
            )

    async def preflight_close(self) -> None:
        pending = tuple(
            task for task in self._detached_finalizers if not task.done()
        )
        if pending:
            _logger.warning(
                "task service close blocked by detached finalizers: tasks=%s",
                len(pending),
            )
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "phase": "task_service_preflight_close",
                    "pending_finalizers": len(pending),
                },
            )
        failure = self._detached_finalizer_failure
        if failure is not None:
            raise AIError(
                failure.code,
                category=failure.category,
                retryable=failure.retryable,
                operation_id=failure.operation_id,
                safe_details=dict(failure.safe_details),
                diagnostics=failure.diagnostics,
            )


def _validate_event_window(after_sequence: int, limit: int) -> None:
    if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise AIError(ErrorCode.PAGE_LIMIT_INVALID)


def _validate_event_page(
    graph_id: str,
    after_sequence: int,
    page: Page[TaskEvent],
) -> None:
    expected = after_sequence
    for event in page.items:
        if event.graph_id != graph_id or event.sequence != expected + 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected = event.sequence


def _task_service_failure(
    error: BaseException,
    *,
    phase: str,
    graph_id: str,
    durable_admitted: bool = False,
) -> AIError:
    details = dict(error.safe_details) if isinstance(error, AIError) else {}
    details.setdefault("phase", phase)
    details.setdefault("graph_id", graph_id)
    if durable_admitted:
        details.setdefault("durable_admitted", True)
    if isinstance(error, AIError):
        return AIError(
            error.code,
            category=error.category,
            retryable=error.retryable,
            operation_id=error.operation_id,
            safe_details=details,
            diagnostics=error.diagnostics,
        )
    return AIError(
        ErrorCode.STORAGE_RECOVERY_REQUIRED,
        safe_details=details,
    )


def _terminal(status: TaskStatus) -> bool:
    return status in {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.BLOCKED,
    }


def _nearest_live_lease_delay(snapshot: TaskGraphSnapshot) -> "float | None":
    now = datetime.now(timezone.utc)
    expiries = tuple(
        state.lease_expires_at
        for state in snapshot.node_states
        if state.status is TaskStatus.RUNNING
        and state.lease_expires_at is not None
        and state.lease_expires_at > now
    )
    if not expiries:
        return None
    return max(0.0, (min(expiries) - now).total_seconds())


def _lease_phase(node: TaskNodeView, now: datetime) -> str:
    if node.lease_expires_at is None:
        return "none"
    if node.lease_expires_at <= now:
        return "expired"
    return "live"


def _observation_fingerprint(snapshot: TaskGraphSnapshot) -> str:
    now = datetime.now(timezone.utc)
    return canonical_sha256(
        {
            "status": snapshot.status.value,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "status": node.status.value,
                    "owner": node.owner,
                    "fence": node.fence,
                    "execution_id": node.execution_id,
                    "result_digest": node.result_digest,
                    "error_code": node.error_code,
                    "error_digest": node.error_digest,
                    "lease_phase": _lease_phase(node, now),
                }
                for node in snapshot.node_states
            ],
        }
    )


def _snapshot_result(snapshot: TaskGraphSnapshot) -> TaskGraphResult:
    return _node_result(snapshot.graph_id, snapshot.status, snapshot.node_states)


def _node_result(
    graph_id: str,
    status: TaskStatus,
    nodes: tuple[TaskNodeView, ...],
) -> TaskGraphResult:
    results = tuple(
        TaskNodeResult(
            node.node_id,
            node.status,
            node.result_digest,
            node.execution_id,
            node.error_code,
            node.error_digest,
        )
        for node in nodes
    )
    execution_ids = tuple(
        item.execution_id
        for item in results
        if item.status is TaskStatus.SUCCEEDED and item.execution_id is not None
    )
    return TaskGraphResult(graph_id, status, execution_ids, results)


def _stable_operation_error(error_code: "str | None") -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["DefaultTaskService", "TaskPersistence"]
