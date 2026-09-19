#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only Runtime composition for persisted execution history."""

import json
import heapq
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

from linktools.core import environ

from ..agent import AgentBindingSnapshot, restore_output
from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorSigner,
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    JsonValue,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    TenantAuthorizationPolicy,
    canonical_sha256,
    normalize_json_value,
    validate_page_limit,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..storage import ObjectStore, StoredPayload, read_object
from ..task import (
    TaskBindingSnapshot,
    TaskEvent,
    TaskGraphInfo,
    TaskResultRecord,
    TaskResultRef,
)
from ._artifact import DefaultArtifactService
from ._cursor import decode_cursor as decode_runtime_cursor
from ._cursor import encode_cursor as encode_runtime_cursor
from ._history import StepExecutionHistoryReader
from ._history_service import DefaultExecutionHistoryService
from ._runtime_identity import grant_key
from .service_api import (
    AttachmentFact,
    ArtifactService,
    ArtifactView,
    ExecutionEvent,
    ExecutionHistoryItem,
    ExecutionHistoryService,
    ExecutionResult,
    ExecutionTraceItem,
    ExecutionView,
    ListExecutionRequest,
    ModelInteractionItem,
    SessionView,
    TranscriptItem,
    UsageSummary,
)
from .state import RuntimeDomain, RuntimeState
from .state._contracts import (
    EventRepository,
    ExecutionRecord,
    ExecutionRepository,
    SessionRecord,
    SessionRepository,
    TaskRepository,
)

_logger = environ.get_logger("ai.runtime.history")


@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    """Execution metadata required by local diagnostics."""

    execution_id: str
    binding_kind: str
    agent_id: str | None
    task_type: str | None
    status: ExecutionStatus
    lineage_kind: ExecutionLineageKind
    parent_execution_id: str | None
    root_execution_id: str
    parent_invocation_id: str | None
    session_id: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    terminal_at: datetime | None
    binding_digest: str
    input_digest: str
    output_fingerprint: str
    output_digest: str | None
    usage: UsageSummary | None
    error_code: str | None
    safe_error_details: Mapping[str, JsonValue] = field(default_factory=dict)
    error_diagnostics: ErrorDiagnostics | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "safe_error_details", dict(self.safe_error_details))


def _merge_usage_summaries(
    values: "list[UsageSummary]",
    *,
    unrecorded_executions: int = 0,
) -> UsageSummary:
    transport_retries: int | None = 0
    cutoffs = []
    totals = {
        "logical_requests": 0,
        "succeeded_requests": 0,
        "failed_requests": 0,
        "cancelled_requests": 0,
        "output_correction_retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "model_duration_ns": 0,
        "unknown_usage_requests": 0,
    }
    for value in values:
        totals["logical_requests"] += value.logical_requests
        totals["succeeded_requests"] += value.succeeded_requests
        totals["failed_requests"] += value.failed_requests
        totals["cancelled_requests"] += value.cancelled_requests
        totals["output_correction_retries"] += value.output_correction_retries
        totals["input_tokens"] += value.input_tokens
        totals["output_tokens"] += value.output_tokens
        totals["cache_read_tokens"] += value.cache_read_tokens
        totals["cache_write_tokens"] += value.cache_write_tokens
        totals["model_duration_ns"] += value.model_duration_ns
        totals["unknown_usage_requests"] += value.unknown_usage_requests
        cutoffs.extend(value.cutoffs)
        if transport_retries is not None:
            if value.transport_retries is None:
                transport_retries = None
            else:
                transport_retries += value.transport_retries
        unrecorded_executions += value.unrecorded_executions
    return UsageSummary(
        **totals,
        transport_retries=transport_retries,
        unrecorded_executions=unrecorded_executions,
        cutoffs=tuple(cutoffs),
    )


def _project_session_view(record: SessionRecord) -> SessionView:
    return SessionView(
        session_id=record.session_id,
        agent_id=record.agent_id,
        status=record.status,
        revision=record.revision,
        cwd=record.cwd,
        active_execution_ids=(
            () if record.active_execution_id is None else (record.active_execution_id,)
        ),
        metadata=record.metadata,
        history_quality=record.history_quality,
    )


def _project_execution_info(
    record: ExecutionRecord,
    *,
    usage: UsageSummary | None = None,
) -> ExecutionInfo:
    result = record.result
    return ExecutionInfo(
        execution_id=record.execution_id,
        binding_kind=record.binding_kind,
        agent_id=record.agent_id,
        task_type=record.task_type,
        status=record.status,
        lineage_kind=record.lineage_kind,
        parent_execution_id=record.parent_execution_id,
        root_execution_id=record.root_execution_id,
        parent_invocation_id=record.parent_invocation_id,
        session_id=record.session_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        terminal_at=None if result is None else result.created_at,
        binding_digest=record.binding_digest,
        input_digest=record.stored_user_input.digest,
        output_fingerprint=_output_fingerprint(record),
        output_digest=(
            None
            if result is None or result.output is None
            else result.output.digest
        ),
        usage=usage,
        error_code=record.error_code,
        safe_error_details=record.safe_error_details,
        error_diagnostics=None,
    )


class RuntimeHistory:
    """Stable read-only composition for persisted execution projections."""

    def __init__(
        self,
        service: ExecutionHistoryService,
        *,
        tenant_id: str,
        executions: "ExecutionRepository | None" = None,
        events: "EventRepository | None" = None,
        sessions: "SessionRepository | None" = None,
        tasks: "TaskRepository | None" = None,
        authorization: "AuthorizationPolicy | None" = None,
        cursor_signer: "CursorSigner | None" = None,
        namespace: "str | None" = None,
        execution_objects: "ObjectStore | None" = None,
        task_objects: "ObjectStore | None" = None,
        artifacts: "ArtifactService | None" = None,
    ) -> None:
        self._service = service
        self._tenant_id = tenant_id
        self._executions = executions
        self._events = events
        self._sessions = sessions
        self._tasks = tasks
        self._authorization = authorization
        self._cursor_signer = cursor_signer
        self._namespace = (
            None if namespace is None else validate_persistence_namespace(namespace)
        )
        self._execution_objects = execution_objects
        self._task_objects = task_objects
        self._artifacts = artifacts

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    async def inspect_execution(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionInfo:
        record = await self._authorized_record(execution_id, principal)
        return _project_execution_info(
            record,
            usage=await self._service.usage(
                execution_id,
                principal=principal,
            ),
        )

    async def result(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult:
        record = await self._authorized_record(execution_id, principal)
        if record.status is ExecutionStatus.RECOVERY_REQUIRED:
            if record.error_code is None or record.error_diagnostics is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                code = ErrorCode(record.error_code)
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            raise AIError(code, safe_details=record.safe_error_details)
        if record.status not in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            raise AIError(ErrorCode.EXECUTION_NOT_READY)

        executions, _authorization = self._require_direct_reader()
        stored = await executions.get_result(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if record.status is not ExecutionStatus.SUCCEEDED:
            if stored.output is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            error_code = _terminal_error_code(record)
            return ExecutionResult(
                record.execution_id,
                record.status,
                None,
                None,
                stored.usage,
                error_code,
                record.safe_error_details,
                None,
            )

        if stored.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        output = await self._read_payload(
            stored.output,
            self._execution_objects,
        )
        return ExecutionResult(
            record.execution_id,
            record.status,
            output,
            _output_fingerprint(record),
            stored.usage,
        )

    async def task_result_ref(
        self,
        graph_id: str,
        node_id: str,
        *,
        principal: Principal,
    ) -> TaskResultRef:
        record = await self._task_result_record(
            graph_id,
            node_id,
            principal=principal,
        )
        if self._namespace is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return TaskResultRef(
            self._namespace,
            principal.tenant_id,
            graph_id,
            node_id,
            record.result_digest,
        )

    async def task_result(
        self,
        graph_id: str,
        node_id: str,
        *,
        principal: Principal,
    ) -> JsonValue:
        record = await self._task_result_record(
            graph_id,
            node_id,
            principal=principal,
        )
        if record.execution_id is not None:
            execution = await self.result(
                record.execution_id,
                principal=principal,
            )
            if execution.status is not ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            output = execution.output
        elif record.payload is not None:
            output = await self._read_payload(record.payload, self._task_objects)
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if canonical_sha256(output) != record.result_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return output

    async def artifacts(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[ArtifactView]:
        if self._artifacts is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._artifacts.list(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
        )

    async def recent_executions(
        self,
        *,
        principal: Principal,
        limit: int = 20,
    ) -> tuple[ExecutionInfo, ...]:
        """Return the exact newest executions with O(limit) memory."""
        validate_page_limit(limit)
        executions, authorization = self._require_direct_reader()
        recent: list[tuple[datetime, str, ExecutionInfo]] = []
        cursor: str | None = None

        while True:
            page = await executions.list_candidates(
                tenant_id=principal.tenant_id,
                session_id=None,
                parent_execution_id=None,
                cursor=cursor,
                limit=1000,
            )
            if not page.items:
                break
            for candidate in page.items:
                record = candidate.record
                resource = ResourceRef(
                    ResourceKind.EXECUTION,
                    record.execution_id,
                    principal.tenant_id,
                )
                try:
                    await authorization.authorize(
                        principal,
                        AuthorizationAction.EXECUTION_READ,
                        resource,
                    )
                except AIError as error:
                    if error.code is ErrorCode.AUTHORIZATION_DENIED:
                        continue
                    raise
                info = _project_execution_info(record)
                entry = (record.created_at, record.execution_id, info)
                if len(recent) < limit:
                    heapq.heappush(recent, entry)
                elif entry[:2] > recent[0][:2]:
                    heapq.heapreplace(recent, entry)
            cursor = page.items[-1].cursor
            if not page.has_more:
                break

        recent.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return tuple(value[2] for value in recent)

    async def list_executions(
        self, request: ListExecutionRequest
    ) -> Page[ExecutionView]:
        return await self._service.list(request)

    async def task_graph(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> TaskGraphInfo:
        tasks, authorization = self._require_task_reader()
        header = await tasks.get_header(graph_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal,
            AuthorizationAction.TASK_READ,
            header,
        )
        snapshot = await tasks.snapshot_graph(
            graph_id,
            tenant_id=principal.tenant_id,
        )
        if snapshot is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return TaskGraphInfo.from_snapshot(snapshot)

    async def task_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> Page[TaskEvent]:
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > 1000
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        tasks, authorization = self._require_task_reader()
        header = await tasks.get_header(graph_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal,
            AuthorizationAction.TASK_READ,
            header,
        )
        return await tasks.list_events(
            graph_id,
            tenant_id=principal.tenant_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    async def list_events(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[ExecutionEvent]:
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        validate_page_limit(limit)
        record = await self._authorized_record(execution_id, principal)
        events, signer = self._require_event_reader()
        if cursor is None:
            high_water = record.event_sequence
            after_sequence = 0
        else:
            payload = decode_runtime_cursor(
                cursor,
                signer,
                tenant_id=record.tenant_id,
                resource_kind="EXECUTION_EVENTS",
                filter_digest=canonical_sha256(
                    {
                        "execution_id": execution_id,
                        "include_content": include_content,
                    }
                ),
            )
            if payload.revision != 0:
                raise AIError(ErrorCode.CURSOR_INVALID)
            try:
                high_water_raw, after_raw = payload.position.split(":", 1)
                high_water = int(high_water_raw)
                after_sequence = int(after_raw)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.CURSOR_INVALID) from error
            if (
                high_water < 0
                or after_sequence < 0
                or after_sequence > high_water
            ):
                raise AIError(ErrorCode.CURSOR_INVALID)
        if after_sequence >= high_water:
            return Page((), None)
        page_limit = min(limit, high_water - after_sequence)
        page = await events.list(
            execution_id,
            tenant_id=record.tenant_id,
            after_sequence=after_sequence,
            limit=page_limit,
        )
        expected = after_sequence
        projected: list[ExecutionEvent] = []
        for event in page.items:
            expected += 1
            if (
                event.execution_id != execution_id
                or event.sequence != expected
                or event.sequence > high_water
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            projected.append(
                ExecutionEvent(
                    event.execution_id,
                    event.sequence,
                    event.event_type,
                    event.payload if include_content else {},
                )
            )
        if len(projected) != page_limit:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        next_after = after_sequence + len(projected)
        next_cursor = (
            None
            if next_after >= high_water
            else encode_runtime_cursor(
                signer,
                tenant_id=record.tenant_id,
                resource_kind="EXECUTION_EVENTS",
                filter_digest=canonical_sha256(
                    {
                        "execution_id": execution_id,
                        "include_content": include_content,
                    }
                ),
                position=f"{high_water}:{next_after}",
            )
        )
        return Page(tuple(projected), next_cursor)

    async def attachment_facts(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[AttachmentFact]:
        return await self._service.attachment_facts(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
        )

    async def usage(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> UsageSummary:
        return await self._service.usage(
            execution_id,
            principal=principal,
        )

    async def graph_usage(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> UsageSummary:
        graph = await self.task_graph(
            graph_id,
            principal=principal,
        )
        executions, _authorization = self._require_direct_reader()
        roots = tuple(sorted({
            state.execution_id
            for state in graph.node_states
            if state.execution_id is not None
        }))
        unrecorded = sum(
            1
            for state in graph.node_states
            if state.execution_id is None
            and state.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}
        )

        pending = list(roots)
        seen: set[str] = set()
        values: list[UsageSummary] = []
        while pending:
            execution_id = pending.pop(0)
            if execution_id in seen:
                continue
            record = await executions.get(
                execution_id,
                tenant_id=principal.tenant_id,
            )
            if record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            seen.add(execution_id)
            values.append(
                await self._service.usage(
                    execution_id,
                    principal=principal,
                )
            )
            children = await executions.list_children(
                execution_id,
                tenant_id=principal.tenant_id,
            )
            for child in children:
                if (
                    child.parent_execution_id != execution_id
                    or child.tenant_id != principal.tenant_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if child.execution_id not in seen:
                    pending.append(child.execution_id)

        return _merge_usage_summaries(
            values,
            unrecorded_executions=unrecorded,
        )

    async def recent_sessions(
        self,
        *,
        principal: Principal,
        limit: int = 20,
    ) -> tuple[SessionView, ...]:
        validate_page_limit(limit)
        sessions, authorization = self._require_session_reader()
        recent: list[tuple[datetime, str, SessionView]] = []
        cursor: str | None = None
        snapshot: int | None = None
        while True:
            snapshot, page = await sessions.list_page(
                tenant_id=principal.tenant_id,
                owner_principal_id=principal.principal_id,
                cursor=cursor,
                limit=1000,
                snapshot=snapshot,
            )
            for record in page.items:
                resource = ResourceRef(
                    ResourceKind.SESSION,
                    record.session_id,
                    record.tenant_id,
                    record.owner_principal_id,
                )
                try:
                    await authorization.authorize(
                        principal,
                        AuthorizationAction.SESSION_READ,
                        resource,
                    )
                except AIError as error:
                    if error.code is ErrorCode.AUTHORIZATION_DENIED:
                        continue
                    raise
                info = _project_session_view(record)
                entry = (record.updated_at, record.session_id, info)
                if len(recent) < limit:
                    heapq.heappush(recent, entry)
                elif entry[:2] > recent[0][:2]:
                    heapq.heapreplace(recent, entry)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        recent.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return tuple(value[2] for value in recent)

    async def inspect_session(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> SessionView:
        sessions, authorization = self._require_session_reader()
        header = await sessions.get_header(
            session_id,
            tenant_id=principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal,
            AuthorizationAction.SESSION_READ,
            header,
        )
        record = await sessions.get(
            session_id,
            tenant_id=principal.tenant_id,
        )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return _project_session_view(record)

    @classmethod
    def open(
        cls,
        namespace: str,
        *,
        state: RuntimeState,
        tenant_id: "str | None" = None,
        authorization: "AuthorizationPolicy | None" = None,
    ) -> AbstractAsyncContextManager["RuntimeHistory"]:
        return _open_runtime_history(
            namespace,
            state=state,
            tenant_id=tenant_id,
            authorization=authorization,
        )

    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[ExecutionHistoryItem]:
        return await self._service.history(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
        )

    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[ExecutionTraceItem]:
        return await self._service.trace(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
        )

    async def transcript(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[TranscriptItem]:
        return await self._service.transcript(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
        )

    async def model_interactions(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[ModelInteractionItem]:
        return await self._service.model_interactions(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
        )

    def _require_session_reader(
        self,
    ) -> tuple[SessionRepository, AuthorizationPolicy]:
        if self._sessions is None or self._authorization is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._sessions, self._authorization

    def _require_direct_reader(
        self,
    ) -> tuple[ExecutionRepository, AuthorizationPolicy]:
        if self._executions is None or self._authorization is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._executions, self._authorization

    def _require_event_reader(
        self,
    ) -> tuple[EventRepository, CursorSigner]:
        if self._events is None or self._cursor_signer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._events, self._cursor_signer

    def _require_task_reader(
        self,
    ) -> tuple[TaskRepository, AuthorizationPolicy]:
        if self._tasks is None or self._authorization is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._tasks, self._authorization

    async def _task_result_record(
        self,
        graph_id: str,
        node_id: str,
        *,
        principal: Principal,
    ) -> TaskResultRecord:
        graph = await self.task_graph(graph_id, principal=principal)
        state = next(
            (value for value in graph.node_states if value.node_id == node_id),
            None,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if state.status in {
            TaskStatus.PENDING,
            TaskStatus.READY,
            TaskStatus.RUNNING,
            TaskStatus.WAITING,
            TaskStatus.RECOVERY_REQUIRED,
        }:
            raise AIError(ErrorCode.TASK_NOT_READY)
        if state.status in {
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }:
            details: dict[str, JsonValue] = {
                "graph_id": graph_id,
                "node_id": node_id,
                "status": state.status.value,
            }
            if state.error_code is not None:
                details["error_code"] = state.error_code
            raise AIError(ErrorCode.TASK_NODE_FAILED, safe_details=details)
        if state.status is not TaskStatus.SUCCEEDED or state.result_digest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        tasks, _authorization = self._require_task_reader()
        records = await tasks.get_results(
            graph_id,
            (node_id,),
            tenant_id=principal.tenant_id,
        )
        record = records.get(node_id)
        if record is None or record.result_digest != state.result_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return record

    async def _read_payload(
        self,
        payload: StoredPayload,
        objects: "ObjectStore | None",
    ) -> JsonValue:
        try:
            if payload.kind == "inline":
                value = payload.decode()
            else:
                if objects is None or payload.ref is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                raw = await read_object(
                    objects,
                    payload.ref.key,
                    expected_digest=payload.ref.digest,
                    expected_size=payload.ref.size,
                )
                value = json.loads(raw.decode("utf-8"))
            return normalize_json_value(value)
        except AIError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def _authorized_record(
        self,
        execution_id: str,
        principal: Principal,
    ) -> ExecutionRecord:
        executions, authorization = self._require_direct_reader()
        header = await executions.get_header(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal,
            AuthorizationAction.EXECUTION_READ,
            header,
        )
        record = await executions.get(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


@asynccontextmanager
async def _open_runtime_history(
    namespace: str,
    *,
    state: RuntimeState,
    tenant_id: "str | None",
    authorization: "AuthorizationPolicy | None",
) -> AsyncIterator[RuntimeHistory]:
    resolved_namespace = validate_persistence_namespace(namespace)
    effective_tenant_id = (
        "default" if tenant_id is None else validate_tenant_id(tenant_id)
    )
    if not isinstance(state, RuntimeState):
        raise TypeError("state must be RuntimeState")
    selected_state = state
    initialized = False
    body_error: BaseException | None = None
    try:
        await selected_state.initialize(
            namespace=resolved_namespace,
            tenant_id=effective_tenant_id,
            read_only=True,
        )
        initialized = True
        if (
            selected_state.namespace != resolved_namespace
            or selected_state.tenant_id != effective_tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        reader = StepExecutionHistoryReader(
            namespace=resolved_namespace,
            executions=selected_state.execution.executions,
            store=selected_state.steps.read_store(RuntimeDomain.EXECUTION),
            cursor_signer=HmacCursorSigner(
                "execution-history",
                grant_key(resolved_namespace),
            ),
        )
        effective_authorization = (
            TenantAuthorizationPolicy(effective_tenant_id)
            if authorization is None
            else authorization
        )
        service = DefaultExecutionHistoryService(
            selected_state.execution.executions,
            effective_authorization,
            reader,
            cursor_signer=HmacCursorSigner("execution", grant_key(resolved_namespace)),
        )
        artifacts = DefaultArtifactService(
            selected_state.artifact,
            effective_authorization,
            grant_key=grant_key(resolved_namespace),
            cursor_signer=HmacCursorSigner("artifact", grant_key(resolved_namespace)),
        )
        yield RuntimeHistory(
            service,
            tenant_id=effective_tenant_id,
            executions=selected_state.execution.executions,
            events=selected_state.execution.events,
            sessions=selected_state.conversation.sessions,
            tasks=selected_state.task.tasks,
            authorization=effective_authorization,
            namespace=resolved_namespace,
            execution_objects=selected_state.object_store(RuntimeDomain.EXECUTION),
            task_objects=selected_state.object_store(RuntimeDomain.TASK),
            artifacts=artifacts,
            cursor_signer=HmacCursorSigner(
                "runtime-history",
                grant_key(resolved_namespace),
            ),
        )
    except BaseException as error:
        body_error = error
        raise
    finally:
        if initialized:
            try:
                await selected_state.close()
            except BaseException as error:
                if body_error is None:
                    raise
                _log_secondary_cleanup("history.close", error)


def _log_secondary_cleanup(phase: str, error: BaseException) -> None:
    code = error.code.value if isinstance(error, AIError) else None
    _logger.error(
        "secondary cleanup failed: phase=%s code=%s exception_type=%s",
        phase,
        code,
        type(error).__name__,
    )


def _output_fingerprint(record: ExecutionRecord) -> str:
    binding = record.binding
    if isinstance(binding, TaskBindingSnapshot):
        return binding.output_fingerprint
    if isinstance(binding, AgentBindingSnapshot):
        return restore_output(binding.output_mode, binding.output_schema).fingerprint
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _terminal_error_code(record: ExecutionRecord) -> str:
    if record.status is ExecutionStatus.CANCELLED:
        if record.error_code != ErrorCode.EXECUTION_CANCELLED.value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return record.error_code
    if record.status is not ExecutionStatus.FAILED or record.error_code is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(record.error_code)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if code is ErrorCode.EXECUTION_CANCELLED:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return code.value


__all__ = ["ExecutionInfo", "RuntimeHistory"]
