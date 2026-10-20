#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only Runtime composition for persisted execution history."""

from ..storage import StorageRetentionMode

import json
import math
import time
import heapq
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import cast

from pydantic_ai.messages import ModelMessage

from linktools.core import environ

from ..agent import AgentBindingContract
from ._features import RuntimeFeatures, RuntimeFeaturesCapture
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
    SessionStatus,
    TaskStatus,
    TenantAuthorizationPolicy,
    canonical_sha256,
    normalize_json_value,
    principal_identity_payload,
    validate_page_limit,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..storage import CacheBackend, ObjectRef, ObjectStore, StoredPayload, read_object
from ..session import SessionDetail, SessionManager, SessionPendingExecutionLookup
from ..spec import CapabilityRef
from ..task import (
    TaskEvent,
    TaskGraphInfo,
    TaskResultRecord,
    TaskResultRef,
)
from ._artifact import DefaultArtifactService
from ._persistence_configuration import persistence_storage_configuration, capability_storage_configuration, execution_configuration
from ._pending_coordination import _PendingTurnCoordination
from .state._object_router import runtime_object_source_reads
from ._event import project_event_payload
from ..core import decode_cursor as decode_runtime_cursor
from ..core import encode_cursor as encode_runtime_cursor
from ._history_projection import StepExecutionHistoryReader
from ._history_service import DefaultExecutionHistoryService, _read_model_interaction_updates
from ._runtime_identity import token_seed
from ._transient_history import TransientExecutionHistoryStore
from ._session_display_cache import SessionDisplayCache
from .state._store import active_state_scope
from ._session_timeline import (
    SessionTimelineTranscriptStore,
    capture_session_timeline,
    project_captured_session_timeline,
)
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
    ModelInteractionReadBoundary,
    ModelInteractionSubscription,
    SessionTurn,
    SessionView,
    TranscriptItem,
    UsageReadCutoff,
    UsageSummary,
)
from .state import RuntimeDomain, RuntimeStorage, RuntimeState
from .state._execution_repositories import ExecutionRepositoryImpl
from .state._conversation_repositories import SessionRepositoryImpl
from .state._step_archive import StagingAgentRunStore
from .state._contracts import (
    ConversationRepositories,
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
    task_id: str | None
    status: ExecutionStatus
    lineage_kind: ExecutionLineageKind
    parent_execution_id: str | None
    root_execution_id: str
    parent_invocation_id: str | None
    session_id: str | None
    created_at: datetime
    updated_at: datetime
    error_code: str | None
    safe_error_details: Mapping[str, JsonValue] = field(default_factory=dict)
    error_diagnostics: ErrorDiagnostics | None = None
    started_at: datetime | None = None
    terminal_at: datetime | None = None
    binding_digest: str | None = None
    usage: UsageSummary | None = None
    capability_refs: tuple[CapabilityRef, ...] | None = None

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
        "running_requests": 0,
        "interrupted_requests": 0,
        "output_correction_retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "model_duration_ns": 0,
        "unknown_usage_requests": 0,
        "unknown_duration_requests": 0,
    }
    for value in values:
        totals["logical_requests"] += value.logical_requests
        totals["succeeded_requests"] += value.succeeded_requests
        totals["failed_requests"] += value.failed_requests
        totals["cancelled_requests"] += value.cancelled_requests
        totals["running_requests"] += value.running_requests
        totals["interrupted_requests"] += value.interrupted_requests
        totals["output_correction_retries"] += value.output_correction_retries
        totals["input_tokens"] += value.input_tokens
        totals["output_tokens"] += value.output_tokens
        totals["cache_read_tokens"] += value.cache_read_tokens
        totals["cache_write_tokens"] += value.cache_write_tokens
        totals["model_duration_ns"] += value.model_duration_ns
        totals["unknown_usage_requests"] += value.unknown_usage_requests
        totals["unknown_duration_requests"] += value.unknown_duration_requests
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


def _execution_cutoffs(
    cutoffs: "tuple[UsageReadCutoff, ...] | None",
    execution_id: str,
) -> "tuple[UsageReadCutoff, ...] | None":
    if cutoffs is None:
        return None
    if any(not isinstance(value, UsageReadCutoff) for value in cutoffs):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return tuple(
        value for value in cutoffs if value.execution_id == execution_id
    )


def _project_session_view(record: SessionRecord) -> SessionView:
    return SessionView(
        session_id=record.session_id,
        agent_id=record.agent_id,
        status=record.status,
        revision=record.revision,
        cwd=record.cwd,
        active_execution_id=record.active_execution_id,
        metadata=record.metadata,
        history_quality=record.history_quality,
    )


def _project_execution_info(
    record: ExecutionRecord,
    *,
    usage: UsageSummary | None = None,
) -> ExecutionInfo:
    result = record.result
    binding = record.binding
    capability_refs = None
    if isinstance(binding, AgentBindingContract):
        capability_refs = binding.capability_refs
    return ExecutionInfo(
        execution_id=record.execution_id,
        binding_kind=record.binding_kind,
        agent_id=record.agent_id,
        task_id=record.task_id,
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
        usage=usage,
        capability_refs=capability_refs,
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
        object_reader: "Callable[[RuntimeDomain, ObjectRef], Awaitable[bytes]] | None" = None,
        artifacts: "ArtifactService | None" = None,
        conversation: "ConversationRepositories | None" = None,
        session_transcript_store: "SessionTimelineTranscriptStore | None" = None,
        pending_session_execution: SessionPendingExecutionLookup[ExecutionRecord] | None = None,
        pending_execution: Callable[..., Awaitable[ExecutionRecord | None]] | None = None,
        cache: CacheBackend | None = None,
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
        self._object_reader = object_reader
        self._artifacts = artifacts
        self._conversation = conversation
        self._session_transcript_store = session_transcript_store
        self._pending_session_execution = pending_session_execution
        self._pending_execution = pending_execution
        self._session_cursor_signer = (cursor_signer if namespace is None else
                                       HmacCursorSigner("session", token_seed(namespace)))
        self._display_cache = (
            None if conversation is None or executions is None or self._session_cursor_signer is None
            else SessionDisplayCache(conversation, executions, session_transcript_store,
                                     self._session_cursor_signer, cache, now=lambda: time.time())
        )

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    async def inspect_execution(
        self, execution_id: str, *, principal: Principal, include_usage: bool = True
    ) -> ExecutionInfo:
        if not isinstance(include_usage, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        record = await self._authorized_record(execution_id, principal)
        return _project_execution_info(
            record,
            usage=(await self._service.usage(execution_id, principal=principal)
                   if include_usage else None),
        )

    async def result(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult:
        record = await self._authorized_record(execution_id, principal, committed=True)
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
        stored = (record.result if type(executions) is ExecutionRepositoryImpl else
                  await executions.get_result(execution_id, tenant_id=principal.tenant_id))
        if stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not record.retention_closed and record.conversation and record.session_id is not None:
            sessions, _ = self._require_session_reader()
            await sessions.state_store.read(
                lambda transaction: sessions.pending_turn_commit_in_transaction(
                    transaction, session_id=record.session_id, execution_id=record.execution_id,
                    allow_unadmitted=record.started_at is None,
                )
            )

        if record.status is not ExecutionStatus.SUCCEEDED:
            if stored.output is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            error_code = _terminal_error_code(record)
            try:
                return ExecutionResult(
                    record.execution_id,
                    record.status,
                    None,
                    stored.usage,
                    error_code,
                    record.safe_error_details,
                    record.error_diagnostics,
                )
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

        if stored.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        output = await self._read_payload(stored.output)
        return ExecutionResult(
            record.execution_id,
            record.status,
            output,
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
        execution = await self.result(
            record.execution_id,
            principal=principal,
        )
        if execution.status is not ExecutionStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        output = execution.output
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
        """Scan visible executions and return the newest; O(N) time, O(limit) memory."""
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
        state = await tasks.graph_state(
            graph_id,
            tenant_id=principal.tenant_id,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return TaskGraphInfo.from_state(state)

    async def list_task_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
        limit: int = 100,
    ) -> Page[TaskEvent]:
        if (
            isinstance(after_event_seq, bool)
            or not isinstance(after_event_seq, int)
            or after_event_seq < 0
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
            after_event_seq=after_event_seq,
            limit=limit,
        )

    async def list_execution_events(
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
            high_water = record.event_seq
            after_event_seq = 0
        else:
            payload = decode_runtime_cursor(
                cursor,
                signer,
                tenant_id=principal.tenant_id,
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
                after_event_seq = int(after_raw)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.CURSOR_INVALID) from error
            if (
                high_water < 0
                or after_event_seq < 0
                or after_event_seq > high_water
            ):
                raise AIError(ErrorCode.CURSOR_INVALID)
        if after_event_seq >= high_water:
            return Page((), None)
        page_limit = min(limit, high_water - after_event_seq)
        page = await events.list(
            execution_id,
            tenant_id=principal.tenant_id,
            after_event_seq=after_event_seq,
            limit=page_limit,
        )
        expected = after_event_seq
        projected: list[ExecutionEvent] = []
        for event in page.items:
            expected += 1
            if (
                event.execution_id != execution_id
                or event.event_seq != expected
                or event.event_seq > high_water
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            projected.append(
                ExecutionEvent(
                    event.execution_id,
                    event.event_seq,
                    event.event_type,
                    project_event_payload(
                        event.event_type, event.payload, include_content=include_content
                    ),
                )
            )
        if len(projected) != page_limit:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        next_after = after_event_seq + len(projected)
        next_cursor = (
            None
            if next_after >= high_water
            else encode_runtime_cursor(
                signer,
                tenant_id=principal.tenant_id,
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
        include_children: bool = False,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> UsageSummary:
        if not include_children:
            selected = _execution_cutoffs(cutoffs, execution_id)
            return await self._service.usage(
                execution_id,
                principal=principal,
                cutoffs=selected,
            )

        executions, _authorization = self._require_direct_reader()
        await self._authorized_record(execution_id, principal)
        explicit_ids = (
            None if cutoffs is None else {value.execution_id for value in cutoffs}
        )
        pending = [execution_id]
        seen: set[str] = set()
        values: list[UsageSummary] = []
        while pending:
            current_id = pending.pop(0)
            if current_id in seen:
                continue
            record = await executions.get(
                current_id,
                tenant_id=principal.tenant_id,
            )
            if record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            seen.add(current_id)
            if explicit_ids is None or current_id in explicit_ids:
                values.append(
                    await self._service.usage(
                        current_id,
                        principal=principal,
                        cutoffs=_execution_cutoffs(cutoffs, current_id),
                    )
                )
            children = await executions.list_children(
                current_id,
                tenant_id=principal.tenant_id,
            )
            for child in children:
                if child.parent_execution_id != current_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if child.execution_id not in seen:
                    pending.append(child.execution_id)
        if explicit_ids is not None and not explicit_ids.issubset(seen):
            raise AIError(ErrorCode.CURSOR_INVALID)
        return _merge_usage_summaries(values)

    async def graph_usage(
        self,
        graph_id: str,
        *,
        principal: Principal,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
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
        explicit_ids = (
            None if cutoffs is None else {value.execution_id for value in cutoffs}
        )
        pending = list(roots)
        seen: set[str] = set()
        values: list[UsageSummary] = []
        while pending:
            current_id = pending.pop(0)
            if current_id in seen:
                continue
            record = await executions.get(
                current_id,
                tenant_id=principal.tenant_id,
            )
            if record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            seen.add(current_id)
            if explicit_ids is None or current_id in explicit_ids:
                values.append(
                    await self._service.usage(
                        current_id,
                        principal=principal,
                        cutoffs=_execution_cutoffs(cutoffs, current_id),
                    )
                )
            children = await executions.list_children(
                current_id,
                tenant_id=principal.tenant_id,
            )
            for child in children:
                if child.parent_execution_id != current_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if child.execution_id not in seen:
                    pending.append(child.execution_id)
        if explicit_ids is not None and not explicit_ids.issubset(seen):
            raise AIError(ErrorCode.CURSOR_INVALID)
        return _merge_usage_summaries(
            values,
            unrecorded_executions=unrecorded,
        )

    async def list_sessions(
        self,
        *,
        principal: Principal,
        cursor: str | None = None,
        limit: int = 100,
    ) -> Page[SessionView]:
        """Read one metadata page in repository order, without loading a Runtime.

        Cursors retain the repository snapshot. Authorization may yield a short
        or empty page with a next cursor; callers must follow that cursor.
        """
        limit = validate_page_limit(limit)
        sessions, authorization = self._require_session_reader()
        signer = self._cursor_signer
        if signer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, "list", principal.tenant_id),
        )
        digest = canonical_sha256({"principal": principal_identity_payload(principal)})
        position = None
        snapshot = None
        if cursor is not None:
            decoded = decode_runtime_cursor(
                cursor, signer, tenant_id=principal.tenant_id,
                resource_kind="HISTORY_SESSIONS", filter_digest=digest,
            )
            position, snapshot = decoded.position, decoded.revision
        snapshot, page = await sessions.list_page(
            tenant_id=principal.tenant_id,
            owner_principal_id=principal.principal_id,
            cursor=position, limit=limit, snapshot=snapshot,
        )
        active_ids = tuple(dict.fromkeys(record.active_execution_id for record in page.items
                                         if record.active_execution_id is not None))
        active_records = {}
        if active_ids:
            executions, _ = self._require_direct_reader()
            active_records = await executions.get_many(active_ids, tenant_id=principal.tenant_id)
        values: list[SessionView] = []
        for record in page.items:
            try:
                await authorization.authorize(
                    principal, AuthorizationAction.SESSION_READ,
                    ResourceRef(ResourceKind.SESSION, record.session_id,
                                principal.tenant_id, record.owner_principal_id),
                )
            except AIError as error:
                if error.code is ErrorCode.AUTHORIZATION_DENIED:
                    continue
                raise
            values.append(await self._observed_session_view(record, principal, active_records=active_records))
        next_cursor = None if page.next_cursor is None else encode_runtime_cursor(
            signer, tenant_id=principal.tenant_id,
            resource_kind="HISTORY_SESSIONS", filter_digest=digest,
            position=page.next_cursor, revision=snapshot,
        )
        return Page(tuple(values), next_cursor)

    async def messages(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        limit: int = 100,
    ) -> Page[ModelMessage]:
        """Read retained raw messages, oldest first, including all message parts.

        The first page fixes a committed prefix; later appends are excluded.
        Compression changes the model's working context, not this transcript.
        Pending or paused staging is excluded. Failed/cancelled turns expose
        only their committed messages; zero-message outcomes remain available
        through ``session_timeline``.
        """
        limit = validate_page_limit(limit)
        sessions, authorization = self._require_session_reader()
        signer, store = self._cursor_signer, self._session_transcript_store
        if signer is None or store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        header = await sessions.get_header(session_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(principal, AuthorizationAction.SESSION_READ, header)
        record = await sessions.get(session_id, tenant_id=principal.tenant_id)
        if (record is None or record.owner_principal_id != header.owner_principal_id
                or header.id != record.session_id or header.tenant_id != principal.tenant_id):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if record.history_quality == "unavailable":
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        digest = canonical_sha256({
            "session_id": session_id,
            "principal": principal_identity_payload(principal),
        })
        continuation = record.continuation
        if continuation is None:
            if cursor is not None:
                raise AIError(ErrorCode.CURSOR_INVALID)
            return Page((), None)
        history_id = continuation.history_id or record.history_id
        agent_run_id = continuation.agent_run_id
        start = 0
        if cursor is None:
            high_water = continuation.message_count
            if high_water is None:
                high_water = await store.committed_conversation_message_count(
                    history_id=history_id, agent_run_id=agent_run_id,
                    tenant_id=principal.tenant_id,
                )
        else:
            decoded = decode_runtime_cursor(
                cursor, signer, tenant_id=principal.tenant_id,
                resource_kind="SESSION_MESSAGES", filter_digest=digest,
            )
            try:
                coordinate = json.loads(decoded.position)
            except (ValueError, TypeError) as error:
                raise AIError(ErrorCode.CURSOR_INVALID) from error
            if (
                decoded.revision != 0 or not isinstance(coordinate, list)
                or len(coordinate) != 4 or coordinate[0] != history_id
                or not isinstance(coordinate[1], str) or not coordinate[1]
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in coordinate[2:])
                or coordinate[2] < 0 or not 0 <= coordinate[3] <= coordinate[2]
            ):
                raise AIError(ErrorCode.CURSOR_INVALID)
            _, agent_run_id, high_water, start = coordinate
            if history_id is None and agent_run_id != continuation.agent_run_id:
                raise AIError(ErrorCode.CURSOR_INVALID)
            if continuation.message_count is not None and high_water > continuation.message_count:
                raise AIError(ErrorCode.CURSOR_INVALID)
        if isinstance(high_water, bool) or not isinstance(high_water, int) or high_water < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        end = min(start + limit, high_water)
        if start == end:
            return Page((), None)
        messages = tuple([
            message async for message in store.iter_conversation_message_range(
                history_id=history_id, agent_run_id=agent_run_id,
                tenant_id=principal.tenant_id, start=start, end=end,
            )
        ])
        if len(messages) != end - start:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        next_cursor = None if end == high_water else encode_runtime_cursor(
            signer, tenant_id=principal.tenant_id,
            resource_kind="SESSION_MESSAGES", filter_digest=digest,
            position=json.dumps([history_id, agent_run_id, high_water, end],
                                separators=(",", ":")),
        )
        return Page(cast(tuple[ModelMessage, ...], messages), next_cursor)

    async def recent_sessions(
        self,
        *,
        principal: Principal,
        limit: int = 20,
    ) -> tuple[SessionView, ...]:
        """Scan visible sessions and return the newest; O(N) time, O(limit) memory."""
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
                    principal.tenant_id,
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

    async def _observed_session_view(
        self, record: SessionRecord, principal: Principal,
        *, active_records: Mapping[str, ExecutionRecord] | None = None,
    ) -> SessionView:
        header = ResourceRef(ResourceKind.SESSION, record.session_id,
                             principal.tenant_id, record.owner_principal_id)
        pending = (await self._pending_session(header)
                   if record.status is SessionStatus.OPEN and record.active_execution_id is None else None)

        async def read() -> SessionView:
            current = record
            if pending is not None:
                sessions, _ = self._require_session_reader()
                current = await sessions.get(record.session_id, tenant_id=principal.tenant_id)
                if current is None or current.owner_principal_id != header.owner_principal_id:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            active = current.active_execution_id
            if active is not None:
                executions, _ = self._require_direct_reader()
                execution = (active_records.get(active) if active_records is not None and pending is None
                             else await executions.get(active, tenant_id=principal.tenant_id))
                observed = SessionManager.validate_admission(current, execution)
                active = None if observed is None else observed.execution_id
            elif (pending is not None and current.status is SessionStatus.OPEN
                  and pending.created_at >= current.created_at):
                executions, _ = self._require_direct_reader()
                if isinstance(executions, ExecutionRepositoryImpl):
                    records = await executions.get_many_fresh(
                        (pending.execution_id,), tenant_id=principal.tenant_id,
                    )
                    canonical = records.get(pending.execution_id)
                else:
                    canonical = await executions.get(pending.execution_id, tenant_id=principal.tenant_id)
                if canonical is None:
                    active = pending.execution_id
                else:
                    sessions, _ = self._require_session_reader()
                    current = (await sessions.state_store.read(
                        lambda _transaction: sessions.get(record.session_id, tenant_id=principal.tenant_id)
                    ) if isinstance(sessions, SessionRepositoryImpl) else
                        await sessions.get(record.session_id, tenant_id=principal.tenant_id))
                    if current is None or current.owner_principal_id != header.owner_principal_id:
                        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                    observed = SessionManager.validate_admission(
                        current, canonical if current.active_execution_id == canonical.execution_id else None,
                    )
                    active = None if observed is None else observed.execution_id
            return replace(_project_session_view(current), active_execution_id=active)

        if pending is None or self._display_cache is not None and self._display_cache.supported():
            return await read()
        sessions, _ = self._require_session_reader()
        executions, _ = self._require_direct_reader()
        if (type(sessions) is SessionRepositoryImpl and type(executions) is ExecutionRepositoryImpl
                and sessions.state_store.storage_group is executions.state_store.storage_group):
            return await sessions.state_store.storage_group.read_many(
                (sessions.state_store, executions.state_store), lambda _group: read(),
            )
        return await read()

    async def _pending_session(self, header: ResourceRef) -> ExecutionRecord | None:
        if self._pending_session_execution is None:
            return None
        pending = await self._pending_session_execution(header.id, tenant_id=header.tenant_id)
        if (pending is None or pending.session_id != header.id
                or pending.principal_id != header.owner_principal_id
                or pending.parent_execution_id is not None or pending.result is not None
                or pending.status not in {ExecutionStatus.PENDING_START, ExecutionStatus.STARTED,
                                          ExecutionStatus.CANCELLING}):
            return None
        return pending

    async def session_detail(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        limit: int = 100,
        max_age: float = 0,
    ) -> SessionDetail:
        """Read a coherent display with live coordination observations, without writes.

        Durable admission and terminal commits take precedence over observations.
        Positive max_age reuses the shared display cache with current authorization
        and membership checks; zero forces an authoritative metadata capture.
        """
        if isinstance(max_age, bool) or not isinstance(max_age, (int, float)) or not math.isfinite(max_age) or max_age < 0:
            raise ValueError("max_age must be finite and non-negative")
        started = time.time()
        validate_page_limit(limit)
        if self._conversation is None or self._session_transcript_store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        signer = self._session_cursor_signer
        if signer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        executions, authorization = self._require_direct_reader()
        sessions = self._conversation.sessions
        header = await sessions.get_header(session_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(principal, AuthorizationAction.SESSION_READ, header)
        pending = await self._pending_session(header)
        cache_key = None
        display_cache = self._display_cache
        if (display_cache is not None and display_cache.supported()
                and max_age > 0 and active_state_scope() is None
                and (pending is None or cursor is None)):
            cacheable = True
            if cursor is not None:
                try:
                    cacheable = json.loads(signer.decode(cursor).position)[0] == session_id
                except (AIError, TypeError, ValueError, IndexError, KeyError):
                    cacheable = False
            if cacheable:
                cache_key = "session-display:v1:" + canonical_sha256({
                    "namespace": sessions.namespace,
                    "principal": [principal.tenant_id, principal.principal_id, principal.kind],
                    "session_id": session_id, "cursor": cursor, "limit": limit,
                })
                cached = await display_cache.read(
                    cache_key, header, principal, cursor=cursor, max_age=max_age, pending=pending,
                )
                if cached is not None:
                    return cached
        state = await capture_session_timeline(
            self._conversation, executions, session_id,
            tenant_id=principal.tenant_id, cursor=cursor, cursor_signer=signer,
            limit=limit, pending_execution=pending, authorized_header=header,
            include_active=True,
            transcript_store=self._session_transcript_store,
        )
        record = state.root
        active = SessionManager.validate_admission(
            record, None if record.active_execution_id is None
            else state.executions.get(record.active_execution_id),
        )
        view = replace(_project_session_view(record), active_execution_id=(
            state.pending_execution.execution_id if state.pending_execution is not None
            else None if active is None else active.execution_id
        ))
        with runtime_object_source_reads():
            timeline = await project_captured_session_timeline(
                state, self._session_transcript_store, signer,
                session_id=session_id, tenant_id=principal.tenant_id,
                transcript_tenant_id=sessions.tenant_id,
            )
        detail = SessionDetail(view, timeline)
        if cache_key is not None and record.timeline_parent_session_id is None:
            await display_cache.write(cache_key, state, detail, started, max_age)
        return detail

    async def session_timeline(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[SessionTurn]:
        return (await self.session_detail(
            session_id, principal=principal, cursor=cursor, limit=limit,
        )).timeline

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
        return await self._observed_session_view(record, principal)

    @classmethod
    def open(
        cls,
        namespace: str,
        *,
        storage: RuntimeStorage,
        features: RuntimeFeatures[object] | RuntimeFeaturesCapture[object] | None = None,
        tenant_id: "str | None" = None,
    ) -> AbstractAsyncContextManager["RuntimeHistory"]:
        return _open_runtime_history(
            namespace,
            storage=storage,
            features=features,
            tenant_id=tenant_id,
        )

    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> Page[ExecutionHistoryItem]:
        return await self._service.history(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
            agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq,
            step_index=step_index,
            tool_call_id=tool_call_id,
            message_seq=message_seq,
            part_index=part_index,
        )

    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> Page[ExecutionTraceItem]:
        return await self._service.trace(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
            agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq,
            step_index=step_index,
            tool_call_id=tool_call_id,
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

    async def read_model_interaction_updates(
        self, execution_id: str, *, principal: Principal,
        after_cutoffs: tuple[UsageReadCutoff, ...] = (), limit: int = 200,
    ) -> tuple[ModelInteractionReadBoundary, tuple[ModelInteractionItem, ...]]:
        """Read a fresh boundary and a bounded suffix without changing older service contracts."""
        if type(self) is not RuntimeHistory:
            return await _read_model_interaction_updates(
                self, execution_id, principal=principal, after_cutoffs=after_cutoffs, limit=limit,
            )
        read = getattr(self._service, "read_model_interaction_updates", None)
        if callable(read):
            return await read(execution_id, principal=principal, after_cutoffs=after_cutoffs, limit=limit)
        return await _read_model_interaction_updates(
            self._service, execution_id, principal=principal, after_cutoffs=after_cutoffs, limit=limit,
        )

    async def capture_model_interaction_cutoffs(
        self, execution_id: str, *, principal: Principal,
    ) -> ModelInteractionReadBoundary:
        return await self._service.capture_model_interaction_cutoffs(execution_id, principal=principal)

    async def read_model_interaction_metadata(
        self, execution_id: str, *, principal: Principal, agent_run_seq: int,
        after_model_request_seq: int, through_model_request_seq: int, limit: int = 200,
    ) -> tuple[ModelInteractionItem, ...]:
        return await self._service.read_model_interaction_metadata(
            execution_id, principal=principal, agent_run_seq=agent_run_seq,
            after_model_request_seq=after_model_request_seq,
            through_model_request_seq=through_model_request_seq, limit=limit,
        )

    async def subscribe_model_interactions(
        self, execution_id: str, *, principal: Principal,
    ) -> ModelInteractionSubscription | None:
        return await self._service.subscribe_model_interactions(execution_id, principal=principal)

    async def model_interactions(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        include_client_requests: bool = False,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        limit: int = 100,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> Page[ModelInteractionItem]:
        return await self._service.model_interactions(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            **({"include_client_requests": include_client_requests} if include_client_requests is not False else {}),
            **({"agent_run_seq": agent_run_seq} if agent_run_seq is not None else {}),
            **({"model_request_seq": model_request_seq} if model_request_seq is not None else {}),
            limit=limit,
            cutoffs=cutoffs,
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
        if (
            record is None
            or record.result_digest != state.result_digest
            or record.execution_id != state.execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return record

    async def _read_payload(
        self,
        payload: StoredPayload,
    ) -> JsonValue:
        try:
            if payload.kind == "inline":
                value = payload.decode()
            else:
                if payload.ref is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                if self._object_reader is not None:
                    with runtime_object_source_reads():
                        raw = await self._object_reader(RuntimeDomain.EXECUTION, payload.ref)
                else:
                    if self._execution_objects is None:
                        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                    raw = await read_object(
                        self._execution_objects,
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
        *, committed: bool = False,
    ) -> ExecutionRecord:
        executions, authorization = self._require_direct_reader()
        pending = (None if self._pending_execution is None else
                   await self._pending_execution(execution_id, tenant_id=principal.tenant_id))
        header = await executions.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None and pending is not None:
            header = ResourceRef(ResourceKind.EXECUTION, execution_id,
                                 principal.tenant_id, pending.principal_id)
        if header is None and isinstance(executions, ExecutionRepositoryImpl):
            # A remote terminal commit may have removed its coordination entry
            # while this reader still caches the earlier canonical absence.
            header = await executions.get_header_fresh(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(principal, AuthorizationAction.EXECUTION_READ, header)
        record = (
            await executions.state_store.read(
                lambda _transaction: executions.get(execution_id, tenant_id=principal.tenant_id)
            )
            if committed or pending is not None and isinstance(executions, ExecutionRepositoryImpl)
            else await executions.get(execution_id, tenant_id=principal.tenant_id)
        )
        if record is not None:
            if header.owner_principal_id is not None and record.principal_id != header.owner_principal_id:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return record
        if pending is not None:
            if header.owner_principal_id is not None and pending.principal_id != header.owner_principal_id:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return pending
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)



@asynccontextmanager
async def _open_runtime_history(
    namespace: str,
    *,
    storage: RuntimeStorage,
    features: RuntimeFeatures[object] | RuntimeFeaturesCapture[object] | None,
    tenant_id: "str | None",
) -> AsyncIterator[RuntimeHistory]:
    resolved_namespace = validate_persistence_namespace(namespace)
    effective_tenant_id = (
        "default" if tenant_id is None else validate_tenant_id(tenant_id)
    )
    if not isinstance(storage, RuntimeStorage):
        raise TypeError("storage must be RuntimeStorage")
    if features is not None and not isinstance(features, (RuntimeFeatures, RuntimeFeaturesCapture)):
        raise TypeError("features must be RuntimeFeatures or RuntimeFeaturesCapture")
    captured = await features.capture() if isinstance(features, RuntimeFeatures) else features
    authorization = None if captured is None else captured.authorization
    if captured is not None:
        await captured.verify_source_revision()
    capabilities = () if captured is None else tuple(
        item.value for item in captured.contributions if item.scope == "runtime"
    )
    execution_policy = execution_configuration(capabilities)
    pending_coordination = _PendingTurnCoordination.select(
        execution_policy.coordination, namespace=resolved_namespace, tenant_id=effective_tenant_id,
    )
    module_routes = persistence_storage_configuration(capabilities)
    capability_routes = capability_storage_configuration(
        () if captured is None else tuple((item.id, item.value) for item in captured.contributions if item.scope == "runtime"),
        default_route=storage.plan.route(RuntimeDomain.EXECUTION),
    )
    selected_state = RuntimeState(storage)
    initialized = False
    body_error: BaseException | None = None
    try:
        await selected_state.initialize(
            namespace=resolved_namespace,
            tenant_id=effective_tenant_id,
            read_only=True,
            module_routes=module_routes,
            capability_routes=capability_routes,
        )
        initialized = True
        if captured is not None:
            await captured.verify_source_revision()
        if (
            selected_state.namespace != resolved_namespace
            or selected_state.tenant_id != effective_tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        history_store = selected_state.run_store.read_store(RuntimeDomain.TRACE)
        if selected_state.plan.route(RuntimeDomain.TRACE).retention is StorageRetentionMode.TRANSIENT:
            if not isinstance(history_store, StagingAgentRunStore):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            history_store = TransientExecutionHistoryStore(history_store, selected_state.run_store)
        reader = StepExecutionHistoryReader(
            namespace=resolved_namespace,
            executions=selected_state.execution.executions,
            store=history_store,
            cursor_signer=HmacCursorSigner(
                "execution-history",
                token_seed(resolved_namespace),
            ),
            tool_operations=selected_state.recovery.tools,
            durable_history_available=selected_state.run_store.model_interaction_history_available,
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
            cursor_signer=HmacCursorSigner("execution", token_seed(resolved_namespace)),
            pending_execution=None if pending_coordination is None else pending_coordination.read_execution,
        )
        artifacts = DefaultArtifactService(
            selected_state.artifact,
            effective_authorization,
            token_seed=token_seed(resolved_namespace),
            cursor_signer=HmacCursorSigner("artifact", token_seed(resolved_namespace)),
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
            object_reader=selected_state.read_object,
            artifacts=artifacts,
            cursor_signer=HmacCursorSigner(
                "runtime-history",
                token_seed(resolved_namespace),
            ),
            conversation=selected_state.conversation,
            session_transcript_store=selected_state.run_store.conversation_persistence,
            pending_session_execution=(None if pending_coordination is None
                                       else pending_coordination.read_session_execution),
            pending_execution=None if pending_coordination is None else pending_coordination.read_execution,
            cache=(selected_state.plan.route(RuntimeDomain.CONVERSATION).cache if all(
                selected_state.plan.route(domain).retention is StorageRetentionMode.DURABLE
                for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION)
            ) else None),
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
