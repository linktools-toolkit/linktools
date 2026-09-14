#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session query API and persistence-backed session service."""

import asyncio
import json
import time
from binascii import Error as Base64Error
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol, cast

from linktools.core import environ
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorPayload,
    CursorSigner,
    ExecutionStatus,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    canonical_sha256,
    idempotency_key_digest,
    validate_agent_id,
)
from ..errors import AIError, ErrorCode
from ..capability import WorkspaceAccess
from .service_api import (
    CancelExecutionRequest,
    CloseSessionRequest,
    CreateSessionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionService,
    ForkSessionRequest,
    ListSessionRequest,
    LoadedSession,
    ResumeSessionRequest,
    SessionHistoryItem,
    SessionHistoryReader,
    SessionTurn,
    SessionTurnItem,
    SessionView,
    UpdateSessionRequest,
)
from .state._contracts import (
    ConversationState,
    ExecutionRecord,
    ExecutionRepository,
    SessionRecord,
    SessionTurnCommitRef,
    SessionTurnRef,
)
from .state._step_contracts import (
    ContinuableSnapshot,
)
from .state._contracts import (
    ConversationCursor,
)
from ._input import stored_user_input_view
from .state._views import project_session_history_message

_TIMELINE_PROJECTION_VERSION = 1


def _timeline_items(messages: tuple[ModelMessage, ...]) -> tuple[SessionTurnItem, ...]:
    values: list[SessionTurnItem] = []
    for message in messages:
        projected = project_session_history_message(message)
        if isinstance(message, ModelRequest):
            projected = tuple(
                item for item in projected if item.item_kind == "tool_result"
            )
        elif not isinstance(message, ModelResponse):
            continue
        for item in projected:
            values.append(
                SessionTurnItem(
                    len(values) + 1,
                    item.item_kind,
                    item.content,
                    item.tool_name,
                    item.tool_call_id,
                )
            )
    return tuple(values)


def _timeline_filter_digest(session_id: str) -> str:
    return canonical_sha256(
        {
            "session_id": session_id,
            "projection_version": _TIMELINE_PROJECTION_VERSION,
        }
    )


def _timeline_cursor(
    *,
    tenant_id: str,
    session_id: str,
    source_session_id: str,
    before_sequence: int,
    signer: CursorSigner,
) -> str:
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "SESSION_TIMELINE",
            _timeline_filter_digest(session_id),
            json.dumps(
                [source_session_id, before_sequence],
                separators=(",", ":"),
            ),
            0,
            int(time.time()) + 3600,
            projection_version=_TIMELINE_PROJECTION_VERSION,
        )
    )


def _decode_timeline_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    session_id: str,
    signer: CursorSigner,
) -> "tuple[str, int] | None":
    if cursor is None:
        return None
    try:
        payload = signer.decode(cursor)
        coordinate = json.loads(payload.sort_key)
    except (AIError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.cursor_version != 1
        or payload.tenant_id != tenant_id
        or payload.resource_kind != "SESSION_TIMELINE"
        or payload.filter_digest != _timeline_filter_digest(session_id)
        or payload.snapshot_or_store_revision != 0
        or payload.projection_version != _TIMELINE_PROJECTION_VERSION
        or not isinstance(coordinate, list)
        or len(coordinate) != 2
        or not isinstance(coordinate[0], str)
        or not coordinate[0]
        or isinstance(coordinate[1], bool)
        or not isinstance(coordinate[1], int)
        or coordinate[1] < 1
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return coordinate[0], coordinate[1]


_logger = environ.get_logger("ai.runtime.session")


class _SessionReleaseCallback(Protocol):
    async def __call__(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation: ConversationCursor | None,
    ) -> None: ...


class _SessionExecutionService(ExecutionService, Protocol):
    async def start_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
    ) -> ExecutionHandle: ...


class _SessionTranscriptStore(Protocol):
    async def iter_messages(self, *, run_id: str) -> AsyncIterator[object]: ...

    async def iter_session_messages(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> AsyncIterator[object]: ...

    async def iter_session_message_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[ModelMessage]: ...

    async def load_session_model_context(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> tuple[object, ...]: ...

    async def load_model_context(
        self,
        *,
        run_id: str,
    ) -> tuple[object, ...]: ...

    async def latest_snapshot(
        self,
        *,
        run_id: str,
        include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None: ...


async def _no_release_terminal(
    session_id: str, *, tenant_id: str, continuation: ConversationCursor | None
) -> None:
    del session_id, tenant_id, continuation


@dataclass
class _SessionHandoffState:
    active_consumers: int = 0
    release_requested: bool = False
    release_in_progress: bool = False
    continuation: ConversationCursor | None = None


class DefaultSessionService:
    """Enforce session ownership, Agent identity immutability, and revision CAS."""

    def __init__(
        self,
        conversation: ConversationState,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        execution: _SessionExecutionService,
        cursor_signer: CursorSigner,
        *,
        history_reader: SessionHistoryReader,
        transcript_store: "_SessionTranscriptStore | None" = None,
        release_terminal: _SessionReleaseCallback | None = None,
        workspace_access: WorkspaceAccess | None = None,
    ) -> None:
        self._conversation = conversation
        self._executions = executions
        self._authorization = authorization
        self._execution = execution
        self._cursor_signer = cursor_signer
        self._history_reader = history_reader
        self._transcript_store = transcript_store
        self._release_terminal = release_terminal or _no_release_terminal
        self._workspace_access = workspace_access
        self._handoff_states: dict[tuple[str, str], _SessionHandoffState] = {}
        self._handoff_condition = asyncio.Condition()

    async def create(self, agent_id: str, request: CreateSessionRequest) -> SessionView:
        try:
            agent_id = validate_agent_id(agent_id)
        except AIError:
            raise
        except TypeError as error:
            raise AIError(ErrorCode.AGENT_ID_INVALID) from error
        if any(key.startswith("linktools.ai.") for key in request.metadata):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        cwd = await self._canonicalize_cwd(request.cwd)
        resource = ResourceRef(
            ResourceKind.SESSION,
            request.session_id,
            request.principal.tenant_id,
            request.principal.principal_id,
        )
        await self._authorization.authorize(
            request.principal, AuthorizationAction.SESSION_CREATE, resource
        )
        digest = canonical_sha256(
            {
                "action": "session.create",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "session_id": request.session_id,
                "agent_id": agent_id,
                "cwd": cwd,
                "metadata": dict(request.metadata),
            }
        )
        now = datetime.now(timezone.utc)
        record = SessionRecord(
            session_id=request.session_id,
            tenant_id=request.principal.tenant_id,
            owner_principal_id=request.principal.principal_id,
            status=SessionStatus.OPEN,
            revision=0,
            cwd=cwd,
            metadata=dict(request.metadata),
            created_at=now,
            updated_at=now,
            closed_at=None,
            active_execution_id=None,
            continuation=None,
            agent_id=agent_id,
        )
        operation = self._session_terminal_operation(
            request.idempotency_key,
            request.principal.tenant_id,
            request.session_id,
            OperationKind.SESSION_CREATE,
            digest,
            record.revision,
        )
        record, _ = await self._conversation.sessions.create_with_operation(
            record,
            operation=operation,
        )
        _logger.debug(
            "session created: session=%s tenant=%s",
            record.session_id,
            request.principal.tenant_id,
        )
        return await self._view(record, request.principal)

    async def get(self, session_id: str, *, principal: Principal) -> SessionView:
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            return await self._view(record, principal)

    async def history(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionHistoryItem]":
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(
                session_id,
                principal,
                AuthorizationAction.SESSION_READ,
            )
            continuation = (
                None if record.continuation is None else record.continuation.step_run_id
            )
            continuation_history_id = (
                None
                if record.continuation is None
                else record.continuation.history_id or record.history_id
            )
            return await self._history_reader.history(
                session_id,
                tenant_id=principal.tenant_id,
                continuation_step_run_id=continuation,
                continuation_history_id=continuation_history_id,
                cursor=cursor,
                limit=limit,
            )

    async def timeline(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[SessionTurn]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 200
        ):
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        async with self._session_consumer(session_id, principal.tenant_id):
            root = await self._authorized(
                session_id,
                principal,
                AuthorizationAction.SESSION_READ,
            )
            if self._transcript_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            coordinate = _decode_timeline_cursor(
                cursor,
                tenant_id=principal.tenant_id,
                session_id=session_id,
                signer=self._cursor_signer,
            )
            blocks, next_coordinate = await self._timeline_blocks(
                root,
                coordinate=coordinate,
                limit=limit,
            )
            refs = tuple(ref for _record, values in blocks for ref in values)
            if not refs:
                return Page((), None)
            execution_ids = tuple(dict.fromkeys(ref.execution_id for ref in refs))
            executions = await self._executions.get_many(
                execution_ids,
                tenant_id=principal.tenant_id,
            )
            if len(executions) != len(execution_ids):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            commits: dict[tuple[str, int], SessionTurnCommitRef] = {}
            messages: dict[str, tuple[ModelMessage, ...]] = {}
            message_bases: dict[str, int] = {}
            for record, values in blocks:
                if not values:
                    continue
                start = values[0].sequence
                end = values[-1].sequence + 1
                committed = await self._conversation.sessions.list_timeline_commits(
                    record.session_id,
                    tenant_id=record.tenant_id,
                    start_sequence=start,
                    end_sequence=end,
                )
                for item in committed:
                    commits[(record.session_id, item.sequence)] = item
                if not committed:
                    continue
                if record.history_id is None:
                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
                range_start = min(item.start_message_index for item in committed)
                range_end = max(item.end_message_index for item in committed)
                loaded = cast(
                    tuple[ModelMessage, ...],
                    tuple(
                        [
                            item
                            async for item in self._transcript_store.iter_session_message_range(
                                record.history_id,
                                tenant_id=record.tenant_id,
                                start=range_start,
                                end=range_end,
                            )
                        ]
                    ),
                )
                if len(loaded) != range_end - range_start:
                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
                messages[record.session_id] = loaded
                message_bases[record.session_id] = range_start

            turns: list[SessionTurn] = []
            for ref in refs:
                execution = executions[ref.execution_id]
                if (
                    execution.session_id != ref.session_id
                    or execution.parent_execution_id is not None
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                commit = commits.get((ref.session_id, ref.sequence))
                items: tuple[SessionTurnItem, ...] = ()
                if commit is not None:
                    base = message_bases[ref.session_id]
                    source = messages[ref.session_id]
                    start = commit.start_message_index - base
                    end = commit.end_message_index - base
                    if start < 0 or end > len(source) or end <= start:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    items = _timeline_items(source[start:end])
                elif execution.status is ExecutionStatus.SUCCEEDED:
                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
                turns.append(
                    SessionTurn(
                        execution.execution_id,
                        execution.status,
                        execution.created_at,
                        execution.updated_at,
                        stored_user_input_view(execution.stored_user_input),
                        commit is not None,
                        items,
                        execution.error_code,
                        execution.safe_error_details,
                    )
                )
            next_cursor = (
                None
                if next_coordinate is None
                else _timeline_cursor(
                    tenant_id=principal.tenant_id,
                    session_id=session_id,
                    source_session_id=next_coordinate[0],
                    before_sequence=next_coordinate[1],
                    signer=self._cursor_signer,
                )
            )
            return Page(tuple(turns), next_cursor)

    async def _timeline_blocks(
        self,
        root: SessionRecord,
        *,
        coordinate: "tuple[str, int] | None",
        limit: int,
    ) -> tuple[
        tuple[tuple[SessionRecord, tuple[SessionTurnRef, ...]], ...],
        "tuple[str, int] | None",
    ]:
        tenant_id = root.tenant_id
        source_id = root.session_id if coordinate is None else coordinate[0]
        source_before = None if coordinate is None else coordinate[1]
        remaining = limit
        newest_first: list[tuple[SessionRecord, tuple[SessionTurnRef, ...]]] = []
        visited: set[str] = set()
        current: SessionRecord | None = root if source_id == root.session_id else None
        while remaining > 0:
            if source_id in visited:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visited.add(source_id)
            if current is None or current.session_id != source_id:
                current = await self._conversation.sessions.get(
                    source_id, tenant_id=tenant_id
                )
                if current is None:
                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
                if (
                    current.tenant_id != root.tenant_id
                    or current.owner_principal_id != root.owner_principal_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head = await self._conversation.sessions.timeline_head(
                source_id, tenant_id=tenant_id
            )
            before = head + 1 if source_before is None else source_before
            if before < 1 or before > head + 1:
                raise AIError(ErrorCode.CURSOR_INVALID)
            available = before - 1
            if available:
                count = min(remaining, available)
                start = before - count
                values = await self._conversation.sessions.list_timeline_turns(
                    source_id,
                    tenant_id=tenant_id,
                    start_sequence=start,
                    end_sequence=before,
                )
                newest_first.append((current, values))
                remaining -= len(values)
                before = start
            source_before = before
            if remaining == 0:
                break
            if before > 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            parent = current.timeline_parent_session_id
            if parent is None:
                source_id = ""
                break
            source_id = parent
            source_before = current.timeline_parent_turn_sequence + 1
            current = None

        next_coordinate: tuple[str, int] | None = None
        if source_id:
            if current is None or current.session_id != source_id:
                current = await self._conversation.sessions.get(
                    source_id, tenant_id=tenant_id
                )
                if current is None:
                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            before = 1 if source_before is None else source_before
            if before > 1 or current.timeline_parent_session_id is not None:
                next_coordinate = (source_id, before)
        return tuple(reversed(newest_first)), next_coordinate

    async def list(self, request: ListSessionRequest) -> Page[SessionView]:
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, "list", request.principal.tenant_id),
        )
        if not 1 <= request.limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        cursor, snapshot = _decode_session_cursor(
            request.cursor,
            request.principal.tenant_id,
            request.principal.principal_id,
            self._cursor_signer,
        )
        snapshot, page = await self._conversation.sessions.list_page(
            tenant_id=request.principal.tenant_id,
            owner_principal_id=request.principal.principal_id,
            cursor=cursor,
            limit=request.limit,
            snapshot=snapshot,
        )
        values = page.items
        active_ids = tuple(
            dict.fromkeys(
                record.active_execution_id
                for record in values
                if record.active_execution_id is not None
            )
        )
        active_by_id = (
            {}
            if not active_ids
            else await self._executions.get_many(
                active_ids,
                tenant_id=request.principal.tenant_id,
            )
        )
        views = tuple(
            await asyncio.gather(
                *(
                    self._view(
                        record,
                        request.principal,
                        active=self._active_execution_ids(
                            record,
                            None
                            if record.active_execution_id is None
                            else active_by_id.get(record.active_execution_id),
                        ),
                    )
                    for record in values
                )
            )
        )
        next_cursor = _make_cursor(
            snapshot,
            request.principal.tenant_id,
            request.principal.principal_id,
            page.next_cursor,
            self._cursor_signer,
        )
        return Page(views, next_cursor)

    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession:
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            record = await self._reconcile_terminal_admission(record)
            active = (
                ()
                if record.active_execution_id is None
                else (record.active_execution_id,)
            )
            return LoadedSession(
                await self._view(record, principal, active=active),
                active,
            )

    async def load_model_context(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> tuple[object, ...]:
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            if self._transcript_store is None or record.continuation is None:
                return ()
            history_id = record.continuation.history_id or record.history_id
            if history_id is not None:
                return await self._transcript_store.load_session_model_context(
                    history_id,
                    tenant_id=record.tenant_id,
                )
            return await self._transcript_store.load_model_context(
                run_id=record.continuation.step_run_id,
            )

    async def _iter_session_messages(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[object]:
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            if self._transcript_store is None or record.continuation is None:
                return
            history_id = record.continuation.history_id or record.history_id
            if history_id is not None:
                async for message in self._transcript_store.iter_session_messages(
                    history_id,
                    tenant_id=record.tenant_id,
                ):
                    yield message
                return
            async for message in self._transcript_store.iter_messages(
                run_id=record.continuation.step_run_id,
            ):
                yield message

    def iter_session_messages(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[object]:
        return self._iter_session_messages(session_id, principal=principal)

    async def resume(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
    ) -> ExecutionHandle:
        return await self._resume(agent_id, binding_digest, session_id, request)

    async def _resume(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
    ) -> ExecutionHandle:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            record = await self._authorized(
                session_id, request.principal, AuthorizationAction.SESSION_READ
            )
            record = await self._reconcile_terminal_admission(record)
            await self._authorization.authorize(
                request.principal,
                AuthorizationAction.EXECUTION_RUN,
                ResourceRef(
                    ResourceKind.EXECUTION,
                    session_id,
                    request.principal.tenant_id,
                ),
            )
            if record.agent_id != agent_id:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
            execution_request = ExecutionRequest(
                user_prompt=request.user_prompt,
                principal=request.principal,
                idempotency_key=request.idempotency_key,
                memory_scope=request.memory_scope,
                mode=request.mode,
                planning=request.planning,
                thinking=request.thinking,
                correlation=request.correlation,
                files=request.files,
            )
            return await self._execution.start_for_session(
                agent_id,
                binding_digest,
                session_id,
                execution_request,
            )

    async def fork(
        self, agent_id: str, session_id: str, request: ForkSessionRequest
    ) -> SessionView:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            source = await self._authorized(
                session_id, request.principal, AuthorizationAction.SESSION_READ
            )
            source = await self._reconcile_terminal_admission(source)
            await self._authorization.authorize(
                request.principal,
                AuthorizationAction.SESSION_CREATE,
                ResourceRef(
                    ResourceKind.SESSION,
                    request.new_session_id,
                    request.principal.tenant_id,
                    request.principal.principal_id,
                ),
            )
            if source.agent_id != agent_id:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
            digest = canonical_sha256(
                {
                    "action": "session.fork",
                    "tenant_id": request.principal.tenant_id,
                    "principal_id": request.principal.principal_id,
                    "source": session_id,
                    "target": request.new_session_id,
                    "agent_id": source.agent_id,
                    "cwd": (
                        source.cwd
                        if request.cwd is None
                        else await self._canonicalize_cwd(request.cwd)
                    ),
                }
            )
            target_cwd = (
                source.cwd
                if request.cwd is None
                else await self._canonicalize_cwd(request.cwd)
            )
            now = datetime.now(timezone.utc)
            target_metadata = dict(source.metadata)
            target = SessionRecord(
                session_id=request.new_session_id,
                tenant_id=source.tenant_id,
                owner_principal_id=source.owner_principal_id,
                status=SessionStatus.OPEN,
                revision=0,
                cwd=target_cwd,
                metadata=target_metadata,
                created_at=now,
                updated_at=now,
                closed_at=None,
                active_execution_id=None,
                continuation=source.continuation,
                agent_id=source.agent_id,
            )
            operation = self._session_terminal_operation(
                request.idempotency_key,
                request.principal.tenant_id,
                request.new_session_id,
                OperationKind.SESSION_FORK,
                digest,
                target.revision,
            )
            if source.history_id is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            target, _ = await self._conversation.sessions.create_fork_with_operation(
                session_id,
                target,
                expected_source_revision=source.revision,
                operation=operation,
            )
            _logger.debug(
                "session forked: source=%s target=%s", session_id, target.session_id
            )
            return await self._view(target, request.principal)

    async def update(
        self, agent_id: str, session_id: str, request: UpdateSessionRequest
    ) -> SessionView:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            return await self._update(agent_id, session_id, request)

    async def _update(
        self, agent_id: str, session_id: str, request: UpdateSessionRequest
    ) -> SessionView:
        current = await self._authorized(
            session_id, request.principal, AuthorizationAction.SESSION_UPDATE
        )
        if current.agent_id != agent_id:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
        if any(key.startswith("linktools.ai.") for key in request.metadata):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        requested_cwd = (
            current.cwd
            if request.cwd is None
            else await self._canonicalize_cwd(request.cwd)
        )
        digest = canonical_sha256(
            {
                "action": "session.update",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "session_id": session_id,
                "expected_revision": request.expected_revision,
                "metadata": request.metadata,
                "cwd": requested_cwd,
            }
        )
        now = datetime.now(timezone.utc)
        next_record = replace(
            current,
            revision=current.revision + 1,
            cwd=requested_cwd,
            metadata=dict(request.metadata),
            updated_at=now,
        )
        operation = self._session_terminal_operation(
            request.idempotency_key,
            request.principal.tenant_id,
            session_id,
            OperationKind.SESSION_UPDATE,
            digest,
            request.expected_revision + 1,
        )
        updated, _ = await self._conversation.sessions.compare_and_swap_with_operation(
            session_id,
            tenant_id=request.principal.tenant_id,
            expected_revision=request.expected_revision,
            next_record=next_record,
            operation=operation,
        )
        _logger.debug(
            "session updated: session=%s revision=%s", session_id, updated.revision
        )
        return await self._view(updated, request.principal)

    async def _canonicalize_cwd(self, value: str | None) -> str | None:
        if value is None:
            return None
        if self._workspace_access is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        try:
            return await self._workspace_access.canonicalize_path(value)
        except AIError:
            raise
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error

    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            return await self._close(session_id, request)

    async def _close(
        self, session_id: str, request: CloseSessionRequest
    ) -> SessionView:
        await self._authorized(
            session_id, request.principal, AuthorizationAction.SESSION_CLOSE
        )
        digest = canonical_sha256(
            {
                "action": "session.close",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "session_id": session_id,
                "force": request.force,
                "wait_timeout_seconds": request.wait_timeout_seconds,
            }
        )
        operation = await self._begin_close_operation(
            request.idempotency_key,
            request.principal.tenant_id,
            session_id,
            digest,
        )
        if operation.status is OperationStatus.SUCCEEDED:
            closed = await self._conversation.sessions.get(
                session_id, tenant_id=request.principal.tenant_id
            )
            if closed is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_close_replay(operation, closed, session_id)
            view = await self._view(closed, request.principal)
            await self._request_session_release(
                session_id,
                request.principal.tenant_id,
                closed.continuation,
            )
            return view
        current = await self._conversation.sessions.get(
            session_id,
            tenant_id=request.principal.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = await self._reconcile_terminal_admission(current)
        if current.status is SessionStatus.CLOSED:
            await self._complete_close_operation(
                operation,
                request.principal.tenant_id,
                session_id,
                canonical_sha256(
                    {"session_id": session_id, "revision": current.revision}
                ),
            )
            view = await self._view(current, request.principal)
            await self._request_session_release(
                session_id,
                request.principal.tenant_id,
                current.continuation,
            )
            return view
        if current.status is SessionStatus.CLEANUP_REQUIRED and not request.force:
            raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED)

        if not request.force:
            if current.status is SessionStatus.OPEN:
                try:
                    current = await self._conversation.sessions.transition_status(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                        expected=frozenset({SessionStatus.OPEN}),
                        next_status=SessionStatus.CLOSING,
                        require_no_active=True,
                    )
                except AIError as error:
                    if error.code not in {
                        ErrorCode.SESSION_ACTIVE_EXECUTIONS,
                        ErrorCode.SESSION_CONFLICT,
                    }:
                        raise
                    latest = await self._conversation.sessions.get(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                    )
                    if latest is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if (
                        latest.status is SessionStatus.OPEN
                        and latest.active_execution_id is not None
                    ):
                        raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
                    if latest.status is SessionStatus.OPEN:
                        current = await self._conversation.sessions.transition_status(
                            session_id,
                            tenant_id=request.principal.tenant_id,
                            expected=frozenset({SessionStatus.OPEN}),
                            next_status=SessionStatus.CLOSING,
                            require_no_active=True,
                        )
                    elif latest.status is SessionStatus.CLEANUP_REQUIRED:
                        raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED)
                    else:
                        current = latest
            if current.status is SessionStatus.CLOSING:
                current = await self._close_idle_session(
                    session_id,
                    request.principal.tenant_id,
                    current,
                )
        else:
            if current.status is SessionStatus.OPEN:
                try:
                    current = await self._conversation.sessions.transition_status(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                        expected=frozenset({SessionStatus.OPEN}),
                        next_status=SessionStatus.CLOSING,
                    )
                except AIError as error:
                    if error.code is ErrorCode.STORAGE_CONFLICT:
                        raise
                    if error.code is not ErrorCode.SESSION_CONFLICT:
                        raise
                    current = await self._conversation.sessions.get(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                    )
                    if current is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if current.status is SessionStatus.OPEN:
                        raise
                    if current.status not in {
                        SessionStatus.CLOSING,
                        SessionStatus.CLEANUP_REQUIRED,
                        SessionStatus.CLOSED,
                    }:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            while True:
                current = await self._conversation.sessions.get(
                    session_id,
                    tenant_id=request.principal.tenant_id,
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current = await self._reconcile_terminal_admission(current)
                if current.active_execution_id is None:
                    break
                execution = await self._active_admitted_execution(current)
                if execution is None:
                    continue
                if execution.status is not ExecutionStatus.FINALIZING:
                    await self._execution.cancel(
                        execution.execution_id,
                        CancelExecutionRequest(
                            request.principal,
                            f"{request.idempotency_key}/{execution.execution_id}",
                            True,
                        ),
                    )
                try:
                    await asyncio.wait_for(
                        self._wait_for_no_active(
                            session_id,
                            request.principal.tenant_id,
                        ),
                        timeout=request.wait_timeout_seconds,
                    )
                except asyncio.TimeoutError as error:
                    current = await self._conversation.sessions.get(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                    )
                    if current is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if current.status is SessionStatus.CLOSING:
                        await self._conversation.sessions.transition_status(
                            session_id,
                            tenant_id=request.principal.tenant_id,
                            expected=frozenset({SessionStatus.CLOSING}),
                            next_status=SessionStatus.CLEANUP_REQUIRED,
                        )
                    raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED) from error
                break
            current = await self._conversation.sessions.get(
                session_id,
                tenant_id=request.principal.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in {
                SessionStatus.CLOSING,
                SessionStatus.CLEANUP_REQUIRED,
            }:
                current = await self._close_idle_session(
                    session_id,
                    request.principal.tenant_id,
                    current,
                )

        await self._complete_close_operation(
            operation,
            request.principal.tenant_id,
            session_id,
            canonical_sha256({"session_id": session_id, "revision": current.revision}),
        )
        _logger.debug(
            "session closed: session=%s revision=%s force=%s",
            session_id,
            current.revision,
            request.force,
        )
        view = await self._view(current, request.principal)
        await self._request_session_release(
            session_id,
            request.principal.tenant_id,
            current.continuation,
        )
        return view

    async def _close_idle_session(
        self,
        session_id: str,
        tenant_id: str,
        current: SessionRecord,
    ) -> SessionRecord:
        if current.active_execution_id is not None:
            raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
        try:
            return await self._conversation.sessions.transition_status(
                session_id,
                tenant_id=tenant_id,
                expected=frozenset({current.status}),
                next_status=SessionStatus.CLOSED,
                closed_at=datetime.now(timezone.utc),
                require_no_active=True,
            )
        except AIError as error:
            if error.code not in {
                ErrorCode.SESSION_ACTIVE_EXECUTIONS,
                ErrorCode.SESSION_CONFLICT,
            }:
                raise
            latest = await self._conversation.sessions.get(
                session_id,
                tenant_id=tenant_id,
            )
            if latest is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if latest.status is SessionStatus.CLOSED:
                return latest
            if latest.active_execution_id is not None:
                raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
            raise

    @asynccontextmanager
    async def _session_consumer(self, session_id: str, tenant_id: str):
        key = (tenant_id, session_id)
        async with self._handoff_condition:
            while True:
                state = self._handoff_states.get(key)
                if state is None:
                    state = _SessionHandoffState()
                    self._handoff_states[key] = state
                if not state.release_in_progress:
                    state.active_consumers += 1
                    break
                await self._handoff_condition.wait()
        cleanup_owner = False
        try:
            yield state
        finally:
            async with self._handoff_condition:
                state.active_consumers -= 1
                if state.active_consumers < 0:
                    raise RuntimeError("session consumer count became negative")
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
                        session_id, tenant_id=tenant_id, continuation=state.continuation
                    )
                    cleanup_succeeded = True
                except BaseException as error:
                    cleanup_error = error
                    if isinstance(error, Exception):
                        _logger.error(
                            "session transient handoff cleanup failed: session=%s",
                            session_id,
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
                    cleanup_error, Exception
                ):
                    raise cleanup_error

    async def _request_session_release(
        self, session_id: str, tenant_id: str, continuation: ConversationCursor | None
    ) -> None:
        key = (tenant_id, session_id)
        async with self._handoff_condition:
            state = self._handoff_states.get(key)
            if state is None:
                raise RuntimeError("session release requested without consumer")
            state.release_requested = True
            state.continuation = continuation
            self._handoff_condition.notify_all()

    async def _wait_for_no_active(self, session_id: str, tenant_id: str) -> None:
        record = await self._conversation.sessions.get(session_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = await self._active_admitted_execution(record)
        if execution is None:
            if record.active_execution_id is not None:
                await self._reconcile_terminal_admission(record)
            return
        await self._execution.wait(
            execution.execution_id,
            principal=Principal(
                record.owner_principal_id,
                tenant_id,
                PrincipalKind.LOCAL_TRUSTED.value,
            ),
        )
        current = await self._conversation.sessions.get(session_id, tenant_id=tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._reconcile_terminal_admission(current)

    async def _authorized(
        self, session_id: str, principal: Principal, action: AuthorizationAction
    ) -> SessionRecord:
        header = await self._conversation.sessions.get_header(
            session_id, tenant_id=principal.tenant_id
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._conversation.sessions.get(
            session_id, tenant_id=principal.tenant_id
        )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record

    async def _view(
        self,
        record: SessionRecord,
        principal: Principal,
        *,
        active: "tuple[str, ...] | None" = None,
    ) -> SessionView:
        if active is None:
            active_execution = await self._active_admitted_execution(record)
            active = (
                ()
                if active_execution is None
                else (active_execution.execution_id,)
            )
        return SessionView(
            record.session_id,
            record.agent_id,
            record.status,
            record.revision,
            record.cwd,
            active,
            record.metadata,
            record.history_quality,
        )

    @staticmethod
    def _active_execution_ids(
        record: SessionRecord,
        execution: "ExecutionRecord | None",
    ) -> tuple[str, ...]:
        active = DefaultSessionService._validate_active_admitted_execution(
            record,
            execution,
        )
        return () if active is None else (active.execution_id,)

    @staticmethod
    def _validate_active_admitted_execution(
        record: SessionRecord,
        execution: "ExecutionRecord | None",
    ) -> "ExecutionRecord | None":
        execution_id = record.active_execution_id
        if execution_id is None:
            if execution is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return None
        if (
            execution is None
            or execution.execution_id != execution_id
            or execution.tenant_id != record.tenant_id
            or execution.session_id != record.session_id
            or execution.parent_execution_id is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return None
        return execution

    async def _active_admitted_execution(
        self,
        record: SessionRecord,
    ) -> "ExecutionRecord | None":
        execution_id = record.active_execution_id
        if execution_id is None:
            return None
        execution = await self._executions.get(execution_id, tenant_id=record.tenant_id)
        return self._validate_active_admitted_execution(record, execution)

    async def _reconcile_terminal_admission(
        self, record: SessionRecord
    ) -> SessionRecord:
        execution_id = record.active_execution_id
        if execution_id is None:
            return record
        if await self._active_admitted_execution(record) is not None:
            return record
        await self._conversation.sessions.release_execution(
            record.session_id,
            tenant_id=record.tenant_id,
            execution_id=execution_id,
        )
        updated = await self._conversation.sessions.get(
            record.session_id,
            tenant_id=record.tenant_id,
        )
        if updated is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "released stale session admission: session=%s execution=%s",
            record.session_id,
            execution_id,
        )
        return updated

    def _session_terminal_operation(
        self,
        operation_id: str,
        tenant_id: str,
        session_id: str,
        operation_kind: OperationKind,
        request_digest: str,
        revision: int,
    ) -> OperationLedgerInput:
        now = datetime.now(timezone.utc)
        return OperationLedgerInput(
            idempotency_key_digest(operation_id),
            tenant_id,
            ResourceKind.SESSION,
            session_id,
            None,
            operation_kind,
            OperationStatus.SUCCEEDED,
            request_digest,
            session_id,
            canonical_sha256({"session_id": session_id, "revision": revision}),
            None,
            False,
            now,
            now,
        )

    async def _begin_close_operation(
        self,
        operation_id: str,
        tenant_id: str,
        session_id: str,
        request_digest: str,
    ) -> OperationLedgerRecord:
        operation_id = idempotency_key_digest(operation_id)
        now = datetime.now(timezone.utc)
        requested = OperationLedgerInput(
            operation_id,
            tenant_id,
            ResourceKind.SESSION,
            session_id,
            None,
            OperationKind.SESSION_CLOSE,
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
            existing = await self._conversation.operations.append(requested)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            existing = await self._conversation.operations.get(
                operation_id,
                tenant_id=tenant_id,
            )
            if existing is None:
                raise
        if (
            existing.tenant_id != tenant_id
            or existing.resource_kind is not ResourceKind.SESSION
            or existing.resource_id != session_id
            or existing.execution_id is not None
            or existing.operation_kind is not OperationKind.SESSION_CLOSE
            or existing.request_digest != request_digest
            or not existing.compactable
        ):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if existing.status is OperationStatus.PENDING:
            if (
                existing.result_ref is not None
                or existing.result_digest is not None
                or existing.error_code is not None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        if existing.status is OperationStatus.SUCCEEDED:
            if (
                existing.result_ref != session_id
                or existing.result_digest is None
                or existing.error_code is not None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _complete_close_operation(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
        result_ref: str,
        result_digest: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        completed = OperationLedgerRecord(
            operation.operation_id,
            tenant_id,
            operation.resource_kind,
            operation.resource_id,
            operation.execution_id,
            operation.operation_kind,
            OperationStatus.SUCCEEDED,
            operation.request_digest,
            result_ref,
            result_digest,
            None,
            operation.compactable,
            operation.sequence,
            operation.created_at,
            now,
        )
        try:
            await self._conversation.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=OperationStatus.PENDING,
                next_record=completed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._conversation.operations.get(
                operation.operation_id, tenant_id=tenant_id
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            self._validate_close_operation_identity(
                current,
                operation,
                tenant_id=tenant_id,
                session_id=result_ref,
            )
            if current.status is OperationStatus.SUCCEEDED:
                self._validate_succeeded_close_operation(
                    current,
                    tenant_id=tenant_id,
                    session_id=result_ref,
                    request_digest=operation.request_digest,
                    result_digest=result_digest,
                )
                return
            if current.status is OperationStatus.PENDING:
                if (
                    current.result_ref is not None
                    or current.result_digest is not None
                    or current.error_code is not None
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                raise
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @staticmethod
    def _validate_close_operation_identity(
        current: OperationLedgerRecord,
        expected: OperationLedgerRecord,
        *,
        tenant_id: str,
        session_id: str,
    ) -> None:
        if (
            current.operation_id != expected.operation_id
            or current.tenant_id != expected.tenant_id
            or current.tenant_id != tenant_id
            or current.resource_kind is not expected.resource_kind
            or current.resource_kind is not ResourceKind.SESSION
            or expected.resource_kind is not ResourceKind.SESSION
            or current.resource_id != expected.resource_id
            or current.resource_id != session_id
            or expected.resource_id != session_id
            or current.execution_id != expected.execution_id
            or current.execution_id is not None
            or expected.execution_id is not None
            or current.operation_kind is not expected.operation_kind
            or current.operation_kind is not OperationKind.SESSION_CLOSE
            or expected.operation_kind is not OperationKind.SESSION_CLOSE
            or current.request_digest != expected.request_digest
            or current.compactable != expected.compactable
            or current.compactable is not True
            or current.sequence != expected.sequence
            or current.created_at != expected.created_at
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @staticmethod
    def _validate_succeeded_close_operation(
        operation: OperationLedgerRecord,
        *,
        tenant_id: str,
        session_id: str,
        request_digest: str,
        result_digest: str,
    ) -> None:
        if (
            operation.tenant_id != tenant_id
            or operation.resource_kind is not ResourceKind.SESSION
            or operation.resource_id != session_id
            or operation.execution_id is not None
            or operation.operation_kind is not OperationKind.SESSION_CLOSE
            or operation.request_digest != request_digest
            or operation.status is not OperationStatus.SUCCEEDED
            or operation.result_ref != session_id
            or operation.result_digest != result_digest
            or operation.error_code is not None
            or not operation.compactable
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @staticmethod
    def _validate_close_replay(
        operation: OperationLedgerRecord,
        session: SessionRecord,
        session_id: str,
    ) -> None:
        if (
            session.status is not SessionStatus.CLOSED
            or session.active_execution_id is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result_digest = canonical_sha256(
            {"session_id": session_id, "revision": session.revision}
        )
        DefaultSessionService._validate_close_operation_identity(
            operation,
            operation,
            tenant_id=session.tenant_id,
            session_id=session_id,
        )
        DefaultSessionService._validate_succeeded_close_operation(
            operation,
            tenant_id=session.tenant_id,
            session_id=session_id,
            request_digest=operation.request_digest,
            result_digest=result_digest,
        )


__all__ = ["DefaultSessionService"]


def _make_cursor(
    snapshot: int,
    tenant_id: str,
    owner_principal_id: str,
    sort_key: "str | None",
    signer: CursorSigner,
) -> "str | None":
    if sort_key is None:
        return None
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "SESSION",
            canonical_sha256({"owner_principal_id": owner_principal_id}),
            sort_key,
            snapshot,
            int(time.time()) + 3600,
        )
    )


def _decode_session_cursor(
    cursor: "str | None", tenant_id: str, owner_principal_id: str, signer: CursorSigner
) -> "tuple[str | None, int | None]":
    if cursor is None:
        return None, None
    try:
        payload = signer.decode(cursor)
        if (
            payload.cursor_version != 1
            or payload.tenant_id != tenant_id
            or payload.resource_kind != "SESSION"
            or payload.filter_digest
            != canonical_sha256({"owner_principal_id": owner_principal_id})
        ):
            raise ValueError("session cursor identity mismatch")
        if not payload.sort_key.strip():
            raise ValueError("session cursor sort key is empty")
        return payload.sort_key, payload.snapshot_or_store_revision
    except (
        Base64Error,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
