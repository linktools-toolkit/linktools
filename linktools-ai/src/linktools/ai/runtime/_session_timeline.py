#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session timeline capture, coordinated refresh, and read-only projection."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorSigner,
    ExecutionStatus,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    canonical_sha256,
    validate_page_limit,
)
from ..errors import AIError, ErrorCode
from ._cursor import decode_cursor as decode_runtime_cursor
from ._cursor import encode_cursor as encode_runtime_cursor
from ._input import stored_user_input_view
from .service_api import SessionTurn, SessionTurnItem
from .state._contracts import (
    ConversationRepositories,
    ExecutionRecord,
    ExecutionRepository,
    SessionRecord,
    SessionTurnCommitRef,
    SessionTurnRef,
)
from .state._conversation_repositories import SessionRepositoryImpl
from .state._execution_repositories import ExecutionRepositoryImpl
from .state._store import StateGroupTransaction
from .state._views import project_session_history_message

_TIMELINE_PROJECTION_VERSION = 1
_TIMELINE_CURSOR_KIND = "SESSION_TIMELINE"


class SessionTimelineTranscriptStore(Protocol):
    def iter_conversation_message_range(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]: ...


def _timeline_items(messages: tuple[ModelMessage, ...]) -> tuple[SessionTurnItem, ...]:
    values: list[SessionTurnItem] = []
    for message in messages:
        projected = project_session_history_message(message)
        if isinstance(message, ModelRequest):
            projected = tuple(
                item for item in projected if item.item_kind == "tool_result"
            )
        elif isinstance(message, ModelResponse):
            projected = tuple(
                item
                for item in projected
                if item.item_kind in {"assistant", "thinking", "tool_call"}
            )
        else:
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
    before_turn_seq: int,
    signer: CursorSigner,
) -> str:
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind=_TIMELINE_CURSOR_KIND,
        filter_digest=_timeline_filter_digest(session_id),
        position=json.dumps(
            [source_session_id, before_turn_seq],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
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
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind=_TIMELINE_CURSOR_KIND,
        filter_digest=_timeline_filter_digest(session_id),
    )
    try:
        coordinate = json.loads(payload.position)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.revision != 0
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


async def _timeline_blocks(
    conversation: ConversationRepositories,
    root: SessionRecord,
    *,
    coordinate: "tuple[str, int] | None",
    limit: int,
    pending_execution: ExecutionRecord | None = None,
) -> tuple[
    tuple[tuple[SessionRecord, tuple[SessionTurnRef, ...]], ...],
    "tuple[str, int] | None",
]:
    tenant_id = conversation.sessions.tenant_id
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
            current = await conversation.sessions.get(source_id, tenant_id=tenant_id)
            if current is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if current.owner_principal_id != root.owner_principal_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head = await conversation.sessions.timeline_head(source_id, tenant_id=tenant_id)
        pending = pending_execution if coordinate is None and source_id == root.session_id else None
        display_head = head + (pending is not None)
        before = display_head + 1 if source_before is None else source_before
        if before < 1 or before > display_head + 1:
            raise AIError(ErrorCode.CURSOR_INVALID)
        available = before - 1
        if available:
            count = min(remaining, available)
            start = before - count
            values = await conversation.sessions.list_timeline_turns(
                source_id,
                tenant_id=tenant_id,
                start_turn_seq=min(start, head) if pending is not None and head else start,
                end_turn_seq=min(before, head + 1),
            )
            if pending is not None:
                # An observation cannot replace a missing canonical execution
                # behind an existing turn, including a pending-only page.
                if any(value.execution_id == pending.execution_id for value in values):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values = tuple(value for value in values if value.turn_seq >= start)
                values += (SessionTurnRef(source_id, head + 1, pending.execution_id),)
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
        source_before = current.timeline_parent_turn_seq + 1
        current = None

    next_coordinate: tuple[str, int] | None = None
    if source_id:
        if current is None or current.session_id != source_id:
            current = await conversation.sessions.get(source_id, tenant_id=tenant_id)
            if current is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        before = 1 if source_before is None else source_before
        if before > 1 or current.timeline_parent_session_id is not None:
            next_coordinate = (source_id, before)
    return tuple(reversed(newest_first)), next_coordinate


_TimelineBlocks = tuple[tuple[SessionRecord, tuple[SessionTurnRef, ...]], ...]


@dataclass(frozen=True, slots=True)
class SessionTimelineCapture:
    root: SessionRecord
    blocks: _TimelineBlocks
    next_coordinate: tuple[str, int] | None
    executions: Mapping[str, ExecutionRecord]
    commits: dict[tuple[str, int], SessionTurnCommitRef]
    pending_execution: ExecutionRecord | None = None


class _ActiveSessionAdmission(Exception):
    def __init__(self, record: SessionRecord) -> None:
        self.record = record


async def _capture_timeline_state(
    conversation: ConversationRepositories,
    executions: ExecutionRepository,
    session_id: str,
    *,
    tenant_id: str,
    cursor: str | None,
    cursor_signer: CursorSigner,
    limit: int,
    pending_execution: ExecutionRecord | None = None,
    authorized_header: ResourceRef | None = None,
    refresh_phase: Literal["initial", "final"] | None = None,
    transcript_store: SessionTimelineTranscriptStore | None = None,
    validate_active: Callable[[SessionRecord, ExecutionRecord | None], ExecutionRecord | None] | None = None,
) -> SessionTimelineCapture:
    shared_native_scope = (
        type(conversation.sessions) is SessionRepositoryImpl
        and type(executions) is ExecutionRepositoryImpl
        and conversation.sessions.state_store.storage_group
        is executions.state_store.storage_group
    )

    async def read_blocks() -> tuple[SessionRecord, _TimelineBlocks, tuple[str, int] | None]:
        nonlocal pending_execution
        root = await conversation.sessions.get(session_id, tenant_id=tenant_id)
        if root is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if authorized_header is not None and (
                authorized_header.kind is not ResourceKind.SESSION
                or authorized_header.tenant_id != tenant_id
                or root.session_id != authorized_header.id
                or root.owner_principal_id != authorized_header.owner_principal_id):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if refresh_phase == "initial" and root.active_execution_id is not None:
            if not shared_native_scope or validate_active is None:
                raise _ActiveSessionAdmission(root)
            active = await executions.get(root.active_execution_id, tenant_id=tenant_id)
            if validate_active(root, active) is None:
                # Terminal admission cleanup precedes page validation and must
                # leave the snapshot before attempting any mutation.
                raise _ActiveSessionAdmission(root)
        if refresh_phase is not None and transcript_store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if pending_execution is not None:
            # The observation is never admission: canonical state in this same
            # snapshot wins if the owner committed or promoted during capture.
            if (root.status is not SessionStatus.OPEN
                    or root.active_execution_id is not None
                    or pending_execution.session_id != root.session_id
                    or pending_execution.principal_id != root.owner_principal_id
                    or pending_execution.created_at < root.created_at
                    or await executions.get(pending_execution.execution_id, tenant_id=tenant_id) is not None):
                pending_execution = None
        page_limit = validate_page_limit(limit)
        coordinate = _decode_timeline_cursor(
            cursor, tenant_id=tenant_id, session_id=session_id, signer=cursor_signer,
        )
        blocks, next_coordinate = await _timeline_blocks(
            conversation, root, coordinate=coordinate, limit=page_limit,
            pending_execution=pending_execution,
        )
        return root, blocks, next_coordinate

    async def execution_records(
        root: SessionRecord, blocks: _TimelineBlocks,
    ) -> Mapping[str, ExecutionRecord]:
        ids = tuple(ref.execution_id for _record, refs in blocks for ref in refs)
        if refresh_phase is not None and root.active_execution_id is not None:
            ids += (root.active_execution_id,)
        ids = tuple(dict.fromkeys(ids))
        # Only the synthetic newest turn may omit canonical membership.
        # A continuation consists entirely of persisted references.
        pending = {} if pending_execution is None or cursor is not None else {pending_execution.execution_id: pending_execution}
        ids = tuple(value for value in ids if value not in pending)
        records = await executions.get_many(ids, tenant_id=tenant_id) if ids else {}
        if len(records) != len(ids):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return {**pending, **records}

    async def commit_records(
        blocks: _TimelineBlocks,
    ) -> dict[tuple[str, int], SessionTurnCommitRef]:
        commits: dict[tuple[str, int], SessionTurnCommitRef] = {}
        for record, refs in blocks:
            if not refs:
                continue
            values = await conversation.sessions.list_timeline_commits(
                record.session_id, tenant_id=conversation.sessions.tenant_id,
                start_turn_seq=refs[0].turn_seq,
                end_turn_seq=refs[-1].turn_seq + 1,
            )
            for value in values:
                commits[(record.session_id, value.turn_seq)] = value
        return commits

    # Custom owners may schedule child tasks or perform their own I/O boundaries.
    if (type(conversation.sessions) is not SessionRepositoryImpl
            or type(executions) is not ExecutionRepositoryImpl):
        root, blocks, next_coordinate = await read_blocks()
        records = await execution_records(root, blocks)
        commits = await commit_records(blocks)
        return SessionTimelineCapture(root, blocks, next_coordinate, records, commits, pending_execution)

    session_store = conversation.sessions.state_store
    execution_store = executions.state_store
    if session_store.storage_group is execution_store.storage_group:
        async def read(_group: StateGroupTransaction) -> SessionTimelineCapture:
            root, blocks, next_coordinate = await read_blocks()
            records = await execution_records(root, blocks)
            commits = await commit_records(blocks)
            return SessionTimelineCapture(root, blocks, next_coordinate, records, commits, pending_execution)

        return await session_store.storage_group.read_many(
            (session_store, execution_store), read,
        )

    # Separate routes keep their own snapshots; no cross-database scope is implied.
    root, blocks, next_coordinate = await session_store.read(lambda _transaction: read_blocks())
    records = await execution_records(root, blocks)
    commits = await session_store.read(lambda _transaction: commit_records(blocks))
    return SessionTimelineCapture(root, blocks, next_coordinate, records, commits, pending_execution)


async def capture_session_refresh(
    conversation: ConversationRepositories,
    executions: ExecutionRepository,
    session_id: str,
    *,
    tenant_id: str,
    cursor: str | None,
    cursor_signer: CursorSigner,
    limit: int,
    reconcile: Callable[[SessionRecord], Awaitable[SessionRecord]],
    transcript_store: SessionTimelineTranscriptStore | None,
    validate_active: Callable[[SessionRecord, ExecutionRecord | None], ExecutionRecord | None],
    pending_execution: ExecutionRecord | None = None,
    authorized_header: ResourceRef | None = None,
) -> SessionTimelineCapture:
    """Capture authorized metadata, reconciling terminal admission outside read scopes."""
    try:
        return await _capture_timeline_state(
            conversation, executions, session_id, tenant_id=tenant_id,
            cursor=cursor, cursor_signer=cursor_signer, limit=limit,
            pending_execution=pending_execution, authorized_header=authorized_header,
            refresh_phase="initial", transcript_store=transcript_store,
            validate_active=validate_active,
        )
    except _ActiveSessionAdmission as admission:
        await reconcile(admission.record)
    return await _capture_timeline_state(
        conversation, executions, session_id, tenant_id=tenant_id,
        cursor=cursor, cursor_signer=cursor_signer, limit=limit,
        pending_execution=pending_execution, authorized_header=authorized_header,
        refresh_phase="final", transcript_store=transcript_store,
    )


async def project_session_timeline(
    conversation: ConversationRepositories,
    executions: ExecutionRepository,
    authorization: AuthorizationPolicy,
    cursor_signer: CursorSigner,
    transcript_store: SessionTimelineTranscriptStore,
    session_id: str,
    *,
    principal: Principal,
    cursor: "str | None" = None,
    limit: int = 100,
    pending_session_execution: Callable[[ResourceRef], Awaitable[ExecutionRecord | None]] | None = None,
) -> Page[SessionTurn]:
    limit = validate_page_limit(limit)
    header = await conversation.sessions.get_header(
        session_id,
        tenant_id=principal.tenant_id,
    )
    if header is None:
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
    await authorization.authorize(principal, AuthorizationAction.SESSION_READ, header)
    pending = None if pending_session_execution is None else await pending_session_execution(header)
    state = await _capture_timeline_state(
        conversation, executions, session_id,
        tenant_id=principal.tenant_id, cursor=cursor, cursor_signer=cursor_signer,
        limit=limit, pending_execution=pending,
        authorized_header=header if pending_session_execution is not None else None,
    )
    return await project_captured_session_timeline(
        state, transcript_store, cursor_signer, session_id=session_id,
        tenant_id=principal.tenant_id, transcript_tenant_id=conversation.sessions.tenant_id,
    )


async def project_captured_session_timeline(
    state: SessionTimelineCapture,
    transcript_store: SessionTimelineTranscriptStore,
    cursor_signer: CursorSigner,
    *,
    session_id: str,
    tenant_id: str,
    transcript_tenant_id: str,
) -> Page[SessionTurn]:
    """Load transcript bodies after the captured metadata scope has closed."""
    blocks = state.blocks
    next_coordinate = state.next_coordinate
    refs = tuple(ref for _record, values in blocks for ref in values)
    if not refs:
        return Page((), None)
    execution_records = state.executions
    commits = state.commits
    messages: dict[str, tuple[ModelMessage, ...]] = {}
    message_bases: dict[str, int] = {}
    for record, values in blocks:
        if not values:
            continue
        committed = tuple(
            value for (source_id, _sequence), value in commits.items()
            if source_id == record.session_id
        )
        if not committed:
            continue
        if record.continuation is None:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        range_start = min(item.start_message_index for item in committed)
        range_end = max(item.end_message_index for item in committed)
        loaded = cast(
            tuple[ModelMessage, ...],
            tuple(
                [
                    item
                    async for item in transcript_store.iter_conversation_message_range(
                        history_id=record.history_id,
                        agent_run_id=record.continuation.agent_run_id,
                        tenant_id=transcript_tenant_id,
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
        execution = execution_records[ref.execution_id]
        if (
            execution.session_id != ref.session_id
            or execution.parent_execution_id is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        commit = commits.get((ref.session_id, ref.turn_seq))
        items: tuple[SessionTurnItem, ...] = ()
        if commit is not None:
            if commit.execution_id != ref.execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
            tenant_id=tenant_id,
            session_id=session_id,
            source_session_id=next_coordinate[0],
            before_turn_seq=next_coordinate[1],
            signer=cursor_signer,
        )
    )
    return Page(tuple(turns), next_cursor)


__all__ = [
    "SessionTimelineCapture",
    "SessionTimelineTranscriptStore",
    "capture_session_refresh",
    "project_captured_session_timeline",
    "project_session_timeline",
]
