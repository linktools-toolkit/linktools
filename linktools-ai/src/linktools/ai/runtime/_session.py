#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session query API and persistence-backed session service."""

import json
import math
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from datetime import datetime
from typing import Protocol


from ..session import SessionPendingExecutionLookup, SessionReleaseCallback, SessionManager
from ..agent import AgentBindingContract
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
    normalize_json_value,
    validate_page_limit,
)
from ..errors import AIError, ErrorCode
from ..storage import CacheBackend
from ._session_timeline import (
    SessionTimelineCapture,
    capture_session_refresh,
    project_captured_session_timeline,
    project_session_timeline,
)
from ._input import _InputFileSource, stored_user_input_view
from .service_api import (
    CancelExecutionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionService,
    ResumeSessionRequest,
    SessionDetail,
    SessionHistoryItem,
    SessionHistoryReader,
    SessionTurn,
    SessionTurnItem,
    SessionView,
)
from .state._object_router import runtime_object_source_reads
from .state._contracts import (
    ConversationCursor,
    ConversationRepositories,
    ExecutionRecord,
    ExecutionRepository,
    SessionRecord,
    SessionTurnCommit,
)
from .state._codec import decode_domain, encode_domain
from .state._conversation_repositories import ConversationHistoryRepositoryImpl, SessionRepositoryImpl
from .state._execution_repositories import ExecutionRepositoryImpl
from .state._model_interaction_store import ModelInteractionStateStepArchive
from .state._step_archive import StateStepArchive
from .state import ConversationPersistence
from .state._store import active_state_scope





class _SessionExecutionService(ExecutionService, Protocol):
    async def start_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle: ...


class _SessionTranscriptStore(Protocol):
    def iter_conversation_messages(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
    ) -> AsyncIterator[object]: ...

    def iter_conversation_message_range(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]: ...

    async def load_conversation_model_context(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        message_count: int | None = None,
    ) -> tuple[object, ...]: ...


class _SessionExecutionAdapter:
    def __init__(self, execution: _SessionExecutionService) -> None:
        self._execution = execution

    async def start_for_session(self, agent_id: str, binding_digest: str, session_id: str,
                                request: ResumeSessionRequest, *,
                                binding_contract: AgentBindingContract | None = None,
                                dependency_hold_id: str | None = None,
                                requires_task_invocation_capture: bool = False,
                                budget_scope_id: str | None = None) -> ExecutionHandle:
        execution_request = ExecutionRequest(
            user_prompt=request.user_prompt, principal=request.principal,
            idempotency_key=request.idempotency_key, memory_scope=request.memory_scope,
            mode=request.mode, planning=request.planning, thinking=request.thinking,
            correlation=request.correlation, files=request.files, budget=request.budget,
        )
        return await self._execution.start_for_session(
            agent_id, binding_digest, session_id, execution_request,
            binding_contract=binding_contract, dependency_hold_id=dependency_hold_id,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
        )

    async def cancel(self, execution_id: str, *, principal: Principal, idempotency_key: str) -> None:
        await self._execution.cancel(execution_id, CancelExecutionRequest(principal, idempotency_key, True))

    async def wait(self, execution_id: str, *, principal: Principal) -> None:
        await self._execution.wait(execution_id, principal=principal)


class RuntimeSessionService(SessionManager[ExecutionHandle]):
    """Enforce session ownership, Agent identity immutability, and revision CAS."""

    def __init__(
        self,
        conversation: ConversationRepositories,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        execution: _SessionExecutionService,
        cursor_signer: CursorSigner,
        *,
        history_reader: SessionHistoryReader,
        enabled: bool = True,
        transcript_store: "_SessionTranscriptStore | None" = None,
        release_terminal: SessionReleaseCallback | None = None,
        workspace_access: _InputFileSource | None = None,
        cache: CacheBackend | None = None,
        pending_session_execution: SessionPendingExecutionLookup | None = None,
    ) -> None:
        super().__init__(
            conversation, executions, authorization, _SessionExecutionAdapter(execution), cursor_signer,
            enabled=enabled, release_terminal=release_terminal,
            canonicalize_path=None if workspace_access is None else workspace_access.canonicalize_path,
            pending_session_execution=pending_session_execution,
        )
        self._history_reader = history_reader
        self._transcript_store = transcript_store
        self._cache = cache if isinstance(cache, CacheBackend) else None

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
                None if record.continuation is None else record.continuation.agent_run_id
            )
            continuation_history_id = (
                None
                if record.continuation is None
                else record.continuation.history_id or record.history_id
            )
            return await self._history_reader.history(
                session_id,
                tenant_id=principal.tenant_id,
                continuation_agent_run_id=continuation,
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
        async with self._session_consumer(session_id, principal.tenant_id):
            if self._transcript_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            return await project_session_timeline(
                self._conversation,
                self._executions,
                self._authorization,
                self._cursor_signer,
                self._transcript_store,
                session_id,
                principal=principal,
                cursor=cursor,
                limit=limit,
                pending_session_execution=(
                    self._pending_display_execution if self._pending_session_execution is not None else None
                ),
            )

    async def refresh(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        limit: int = 100,
        max_age: float = 0,
    ) -> SessionDetail:
        """Read a complete display snapshot; zero age retains strong refresh behavior.

        Positive ages allow stale display and delayed detection of independently
        removed or damaged bodies. Admission changes and terminal active executions
        invalidate a cached display. Authorization is checked on every call;
        resource identity and membership use the store's bounded record freshness.
        Control operations never use this cache.
        """
        if isinstance(max_age, bool) or not isinstance(max_age, (int, float)) or not math.isfinite(max_age) or max_age < 0:
            raise ValueError("max_age must be finite and non-negative")
        started = time.time()
        cache_key = None
        async with self._session_consumer(
            session_id, principal.tenant_id,
            display_read=max_age > 0 and self._cache is not None,
        ):
            header = await self._conversation.sessions.get_header(
                session_id, tenant_id=principal.tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                principal, AuthorizationAction.SESSION_READ, header,
            )
            pending = await self._pending_display_execution(header)
            cacheable_request = (pending is None or cursor is None) and max_age > 0 and active_state_scope() is None and self._display_cache_supported()
            if cacheable_request:
                try:
                    validate_page_limit(limit)
                    if cursor is not None:
                        coordinate = json.loads(self._cursor_signer.decode(cursor).position)
                        cacheable_request = coordinate[0] == session_id
                except (AIError, TypeError, ValueError, IndexError, KeyError):
                    cacheable_request = False
            if cacheable_request:
                cache_key = "session-display:v1:" + canonical_sha256({
                    "namespace": self._conversation.sessions.namespace,
                    "principal": [principal.tenant_id, principal.principal_id, principal.kind],
                    "session_id": session_id, "cursor": cursor, "limit": limit,
                })
            if cache_key is not None:
                cached = await self._cached_display(
                    cache_key, header, principal, cursor=cursor, max_age=max_age, pending=pending,
                )
                if cached is not None:
                    return cached
            state = await capture_session_refresh(
                self._conversation, self._executions, session_id,
                tenant_id=principal.tenant_id, cursor=cursor,
                cursor_signer=self._cursor_signer, limit=limit,
                reconcile=self._reconcile_terminal_admission,
                transcript_store=self._transcript_store,
                validate_active=self._validate_active_admitted_execution,
                pending_execution=pending,
                authorized_header=header if self._pending_session_execution is not None else None,
            )
            record = state.root
            if cache_key is not None:
                self._check_display_authority(record, header)
            active = self._active_execution_id(
                record,
                None if record.active_execution_id is None
                else state.executions.get(record.active_execution_id),
            )
            if state.pending_execution is not None:
                active = state.pending_execution.execution_id
            session = await self._view(record, principal, active=active)
            if self._transcript_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            with runtime_object_source_reads():
                timeline = await project_captured_session_timeline(
                    state, self._transcript_store, self._cursor_signer,
                    session_id=session_id, tenant_id=principal.tenant_id,
                    transcript_tenant_id=self._conversation.sessions.tenant_id,
                )
            detail = SessionDetail(session, timeline)
            if cache_key is not None and record.timeline_parent_session_id is None:
                await self._cache_display(cache_key, state, detail, started, max_age)
            return detail

    def _display_cache_supported(self) -> bool:
        sessions = self._conversation.sessions
        histories = self._conversation.histories
        executions = self._executions
        if (self._cache is None
                or type(sessions) is not SessionRepositoryImpl
                or type(histories) is not ConversationHistoryRepositoryImpl
                or type(executions) is not ExecutionRepositoryImpl
                or type(self._transcript_store) is not ConversationPersistence):
            return False
        archive = self._transcript_store.archive
        return (
            type(archive) in (StateStepArchive, ModelInteractionStateStepArchive)
            and archive.state_store is sessions.state_store
            and histories.state_store is sessions.state_store
            and sessions.state_store.storage_group is executions.state_store.storage_group
        )

    @staticmethod
    def _display_session_identity(record: SessionRecord) -> tuple[object, ...]:
        return (record.session_id, record.owner_principal_id, record.agent_id,
                record.history_id, record.created_at)

    @staticmethod
    def _display_execution_identity(record: ExecutionRecord) -> tuple[object, ...]:
        return (record.execution_id, record.session_id, record.parent_execution_id,
                record.created_at, record.principal_id, record.principal_kind,
                record.binding_digest)


    @staticmethod
    def _display_commit_identity(record: SessionTurnCommit) -> tuple[int, str, str, datetime]:
        return record.turn_seq, record.session_id, record.execution_id, record.created_at

    @staticmethod
    def _decode_display(value: object) -> SessionDetail:
        # These public view dataclasses intentionally do not enforce wire types.
        # Validate the disposable representation without registering durable types.
        def text(value: object, *, optional: bool = False) -> None:
            if not isinstance(value, str) and not (optional and value is None):
                raise TypeError("invalid display text")

        session, timeline = value["session"], value["timeline"]
        for name in ("session_id", "agent_id"):
            text(session[name])
        for name in ("cwd", "active_execution_id"):
            text(session[name], optional=True)
        if (type(session["revision"]) is not int or session["revision"] < 0
                or session["history_quality"] not in {"complete", "conservative", "unavailable"}
                or not isinstance(session["metadata"], dict)):
            raise ValueError("invalid session display")
        view = SessionView(**{
            **session, "status": SessionStatus(session["status"]),
            "metadata": normalize_json_value(session["metadata"]),
        })
        text(timeline["next_cursor"], optional=True)
        turns = []
        for turn in timeline["items"]:
            text(turn["execution_id"])
            text(turn["error_code"], optional=True)
            if (not all(isinstance(turn[key], datetime) for key in ("created_at", "updated_at"))
                    or type(turn["conversation_committed"]) is not bool
                    or not isinstance(turn["safe_error_details"], dict)):
                raise ValueError("invalid turn display")
            items = []
            for item in turn["items"]:
                if type(item["ordinal"]) is not int:
                    raise TypeError("invalid item ordinal")
                text(item["item_kind"])
                text(item["tool_name"], optional=True)
                text(item["tool_call_id"], optional=True)
                items.append(SessionTurnItem(**{**item, "content": normalize_json_value(item["content"])}))
            turns.append(SessionTurn(**{
                **turn, "status": ExecutionStatus(turn["status"]), "items": tuple(items),
                "user_input": normalize_json_value(turn["user_input"]),
                "output": normalize_json_value(turn["output"]),
                "safe_error_details": normalize_json_value(turn["safe_error_details"]),
            }))
        return SessionDetail(view, Page(tuple(turns), timeline["next_cursor"]))

    async def _cached_display(
        self, key: str, header: ResourceRef, principal: Principal,
        *, cursor: str | None, max_age: float, pending: ExecutionRecord | None = None,
    ) -> SessionDetail | None:
        try:
            raw = await self._cache.get(key)
            if raw is None:
                return None
            envelope = json.loads(raw)
            if canonical_sha256(envelope["payload"]) != envelope["digest"]:
                return None
            entry = decode_domain(envelope["payload"], dict[str, object])
            started, expires = entry["started"], entry["expires"]
            if (entry["key"] != key or not isinstance(started, (float, int))
                    or not isinstance(expires, (float, int))
                    or not math.isfinite(started) or not math.isfinite(expires)
                    or not started <= time.time() < min(expires, started + max_age)):
                return None
            pending_identity = None if pending is None else self._display_execution_identity(pending)
            if entry.get("pending") != pending_identity:
                return None
            detail = self._decode_display(entry["detail"])
            if detail.session.session_id != header.id:
                return None
            # The exact input token was validated when this entry was captured;
            # decode it again for expiration/key rotation, without extending it.
            for token in (cursor, detail.timeline.next_cursor):
                if token is not None:
                    self._cursor_signer.decode(token)
            identities = entry["executions"]
            commits = entry["commits"]
            if (not isinstance(identities, dict) or not all(isinstance(key, str) for key in identities)
                    or not isinstance(commits, dict) or not all(isinstance(key, str) for key in commits)
                    or set(identities) & set(commits)):
                return None
            for execution_id, identity in commits.items():
                sequence, source_session_id, committed_execution_id, created_at = identity
                if (type(sequence) is not int or sequence < 1
                        or source_session_id != header.id or committed_execution_id != execution_id
                        or not isinstance(created_at, datetime)):
                    return None
            if len({identity[0] for identity in commits.values()}) != len(commits):
                return None
            if (detail.session.active_execution_id is not None and detail.session.active_execution_id not in identities
                    or not all(
                        turn.execution_id in (commits if turn.conversation_committed else identities)
                        for turn in detail.timeline.items
                    )
                    or detail.session.session_id != entry["root"][0]
                    or detail.session.agent_id != entry["root"][2]):
                return None
            required_end = entry["message_end"]
            if not isinstance(required_end, int) or required_end < 0:
                return None
        except Exception:  # noqa: BLE001 -- a disposable cache never replaces origin errors
            return None

        sessions = self._conversation.sessions
        histories = self._conversation.histories
        async def validate() -> bool:
            record = await sessions.get(header.id, tenant_id=principal.tenant_id)
            if record is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            self._check_display_authority(record, header)
            if (record.timeline_parent_session_id is not None
                    or self._display_session_identity(record) != entry["root"]
                    or record.revision > detail.session.revision):
                return False
            ids = tuple(identities)
            current_revision = record.revision == detail.session.revision
            if current_revision and record.active_execution_id is not None and record.active_execution_id not in identities:
                ids += (record.active_execution_id,)
            records = await self._executions.get_many(ids, tenant_id=principal.tenant_id) if ids else {}
            if len(records) != len(ids):
                # A pre-admission negative cache entry can outlive the source
                # capture. Only authoritative absence is an integrity failure.
                records = await self._executions.get_many_fresh(ids, tenant_id=principal.tenant_id)
            if len(records) != len(ids):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current_revision:
                active = self._validate_active_admitted_execution(
                    record, records.get(record.active_execution_id) if record.active_execution_id else None,
                )
                if (record.active_execution_id != detail.session.active_execution_id
                        or record.active_execution_id is not None and active is None):
                    return False
            # A source-captured display can be newer than another reader's
            # bounded session cache. Do not replace that newer whole snapshot
            # merely because the same resource's older revision is still hot.
            if any(value.session_id != record.session_id or value.parent_execution_id is not None
                   for value in records.values()):
                return False
            if any(self._display_execution_identity(records[key]) != identity
                   for key, identity in identities.items()):
                return False
            if commits:
                sequences = tuple(identity[0] for identity in commits.values())
                current_commits = await sessions.list_timeline_commits(
                    record.session_id, tenant_id=principal.tenant_id,
                    start_turn_seq=min(sequences), end_turn_seq=max(sequences) + 1,
                )
                retained = {value.execution_id: self._display_commit_identity(value) for value in current_commits}
                if any(retained.get(key) != identity for key, identity in commits.items()):
                    return False
            if required_end == 0:
                return True
            if record.history_id is None:
                return False
            value = await histories.get_with_head(
                record.history_id, tenant_id=principal.tenant_id,
            )
            if (value is None or value[0].session_id != record.session_id
                    or value[0].parent_history_id is not None or value[1].message_count < required_end):
                # A pending-page head may predate the committed display. Confirm
                # source membership and replace that older ordinary cache batch.
                value = await histories.get_with_head_fresh(record.history_id, tenant_id=principal.tenant_id)
            if value is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            history, head = value
            return (history.session_id == record.session_id and history.parent_history_id is None
                    and head.message_count >= required_end)

        # Reevaluate policy above on every request; only immutable resource
        # identity/membership and display state use the existing bounded cache.
        # A mismatch rebuilds the page in the authoritative capture below.
        valid = await validate()
        if valid and time.time() < min(expires, started + max_age):
            if pending is not None:
                # Only the canonical page was cached. Progress and status always
                # come from this request's authorized pending publication.
                turn = SessionTurn(
                    pending.execution_id, pending.status, pending.created_at, pending.updated_at,
                    stored_user_input_view(pending.stored_user_input), False, (),
                    pending.error_code, pending.safe_error_details,
                )
                return SessionDetail(
                    replace(detail.session, active_execution_id=pending.execution_id),
                    Page((*detail.timeline.items, turn), detail.timeline.next_cursor),
                )
            return detail
        return None

    async def _cache_display(
        self, key: str, state: SessionTimelineCapture, detail: SessionDetail,
        started: float, max_age: float,
    ) -> None:
        if state.root.history_id is None or time.time() >= started + max_age:
            return
        pending = state.pending_execution
        if pending is not None:
            # Root pages reserve exactly one slot for the live pending turn.
            # Its mutable content is never included in the cached page.
            if not detail.timeline.items or detail.timeline.items[-1].execution_id != pending.execution_id:
                return
            detail = SessionDetail(
                replace(detail.session, active_execution_id=None),
                Page(detail.timeline.items[:-1], detail.timeline.next_cursor),
            )
        try:
            entry = {
                "key": key, "started": started, "expires": started + max_age,
                "root": self._display_session_identity(state.root),
                "executions": {key: self._display_execution_identity(value)
                               for key, value in state.executions.items()
                               if pending is None or key != pending.execution_id},
                "commits": {value.execution_id: self._display_commit_identity(value)
                            for value in state.commits.values()},
                "pending": None if pending is None else self._display_execution_identity(pending),
                "message_end": max((value.end_message_index for value in state.commits.values()), default=0),
                "detail": asdict(detail),
            }
            payload = encode_domain(entry)
            envelope = {"payload": payload, "digest": canonical_sha256(payload)}
            await self._cache.put(key, json.dumps(envelope, separators=(",", ":")).encode())
        except Exception:  # noqa: BLE001 -- best-effort serialization/publication only
            return



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
            if record.history_quality == "unavailable":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if self._transcript_store is None or record.continuation is None:
                return ()
            history_id = record.continuation.history_id or record.history_id
            return await self._transcript_store.load_conversation_model_context(
                history_id=history_id,
                agent_run_id=record.continuation.agent_run_id,
                tenant_id=self._conversation.sessions.tenant_id,
                message_count=record.continuation.message_count,
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
            message_count = record.continuation.message_count
            if message_count is None:
                messages = self._transcript_store.iter_conversation_messages(
                    history_id=history_id,
                    agent_run_id=record.continuation.agent_run_id,
                    tenant_id=self._conversation.sessions.tenant_id,
                )
            else:
                messages = self._transcript_store.iter_conversation_message_range(
                    history_id=history_id,
                    agent_run_id=record.continuation.agent_run_id,
                    tenant_id=self._conversation.sessions.tenant_id,
                    start=0,
                    end=message_count,
                )
            async for message in messages:
                yield message

    def iter_session_messages(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[object]:
        return self._iter_session_messages(session_id, principal=principal)















    async def _load_session(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        repository = self._conversation.sessions
        return await repository.state_store.read(
            lambda _transaction: repository.get(session_id, tenant_id=tenant_id)
        )


    async def _pending_display_execution(self, header: ResourceRef) -> ExecutionRecord | None:
        if self._pending_session_execution is None:
            return None
        pending = await self._pending_session_execution(header.id, tenant_id=header.tenant_id)
        if (pending is None or pending.session_id != header.id
                or pending.principal_id != header.owner_principal_id
                or pending.parent_execution_id is not None
                or pending.result is not None
                or pending.status not in {ExecutionStatus.PENDING_START, ExecutionStatus.STARTED, ExecutionStatus.CANCELLING}):
            return None
        return pending

    async def _observed_view(
        self, record: SessionRecord, principal: Principal, *, active: str | None,
    ) -> SessionView:
        if record.active_execution_id is not None or record.status is not SessionStatus.OPEN:
            return await self._view(record, principal, active=active)
        header = ResourceRef(ResourceKind.SESSION, record.session_id, principal.tenant_id, record.owner_principal_id)
        pending = await self._pending_display_execution(header)
        if pending is None:
            return await self._view(record, principal, active=active)

        async def read() -> SessionView:
            # Observation may settle a turn. Re-read committed metadata before
            # using it; display reads retain the store's bounded freshness,
            # while control/admission reads stay authoritative.
            current = await self._conversation.sessions.get(record.session_id, tenant_id=principal.tenant_id)
            if current is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            self._check_display_authority(current, header)
            admitted = await self._active_admitted_execution(current)
            observed = None if admitted is None else admitted.execution_id
            if (current.status is SessionStatus.OPEN and current.active_execution_id is None
                    and pending.created_at >= current.created_at
                    and await self._executions.get(pending.execution_id, tenant_id=principal.tenant_id) is None):
                observed = pending.execution_id
            return await self._view(current, principal, active=observed)

        sessions, executions = self._conversation.sessions, self._executions
        if (self._cache is not None and self._display_cache_supported()
                and active_state_scope() is None):
            return await read()
        if (type(sessions) is SessionRepositoryImpl and type(executions) is ExecutionRepositoryImpl
                and sessions.state_store.storage_group is executions.state_store.storage_group):
            return await sessions.state_store.storage_group.read_many(
                (sessions.state_store, executions.state_store), lambda _group: read(),
            )
        return await read()













__all__ = ["RuntimeSessionService"]
