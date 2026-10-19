#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session query API and persistence-backed session service."""

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ

from ..agent import AgentBindingContract
from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
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
    normalize_json_value,
    validate_agent_id,
    validate_page_limit,
)
from ..errors import AIError, ErrorCode
from ..storage import CacheBackend
from ._cursor import decode_cursor as decode_runtime_cursor
from ._cursor import encode_cursor as encode_runtime_cursor
from ._handoff import HandoffGate, HandoffState
from ._session_timeline import (
    SessionTimelineCapture,
    capture_session_refresh,
    project_captured_session_timeline,
    project_session_timeline,
)
from ._input import _InputFileSource, stored_user_input_view
from .service_api import (
    CancelExecutionRequest,
    CloseSessionRequest,
    CreateSessionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionService,
    ForkSessionRequest,
    ListSessionRequest,
    ResumeSessionRequest,
    SessionDetail,
    SessionHistoryItem,
    SessionHistoryReader,
    SessionTurn,
    SessionTurnItem,
    SessionView,
    UpdateSessionRequest,
)
from .state._object_router import runtime_object_source_reads
from .state._contracts import (
    ConversationCursor,
    ConversationRepositories,
    ExecutionRecord,
    ExecutionRepository,
    SessionRecord,
)
from .state._codec import decode_domain, encode_domain
from .state._conversation_repositories import ConversationHistoryRepositoryImpl, SessionRepositoryImpl
from .state._execution_repositories import ExecutionRepositoryImpl
from .state._model_interaction_store import ModelInteractionStateStepArchive
from .state._step_archive import StateStepArchive
from .state import ConversationPersistence
from .state._store import active_state_scope

_SESSION_CURSOR_KIND = "SESSION"


_logger = environ.get_logger("ai.runtime.session")
_VIEW_ACTIVE_UNSET = object()


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
        *,
        binding_contract: "AgentBindingContract | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
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


async def _no_release_terminal(
    session_id: str, *, tenant_id: str, continuation: ConversationCursor | None
) -> None:
    del session_id, tenant_id, continuation


class DefaultSessionService:
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
        transcript_store: "_SessionTranscriptStore | None" = None,
        release_terminal: _SessionReleaseCallback | None = None,
        workspace_access: _InputFileSource | None = None,
        cache: object | None = None,
        pending_session_execution: Callable[..., Awaitable[ExecutionRecord | None]] | None = None,
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
        self._cache = cache if isinstance(cache, CacheBackend) else None
        self._pending_session_execution = pending_session_execution
        self._display_reads: set[asyncio.Future[None]] = set()
        self._display_close_started = False
        self._handoff = HandoffGate[tuple[str, str], ConversationCursor]()

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
            return await self._observed_view(
                record, principal, active=record.active_execution_id,
            )

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
    def _check_display_authority(record: SessionRecord, header: ResourceRef) -> None:
        # An authorization callback may have deleted/recreated the resource.
        if (header.kind is not ResourceKind.SESSION
                or record.session_id != header.id
                or record.owner_principal_id != header.owner_principal_id):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

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
            if not isinstance(identities, dict) or not all(isinstance(key, str) for key in identities):
                return None
            if (detail.session.active_execution_id is not None and detail.session.active_execution_id not in identities
                    or not all(turn.execution_id in identities for turn in detail.timeline.items)
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
                "pending": None if pending is None else self._display_execution_identity(pending),
                "message_end": max((value.end_message_index for value in state.commits.values()), default=0),
                "detail": asdict(detail),
            }
            payload = encode_domain(entry)
            envelope = {"payload": payload, "digest": canonical_sha256(payload)}
            await self._cache.put(key, json.dumps(envelope, separators=(",", ":")).encode())
        except Exception:  # noqa: BLE001 -- best-effort serialization/publication only
            return

    async def list(self, request: ListSessionRequest) -> Page[SessionView]:
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, "list", request.principal.tenant_id),
        )
        limit = validate_page_limit(request.limit)
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
            limit=limit,
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
                    self._observed_view(
                        record,
                        request.principal,
                        active=self._active_execution_id(
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

    async def reconcile(self, session_id: str, *, principal: Principal) -> SessionView:
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            record = await self._reconcile_terminal_admission(record)
            return await self._view(record, principal)

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

    async def resume(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        return await self._resume(
            agent_id,
            binding_digest,
            session_id,
            request,
            binding_contract=binding_contract,
            dependency_hold_id=dependency_hold_id,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
        )

    async def _resume(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            record = await self._authorized(
                session_id, request.principal, AuthorizationAction.SESSION_READ
            )
            record = await self._reconcile_terminal_admission(record)
            if record.cwd is not None and self._workspace_access is None:
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={"field": "cwd", "reason": "workspace_required"},
                )
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
                budget=request.budget,
            )
            return await self._execution.start_for_session(
                agent_id,
                binding_digest,
                session_id,
                execution_request,
                binding_contract=binding_contract,
                dependency_hold_id=dependency_hold_id,
                requires_task_invocation_capture=requires_task_invocation_capture,
                budget_scope_id=budget_scope_id,
            )

    async def fork(
        self, agent_id: str, session_id: str, request: ForkSessionRequest
    ) -> SessionView:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            source = await self._authorized(
                session_id, request.principal, AuthorizationAction.SESSION_READ
            )
            source = await self._reconcile_terminal_admission(source)
            if source.history_quality == "unavailable":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
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
            target_cwd = (
                source.cwd
                if request.cwd is None
                else await self._canonicalize_cwd(request.cwd)
            )
            if target_cwd is not None and self._workspace_access is None:
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={"field": "cwd", "reason": "workspace_required"},
                )
            digest = canonical_sha256(
                {
                    "action": "session.fork",
                    "tenant_id": request.principal.tenant_id,
                    "principal_id": request.principal.principal_id,
                    "source": session_id,
                    "target": request.new_session_id,
                    "agent_id": source.agent_id,
                    "cwd": target_cwd,
                }
            )
            now = datetime.now(timezone.utc)
            target_metadata = dict(source.metadata)
            target = SessionRecord(
                session_id=request.new_session_id,
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
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "cwd", "reason": "workspace_required"},
            )
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
        if self._pending_session_execution is not None:
            pending = await self._pending_session_execution(session_id, tenant_id=request.principal.tenant_id)
            if pending is not None:
                if not request.force:
                    raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
                await self._execution.cancel(
                    pending.execution_id,
                    CancelExecutionRequest(request.principal, f"{request.idempotency_key}/{pending.execution_id}", True),
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
            closed = await self._load_session(
                session_id, tenant_id=request.principal.tenant_id
            )
            if closed is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_close_replay(
                operation,
                closed,
                session_id,
                tenant_id=request.principal.tenant_id,
            )
            view = await self._view(closed, request.principal)
            await self._request_session_release(
                session_id,
                request.principal.tenant_id,
                closed.continuation,
            )
            return view
        current = await self._load_session(
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
                    latest = await self._load_session(
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
                    current = await self._load_session(
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
                current = await self._load_session(
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
                    current = await self._load_session(
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
            current = await self._load_session(
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
            latest = await self._load_session(
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

    async def drain_display_reads(self) -> None:
        """Stop display-cache admission and settle requests before closing sources."""
        self._display_close_started = True
        while self._display_reads:
            await asyncio.wait(tuple(self._display_reads))

    @asynccontextmanager
    async def _session_consumer(
        self, session_id: str, tenant_id: str, *, display_read: bool = False,
    ):
        completed = None
        if display_read:
            if self._display_close_started:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            completed = asyncio.get_running_loop().create_future()
            self._display_reads.add(completed)
        try:
            key = (tenant_id, session_id)
            state = await self._handoff.enter(key)
            try:
                yield state
            finally:
                owner = await self._handoff.leave(key, state)
                if owner:
                    await self._run_session_release(session_id, tenant_id, state)
        finally:
            if completed is not None:
                completed.set_result(None)
                self._display_reads.discard(completed)

    async def _request_session_release(
        self, session_id: str, tenant_id: str, continuation: ConversationCursor | None
    ) -> None:
        key = (tenant_id, session_id)
        state, owner = await self._handoff.request_release(
            key,
            value=continuation,
            require_existing=True,
        )
        if owner:
            await self._run_session_release(session_id, tenant_id, state)

    async def _run_session_release(
        self,
        session_id: str,
        tenant_id: str,
        state: HandoffState[ConversationCursor],
    ) -> None:
        succeeded = False
        cleanup_error: BaseException | None = None
        try:
            await self._release_terminal(
                session_id,
                tenant_id=tenant_id,
                continuation=state.release_value,
            )
            succeeded = True
        except BaseException as error:
            cleanup_error = error
            if isinstance(error, Exception):
                _logger.error(
                    "session transient handoff cleanup failed: session=%s",
                    session_id,
                    exc_info=environ.debug,
                )
        finally:
            await self._handoff.finish_release(
                (tenant_id, session_id),
                state,
                succeeded=succeeded,
            )
        if cleanup_error is not None and not isinstance(cleanup_error, Exception):
            raise cleanup_error

    async def _wait_for_no_active(self, session_id: str, tenant_id: str) -> None:
        record = await self._load_session(session_id, tenant_id=tenant_id)
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
        current = await self._load_session(session_id, tenant_id=tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._reconcile_terminal_admission(current)

    async def _load_session(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        repository = self._conversation.sessions
        return await repository.state_store.read(
            lambda _transaction: repository.get(session_id, tenant_id=tenant_id)
        )

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
        self._check_display_authority(record, header)
        return record

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

    async def _view(
        self,
        record: SessionRecord,
        principal: Principal,
        *,
        active: object = _VIEW_ACTIVE_UNSET,
    ) -> SessionView:
        if active is _VIEW_ACTIVE_UNSET:
            active_execution = await self._active_admitted_execution(record)
            active = None if active_execution is None else active_execution.execution_id
        if active is not None and not isinstance(active, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
    def _active_execution_id(
        record: SessionRecord,
        execution: "ExecutionRecord | None",
    ) -> "str | None":
        active = DefaultSessionService._validate_active_admitted_execution(
            record,
            execution,
        )
        return None if active is None else active.execution_id

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
        execution = await self._executions.get(
            execution_id,
            tenant_id=self._conversation.sessions.tenant_id,
        )
        return self._validate_active_admitted_execution(record, execution)

    async def _reconcile_terminal_admission(
        self, record: SessionRecord
    ) -> SessionRecord:
        execution_id = record.active_execution_id
        if execution_id is None:
            return record
        execution = await self._executions.get(
            execution_id, tenant_id=self._conversation.sessions.tenant_id,
        )
        if self._validate_active_admitted_execution(record, execution) is not None:
            return record
        await self._conversation.sessions.release_execution(
            record.session_id,
            tenant_id=self._conversation.sessions.tenant_id,
            execution_id=execution_id,
            history_unavailable=(execution is not None and execution.invalidates_session_history),
        )
        updated = await self._conversation.sessions.get(
            record.session_id,
            tenant_id=self._conversation.sessions.tenant_id,
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
        *,
        tenant_id: str,
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
            tenant_id=tenant_id,
            session_id=session_id,
        )
        DefaultSessionService._validate_succeeded_close_operation(
            operation,
            tenant_id=tenant_id,
            session_id=session_id,
            request_digest=operation.request_digest,
            result_digest=result_digest,
        )


__all__ = ["DefaultSessionService"]


def _session_filter_digest(owner_principal_id: str) -> str:
    return canonical_sha256({"owner_principal_id": owner_principal_id})


def _make_cursor(
    snapshot: int,
    tenant_id: str,
    owner_principal_id: str,
    sort_key: "str | None",
    signer: CursorSigner,
) -> "str | None":
    if sort_key is None:
        return None
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind=_SESSION_CURSOR_KIND,
        filter_digest=_session_filter_digest(owner_principal_id),
        position=sort_key,
        revision=snapshot,
    )


def _decode_session_cursor(
    cursor: "str | None", tenant_id: str, owner_principal_id: str, signer: CursorSigner
) -> "tuple[str | None, int | None]":
    if cursor is None:
        return None, None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind=_SESSION_CURSOR_KIND,
        filter_digest=_session_filter_digest(owner_principal_id),
    )
    return payload.position, payload.revision
