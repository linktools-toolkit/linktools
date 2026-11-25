#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session query API and persistence-backed session service."""

import asyncio
import json
import math
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, TypeVar


from ._feature_policy import _RuntimeFeaturePolicy, _runtime_feature_policy
from ._execution import DefaultExecutionService, _SessionStartRead
from ..session import SessionPendingExecutionLookup, SessionReleaseCallback, SessionManager, SessionRecordReader
from ..agent import AgentBinding, AgentBindingContract
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
from ..storage import CacheBackend, TurnCoordination, sql_observation_scope
from ..history import SessionDisplayCache
from ..history import (
    capture_session_refresh,
    project_captured_session_timeline,
    project_materialized_session_turn,
    project_session_timeline,
)
from ..inputs import InputFileSource, stored_user_input_view
from ..execution import (
    CancelExecutionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionService,
    LocalExecutionWaiter,
)
from ..session import CreateSessionRequest, ResumeSessionRequest, SessionDetail, SessionHistoryItem, SessionHistoryReader, SessionTurn, SessionView
from ..persistence.objects import runtime_object_source_reads
from ..session import ConversationCursor, SessionTurnNotice, SessionRecord, SessionAdmission, SessionTurnRef, PublishedSessionTurn
from ..persistence.contracts import ConversationRepositories, ExecutionRepository, ExecutionTerminalPublication
from ..execution import ExecutionRecord
from ..persistence.repositories import SessionRepositoryImpl
from ..persistence.repositories import ExecutionRepositoryImpl
from ..persistence.state import active_state_scope

if TYPE_CHECKING:
    from ._regenerable_turn import _RegenerableTurnRunner





class _SessionExecutionService(ExecutionService, Protocol):
    async def lookup_session_admission(
        self, session_id: str, *, principal: Principal, idempotency_key: str,
    ) -> SessionAdmission | None: ...

    async def start_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
        *,
        session: SessionRecord | None = None,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: "AgentBinding | None" = None,
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
        agent_run_id: str,
        tenant_id: str,
        checkpoint_sequence: int | None = None,
        projection_digest: str | None = None,
    ) -> tuple[object, ...]: ...


class _SessionExecutionAdapter:
    def __init__(self, execution: _SessionExecutionService) -> None:
        self._execution = execution
        self.start_read: _SessionStartRead | None = None

    async def start_for_session(self, agent_id: str, binding_digest: str, session_id: str,
                                request: ResumeSessionRequest, *,
                                session: SessionRecord | None = None,
                                binding_contract: AgentBindingContract | None = None,
                                resolved_binding: AgentBinding | None = None,
                                dependency_hold_id: str | None = None,
                                requires_task_invocation_capture: bool = False,
                                budget_scope_id: str | None = None) -> ExecutionHandle:
        execution_request = ExecutionRequest(
            user_prompt=request.user_prompt, principal=request.principal,
            idempotency_key=request.idempotency_key, memory_scope=request.memory_scope,
            mode=request.mode, planning=request.planning, thinking=request.thinking,
            correlation=request.correlation, files=request.files, budget=request.budget,
            tool_effects=request.tool_effects,
            session_context=request.session_context,
        )
        return await self._execution.start_for_session(
            agent_id, binding_digest, session_id, execution_request,
            session=session,
            binding_contract=binding_contract, resolved_binding=resolved_binding, dependency_hold_id=dependency_hold_id,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
            **({"_start_read": self.start_read} if isinstance(self._execution, DefaultExecutionService)
               and self.start_read is not None and session is self.start_read.session
               and session is not None else {}),
        )

    async def cancel(self, execution_id: str, *, principal: Principal, idempotency_key: str) -> None:
        await self._execution.cancel(execution_id, CancelExecutionRequest(principal, idempotency_key, True))

    async def wait(self, execution_id: str, *, principal: Principal) -> None:
        await self._execution.wait(execution_id, principal=principal)


_AdmissionT = TypeVar("_AdmissionT")


@dataclass
class _SessionWatch:
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    task: asyncio.Task[None] | None = None


class RuntimeSessionService(SessionManager[ExecutionHandle]):
    """Enforce session ownership, Agent identity immutability, and revision CAS."""

    @property
    def regenerable_turns(self) -> "_RegenerableTurnRunner | None":
        return self._regenerable_turns

    async def require_available(self) -> None:
        if self._current_policy is None:
            await super().require_available()
        elif not (await self._current_policy()).session.enabled:
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING, "Session management is disabled")

    async def _authorize_resume_execution(self, session_id: str, principal: Principal) -> None:
        # The built-in execution owner authorizes the actual new or replayed
        # execution before dispatch. A session ID is not an execution resource.
        # Custom services retain the generic manager's admission contract.
        if type(self._admission_execution) is not DefaultExecutionService:
            await super()._authorize_resume_execution(session_id, principal)

    async def start_existing(self, session_id: str, principal: Principal, idempotency_key: str,
                             start: Callable[[str, Callable[..., Awaitable[ExecutionHandle]]], Awaitable[_AdmissionT]],
                             ) -> _AdmissionT:
        """Authorize an existing session once; keep admission within its retention scope."""
        async with self._retain_session(session_id, principal.tenant_id):
            execution = self._admission_execution
            adapter = self._execution
            read_session = self._read_session
            if isinstance(execution, DefaultExecutionService):
                adapter = _SessionExecutionAdapter(execution)
                async def read_session(identity: str, *, tenant_id: str) -> SessionRecord | None:
                    adapter.start_read = await execution._read_session_start(
                        identity, idempotency_key, tenant_id=tenant_id)
                    return adapter.start_read.session
            record = await self.read_authorized(
                session_id, principal, AuthorizationAction.SESSION_READ, read_session=read_session)
            if record.status is not SessionStatus.OPEN:
                raise AIError(ErrorCode.SESSION_CONFLICT)

            async def resume(agent_id: str, binding_digest: str, identity: str,
                             request: ResumeSessionRequest, **kwargs) -> ExecutionHandle:
                binding = kwargs["resolved_binding"]
                if not _runtime_feature_policy(binding.compiled_agent.runtime_contributions).session.enabled:
                    raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING, "Session management is disabled")
                return await self._resume_record(
                    record, agent_id, binding_digest, identity, request, execution=adapter, **kwargs)

            return await start(record.agent_id, resume)

    async def _resume(self, agent_id: str, binding_digest: str, session_id: str,
                      request: ResumeSessionRequest, **kwargs) -> ExecutionHandle:
        execution = self._admission_execution
        if not isinstance(execution, DefaultExecutionService):
            return await super()._resume(agent_id, binding_digest, session_id, request, **kwargs)
        adapter = _SessionExecutionAdapter(execution)
        async def read_session(identity: str, *, tenant_id: str) -> SessionRecord | None:
            adapter.start_read = await execution._read_session_start(identity, request.idempotency_key, tenant_id=tenant_id)
            return adapter.start_read.session
        return await super()._resume(agent_id, binding_digest, session_id, request,
            read_session=read_session, execution=adapter, **kwargs)

    async def _prepare_resume_session(
        self, agent_id: str, session_id: str, principal: Principal,
        *, read_session: SessionRecordReader | None = None,
    ) -> SessionRecord:
        # Runtime starts preserve lazy session creation, but the authorized
        # record is consumed by this same resume rather than fetched twice.
        try:
            record = await self.read_authorized(session_id, principal, AuthorizationAction.SESSION_READ,
                                                read_session=read_session)
        except AIError as error:
            if error.code not in {ErrorCode.SESSION_NOT_FOUND, ErrorCode.AUTHORIZATION_DENIED}:
                raise
            try:
                await self.create(agent_id, CreateSessionRequest(
                    principal, session_id, secrets.token_urlsafe(32), None, {},
                ))
            except AIError as create_error:
                if create_error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
            # Both a created session and a concurrent winner require a fresh
            # read authorization; a create conflict is not an ownership proof.
            record = await self.read_authorized(session_id, principal, AuthorizationAction.SESSION_READ)
        if record.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if record.agent_id != agent_id:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
        return record

    async def lookup_admission(
        self, session_id: str, *, principal: Principal, idempotency_key: str,
    ) -> SessionAdmission | None:
        await self.require_available()
        await self.read_authorized(session_id, principal, AuthorizationAction.SESSION_READ)
        return await self._admission_execution.lookup_session_admission(
            session_id, principal=principal, idempotency_key=idempotency_key,
        )

    def __init__(
        self,
        conversation: ConversationRepositories,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        execution: _SessionExecutionService,
        cursor_signer: CursorSigner,
        *,
        history_reader: SessionHistoryReader,
        regenerable_turns: "_RegenerableTurnRunner | None" = None,
        enabled: bool = True,
        current_policy: Callable[[], Awaitable[_RuntimeFeaturePolicy]] | None = None,
        transcript_store: "_SessionTranscriptStore | None" = None,
        release_terminal: SessionReleaseCallback | None = None,
        workspace_access: InputFileSource | None = None,
        cache: CacheBackend | None = None,
        pending_session_execution: SessionPendingExecutionLookup[ExecutionRecord] | None = None,
        observe_local_session: Callable[..., tuple[ExecutionRecord | None, int]] | None = None,
        read_session_owner: Callable[..., Awaitable[TurnCoordination | None]] | None = None,
        local_waiter: LocalExecutionWaiter | None = None,
        capture_terminal_publication: Callable[..., AbstractContextManager[Callable[[], ExecutionTerminalPublication | None]]] | None = None,
    ) -> None:
        self._watches: dict[str, _SessionWatch] = {}
        self._watches_closed = False
        self._observe_turns: Callable[[Callable[[SessionTurnNotice], None]], Callable[[], None]] | None = None
        self._change_subscriptions: set[Callable[[], None]] = set()
        self._current_policy = current_policy
        self._admission_execution = execution
        self._runtime_conversation = conversation
        self._runtime_executions = executions
        self._runtime_authorization = authorization
        self._runtime_cursor_signer = cursor_signer
        self._runtime_pending_session_execution = pending_session_execution
        self._observe_local_session = observe_local_session
        self._read_session_owner = read_session_owner
        self._local_waiter = local_waiter
        self._capture_terminal_publication = capture_terminal_publication
        super().__init__(
            conversation, executions, authorization, _SessionExecutionAdapter(execution), cursor_signer,
            enabled=enabled, release_terminal=release_terminal,
            canonicalize_path=None if workspace_access is None else workspace_access.canonicalize_path,
            pending_session_execution=pending_session_execution,
            read_session=self._read_session_record,
            observe_view=self._observe_session_view,
        )
        self._history_reader = history_reader
        self._regenerable_turns = regenerable_turns
        self._transcript_store = transcript_store
        self._cache = cache if isinstance(cache, CacheBackend) else None
        self._display_cache = SessionDisplayCache(
            conversation, executions, transcript_store, cursor_signer, self._cache,
            now=lambda: time.time(),
        )

    @contextmanager
    def capture_published_turn(
        self, session_id: str, execution_id: str, *, principal: Principal,
    ) -> Iterator[Callable[[], Awaitable[PublishedSessionTurn | None]]]:
        """Capture a local terminal after-image while consuming a live turn."""
        capture = self._capture_terminal_publication
        if capture is None or not isinstance(self._admission_execution, DefaultExecutionService):
            async def unavailable() -> None:
                return None
            yield unavailable
            return

        with capture(execution_id, tenant_id=principal.tenant_id) as publication:
            async def read() -> PublishedSessionTurn | None:
                receipt = publication()
                if receipt is None or receipt.turn_after_image is None:
                    return None
                image = receipt.turn_after_image
                commit = image.commit
                execution = receipt.execution
                if (commit.session_id != session_id or commit.execution_id != execution_id
                        or execution.session_id != session_id):
                    return None
                await self._admission_execution.authorize_loaded_execution(
                    execution, principal, AuthorizationAction.EXECUTION_READ, owner_principal_id=None,
                )
                await self._runtime_authorization.authorize(
                    principal, AuthorizationAction.SESSION_READ,
                    ResourceRef(ResourceKind.SESSION, session_id, principal.tenant_id, image.owner_principal_id),
                )
                turn = project_materialized_session_turn(
                    SessionTurnRef(session_id, commit.turn_seq, execution_id), commit, execution,
                    tuple(image.messages), inline_output=image.inline_output,
                )
                return None if turn is None else PublishedSessionTurn(turn, commit.turn_seq, image.session_revision)
            yield read

    def bind_turn_observer(self, subscribe: Callable[[Callable[[SessionTurnNotice], None]], Callable[[], None]]) -> None:
        self._observe_turns = subscribe

    def observe_turns(self, on_notice: Callable[[SessionTurnNotice], None]) -> Callable[[], None]:
        if self._watches_closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._observe_turns is None:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        remove = self._observe_turns(on_notice)

        def unsubscribe() -> None:
            self._change_subscriptions.discard(unsubscribe)
            remove()

        self._change_subscriptions.add(unsubscribe)
        return unsubscribe

    def invalidate(self, session_id: str) -> None:
        if self._watches_closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        repository = self._runtime_conversation.sessions
        if type(repository) is not SessionRepositoryImpl:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        repository.notify_session_change(session_id, tenant_id=repository.tenant_id)

    def observe_changes(self, session_id: str, on_invalidate: Callable[[], None]) -> Callable[[], None]:
        if self._watches_closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        repository = self._runtime_conversation.sessions
        if type(repository) is not SessionRepositoryImpl:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)

        def notify() -> None:
            if unsubscribe in self._change_subscriptions:
                on_invalidate()

        remove = repository.observe_changes(session_id, notify)

        def unsubscribe() -> None:
            self._change_subscriptions.discard(unsubscribe)
            remove()

        self._change_subscriptions.add(unsubscribe)
        return unsubscribe

    async def watch(self, session_id: str, *, principal: Principal) -> AsyncIterator[SessionView]:
        """Observe local Runtime session changes without polling.

        Values are observations, not admission reservations. Other processes'
        writes require an explicit refresh. Hints are subscribed before reading;
        one shared read serves concurrent observers of this session.
        """
        repository = self._runtime_conversation.sessions
        if (type(repository) is not SessionRepositoryImpl
                or type(self._runtime_executions) is not ExecutionRepositoryImpl):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        if principal.tenant_id != repository.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if self._watches_closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if active_state_scope() is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        channel = self._watches.setdefault(session_id, _SessionWatch())
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        channel.subscribers.add(queue)
        if channel.task is None:
            unsubscribe = repository.observe_changes(session_id, channel.changed.set)
            channel.task = asyncio.create_task(self._observe_changes(session_id, channel, unsubscribe))
        channel.changed.set()
        previous: SessionView | None = None
        try:
            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                record, active = item
                await self._runtime_authorization.authorize(principal, AuthorizationAction.SESSION_READ,
                    ResourceRef(ResourceKind.SESSION, session_id, principal.tenant_id, record.owner_principal_id))
                view = await self.project_view(record, principal, active=active)
                if view != previous:
                    previous = view
                    yield view
        finally:
            channel.subscribers.discard(queue)
            if not channel.subscribers:
                if self._watches.get(session_id) is channel:
                    self._watches.pop(session_id, None)
                channel.task.cancel()
                await asyncio.gather(channel.task, return_exceptions=True)

    async def _observe_changes(
        self, session_id: str, channel: _SessionWatch, unsubscribe: Callable[[], None],
    ) -> None:
        try:
            while True:
                await channel.changed.wait()
                channel.changed.clear()
                with sql_observation_scope("runtime.session.watch", inherit=False):
                    await self.require_available()
                    record = await self._read_session_record(session_id,
                        tenant_id=self._runtime_conversation.sessions.tenant_id)
                    if record is None:
                        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                    active = record.active_execution_id
                    if active is not None:
                        executions = self._runtime_executions
                        memberships = await executions.read_content(lambda _group: executions.get_membership_many(
                            (active,), tenant_id=self._runtime_conversation.sessions.tenant_id))
                        if channel.changed.is_set():
                            continue
                        self.validate_admission(record, memberships.get(active))
                    if self._observe_local_session is not None:
                        pending, _epoch = self._observe_local_session(session_id,
                            tenant_id=self._runtime_conversation.sessions.tenant_id)
                        if pending is not None:
                            if (pending.session_id != session_id or pending.parent_execution_id is not None
                                    or active is not None and active != pending.execution_id):
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            active = pending.execution_id
                    if channel.changed.is_set():
                        continue
                    self._publish_watch(channel, (record, active))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._publish_watch(channel, error)
            if self._watches.get(session_id) is channel:
                self._watches.pop(session_id, None)
        finally:
            unsubscribe()

    @staticmethod
    def _publish_watch(channel: _SessionWatch, value: object) -> None:
        for queue in channel.subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(value)

    async def close_watches(self) -> None:
        self._watches_closed = True
        if self._regenerable_turns is not None:
            await self._regenerable_turns.close()
        for unsubscribe in tuple(self._change_subscriptions):
            unsubscribe()
        tasks = []
        for channel in self._watches.values():
            self._publish_watch(channel, AIError(ErrorCode.STORAGE_CLOSED))
            if channel.task is not None:
                channel.task.cancel()
                tasks.append(channel.task)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_inactive(
        self, session_id: str, *, principal: Principal,
        timeout_seconds: float | None = None,
        observation_timeout_seconds: float | None = None,
    ) -> SessionView:
        """Observe authoritative inactivity without consuming or reconciling a turn.

        Success is a point-in-time observation, not an admission reservation.
        A local worker completion is only a wake-up hint. Remote observations
        retain a bounded fallback; timeout/cancellation stop only this observer.
        The optional observation budget excludes passive waits and raises
        TimeoutError rather than the overall deadline's WAIT_TIMEOUT.
        """
        await self.require_available()
        if (timeout_seconds is not None and (
            isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds < 0
        )):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (observation_timeout_seconds is not None and (
            isinstance(observation_timeout_seconds, bool)
            or not isinstance(observation_timeout_seconds, (int, float))
            or not math.isfinite(observation_timeout_seconds) or observation_timeout_seconds < 0
        )):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        sessions, executions = self._runtime_conversation.sessions, self._runtime_executions
        observe = self._observe_local_session
        if (type(sessions) is not SessionRepositoryImpl
                or type(executions) is not ExecutionRepositoryImpl or observe is None):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                          safe_details={"reason": "session_inactivity_not_supported"})
        if principal.tenant_id != sessions.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if active_state_scope() is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                          safe_details={"reason": "session_wait_inside_state_transaction"})

        async def read_session() -> tuple[SessionRecord, int]:
            async def read(transaction):
                stored = await transaction.get_record(sessions.session_record_key(session_id))
                if stored is None:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                record = await sessions.get_in_transaction(transaction, session_id, tenant_id=principal.tenant_id)
                if record.session_id != session_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return record, stored.storage_revision
            return await sessions.state_store.read(read)

        async def read_execution(execution_id: str) -> tuple[ExecutionRecord, int]:
            async def read(transaction):
                stored = await transaction.get_record(executions.execution_record_key(execution_id))
                if stored is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                record = await executions.get_in_transaction(transaction, execution_id, tenant_id=principal.tenant_id)
                if record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return record, stored.storage_revision
            return await executions.state_store.read(read)

        async def observe_once() -> tuple[SessionView | None, str | None]:
            first, session_revision = await read_session()
            header = ResourceRef(ResourceKind.SESSION, session_id, principal.tenant_id, first.owner_principal_id)
            await self._runtime_authorization.authorize(principal, AuthorizationAction.SESSION_READ, header)
            local, epoch = observe(session_id, tenant_id=principal.tenant_id)
            if local is not None and (local.session_id != session_id or local.parent_execution_id is not None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            execution, execution_revision = (
                (None, None) if first.active_execution_id is None
                else await read_execution(first.active_execution_id)
            )
            active = self.validate_admission(first, execution)
            if execution is not None:
                await self._runtime_authorization.authorize(
                    principal, AuthorizationAction.EXECUTION_READ,
                    ResourceRef(ResourceKind.EXECUTION, execution.execution_id, principal.tenant_id),
                )
            owner = (None if self._read_session_owner is None else
                     await self._read_session_owner(session_id, tenant_id=principal.tenant_id))
            if local is None and active is None and owner is None:
                second, confirmed_revision = await read_session()
                if (second.owner_principal_id, second.history_id, second.created_at) != (
                    first.owner_principal_id, first.history_id, first.created_at,
                ):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                stable = session_revision == confirmed_revision
                if stable and execution is not None:
                    confirmed, revision = await read_execution(execution.execution_id)
                    stable = (revision == execution_revision and confirmed.execution_id == execution.execution_id
                              and confirmed.session_id == session_id and confirmed.status is execution.status
                              and confirmed.principal_id == execution.principal_id
                              and confirmed.created_at == execution.created_at)
                pending, confirmed_epoch = observe(session_id, tenant_id=principal.tenant_id)
                if stable and pending is None and epoch == confirmed_epoch:
                    return await self.project_view(second, principal, active=None, authorized_header=header), None
                return None, None
            execution_id = (local.execution_id if local is not None else
                            active.execution_id if active is not None else owner.execution_id)
            return None, execution_id

        async def observe_inactivity() -> SessionView:
            retry_delay = 1.0
            while True:
                observation = asyncio.timeout(observation_timeout_seconds)
                try:
                    async with observation:
                        view, execution_id = await observe_once()
                except TimeoutError as error:
                    if not observation.expired():
                        raise
                    raise TimeoutError("session inactivity observation timed out") from error
                if observation.expired():
                    raise TimeoutError("session inactivity observation timed out")
                if view is not None:
                    return view
                if execution_id is None:
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(30.0, retry_delay * 2)
                    continue
                waiter = self._local_waiter
                if waiter is not None and waiter.owns_execution(execution_id, tenant_id=principal.tenant_id):
                    wake = asyncio.timeout(30)
                    try:
                        async with wake:
                            await waiter.wait_terminal(execution_id, tenant_id=principal.tenant_id)
                    except TimeoutError:
                        if not wake.expired():
                            raise
                else:
                    await asyncio.sleep(30)
                retry_delay = 1.0

        deadline = asyncio.timeout(timeout_seconds)
        try:
            async with deadline:
                view = await observe_inactivity()
            if deadline.expired():
                raise TimeoutError("session inactivity wait timed out")
            return view
        except TimeoutError as error:
            if not deadline.expired():
                raise
            raise AIError(ErrorCode.WAIT_TIMEOUT, safe_details={"scope": "session", "resource_id": session_id}) from error

    async def history(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionHistoryItem]":
        async with self.consume_session(session_id, principal.tenant_id):
            record = await self.read_authorized(
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
        async with self.consume_session(session_id, principal.tenant_id):
            if self._transcript_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            return await project_session_timeline(
                self._runtime_conversation,
                self._runtime_executions,
                self._runtime_authorization,
                self._runtime_cursor_signer,
                self._transcript_store,
                session_id,
                principal=principal,
                cursor=cursor,
                limit=limit,
                pending_session_execution=(
                    self._pending_display_execution if self._runtime_pending_session_execution is not None else None
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
        async with self.consume_session(
            session_id, principal.tenant_id,
            display_read=max_age > 0 and self._cache is not None,
        ):
            header = await self._runtime_conversation.sessions.get_header(
                session_id, tenant_id=principal.tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._runtime_authorization.authorize(
                principal, AuthorizationAction.SESSION_READ, header,
            )
            pending = await self._pending_display_execution(header)
            cacheable_request = (pending is None or cursor is None) and max_age > 0 and active_state_scope() is None and self._display_cache.supported()
            if cacheable_request:
                try:
                    validate_page_limit(limit)
                    if cursor is not None:
                        coordinate = json.loads(self._runtime_cursor_signer.decode(cursor).position)
                        cacheable_request = coordinate[0] == session_id
                except (AIError, TypeError, ValueError, IndexError, KeyError):
                    cacheable_request = False
            if cacheable_request:
                cache_key = "session-display:v1:" + canonical_sha256({
                    "namespace": self._runtime_conversation.sessions.namespace,
                    "principal": [principal.tenant_id, principal.principal_id, principal.kind],
                    "session_id": session_id, "cursor": cursor, "limit": limit,
                })
            if cache_key is not None:
                cached = await self._display_cache.read(
                    cache_key, header, principal, cursor=cursor, max_age=max_age, pending=pending,
                )
                if cached is not None:
                    return cached
            state = await capture_session_refresh(
                self._runtime_conversation, self._runtime_executions, session_id,
                tenant_id=principal.tenant_id, cursor=cursor,
                cursor_signer=self._runtime_cursor_signer, limit=limit,
                reconcile=self.reconcile_admission,
                transcript_store=self._transcript_store,
                validate_active=self.validate_admission,
                pending_execution=pending,
                authorized_header=header if self._runtime_pending_session_execution is not None else None,
            )
            record = state.root
            session = await self.project_view(
                record, principal, active=None,
                authorized_header=header if cache_key is not None else None,
            )
            admitted = self.validate_admission(
                record,
                None if record.active_execution_id is None
                else state.executions.get(record.active_execution_id),
            )
            active = None if admitted is None else admitted.execution_id
            if state.pending_execution is not None:
                active = state.pending_execution.execution_id
            session = replace(session, active_execution_id=active)
            if self._transcript_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            with runtime_object_source_reads():
                timeline = await project_captured_session_timeline(
                    state, self._transcript_store, self._runtime_cursor_signer,
                    session_id=session_id, tenant_id=principal.tenant_id,
                    transcript_tenant_id=self._runtime_conversation.sessions.tenant_id,
                )
            detail = SessionDetail(session, timeline)
            if cache_key is not None and record.timeline_parent_session_id is None:
                await self._display_cache.write(cache_key, state, detail, started, max_age)
            return detail

    async def load_model_context(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> tuple[object, ...]:
        async with self.consume_session(session_id, principal.tenant_id):
            record = await self.read_authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            if record.history_quality == "unavailable":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if self._transcript_store is None or record.continuation is None:
                return ()
            return await self._transcript_store.load_conversation_model_context(
                agent_run_id=record.continuation.agent_run_id,
                tenant_id=self._runtime_conversation.sessions.tenant_id,
                checkpoint_sequence=record.continuation.checkpoint_sequence,
                projection_digest=record.continuation.projection_digest,
            )

    async def _iter_session_messages(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[object]:
        async with self.consume_session(session_id, principal.tenant_id):
            record = await self.read_authorized(
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
                    tenant_id=self._runtime_conversation.sessions.tenant_id,
                )
            else:
                messages = self._transcript_store.iter_conversation_message_range(
                    history_id=history_id,
                    agent_run_id=record.continuation.agent_run_id,
                    tenant_id=self._runtime_conversation.sessions.tenant_id,
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















    async def _read_session_record(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        repository = self._runtime_conversation.sessions
        return await repository.state_store.read(
            lambda _transaction: repository.get(session_id, tenant_id=tenant_id)
        )


    async def _pending_display_execution(self, header: ResourceRef) -> ExecutionRecord | None:
        if self._runtime_pending_session_execution is None:
            return None
        pending = await self._runtime_pending_session_execution(header.id, tenant_id=header.tenant_id)
        if (pending is None or pending.session_id != header.id
                or pending.parent_execution_id is not None
                or pending.result is not None
                or pending.status not in {ExecutionStatus.PENDING_START, ExecutionStatus.STARTED, ExecutionStatus.CANCELLING}):
            return None
        return pending

    async def _observe_session_view(
        self, record: SessionRecord, principal: Principal, *, active: str | None,
    ) -> SessionView:
        if record.active_execution_id is not None or record.status is not SessionStatus.OPEN:
            return await self.project_view(record, principal, active=active)
        header = ResourceRef(ResourceKind.SESSION, record.session_id, principal.tenant_id, record.owner_principal_id)
        pending = await self._pending_display_execution(header)
        if pending is None:
            return await self.project_view(record, principal, active=active)

        async def read() -> SessionView:
            # Observation may settle a turn. Re-read committed metadata before
            # using it; display reads retain the store's bounded freshness,
            # while control/admission reads stay authoritative.
            current = await self._runtime_conversation.sessions.get(record.session_id, tenant_id=principal.tenant_id)
            if current is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            view = await self.project_view(current, principal, authorized_header=header)
            if (current.status is SessionStatus.OPEN and current.active_execution_id is None
                    and pending.created_at >= current.created_at
                    and await self._runtime_executions.get(pending.execution_id, tenant_id=principal.tenant_id) is None):
                return replace(view, active_execution_id=pending.execution_id)
            return view

        sessions, executions = self._runtime_conversation.sessions, self._runtime_executions
        if (self._cache is not None and self._display_cache.supported()
                and active_state_scope() is None):
            return await read()
        if (type(sessions) is SessionRepositoryImpl and type(executions) is ExecutionRepositoryImpl
                and sessions.state_store.storage_group is executions.state_store.storage_group):
            return await sessions.state_store.storage_group.read_many(
                (sessions.state_store, executions.state_store), lambda _group: read(),
            )
        return await read()













__all__ = ["RuntimeSessionService"]
