#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission and settlement for ordinary, regenerable model turns."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone

from linktools.core import environ
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved
from threading import BoundedSemaphore
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy.exc import IntegrityError

from ..agent import AgentBindingContract, bind_output, restore_output
from ..capability import AgentContext
from ..core import AuthorizationAction, ExecutionDeltaType, ExecutionEventType, ImmutableJsonMapping, JsonValue, Principal, ResourceKind, ResourceRef, ToolOperationStatus, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..inputs import stored_input_attachment_views
from ..model import MaterializedUserContent, StoredUserInput, native_graph_input_content, normalize_input_files, validate_model_inputs
from ..persistence.codec import decode_domain, encode_domain
from ..persistence.contracts import ToolOperationRecord
from ..persistence.repositories import ToolRepositoryImpl
from ..persistence.turns import (
    MAX_REMEMBERED_SCOPE_BYTES, MAX_REMEMBERED_SCOPES, CanonicalDeferredFrontier, SessionInstructionFrontier,
    SessionTurnKey, SessionTurnSegment, decode_instruction_scopes, decode_session_lineage,
)
from ..session import SessionContext, SessionTurnAdmission, project_tool_process_event
from ..spec import RepositoryInstructions
from ._agent_executor import DurableBoundary, LiveDelta, _NativeSubagentLeaf
from .agent_execution import resolve_execution_policy
from ._regenerable_transcript import prepare_native_turn_transcript, reply_from_turn, restore_turn_prompt
from ._step_store import RuntimeStepStore
from ._repository_instructions import (
    _RepositoryInstructionBoundary,
    _discover_repository_overlay, _instruction_paths, _merge_repository_instructions,
)
from ._feature_policy import _runtime_feature_policy

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic_ai.messages import ModelMessage, UserContent

    from ..agent import AgentBinding
    from ..core import BudgetUsage, RunBudget, UsageMetrics
    from ._budget import RunBudgetContext
    from ._canonical_split_step import _CanonicalSplitStep
    from ..model import CanonicalUserInput
    from ..capability.steps import AtomicStepPersistence
    from ._tool import RuntimeToolOperationBridge
    from ..capability.planning import PlanningStore
    from ..core import AuthorizationPolicy, CorrelationData, ExecutionMode, PromptLimits, ThinkingValue
    from ..inputs import ExecutionInputMaterializer
    from ..spec import RepositoryInstructionResolver
    from ..workspace import Workspace
    from ..persistence.turns import SqlSessionTurns
    from ..persistence.repositories import ExecutionRepositoryImpl
    from ..persistence.runs import StateStepArchive
    from ._canonical_trace_interactions import _CapturedTurnInteractions
    from ..execution import ExecutionStartReservation, ExecutionStartReservationResult, PendingToolContinuation, PendingDeferredCall
    from ..persistence.commands import StateCommands
    from ..persistence._task_repository import TaskRepositoryImpl
    from ..task import TaskGraphAdmission, TaskNodeInvocation
    from ._agent_binding_resolver import _AgentBindingResolver
    from ._agent_executor import AgentEmission, AgentExecutor, EventSink

_TURN_BYTES = 64 * 1024
_CONTEXT_BYTES = 1024 * 1024
_TERMINAL_RESERVE = 1024
_LEASE_SECONDS = 60
_TURN_SLOTS = BoundedSemaphore(4)
_OBSERVER_SLOTS = BoundedSemaphore(128)
_T = TypeVar("_T", bool, None)
_logger = environ.get_logger(__name__)


async def _resolve_initial_instruction_scopes(
    resolver: RepositoryInstructionResolver | None,
    cwd: str | None,
    scopes: tuple[str, ...],
    files: Sequence[Mapping[str, JsonValue]] = (),
) -> RepositoryInstructions | None:
    if resolver is None:
        return None
    documents = {}
    for path in dict.fromkeys((cwd or ".", *scopes, *(file["path"] for file in files))):
        resolved = await resolver.resolve(path, exclude_sources=frozenset(documents))
        for document in resolved.documents:
            documents.setdefault(document.source, document)
    instructions = RepositoryInstructions(tuple(documents.values()))
    resolver.validate(instructions)
    return instructions


def _merging_instruction_scopes(scopes: tuple[str, ...], instructions: RepositoryInstructions | None) -> tuple[str, ...]:
    if instructions is None:
        return scopes
    merged = set(scopes)
    byte_size = len(canonical_json_bytes(scopes))
    for document in instructions.documents:
        if len(merged) == MAX_REMEMBERED_SCOPES:
            break
        scope = document.scope
        if scope not in merged:
            size = len(canonical_json_bytes(scope)) + bool(merged)
            if byte_size + size <= MAX_REMEMBERED_SCOPE_BYTES:
                merged.add(scope)
                byte_size += size
    return tuple(sorted(merged))


@dataclass(frozen=True, slots=True)
class _AdmittedTurn:
    key: SessionTurnKey
    number: int
    prompt: str
    history: list[ModelMessage]
    terminal_bytes: int
    revision: int
    repository_instructions: RepositoryInstructions | None
    generation: int = 1
    session_context: SessionContext = "all"
    model_prompt: Sequence[UserContent] | None = None
    attachments: Sequence[Mapping[str, JsonValue]] = ()
    stored_input: JsonValue = None
    native_step: tuple[AtomicStepPersistence, RuntimeToolOperationBridge] | None = None
    continuation: tuple[ModelMessage, ...] | None = None
    deferred_results: DeferredToolResults | None = None
    deferred_approvals: tuple[PendingDeferredCall, ...] = ()
    native_subagents: bool = False
    previous_turn_interrupted: bool = False
    settled_native_subagents: bool = False
    recover_native_results: Callable[[], Awaitable[DeferredToolResults | None]] | None = None
    instruction_frontier: SessionInstructionFrontier | None = None
    instruction_scopes: tuple[str, ...] = ()
    completed_checkpoint: bool = False
    output_retries_used: int = 0
    mode: ExecutionMode = "run"
    planning: bool = False
    plan_store: PlanningStore | None = None
    budget: RunBudgetContext | None = None
    trace_enabled: bool = False
    split_source: _CanonicalSplitStep | None = None
    preload_skill_inputs: tuple[Mapping[str, JsonValue], ...] = ()
    native_stopped_delegates: set[tuple[str, str, int, str]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _AttachedGraphTurn:
    key: SessionTurnKey
    generation: int
    number: int
    prompt: str
    history: tuple[ModelMessage, ...]
    terminal_bytes: int
    revision: int
    cwd: str | None
    context_from: int
    context_through: int
    session_metadata: ImmutableJsonMapping
    session_context: SessionContext
    repository_instructions: RepositoryInstructions | None = None
    attachments: Sequence[Mapping[str, JsonValue]] = ()
    stored_input: JsonValue = None
    model_prompt: tuple[UserContent, ...] | None = None


@dataclass(frozen=True, slots=True)
class _SessionTurnStartReceipt:
    turn_id: str
    replayed: bool


class _RegenerableTurnRunner:
    def __init__(
        self, store: SqlSessionTurns, executor: AgentExecutor, *,
        workspace: Workspace | None,
        input_materializer: ExecutionInputMaterializer,
        instruction_resolver: RepositoryInstructionResolver | None,
        instruction_resolver_for: Callable[[RepositoryInstructions], RepositoryInstructionResolver],
        prepare_step: Callable[[AgentBinding, SessionTurnKey, int], Awaitable[
            tuple[AtomicStepPersistence, RuntimeToolOperationBridge] | None]] | None = None,
        abandon_child_repository: ExecutionRepositoryImpl | None = None,
        cancel_subagents: Callable[[SessionTurnKey, Principal, Principal], Awaitable[frozenset[str]]] | None = None,
        begin_child_stop: Callable[[SessionTurnKey, int], None] | None = None,
        prepare_subagents: Callable[
            [_AdmittedTurn, AgentBinding, Principal, Callable[[], RepositoryInstructions | None]],
            AbstractAsyncContextManager[_NativeSubagentLeaf],
        ] | None = None,
        recover_subagent: Callable[
            [SessionTurnKey, int, Principal, AgentBinding, ExecutionMode, RuntimeToolOperationBridge,
             ToolOperationRecord, Mapping[str, JsonValue], bool], Awaitable[ToolOperationRecord],
        ] | None = None,
        prepare_plan: Callable[[SessionTurnKey, int], PlanningStore] | None = None,
        prepare_budget: Callable[[SessionTurnKey, int], RunBudgetContext] | None = None,
    ) -> None:
        self._store = store
        self._executor = executor
        self._workspace = workspace
        self._input_materializer = input_materializer
        self._workspace_access = input_materializer.access
        self._instruction_resolver = instruction_resolver
        self._instruction_resolver_for = instruction_resolver_for
        self._prepare_step = prepare_step
        self._abandon_child_repository = abandon_child_repository
        self._cancel_subagents = cancel_subagents
        self._begin_child_stop = begin_child_stop
        self._prepare_subagents = prepare_subagents
        self._recover_subagent = recover_subagent
        self._prepare_plan = prepare_plan
        self._prepare_budget = prepare_budget
        self._publish_deferred = None
        self._stop_deferred = None
        self._split_recovery_source: Callable[[], Awaitable[_CanonicalSplitStep]] | None = None
        self._cancel_graph_turn: Callable[[SessionTurnKey, Principal], Awaitable[bool]] | None = None
        self._prepare_split_wait_resume = None
        from ._lossy_session_trace import LossySessionTrace

        self._trace_archive: StateStepArchive | None = None
        self._lossy_trace = LossySessionTrace()
        self._tasks: set[asyncio.Task[None] | asyncio.Task[bool]] = set()
        self._producers: dict[SessionTurnKey, asyncio.Task[bool]] = {}
        self._closed = False
        self._observers: dict[str, set[Callable[[dict[str, JsonValue]], None]]] = {}
        self._lifecycle_observers: set[Callable[[dict[str, JsonValue]], None]] = set()
        self._observed_revisions: dict[str, int] = {}
        self._cancel_notices: dict[SessionTurnKey, asyncio.Task[None]] = {}
        self._observed_cancellation: dict[str, str] = {}

    def _prepare_turn_budget(self, key, generation, native_step):
        if self._prepare_budget is None or native_step is None:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, "Canonical run budgets require coowned SQL Step")
        budget = self._prepare_budget(key, generation)
        native_step[1].configure_canonical_budget(budget.scope_id, budget.repository)
        return budget

    def _observes_turn_trace(self, binding: AgentBinding) -> bool:
        policy = _runtime_feature_policy(binding.compiled_agent.runtime_contributions)
        return policy.durable_trace and self._trace_archive is not None

    def bind_deferred_publication(self, publish, stop=None) -> None:
        self._publish_deferred = publish
        self._stop_deferred = stop

    def observe_lifecycle(self, callback: Callable[[dict[str, JsonValue]], None]) -> Callable[[], None]:
        """Offer only committed turn identifiers and revisions to a host carrier."""
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)

        def listener(notice: dict[str, JsonValue]) -> None:
            callback(notice)

        self._lifecycle_observers.add(listener)
        return lambda: self._lifecycle_observers.discard(listener)

    def invalidate_session(self, session_id: str, turn_id: str, *, revision: int, phase: str = "admitted") -> None:
        key = SessionTurnKey(session_id, turn_id)
        if phase == "cancelling":
            self._remote_cancel(key)
            return
        if phase == "waiting":
            # A remote carrier is advisory, never a source of occupancy.
            self._publish(session_id, {"type": "gap", "turn_id": turn_id, "phase": "waiting"})
            return
        if session_id not in self._observers or revision <= self._observed_revisions.get(session_id, -1):
            return
        self._observed_revisions[session_id] = revision
        self._gap(key)

    def _remote_cancel(self, key: SessionTurnKey) -> None:
        if self._closed:
            return
        if (key.session_id in self._observers
                and self._observed_cancellation.get(key.session_id) != key.turn_id):
            self._observed_cancellation[key.session_id] = key.turn_id
            self._gap(key, cancelling=True)
        producer = self._producers.get(key)
        if producer is None or producer.done() or producer.cancelling() or key in self._cancel_notices:
            return

        async def confirm() -> None:
            try:
                row = await self._store.get_turn(key)
                if (row is not None and row["state"] == "cancelling"
                        and self._producers.get(key) is producer and not producer.done() and not producer.cancelling()):
                    if self._begin_child_stop is not None:
                        self._begin_child_stop(key, row["generation"])
                    producer.cancel()
            except Exception as error:
                _logger.warning("Remote cancellation confirmation failed: %s", type(error).__name__)

        task = asyncio.create_task(confirm(), name="linktools-remote-turn-cancel")
        self._cancel_notices[key] = task
        task.add_done_callback(lambda _: self._cancel_notices.pop(key, None))

    def invalidate_observed_sessions(self) -> None:
        """Ask current viewers to reconcile after known advisory-channel loss."""
        for session_id in tuple(self._observers):
            self._publish(session_id, {"type": "gap"})

    def _notice_lifecycle(self, key: SessionTurnKey, phase: str, revision: int) -> None:
        if key.session_id in self._observers:
            if phase == "cancelling":
                self._observed_cancellation[key.session_id] = key.turn_id
            else:
                self._observed_revisions[key.session_id] = max(revision, self._observed_revisions.get(key.session_id, -1))
        for callback in tuple(self._lifecycle_observers):
            try:
                callback({
                    "kind": "ordinary_turn", "phase": phase, "session_id": key.session_id,
                    "turn_id": key.turn_id, "revision": revision,
                })
            except (Exception, asyncio.CancelledError) as error:
                _logger.warning("Session lifecycle observer failed: %s", type(error).__name__)

    def observe(
        self, session_id: str, callback: Callable[[dict[str, JsonValue]], None], *, local_tool_feedback: bool = False,
    ) -> Callable[[], None]:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if not _OBSERVER_SLOTS.acquire(blocking=False):
            raise AIError(ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED)

        def listener(frame: dict[str, JsonValue]) -> None:
            if not local_tool_feedback:
                if frame.get("type") == "tool" and isinstance(item := frame.get("item"), dict):
                    frame = {**frame, "item": {**item, "error_message": None}}
                elif frame.get("type") == "terminal" and isinstance(process := frame.get("process"), list):
                    frame = {**frame, "process": [
                        {**item, "error_message": None} if isinstance(item, dict) else item for item in process
                    ]}
            callback(frame)

        listeners = self._observers.setdefault(session_id, set())
        listeners.add(listener)

        def close() -> None:
            if listener not in listeners:
                return
            listeners.remove(listener)
            _OBSERVER_SLOTS.release()
            if not listeners and self._observers.get(session_id) is listeners:
                self._observers.pop(session_id, None)
                self._observed_revisions.pop(session_id, None)
                self._observed_cancellation.pop(session_id, None)

        return close

    def _publish(self, session_id: str, frame: dict[str, JsonValue]) -> None:
        for callback in tuple(self._observers.get(session_id, ())):
            try:
                callback(frame)
            except Exception as error:
                _logger.error("Session turn observer failed: %s", type(error).__name__)

    def _gap(self, key: SessionTurnKey, *, cancelling: bool = False) -> None:
        self._publish(key.session_id, {
            "type": "gap", "turn_id": key.turn_id,
            **({"phase": "cancelling"} if cancelling else {}),
        })

    def has_local_producer(self, key: SessionTurnKey) -> bool:
        producer = self._producers.get(key)
        return producer is not None and not producer.done()

    def _producer_stopped(self, turn: _AdmittedTurn | _AttachedGraphTurn) -> None:
        # Local evidence only: the transport may not report this phase and
        # the existing lease remains the sole durable write authority.
        self._publish(turn.key.session_id, {
            "type": "gap", "phase": "producer_stopped", "turn_id": turn.key.turn_id,
            "generation": turn.generation,
        })

    @staticmethod
    def _notice_prompt(turn: _AdmittedTurn | _AttachedGraphTurn) -> JsonValue:
        if turn.prompt or turn.stored_input is None:
            return turn.prompt
        stored = decode_domain(turn.stored_input, StoredUserInput)
        if stored.codec == "text":
            return turn.prompt
        from ..history import project_ordinary_multipart_prompt

        return project_ordinary_multipart_prompt(stored)

    def _admitted(self, turn: _AdmittedTurn | _AttachedGraphTurn) -> None:
        self._publish(turn.key.session_id, {
            "type": "admitted", "turn_id": turn.key.turn_id, "turn_no": turn.number,
            "generation": turn.generation, "revision": turn.revision, "prompt": self._notice_prompt(turn), "attachments": list(turn.attachments),
        })
        self._notice_lifecycle(turn.key, "admitted", turn.revision)

    def _terminal(
        self, turn: _AdmittedTurn | _AttachedGraphTurn, state: str, *, output: JsonValue = None,
        messages: tuple[ModelMessage, ...] = (), error_code: str | None = None,
        output_ref: Sequence[int] | None = None,
    ) -> None:
        succeeded = state == "succeeded"
        activity = {}
        if succeeded:
            from ..history import project_ordinary_turn_activity

            activity = project_ordinary_turn_activity(messages, output=output, output_ref=output_ref, local_tool_feedback=True)
        self._store.note_terminal_context(
            turn.key, turn_no=turn.number, prompt=(
                "" if turn.stored_input is not None
                and decode_domain(turn.stored_input, StoredUserInput).codec == "text" else turn.prompt
            ),
            stored_input=turn.stored_input, messages=messages, succeeded=succeeded,
            reset_context=turn.session_context == "current_turn",
        )
        self._publish(turn.key.session_id, {
            "type": "terminal", "turn_id": turn.key.turn_id, "turn_no": turn.number,
            "generation": turn.generation, "revision": turn.revision + 1,
            "state": state, "prompt": self._notice_prompt(turn), "attachments": list(turn.attachments), "output": output,
            "error_code": error_code, **activity,
        })
        self._notice_lifecycle(turn.key, "terminal", turn.revision + 1)

    async def canonicalize_cwd(self, cwd: str | None) -> str | None:
        if cwd is None:
            return None
        if not isinstance(cwd, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "cwd"})
        if self._workspace_access is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "cwd", "reason": "workspace_required"})
        try:
            return await self._workspace_access.canonicalize_path(cwd)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "cwd"}) from error

    def _resolver_for(self, binding: AgentBinding) -> RepositoryInstructionResolver | None:
        if binding.repository_instructions is not None:
            return self._instruction_resolver_for(binding.repository_instructions)
        return self._instruction_resolver

    async def create_session(
        self, session_id: str, *, principal: Principal, tenant_id: str,
        authorization: AuthorizationPolicy,
        metadata: dict[str, JsonValue], request_key: str | None = None, cwd: str | None = None,
    ) -> dict[str, JsonValue]:
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(principal, AuthorizationAction.SESSION_CREATE, ResourceRef(
            ResourceKind.SESSION, session_id, tenant_id,
        ))
        cwd = await self.canonicalize_cwd(cwd)
        config: JsonValue = {"metadata": metadata, "create_request_key": request_key, "cwd": cwd}
        try:
            await self._store.create(session_id, config=config)
        except IntegrityError:
            await authorization.authorize(
                principal, AuthorizationAction.SESSION_READ, ResourceRef(ResourceKind.SESSION, session_id, tenant_id),
            )
            existing = await self._store.get_session(session_id)
            if request_key is None or existing is None:
                raise AIError(ErrorCode.SESSION_CONFLICT) from None
            previous_config = existing["config_json"]
            if {key: value for key, value in previous_config.items() if key != "instruction_scopes"} != config:
                raise AIError(ErrorCode.SESSION_CONFLICT) from None
            return {
                "session_id": session_id, "revision": existing["revision"],
                "metadata": metadata, "state": existing["status"], "cwd": cwd,
                "active_turn_id": None if existing["active_turn_id"] is None else existing["active_turn_id"].decode("utf-8"),
            }
        return {
            "session_id": session_id, "revision": 0, "metadata": metadata,
            "state": "open", "active_turn_id": None, "cwd": cwd,
        }

    @staticmethod
    async def _authorize_lineage(
        lineage: tuple[SessionTurnSegment, ...], *, principal: Principal, tenant_id: str,
        authorization: AuthorizationPolicy,
    ) -> None:
        for segment in lineage:
            await authorization.authorize(
                principal, AuthorizationAction.SESSION_READ,
                ResourceRef(ResourceKind.SESSION, segment.session_id, tenant_id),
            )

    async def fork_session(
        self, source_id: str, target_id: str, *, principal: Principal, tenant_id: str,
        authorization: AuthorizationPolicy, expected_revision: int, request_id: str,
    ) -> bool:
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, source_id, tenant_id),
        )
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_CREATE,
            ResourceRef(ResourceKind.SESSION, target_id, tenant_id),
        )
        source = await self._store.get_session(source_id)
        if source is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorize_lineage(
            decode_session_lineage(source["lineage_json"]), principal=principal,
            tenant_id=tenant_id, authorization=authorization,
        )
        config = source["config_json"]
        return await self._store.fork_session(
            source_id, target_id, expected_revision=expected_revision, request_id=request_id,
            config={"metadata": config.get("metadata", {}), "cwd": config.get("cwd")},
        )

    async def lookup_admission(
        self, session_id: str, request_key: str, *, principal: Principal, authorization: AuthorizationPolicy,
    ) -> SessionTurnAdmission | None:
        row = await self._store.read_admission_input(session_id, request_key)
        if row is None:
            return None
        turn_id = row["turn_id"].decode("utf-8")
        await authorization.authorize(
            principal, AuthorizationAction.EXECUTION_READ,
            ResourceRef(ResourceKind.EXECUTION, turn_id, principal.tenant_id, row["principal_id"]),
        )
        stored = await self._store.read_native_admission_input(row, turn_id)
        if stored is not None:
            content = native_graph_input_content(stored)
            if content is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            manifest = row["input"]
            stored = None if manifest is None else decode_domain(manifest, StoredUserInput)
            content = await self._restore_prompt(row["prompt"], stored)
        return SessionTurnAdmission(turn_id, content)

    async def start_session(
        self, session_id: str, *, agent_id: str, principal: Principal, namespace: str, tenant_id: str,
        app: object, authorization: AuthorizationPolicy, resolver: _AgentBindingResolver,
        request_key: str, prompt: CanonicalUserInput, limits: PromptLimits, event_sink: EventSink, thinking: JsonValue,
        files: Sequence[str] = (), session_context: SessionContext = "all", output: type[BaseModel] | None = None,
        correlation: CorrelationData | None = None,
        selected_binding: Callable[[], Awaitable[AgentBinding]] | None = None,
        mode: ExecutionMode = "run", planning: bool | None = None, budget: RunBudget | None = None,
        attachment_display_names: tuple[str, ...] = (),
    ) -> _SessionTurnStartReceipt:
        """Start on the authoritative SQL session after session and execution authorization."""
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        session = await self._store.get_session(session_id)
        if session is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, session_id, tenant_id),
        )
        if session["status"] != "open":
            raise AIError(ErrorCode.SESSION_CONFLICT)
        lineage = decode_session_lineage(session["lineage_json"])
        await self._authorize_lineage(
            lineage, principal=principal, tenant_id=tenant_id, authorization=authorization,
        )

        files = normalize_input_files(files)
        if attachment_display_names and (not isinstance(attachment_display_names, tuple) or isinstance(prompt, str) or any(
            not isinstance(name, str) or not 0 < len(name) <= 255 or not name.isprintable()
            or "/" in name or chr(92) in name for name in attachment_display_names
        )):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "attachment_display_names"})
        if not isinstance(prompt, str):
            prompt = await self._input_materializer.canonicalize_input(prompt)
        output_binding = None if output is None else bind_output(output)
        if budget is not None:
            from ..core import RunBudget
            if not isinstance(budget, RunBudget):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "budget"})
        effective_correlation = {} if correlation is None else dict(correlation)
        resolved_binding = None

        async def bind() -> AgentBinding:
            nonlocal resolved_binding
            if resolved_binding is None:
                resolved_binding = (await resolver.resolve_agent(agent_id, output=output_binding)
                                    if selected_binding is None else await selected_binding())
            return resolved_binding

        async def replay() -> str | None:
            prior = await self._store.find_request(session_id, request_key)
            if prior is None:
                return None
            identity = prior["turn_id"].decode("utf-8")
            request = prior["request_json"]
            await authorization.authorize(
                principal, AuthorizationAction.EXECUTION_RUN,
                ResourceRef(ResourceKind.EXECUTION, identity, tenant_id, request["principal"]["principal_id"]),
            )
            normalized_files = await self._input_materializer.canonicalize_files(files) if files else ()
            if isinstance(prompt, str):
                prior_input = self._stored_input(request)
                prior_files = () if prior_input is None or "input_intent" in request else tuple(
                    item["path"] for item in stored_input_attachment_views(prior_input)
                )
                if prior_input is not None and prior_input.codec == "text":
                    same_input = (not normalized_files and request["prompt"] == ""
                                  and prior_input.payload.digest == self._input_materializer.capture(prompt).payload.digest)
                else:
                    same_input = request["prompt"] == prompt and prior_files == normalized_files
            else:
                same_input = request.get("input_intent") == self._input_materializer.intent(prompt, normalized_files).digest
            if same_input and not isinstance(prompt, str):
                stored = self._stored_input(request)
                saved_names = () if stored is None else tuple(
                    item["display_name"] for item in stored_input_attachment_views(stored) if "display_name" in item
                )
                same_input = saved_names == attachment_display_names
            replay_mode, replay_planning, replay_thinking = resolve_execution_policy(
                mode=mode, planning=planning, thinking=thinking,
                default_planning=prior["default_planning"], default_thinking=prior["default_thinking"],
            )
            output_mismatch = False
            if selected_binding is None:
                schema = prior["default_output_schema"]
                requested_output = output_binding or (
                    bind_output() if schema is None else restore_output("structured", schema)
                )
                output_mismatch = (prior["output_mode"] != requested_output.mode
                                   or prior["output_schema"] != requested_output.schema_definition)
            if (
                not same_input or request["principal"] != asdict(principal)
                or request["thinking"] != replay_thinking
                or request.get("session_context", "all") != session_context
                or request.get("correlation", {}) != effective_correlation
                or request.get("mode", "run") != replay_mode or request.get("planning", False) != replay_planning
                or bool(request.get("budgeted", False)) != (budget is not None)
                or prior["agent_id"] != agent_id
                or (prior.get("binding_digest") is not None) != (selected_binding is not None)
                or output_mismatch
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if selected_binding is not None and prior["binding_digest"] != (await bind()).binding_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if budget is not None:
                if self._prepare_budget is None:
                    raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
                budget_owner = self._prepare_budget(SessionTurnKey(session_id, identity), 1)
                confirmed = await budget_owner.repository.read(budget_owner.scope_id)
                if confirmed.limits != budget:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return identity

        if session["active_turn_id"] is not None:
            if (identity := await replay()) is not None:
                return _SessionTurnStartReceipt(identity, replayed=True)
            raise AIError(ErrorCode.SESSION_BUSY)

        key = SessionTurnKey(session_id, secrets.token_hex(16))
        await authorization.authorize(
            principal, AuthorizationAction.EXECUTION_RUN, ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id),
        )
        files = await self._input_materializer.canonicalize_files(files) if files else ()
        binding = await bind()
        spec = binding.compiled_agent.spec
        resolved_mode, resolved_planning, resolved_thinking = resolve_execution_policy(
            mode=mode, planning=planning, thinking=thinking, default_planning=spec.planning, default_thinking=spec.thinking,
        )
        config = session["config_json"]
        cwd = config.get("cwd")
        context = AgentContext(
            app, principal, namespace, session_id, key.turn_id, config.get("metadata", {}),
            agent_id=agent_id, correlation=effective_correlation,
        )
        try:
            if await self.start(
                key, revision=session["revision"], through=session["last_turn_no"],
                request_key=request_key, prompt=prompt, binding=binding, context=context,
                limits=limits, event_sink=event_sink, thinking=resolved_thinking, cwd=cwd, files=files, lineage=lineage,
                session_context=session_context, mode=resolved_mode, planning=resolved_planning,
                explicit_agent_binding=selected_binding is not None, budget=budget,
                attachment_display_names=attachment_display_names, instruction_scopes=decode_instruction_scopes(config),
            ):
                return _SessionTurnStartReceipt(key.turn_id, replayed=False)
        except (IntegrityError, AIError):
            if (identity := await replay()) is not None:
                return _SessionTurnStartReceipt(identity, replayed=True)
            raise
        if (identity := await replay()) is not None:
            return _SessionTurnStartReceipt(identity, replayed=True)
        raise AIError(ErrorCode.SESSION_CONFLICT)

    async def read_budget_usage(
        self, key: SessionTurnKey, *, principal: Principal, tenant_id: str, authorization: AuthorizationPolicy,
    ) -> BudgetUsage | None:
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_READ, ResourceRef(ResourceKind.SESSION, key.session_id, tenant_id),
        )
        turn = await self._store.get_turn(key)
        if turn is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        request = turn["request_json"]
        await authorization.authorize(
            principal, AuthorizationAction.EXECUTION_READ,
            ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id, request["principal"]["principal_id"]),
        )
        if not request.get("budgeted", False):
            return None
        if self._prepare_budget is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        budget = self._prepare_budget(key, turn["generation"])
        return await budget.repository.read(budget.scope_id)

    async def abandon_session_turn(
        self, key: SessionTurnKey, *, expected_generation: int,
        principal: Principal, tenant_id: str, authorization: AuthorizationPolicy,
    ) -> bool:
        """Release an expired ordinary turn without certifying or repeating its effects."""
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_READ, ResourceRef(ResourceKind.SESSION, key.session_id, tenant_id),
        )
        row = await self._store.get_turn_for_recovery(key)
        if row is None:
            return False
        request = row["request_json"]
        await authorization.authorize(
            principal, AuthorizationAction.EXECUTION_CANCEL,
            ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id, request["principal"]["principal_id"]),
        )
        if row["state"] != "accepted" or row["generation"] != expected_generation or self.has_local_producer(key):
            return False
        frozen = row["binding_json"]
        contract = None if frozen is None else AgentBindingContract.from_payload(frozen)
        if request.get("graph_execution") or request.get("split_step"):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        child_guard = None
        if contract is not None and contract.subagents:
            children = self._abandon_child_repository
            if children is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

            async def require_terminal_children(transaction):
                await children.require_children_terminal_for_abandon(transaction, key.turn_id)

            child_guard = (children.state_store, require_terminal_children)
        session = await self._store.get_session(key.session_id)
        if session is None or session["active_turn_id"] != key.turn_id.encode():
            return False
        code = ErrorCode.STORAGE_RECOVERY_REQUIRED.value
        if not await self._store.finish(
            key, generation=row["generation"], turn_no=row["turn_no"], state="failed",
            result={"error_code": code, "abandoned": True}, _abandon_expired=True,
            _guard_abandon=child_guard,
        ):
            return False
        # Re-read the committed view rather than constructing it from incomplete input or effects.
        self._gap(key)
        self._notice_lifecycle(key, "terminal", session["revision"] + 1)
        return True

    async def cancel_session_turn(
        self, key: SessionTurnKey, *, principal: Principal, tenant_id: str, authorization: AuthorizationPolicy,
    ) -> bool:
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        session = await self._store.get_session(key.session_id)
        if session is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_READ, ResourceRef(ResourceKind.SESSION, key.session_id, tenant_id),
        )
        turn = await self._store.get_turn_for_recovery(key)
        if turn is None:
            return False
        await authorization.authorize(
            principal, AuthorizationAction.EXECUTION_CANCEL,
            ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id, turn["request_json"]["principal"]["principal_id"]),
        )
        if turn["request_json"].get("graph_execution") == key.turn_id:
            if self._cancel_graph_turn is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            return await self._cancel_graph_turn(key, principal)
        split = turn["request_json"].get("split_step", False)
        result = turn["result_json"]
        if (not split and turn["state"] == "cancelling" and isinstance(result, Mapping)
                and result.get("cancel_effects_unresolved") is True):
            await self._finish_cold_cancellation(key, turn, session["revision"], actor=principal)
            return True
        if not split and turn["state"] in ("waiting", "ready"):
            if self._stop_deferred is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            frontier = await self._store.get_waiting_turn(key)
            if frontier is None or frontier.generation != turn["generation"]:
                return await self._cancellation_accepted(key)
            stopped = await self._stop_deferred(frontier)
            if stopped == "lost":
                return await self._cancellation_accepted(key)
            self._gap(key, cancelling=True)
            self._notice_lifecycle(key, "cancelling", session["revision"])
            if stopped == "ready":
                await self._finish_cold_cancellation(key, turn, session["revision"], actor=principal)
            return True
        if turn["state"] in ("succeeded", "failed", "cancelled"):
            return turn["state"] == "cancelled"
        if not split and turn["state"] == "cancelling":
            self._remote_cancel(key)
            return True
        changed = (
            await self._store.request_split_wait_cancel(key, generation=turn["generation"])
            if split and turn["state"] in ("waiting", "ready")
            else await self._store.request_cancel(key)
        )
        if not changed and not (split and turn["state"] == "cancelling"):
            return await self._cancellation_accepted(key)
        producer = self._producers.get(key)
        if changed:
            self._gap(key, cancelling=True)
            self._notice_lifecycle(key, "cancelling", session["revision"])
        if producer is not None and not producer.done() and not producer.cancelling():
            if changed and not split and self._begin_child_stop is not None:
                self._begin_child_stop(key, turn["generation"])
            producer.cancel()
        if producer is None and split:
            if self._split_recovery_source is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            source = await self._split_recovery_source()
            outcome, _, _ = await source.recover_open_turn(
                key, generation=turn["generation"], turn_no=turn["turn_no"], lease_seconds=_LEASE_SECONDS,
                cancelling=True, cancel_result={"error_code": ErrorCode.EXECUTION_CANCELLED.value},
            )
            if outcome == "settled":
                self._gap(key)
                self._notice_lifecycle(key, "terminal", session["revision"] + 1)
            elif outcome != "not_acquired":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return True

    async def _cancellation_accepted(self, key: SessionTurnKey) -> bool:
        current = await self._store.get_turn(key)
        if current is not None and current["state"] == "cancelling":
            self._remote_cancel(key)
        return current is not None and current["state"] in ("cancelling", "cancelled")

    async def _finish_cold_cancellation(
        self, key: SessionTurnKey, row: Mapping[str, object], revision: int, *, actor: Principal,
        authorization: AuthorizationPolicy | None = None,
    ) -> bool:
        request, generation = row["request_json"], row["generation"]
        frozen = row["binding_json"]
        contract = None if frozen is None else AgentBindingContract.from_payload(frozen)
        has_subagents = contract is not None and bool(contract.subagents)
        admitted = Principal(**request["principal"])
        if has_subagents and authorization is not None:
            await authorization.authorize(
                actor, AuthorizationAction.EXECUTION_CANCEL,
                ResourceRef(ResourceKind.EXECUTION, key.turn_id, admitted.tenant_id, admitted.principal_id),
            )
        result = row["result_json"]
        if isinstance(result, Mapping) and result.get("cancel_effects_unresolved") is True:
            if self._stop_deferred is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            frontier = CanonicalDeferredFrontier.from_json(
                key, generation, {name: value for name, value in result.items() if name != "cancel_effects_unresolved"},
            )
            if await self._stop_deferred(frontier) != "ready":
                return False
        if not await self._store.take_over(
            key, generation=generation, lease_seconds=_LEASE_SECONDS, cancelling=True,
        ):
            return False
        if has_subagents:
            if self._cancel_subagents is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            await self._cancel_subagents(key, admitted, actor)
        stored = self._stored_input(request)
        turn = _AdmittedTurn(
            key, row["turn_no"], request["prompt"], [], 0, revision, None, generation + 1,
            attachments=() if stored is None else stored_input_attachment_views(stored),
            stored_input=request.get("input"),
        )
        code = ErrorCode.EXECUTION_CANCELLED.value
        finished = await self._store.finish(
            key, generation=generation + 1, turn_no=row["turn_no"], state="cancelled", result={"error_code": code},
        )
        if finished:
            self._terminal(turn, "cancelled", error_code=code)
        return finished

    async def resume_deferred(
        self, frontier: CanonicalDeferredFrontier, continuation: PendingToolContinuation,
        results: DeferredToolResults, **environment,
    ) -> None:
        frontier.require_continuation(continuation)
        row = await self._store.get_turn_for_recovery(frontier.key)
        if row is None or row["state"] != "ready" or row["generation"] != frontier.generation:
            return
        principal = Principal(**row["request_json"]["principal"])
        await self.recover_session_turn(
            frontier.key, principal=principal, ready=frontier, settled_results=results,
            settled_continuation=continuation, **environment,
        )

    async def recover_session_turn(
        self, key: SessionTurnKey, *, principal: Principal, namespace: str, tenant_id: str,
        app: object, authorization: AuthorizationPolicy, resolver: _AgentBindingResolver,
        limits: PromptLimits, event_sink: EventSink,
        ready: CanonicalDeferredFrontier | None = None,
        settled_results: DeferredToolResults | None = None,
        settled_continuation: PendingToolContinuation | None = None,
    ) -> bool:
        """Explicit recovery or exact authorized decision resumes the admitted request."""
        if ((ready is None) != (settled_results is None) or (ready is None) != (settled_continuation is None)
                or ready is not None and ready.key != key):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)

        async def recover() -> bool:
            session = await self._store.get_session(key.session_id)
            if session is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await authorization.authorize(
                principal, AuthorizationAction.SESSION_READ,
                ResourceRef(ResourceKind.SESSION, key.session_id, tenant_id),
            )
            if session["active_turn_id"] != key.turn_id.encode("utf-8"):
                return False
            row = await self._store.get_turn_for_recovery(key)
            if row is None:
                return False
            if ready is None:
                if row["state"] not in ("accepted", "cancelling") and not (
                    row["state"] == "ready" and row["request_json"].get("split_step", False)
                ):
                    return False
            elif row["state"] != "ready" or row["generation"] != ready.generation:
                return False
            request = row["request_json"]
            admitted = Principal(**request["principal"])
            await authorization.authorize(
                principal, AuthorizationAction.EXECUTION_RECOVER,
                ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id, admitted.principal_id),
            )
            if request.get("graph_execution") == key.turn_id:
                # Only task-owner recovery can restore the real execution and advance both leases.
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            generation = row["generation"]
            mode = request.get("mode", "run")
            planning = request.get("planning", False)
            if mode not in ("run", "plan") or type(planning) is not bool or mode == "plan" and not planning:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if ready is not None and admitted != principal:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            split_recovery = None
            split_wait_resume = False
            if request.get("split_step", False):
                if self._split_recovery_source is None or ready is not None:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                split_recovery = await self._split_recovery_source()
                source = await split_recovery._read_pin(split_recovery._identity(key, generation))
                cancelling = row["state"] == "cancelling"
                split_wait_resume = (row["state"] == "ready"
                                     or source is not None and source.phase in ("resuming", "resume_hold")
                                     or source is None and isinstance(row["result_json"], Mapping)
                                     and "checkpoint_id" in row["result_json"])
                cancel_result = {"error_code": ErrorCode.EXECUTION_CANCELLED.value} if cancelling else None
                if source is not None and source.pin.terminal_state is not None:
                    recovered = await split_recovery.recover_terminal(
                        key, turn_no=row["turn_no"], source=source, cancel_result=cancel_result,
                    )
                elif cancelling:
                    outcome, _, _ = await split_recovery.recover_open_turn(
                        key, generation=generation, turn_no=row["turn_no"], lease_seconds=_LEASE_SECONDS,
                        cancelling=True, require_effects_resolved=split_recovery.require_no_effects,
                        cancel_result=cancel_result,
                    )
                    if outcome not in ("settled", "not_acquired"):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    recovered = outcome == "settled"
                elif source is not None and source.phase == "waiting":
                    frozen_binding = row["binding_json"]
                    if frozen_binding is None:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    max_bytes = _TURN_BYTES - len(canonical_json_bytes({**request, "binding": frozen_binding}))
                    restored_wait = await split_recovery.recover_waiting(
                        key, generation=generation, max_bytes=max_bytes, source=source,
                    )
                    if restored_wait:
                        self._publish(key.session_id, {
                            "type": "waiting", "turn_id": key.turn_id, "turn_no": row["turn_no"],
                            "generation": generation, "revision": session["revision"],
                        })
                        self._notice_lifecycle(key, "waiting", session["revision"])
                    return restored_wait
                else:
                    recovered = None
                if recovered is not None:
                    if recovered:
                        self._gap(key)
                        self._notice_lifecycle(key, "terminal", session["revision"] + 1)
                    return recovered
            if row["state"] == "cancelling":
                return await self._finish_cold_cancellation(
                    key, row, session["revision"], actor=principal, authorization=authorization,
                )

            lineage = decode_session_lineage(session["lineage_json"])
            await self._authorize_lineage(
                lineage, principal=principal, tenant_id=tenant_id, authorization=authorization,
            )
            if lineage and admitted != principal:
                await authorization.authorize(
                    admitted, AuthorizationAction.SESSION_READ,
                    ResourceRef(ResourceKind.SESSION, key.session_id, tenant_id),
                )
                await self._authorize_lineage(
                    lineage, principal=admitted, tenant_id=tenant_id, authorization=authorization,
                )
            prompt = request["prompt"]
            after = row["context_from"] - 1
            through = row["context_through"]
            session_context = request.get("session_context", "all")
            stored = self._stored_input(request)
            history = await self._read_admitted_history(
                key, context_from=row["context_from"], context_through=row["context_through"],
                lineage=lineage, session_context=session_context,
                prompt=prompt, stored=stored,
            )
            frozen_binding = row["binding_json"]
            if frozen_binding is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED, "The admitted Agent binding is unavailable")
            contract = AgentBindingContract.from_payload(frozen_binding)
            split_wait = None
            if split_wait_resume:
                if self._prepare_split_wait_resume is None or split_recovery is None or self._prepare_step is None:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                split_wait = await self._prepare_split_wait_resume(
                    split_recovery, key, generation, frozen_binding, resolver, ready=row["state"] == "ready",
                )
                binding = split_wait.binding
                native_step = await self._prepare_step(binding, key, generation + 1, settled_split=True)
            else:
                binding = await resolver.restore(contract)
                session_policy = _runtime_feature_policy(binding.compiled_agent.runtime_contributions).session
                if session_policy.checkpoints != request.get("session_checkpoints", False):
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED, "The admitted session checkpoint policy changed")
                native_step = None if self._prepare_step is None else await self._prepare_step(binding, key, generation + 1)
            split = (native_step[0].store._canonical_split_source
                     if native_step is not None and isinstance(native_step[0].store, RuntimeStepStore) else None)
            if (split_recovery is None) != (split is None):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if split is not None and (mode != "run" or planning or request.get("budgeted", False)):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            budget = (self._prepare_turn_budget(key, generation + 1, native_step)
                      if request.get("budgeted", False) else None)
            if planning and self._prepare_plan is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            plan_store = self._prepare_plan(key, generation + 1) if planning else None
            subagent_topology = self._allows_native_subagents(
                binding, native_step, mode=mode, planning=planning, budgeted=budget is not None,
                split=split is not None,
            )
            # Cold recovery may settle proven history, but only an exact approval
            # frontier grants a live subagent toolset to a resumed producer.
            native_subagents = ready is not None and subagent_topology
            settle_subagents = (ready is None and subagent_topology
                                and self._recover_subagent is not None)
            self._executor.require_regenerable(
                binding, canonical_step=native_step is not None,
                split_step=split is not None, native_subagents=native_subagents or settle_subagents,
            )
            if "repository_instructions" not in request or "cwd" not in request:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED, "Turn has no admitted workspace evidence")
            cwd = request["cwd"]
            if await self.canonicalize_cwd(cwd) != cwd:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED, "Admitted cwd is no longer canonical")
            frozen = request["repository_instructions"]
            repository_instructions = None if frozen is None else RepositoryInstructions.from_payload(frozen)
            instruction_resolver = self._resolver_for(binding)
            if repository_instructions is not None:
                if instruction_resolver is None:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED, "The admitted instruction policy is unavailable")
                instruction_resolver.validate(repository_instructions)
            try:
                context = AgentContext(
                    app, admitted, namespace, key.session_id, key.turn_id, request["session_metadata"],
                    agent_id=binding.compiled_agent.spec.id, correlation=request.get("correlation", {}),
                )
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            model_prompt = await self._restore_prompt(prompt, stored)
            continuation = deferred_results = recover_native_results = None
            completed_checkpoint = False
            output_retries_used = 0
            instruction_frontier = None
            if native_step is None:
                if ready is not None:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                if not await self._store.take_over(key, generation=generation, lease_seconds=_LEASE_SECONDS):
                    return False
            else:
                capability, tools = native_step
                steps = capability.store
                if not isinstance(steps, RuntimeStepStore):
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                if split_wait is not None:
                    granted, instruction_frontier = await split_recovery.reacquire_waiting_recovery(
                        split_wait, lease_seconds=_LEASE_SECONDS,
                    )
                    if granted is None:
                        return False
                    await steps.bind_canonical_resume(
                        granted.resume.checkpoint, key=key, generation=generation + 1,
                        split_proof=granted.evidence.source, split_approval=granted.approval,
                        settled_only=granted.approval is None,
                    )
                    continuation = granted.resume.messages
                    output_retries_used = granted.resume.output_retries_used
                    deferred_results = granted.results
                elif split_recovery is not None:
                    phase, split_resume, instruction_frontier = await split_recovery.recover_open_turn(
                        key, generation=generation, turn_no=row["turn_no"], lease_seconds=_LEASE_SECONDS,
                        cancelling=False, require_effects_resolved=split_recovery.require_no_effects,
                    )
                    if phase == "settled":
                        # A concurrent producer sealed first; the adapter already
                        # confirmed the exact canonical terminal receipt. Readers
                        # project that durable state instead of inferring success.
                        self._gap(key)
                        self._notice_lifecycle(key, "terminal", session["revision"] + 1)
                        return True
                    if phase == "not_acquired":
                        return False
                    if phase != "resumed" or split_resume is None:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    await steps.bind_canonical_resume(split_resume.checkpoint, key=key, generation=generation + 1)
                    continuation = split_resume.messages
                    output_retries_used = split_resume.output_retries_used
                    completed_checkpoint = True
                else:
                    if ready is None:
                        acquired, checkpoint, instruction_frontier = await self._store.take_over_with_checkpoint(
                            key, generation=generation, lease_seconds=_LEASE_SECONDS, step_store=steps._store,
                        )
                    else:
                        acquired, checkpoint, instruction_frontier = await self._store.take_over_ready_checkpoint(
                            ready, lease_seconds=_LEASE_SECONDS, step_store=steps._store,
                        )
                    if not acquired:
                        return False
                    if checkpoint is None:
                        self._gap(key)
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    await steps.bind_canonical_resume(checkpoint, key=key, generation=generation + 1)
                    from ..model.messages import native_checkpoint_messages
                    from ..persistence.runs import NativeCheckpoint
                    checkpoint = NativeCheckpoint.decode(checkpoint)
                    output_retries_used = (checkpoint.output_retries_used if isinstance(checkpoint, NativeCheckpoint) else 0)
                    continuation = tuple(native_checkpoint_messages(checkpoint.messages, restore=True))
                    latest_response = next((i for i in range(len(continuation) - 1, -1, -1)
                                            if isinstance(continuation[i], ModelResponse)), -1)
                    pending = {}
                    if latest_response >= 0:
                        pending = {part.tool_call_id: part.tool_name for part in continuation[latest_response].parts
                                   if isinstance(part, ToolCallPart)}
                        for message in continuation[latest_response + 1:]:
                            if isinstance(message, ModelRequest):
                                for part in message.parts:
                                    if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                                        pending.pop(part.tool_call_id, None)
                    if ready is not None and checkpoint.state != "complete" and not pending:
                        self._gap(key)
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    if ready is None:
                        if not pending and checkpoint.state == "complete":
                            events = await steps.list_events(run_id=key.turn_id)
                            completed_checkpoint = any(event.kind == "run_completed" for event in events)
                        delegate_pending = "delegate_task" in pending.values()
                        if settle_subagents and not completed_checkpoint and delegate_pending:
                            recover_subagent = self._recover_subagent
                            assert recover_subagent is not None

                            start_reserved = len(pending) == 1

                            async def recover_delegate(original, args):
                                return await recover_subagent(key, generation + 1, admitted, binding,
                                                              mode, tools, original, args, start_reserved)

                            async def recover_native_results():
                                return await tools.resume_canonical_effects(pending, recover_delegate=recover_delegate)
                        if recover_native_results is None:
                            retained_returns = tuple(part for message in continuation if isinstance(message, ModelRequest)
                                                     for part in message.parts if isinstance(part, ToolReturnPart))
                            checkpoint_completed_calls = None
                            if (not pending and not completed_checkpoint and latest_response >= 0
                                    and latest_response == len(continuation) - 2
                                    and isinstance(continuation[-1], ModelRequest)):
                                call_keys = [(part.tool_call_id, part.tool_name)
                                             for part in continuation[latest_response].parts if isinstance(part, ToolCallPart)]
                                returns = tuple(part for part in continuation[-1].parts if isinstance(part, ToolReturnPart))
                                if (call_keys and len({call_id for call_id, _ in call_keys}) == len(call_keys)
                                        and call_keys == [(part.tool_call_id, part.tool_name) for part in returns]
                                        and all(part.outcome == "success" for part in returns)):
                                    checkpoint_completed_calls = dict(call_keys)
                            deferred_results = await tools.resume_canonical_effects(
                                pending, interrupted_generation=generation, retained_returns=retained_returns,
                                checkpoint_completed_calls=checkpoint_completed_calls,
                            )
                            if checkpoint_completed_calls is not None and deferred_results is None:
                                await self._store.expire_stopped_lease(key, generation=generation + 1)
                                self._gap(key)
                                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                            if not pending and not completed_checkpoint:
                                latest_ids = (frozenset(part.tool_call_id for part in continuation[latest_response].parts
                                                       if isinstance(part, ToolCallPart)) if latest_response >= 0 else frozenset())
                                if (continuation and isinstance(continuation[-1], ModelRequest)
                                        and any(isinstance(part, ToolReturnPart) and part.tool_call_id in latest_ids
                                                and tools.has_checkpoint_interruption(part)
                                                for part in continuation[-1].parts)):
                                    # The SDK skips original tool returns already
                                    # present in the retained last request.
                                    deferred_results = DeferredToolResults()
                                elif checkpoint.state != "complete" and deferred_results is None:
                                    self._gap(key)
                                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    else:
                        if (checkpoint.state != "interrupted"
                                or set(pending) != set(settled_results.approvals) | set(settled_results.calls)
                                or set(settled_results.approvals) & set(settled_results.calls)):
                            self._gap(key)
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        for operation in (*settled_continuation.approvals, *settled_continuation.calls):
                            if pending.get(operation.tool_call_id) != operation.tool_name:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        for operation in settled_continuation.approvals:
                            if isinstance(settled_results.approvals.get(operation.tool_call_id), ToolApproved):
                                await tools.resume_canonical_approval(operation, generation=generation)
                        deferred_results = settled_results
            turn = _AdmittedTurn(
                key, row["turn_no"], model_prompt if isinstance(model_prompt, str) else prompt, history,
                _TURN_BYTES - len(canonical_json_bytes({**request, "binding": frozen_binding})),
                session["revision"], repository_instructions, generation + 1, session_context=session_context,
                model_prompt=None if isinstance(model_prompt, str) else model_prompt,
                attachments=() if stored is None else stored_input_attachment_views(stored),
                stored_input=request.get("input"), native_step=native_step,
                continuation=continuation, deferred_results=deferred_results,
                deferred_approvals=() if settled_continuation is None else settled_continuation.approvals,
                native_subagents=native_subagents,
                settled_native_subagents=settle_subagents and not completed_checkpoint,
                recover_native_results=recover_native_results,
                instruction_frontier=instruction_frontier,
                instruction_scopes=_merging_instruction_scopes(
                    decode_instruction_scopes(session["config_json"]),
                    None if instruction_frontier is None else instruction_frontier.overlay,
                ),
                completed_checkpoint=completed_checkpoint, output_retries_used=output_retries_used,
                mode=mode, planning=planning, plan_store=plan_store, budget=budget,
                split_source=split,
                preload_skill_inputs=tuple(request.get("preload_skills", ())),
                previous_turn_interrupted=request.get("previous_turn_interrupted") is True,
            )
            self._admitted(turn)
            try:
                await self._settle_with_lease(
                    turn, binding=binding, context=context, limits=limits, event_sink=event_sink, thinking=request["thinking"],
                )
            except Exception as error:
                if (recover_native_results is not None and isinstance(error, AIError)
                        and error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                        and error.safe_details.get("reason") == "native_child_reservation_missing"
                        and error.safe_details.get("generation") == turn.generation):
                    # The producer has been joined and no original child could be started.
                    await self._store.expire_stopped_lease(key, generation=turn.generation)
                self._producer_stopped(turn)
                raise
            return True

        task = self._spawn_owned(recover, name="linktools-recover-regenerable-turn")
        return await asyncio.shield(task)

    def _spawn_owned(
        self, operation: Callable[[], Coroutine[Any, Any, _T]], *, name: str,
    ) -> asyncio.Task[_T]:
        # A Runtime may be one of several (even on distinct event loops).
        # Reserve before reading input bodies/context, never wait in a queue.
        if not _TURN_SLOTS.acquire(blocking=False):
            raise AIError(ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED)
        coroutine = operation()
        try:
            task = asyncio.create_task(coroutine, name=name)
        except BaseException:
            coroutine.close()
            _TURN_SLOTS.release()
            raise
        self._tasks.add(task)

        def completed(owned: asyncio.Task[_T]) -> None:
            self._tasks.discard(owned)
            _TURN_SLOTS.release()
            if not owned.cancelled():
                owned.exception()

        task.add_done_callback(completed)
        return task

    async def start(
        self, key: SessionTurnKey, *, revision: int, through: int, request_key: str,
        prompt: CanonicalUserInput, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue, cwd: str | None = None,
        files: Sequence[str] = (), lineage: tuple[SessionTurnSegment, ...] = (),
        session_context: SessionContext = "all", mode: ExecutionMode = "run", planning: bool = False,
        explicit_agent_binding: bool = False, budget: RunBudget | None = None,
        attachment_display_names: tuple[str, ...] = (), instruction_scopes: tuple[str, ...] = (),
    ) -> bool:
        """Return only after admission commits; a disconnected observer cannot cancel the producer."""
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)

        accepted: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        async def produce() -> None:
            try:
                turn = await self.admit(
                    key, revision=revision, through=through, request_key=request_key,
                    prompt=prompt, binding=binding, context=context, thinking=thinking, cwd=cwd, files=files, lineage=lineage,
                    session_context=session_context, mode=mode, planning=planning,
                    explicit_agent_binding=explicit_agent_binding, budget=budget,
                    attachment_display_names=attachment_display_names, instruction_scopes=instruction_scopes,
                )
                accepted.set_result(turn is not None)
                if turn is not None:
                    self._admitted(turn)
                    await self._settle_with_lease(
                        turn, binding=binding, context=context, limits=limits,
                        event_sink=event_sink, thinking=thinking,
                    )
            except asyncio.CancelledError:
                if not accepted.done():
                    accepted.cancel()
                raise
            except Exception as error:
                if not accepted.done():
                    accepted.set_exception(error)
                else:
                    self._producer_stopped(turn)
                    cause = error.__cause__ or error
                    frame = cause.__traceback__
                    while frame is not None and frame.tb_next is not None:
                        frame = frame.tb_next
                    location = None if frame is None else (
                        frame.tb_frame.f_globals.get("__name__"),
                        frame.tb_frame.f_code.co_name, frame.tb_lineno,
                    )
                    _logger.error(
                        "Regenerable turn producer stopped: turn=%s code=%s error=%s cause=%s at=%s",
                        key.turn_id, error.code.value if isinstance(error, AIError) else ErrorCode.INTERNAL_ERROR.value,
                        type(error).__name__, type(cause).__name__, location,
                    )

        self._spawn_owned(produce, name="linktools-regenerable-turn")
        try:
            return await asyncio.shield(accepted)
        except asyncio.CancelledError:
            # The source transaction may still commit. Drain the receipt's
            # exception without treating disconnection as a business cancel.
            def consume(receipt: asyncio.Future[bool]) -> None:
                if not receipt.cancelled():
                    receipt.exception()

            accepted.add_done_callback(consume)
            raise

    async def close(self) -> None:
        self._closed = True
        for session_id, listeners in tuple(self._observers.items()):
            self._publish(session_id, {"type": "gap"})
            count = len(listeners)
            listeners.clear()
            _OBSERVER_SLOTS.release(count)
        self._observers.clear()
        self._observed_revisions.clear()
        self._observed_cancellation.clear()
        tasks = (*self._tasks, *self._cancel_notices.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._lifecycle_observers.clear()
        await self._lossy_trace.close()

    async def _settle_with_lease(
        self, turn: _AdmittedTurn, *, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
    ) -> None:
        fields = {"turn_id": turn.key.turn_id, "turn_no": turn.number,
                  "generation": turn.generation, "revision": turn.revision}

        attempts: dict[str, int] = {}

        def model_key(position: Mapping[str, JsonValue]) -> str | None:
            run, request = position.get("agent_run_seq"), position.get("model_request_seq")
            if type(run) is int and run > 0 and type(request) is int and request > 0:
                return f"model:{run}:{request}"
            return None

        def reset_preview(key: str, attempt: int) -> None:
            self._publish(turn.key.session_id, {
                **fields, "type": "preview_reset", "model_key": key, "attempt": attempt,
            })

        async def emit(emission: AgentEmission) -> None:
            if isinstance(emission, LiveDelta) and emission.event_type in (
                ExecutionDeltaType.ASSISTANT_TEXT_DELTA, ExecutionDeltaType.ASSISTANT_THINKING_DELTA,
            ):
                position = emission.position or {}
                key = model_key(position)
                attempt = position.get("stream_attempt_seq")
                scope: dict[str, JsonValue] = {}
                if key is not None and type(attempt) is int and attempt > 0:
                    previous = attempts.get(key)
                    if previous is not None and previous > attempt:
                        await event_sink(emission)
                        return
                    if previous is not None and previous < attempt:
                        reset_preview(key, previous)
                    attempts[key] = attempt
                    scope = {"model_key": key, "attempt": attempt}
                part_index = position.get("part_index")
                if type(part_index) is int and part_index >= 0:
                    scope["part_index"] = part_index
                kind = "thinking" if emission.event_type is ExecutionDeltaType.ASSISTANT_THINKING_DELTA else "text"
                if emission.content:
                    self._publish(turn.key.session_id, {
                        **fields, **scope, "type": "delta", "delta_kind": kind, "text": emission.content,
                    })
            elif isinstance(emission, DurableBoundary):
                item = project_tool_process_event(emission.event_type.value, emission.payload)
                if item is not None:
                    # Host subscribers get only a code unless explicitly opted in.
                    error_code = item.error_code
                    if item.status == "FAILED" and error_code is None:
                        error_code = ErrorCode.TOOL_EXECUTION_FAILED
                    item = replace(item, error_code=error_code, error_message=emission.live_error_message)
                    self._publish(turn.key.session_id, {
                        **fields, "type": "tool", "item": asdict(item),
                    })
                elif (emission.event_type is ExecutionEventType.ASSISTANT_PART_COMPLETED
                      and isinstance(emission.payload, Mapping)):
                    payload = emission.payload
                    key = model_key(payload)
                    attempt, index = payload.get("stream_attempt_seq"), payload.get("part_index")
                    elapsed = payload.get("thinking_elapsed_ms")
                    if (key is not None and type(attempt) is int and attempts.get(key) == attempt
                            and type(index) is int and index >= 0 and type(elapsed) is int and elapsed >= 0):
                        self._publish(turn.key.session_id, {
                            **fields, "type": "thinking_completed", "model_key": key, "attempt": attempt,
                            "part_index": index, "elapsed_ms": elapsed,
                        })
                elif isinstance(emission.payload, Mapping) and emission.event_type in {
                    ExecutionEventType.MODEL_REQUEST_STARTED, ExecutionEventType.MODEL_REQUEST_FINISHED,
                }:
                    # Only actual provider request lifecycle and timing are public
                    # here; never forward usage, request/response, or thinking parts.
                    payload = emission.payload
                    key = model_key(payload)
                    status, duration = payload.get("status"), payload.get("duration_ns")
                    if (key is not None
                            and (status == "RUNNING" if emission.event_type is ExecutionEventType.MODEL_REQUEST_STARTED
                                 else status in ("SUCCEEDED", "FAILED", "CANCELLED"))):
                        if status != "RUNNING":
                            attempt = attempts.pop(key, None)
                            if status != "SUCCEEDED" and attempt is not None:
                                reset_preview(key, attempt)
                        elapsed = ({"elapsed_ms": duration // 1_000_000}
                                   if status != "RUNNING" and type(duration) is int and duration >= 0 else {})
                        self._publish(turn.key.session_id, {
                            **fields, "type": "progress",
                            "item": {"kind": "model", "key": key, "status": status, **elapsed},
                        })
            await event_sink(emission)

        producer = asyncio.create_task(self.settle(
            turn, binding=binding, context=context, limits=limits, event_sink=emit, thinking=thinking,
        ))
        self._producers[turn.key] = producer
        try:
            while not producer.done():
                done, _ = await asyncio.wait((producer,), timeout=_LEASE_SECONDS / 3)
                if done or producer.cancelling():
                    break
                if not await self._store.renew(turn.key, generation=turn.generation, lease_seconds=_LEASE_SECONDS):
                    if not producer.cancelling():
                        producer.cancel()
                    break
            outcome, = await asyncio.gather(producer, return_exceptions=True)
            if outcome is True:
                return
            if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                raise outcome
            # The normal finish path has no post-commit read. When settlement
            # lost its fence, only a durable cancellation authorizes this branch.
            row = await self._store.get_turn(turn.key)
            if row is not None and row["state"] == "cancelling" and row["generation"] == turn.generation:
                result = row["result_json"]
                if isinstance(result, Mapping) and result.get("cancel_effects_unresolved") is True:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                # A worker handoff only drains local tasks. Durable child
                # cancellation is authorized by this committed parent state.
                terminal_children: frozenset[str] = frozenset()
                if binding.binding_contract.subagents:
                    if self._cancel_subagents is None:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    terminal_children = await self._cancel_subagents(turn.key, context.principal, context.principal)
                if turn.native_step is not None and turn.split_source is None:
                    # A bare CancelledError after tool tasks have drained is
                    # not evidence that an ordinary business effect settled.
                    _, tools = turn.native_step
                    if type(tools._repository) is not ToolRepositoryImpl:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    unsettled = await tools._repository.list_by_execution(
                        turn.key.turn_id, tenant_id=context.principal.tenant_id,
                        _states=frozenset({ToolOperationStatus.CLAIMED.value, ToolOperationStatus.EFFECT_UNKNOWN.value}),
                    )
                    if any(
                        operation.status is not ToolOperationStatus.EFFECT_UNKNOWN
                        or operation.producer_generation != turn.generation
                        or operation.tool_name != "delegate_task"
                        or operation.tool_call_id not in terminal_children
                        or (operation.tool_operation_id, operation.owner, operation.fence, operation.tool_call_id)
                           not in turn.native_stopped_delegates
                        for operation in unsettled
                    ):
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                code = ErrorCode.EXECUTION_CANCELLED.value
                if await self._finish_retained_turn(
                    turn, state="cancelled",
                    result={"error_code": code},
                ):
                    self._terminal(turn, "cancelled", error_code=code)
                    return
            self._gap(turn.key)
        finally:
            if self._producers.get(turn.key) is producer:
                self._producers.pop(turn.key, None)
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    @property
    def attachment_limits(self) -> PromptLimits:
        return self._input_materializer.limits

    async def _read_admitted_history(
        self, key: SessionTurnKey, *, context_from: int, context_through: int,
        lineage: tuple[SessionTurnSegment, ...], session_context: SessionContext,
        prompt: str, stored: StoredUserInput | None,
    ) -> tuple[ModelMessage, ...]:
        input_bytes, media_bytes = self._input_usage(prompt, stored)
        remaining = _CONTEXT_BYTES - input_bytes
        remaining_media = self.attachment_limits.max_binary_input_bytes - media_bytes
        after = context_from - 1
        if (not 0 <= context_through - after <= 128 or session_context not in ("all", "current_turn")
                or session_context == "current_turn" and after != context_through):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "The admitted context range is invalid")
        if remaining < 0 or remaining_media < 0:
            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
        history: list[ModelMessage] = []
        positions = await self._store.read_context_range(
            key.session_id, through=context_through, after=after, lineage=lineage,
            missing=ErrorCode.STORAGE_RECOVERY_REQUIRED,
        )
        for position in positions:
            prior = position.request
            prior_input = self._stored_input(prior)
            input_bytes, media_bytes = self._input_usage(prior["prompt"], prior_input)
            remaining -= input_bytes + position.transcript_bytes
            remaining_media -= media_bytes
            if remaining < 0 or remaining_media < 0:
                raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
            restored = await self._restore_prompt(prior["prompt"], prior_input)
            history.extend(self._context_messages(restored, position.messages, interrupted=position.interrupted))
        return tuple(history)

    async def _reserve_attached_graph_turn(
        self, commands: StateCommands, reservation: ExecutionStartReservation, *,
        session_id: str, session: Mapping[str, object], request_key: str, prompt: CanonicalUserInput, principal: Principal, tenant_id: str,
        authorization: AuthorizationPolicy, task_repository: TaskRepositoryImpl,
        task_admission: TaskGraphAdmission, task_invocation: TaskNodeInvocation, binding: AgentBinding,
    ) -> tuple[ExecutionStartReservationResult, _AttachedGraphTurn | None]:
        """Join an already prepared Graph reservation to one authorized native turn."""
        execution = reservation.execution
        lease = task_invocation.task_lease
        if (lease is None or (lease.tenant_id, lease.graph_id, lease.node_id)
                != (tenant_id, task_admission.graph_id, task_invocation.node.node_id)
                or task_invocation.graph_id != task_admission.graph_id
                or task_invocation.principal != principal or task_admission.principal != principal
                or task_admission.budget_scope_id != execution.budget_scope_id
                or task_admission.tool_effects != execution.tool_effects):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if (principal.tenant_id != tenant_id
                or (principal.principal_id, principal.kind) != (execution.principal_id, execution.principal_kind)):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if (not execution.requires_task_invocation_capture or execution.session_id is not None
                or execution.root_execution_id != execution.execution_id or execution.parent_execution_id is not None
                or execution.input_context is not None or execution.repository_instructions is not None
                or not isinstance(execution.binding, AgentBindingContract)):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        key = SessionTurnKey(session_id, execution.execution_id)
        await authorization.authorize(
            principal, AuthorizationAction.EXECUTION_RUN, ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id),
        )
        if session["status"] != "open":
            raise AIError(ErrorCode.SESSION_CONFLICT)
        lineage = decode_session_lineage(session["lineage_json"])
        await self._authorize_lineage(lineage, principal=principal, tenant_id=tenant_id, authorization=authorization)
        config = session["config_json"]
        cwd = config.get("cwd")
        if await self.canonicalize_cwd(cwd) != cwd:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "cwd"})
        captured = execution.stored_user_input
        native_input = None if captured is None else native_graph_input_content(captured)
        if (native_input is None or (native_input != prompt if isinstance(prompt, str)
                                    else isinstance(native_input, str) or tuple(native_input) != tuple(prompt))):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        text = native_input if isinstance(native_input, str) else ""
        stored_input = None if isinstance(native_input, str) else encode_domain(captured)
        instruction_scopes = decode_instruction_scopes(config)
        # The captured paths are provenance, not a request to read or materialize
        # the files again. Recovery reuses the instructions admitted below.
        files = () if captured.view is None else captured.view.get("files", ())
        instructions = await _resolve_initial_instruction_scopes(
            self._resolver_for(binding), cwd, instruction_scopes, files,
        )
        scopes = _merging_instruction_scopes(instruction_scopes, instructions)
        metadata = ImmutableJsonMapping(config.get("metadata", {}))
        request: dict[str, JsonValue] = {
            "prompt": text, "principal": asdict(principal), "thinking": execution.thinking,
            "session_metadata": dict(metadata), "cwd": cwd,
            "repository_instructions": None if instructions is None else instructions.to_payload(),
        }
        if execution.session_context == "current_turn":
            request["session_context"] = "current_turn"
        if execution.correlation:
            request["correlation"] = dict(execution.correlation)
        if execution.mode == "plan":
            request["mode"] = "plan"
        if execution.planning:
            request["planning"] = True
        through = session["last_turn_no"]
        revision = session["revision"]
        terminal_bytes = _TURN_BYTES - len(canonical_json_bytes({
            **request, **({} if stored_input is None else {"input": stored_input}),
            "binding": execution.binding.to_payload(),
        }))
        if terminal_bytes < _TERMINAL_RESERVE:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)
        input_bytes, media_bytes = self._input_usage(text, captured if stored_input is not None else None)
        context_from, history, _, _ = await self._admission_history(
            key, through=through, lineage=lineage, session_context=execution.session_context,
            remaining=_CONTEXT_BYTES - input_bytes,
            remaining_media=self.attachment_limits.max_binary_input_bytes - media_bytes,
        )
        result = await commands._reserve_canonical_graph_start(
            reservation, turns=self._store, key=key, expected_revision=revision,
            context_from=context_from, context_through=through, request_key=request_key,
            request=request, lease_seconds=_LEASE_SECONDS,
            task_repository=task_repository, task_lease=lease,
            instruction_scopes=None if scopes == instruction_scopes else scopes,
        )
        if not result.created:
            return result, None
        return result, _AttachedGraphTurn(
            key, 1, through + 1, text, tuple(history), terminal_bytes, revision + 1,
            cwd, context_from, through, metadata, execution.session_context, repository_instructions=instructions,
            stored_input=stored_input, model_prompt=None if isinstance(native_input, str) else tuple(native_input),
            attachments=stored_input_attachment_views(captured),
        )

    def _allows_native_subagents(
        self, binding: AgentBinding, native_step: tuple[AtomicStepPersistence, RuntimeToolOperationBridge] | None,
        *, mode: ExecutionMode, planning: bool, budgeted: bool, split: bool,
    ) -> bool:
        contract = binding.binding_contract
        return (
            bool(contract.subagents)
            and self._prepare_subagents is not None and self._cancel_subagents is not None
            and native_step is not None and native_step[0].capture_frontier
            and mode == "run" and not planning and not budgeted and not split
            and set(contract.subagent_binding_map) == set(contract.subagent_ids)
        )

    async def admit(
        self, key: SessionTurnKey, *, revision: int, through: int, request_key: str,
        prompt: CanonicalUserInput, binding: AgentBinding, context: AgentContext[object], thinking: ThinkingValue,
        cwd: str | None = None, files: Sequence[str] = (), lineage: tuple[SessionTurnSegment, ...] = (),
        session_context: SessionContext = "all", mode: ExecutionMode = "run", planning: bool = False,
        explicit_agent_binding: bool = False, budget: RunBudget | None = None,
        attachment_display_names: tuple[str, ...] = (), instruction_scopes: tuple[str, ...] = (),
    ) -> _AdmittedTurn | None:
        """Called by the session owner with authorized identity and execution binding."""
        native_step = None if self._prepare_step is None else await self._prepare_step(binding, key, 1)
        split = (native_step[0].store._canonical_split_source
                 if native_step is not None and isinstance(native_step[0].store, RuntimeStepStore) else None)
        if split is not None and (budget is not None or planning or mode != "run"):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        run_budget = None if budget is None else self._prepare_turn_budget(key, 1, native_step)
        if planning and self._prepare_plan is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        plan_store = self._prepare_plan(key, 1) if planning else None
        trace_enabled = self._observes_turn_trace(binding)
        native_subagents = self._allows_native_subagents(
            binding, native_step, mode=mode, planning=planning, budgeted=budget is not None,
            split=split is not None,
        )
        self._executor.require_regenerable(
            binding, canonical_step=native_step is not None,
            split_step=split is not None, native_subagents=native_subagents,
        )
        if await self.canonicalize_cwd(cwd) != cwd:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "cwd"})
        instruction_resolver = self._resolver_for(binding)
        frozen_binding = binding.binding_contract.to_payload()
        # A nonempty scalar owns its own text. A multipart sequence has one
        # complete typed input manifest and no duplicate text field.
        text = prompt if isinstance(prompt, str) else ""
        request: dict[str, JsonValue] = {
            "prompt": text,
            "principal": asdict(context.principal),
            "thinking": thinking,
            "session_metadata": dict(context.session_metadata),
            "cwd": cwd,
        }
        preload_skill_inputs = await self._executor.capture_preload_skills(binding)
        if preload_skill_inputs:
            request["preload_skills"] = [dict(value) for value in preload_skill_inputs]
        if split is not None:
            request["split_step"] = True
        if _runtime_feature_policy(binding.compiled_agent.runtime_contributions).session.checkpoints:
            request["session_checkpoints"] = True
        if run_budget is not None:
            request["budgeted"] = True
        if explicit_agent_binding:
            request["explicit_agent_binding"] = True
        if session_context == "current_turn":
            request["session_context"] = session_context
        if context.correlation:
            request["correlation"] = dict(context.correlation)
        if mode == "plan":
            request["mode"] = mode
        if planning:
            request["planning"] = True
        model_prompt = None
        repository_instructions = None
        attachments = ()
        input_size = len(text.encode("utf-8"))
        media_bytes = 0
        if not isinstance(prompt, str):
            materialized = await self._input_materializer.materialize(prompt, files)
            validate_model_inputs(binding.compiled_agent.model, materialized)
            repository_instructions = await _resolve_initial_instruction_scopes(
                instruction_resolver, cwd, instruction_scopes, materialized.view.get("files", ()),
            )
            model_prompt = tuple(materialized)
            captured = self._input_materializer.capture(materialized)
            attachments = stored_input_attachment_views(captured)
            if attachment_display_names:
                # Upload names annotate occurrences, not provider identifiers.
                if len(attachments) != len(attachment_display_names):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "attachment_display_names"})
                attachments = tuple(
                    {**item, "display_name": name}
                    for item, name in zip(attachments, attachment_display_names, strict=True)
                )
            input_size, media_bytes = self._input_usage("", captured, content=model_prompt)
            if input_size > _CONTEXT_BYTES:
                raise AIError(ErrorCode.PROMPT_TOO_LARGE)
            # The payload already contains the ordered text and native content
            # metadata. Only retain the derived attachment summaries as a view.
            captured = replace(captured, view={"version": 1, "attachments": list(attachments)})
            stored = await self._input_materializer.materialize_payload(captured, tenant_id=context.principal.tenant_id)
            input_size, media_bytes = self._input_usage("", stored)
            request["input"] = encode_domain(stored)
            request["input_intent"] = self._input_materializer.intent(prompt, files).digest
        elif files:
            materialized = await self._input_materializer.materialize(prompt, files)
            validate_model_inputs(binding.compiled_agent.model, materialized)
            repository_instructions = await _resolve_initial_instruction_scopes(
                instruction_resolver, cwd, instruction_scopes, materialized.view.get("files", ()),
            )
            # The materializer prepends attachment parts and retains the text as the last item.
            # Store the text once, separately from replayable file parts.
            model_prompt = tuple(materialized)
            detached = MaterializedUserContent(
                model_prompt[:-1], {"version": 1, "attachments": materialized.view["attachments"]},
            )
            captured = self._input_materializer.capture(detached)
            input_size, media_bytes = self._input_usage(text, captured, content=model_prompt[:-1])
            if input_size > _CONTEXT_BYTES:
                raise AIError(ErrorCode.PROMPT_TOO_LARGE)
            stored = await self._input_materializer.materialize_payload(captured, tenant_id=context.principal.tenant_id)
            if not stored.payload.object_refs():
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "Attachment object storage is required")
            request["input"] = encode_domain(stored)
            attachments = stored_input_attachment_views(stored)
        else:
            repository_instructions = await _resolve_initial_instruction_scopes(instruction_resolver, cwd, instruction_scopes)
        request["repository_instructions"] = None if repository_instructions is None else repository_instructions.to_payload()
        scopes = _merging_instruction_scopes(instruction_scopes, repository_instructions)
        terminal_bytes = _TURN_BYTES - len(canonical_json_bytes({**request, "binding": frozen_binding}))
        if input_size > _CONTEXT_BYTES:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)
        if terminal_bytes < _TERMINAL_RESERVE and isinstance(prompt, str) and not files:
            # Leave completed/competing receipts to the same replay path before
            # placing another body. Inline starts retain the no-extra-read path.
            if await self._store.find_request(key.session_id, request_key) is not None:
                return None
            request["prompt"] = ""
            if _TURN_BYTES - len(canonical_json_bytes({**request, "binding": frozen_binding})) < _TERMINAL_RESERVE:
                raise AIError(ErrorCode.PROMPT_TOO_LARGE)
            source = self._input_materializer.capture(prompt)
            placed = await self._input_materializer.materialize_payload(
                source, tenant_id=context.principal.tenant_id, externalize_text=True,
            )
            request["input"] = encode_domain(placed)
            terminal_bytes = _TURN_BYTES - len(canonical_json_bytes({**request, "binding": frozen_binding}))
        if terminal_bytes < _TERMINAL_RESERVE:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)

        context_from, history, interrupted, baseline_turn_id = await self._admission_history(
            key, through=through, lineage=lineage, session_context=session_context,
            remaining=_CONTEXT_BYTES - input_size,
            remaining_media=self.attachment_limits.max_binary_input_bytes - media_bytes,
        )
        if interrupted:
            request["previous_turn_interrupted"] = True
            terminal_bytes = _TURN_BYTES - len(canonical_json_bytes({**request, "binding": frozen_binding}))
            if terminal_bytes < _TERMINAL_RESERVE:
                raise AIError(ErrorCode.PROMPT_TOO_LARGE)

        if not await self._store.accept(
            key, expected_revision=revision, context_from=context_from, context_through=through,
            request_key=request_key, request=request, binding=frozen_binding, lease_seconds=_LEASE_SECONDS,
            **({} if run_budget is None else {"_budget": (run_budget.repository, budget)}),
            **({} if scopes == instruction_scopes else {"_instruction_scopes": scopes}),
        ):
            return None
        if baseline_turn_id is not None and native_step is not None and split is None:
            native_step[0].store.inherit_canonical_context(key, baseline_turn_id)
        return _AdmittedTurn(
            key, through + 1, text,
            history, terminal_bytes, revision + 1,
            repository_instructions, session_context=session_context, model_prompt=model_prompt, attachments=attachments,
            stored_input=request.get("input"), instruction_scopes=scopes,
            native_step=native_step, trace_enabled=trace_enabled, mode=mode, planning=planning, plan_store=plan_store,
            budget=run_budget, split_source=split, preload_skill_inputs=preload_skill_inputs,
            native_subagents=native_subagents, previous_turn_interrupted=interrupted,
        )

    async def _admission_history(
        self, key: SessionTurnKey, *, through: int, lineage: tuple[SessionTurnSegment, ...],
        session_context: SessionContext, remaining: int, remaining_media: int,
    ) -> tuple[int, list[ModelMessage], bool, str | None]:
        history: list[tuple[ModelMessage, ...]] = []
        baseline_turn_id = None
        context_from = through + 1
        candidates = (() if session_context == "current_turn"
                      else await self._store.recent_context(key.session_id, through=through, lineage=lineage))
        interrupted = bool(candidates and candidates[-1].turn_no == through and candidates[-1].interrupted)
        for candidate in reversed(candidates):
            if not isinstance(candidate.request, dict) or not isinstance(candidate.request.get("prompt"), str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Missing admitted session input")
            candidate_prompt = candidate.request["prompt"]
            stored = self._stored_input(candidate.request)
            input_bytes, media_bytes = self._input_usage(candidate_prompt, stored)
            size = input_bytes + candidate.transcript_bytes
            if size > remaining or media_bytes > remaining_media:
                break
            restored = await self._restore_prompt(candidate_prompt, stored)
            if not history and not candidate.interrupted:
                baseline_turn_id = candidate.turn_id
            history.append(self._context_messages(restored, candidate.messages, interrupted=candidate.interrupted))
            remaining -= size
            remaining_media -= media_bytes
            context_from = candidate.turn_no
        return context_from, [message for messages in reversed(history) for message in messages], interrupted, baseline_turn_id

    @staticmethod
    def _context_messages(
        prompt: str | Sequence[UserContent], messages: Sequence[ModelMessage], *, interrupted: bool,
    ) -> tuple[ModelMessage, ...]:
        if interrupted:
            # Keep an explicit historical boundary: the SDK can merge adjacent
            # requests when reloading an already completed exchange.
            content = prompt if isinstance(prompt, str) else list(prompt)
            timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc)
            return (
                ModelRequest(parts=[UserPromptPart(content, timestamp=timestamp)]),
                ModelResponse(parts=[TextPart("[Runtime: This turn was interrupted before a final answer.]")],
                              timestamp=timestamp),
            )
        return restore_turn_prompt(prompt, messages)

    @staticmethod
    def _stored_input(request: Mapping[str, JsonValue]) -> StoredUserInput | None:
        return decode_domain(request["input"], StoredUserInput) if "input" in request else None

    @staticmethod
    def _input_usage(
        prompt: str, stored: StoredUserInput | None, *, content: Sequence[UserContent] = (),
    ) -> tuple[int, int]:
        """Charge text and metadata separately from the binary data carried by the prompt."""
        text_bytes = len(prompt.encode("utf-8"))
        if stored is None:
            return text_bytes, 0
        payload = stored.payload
        sizes = ([ref.size for ref in payload.object_refs()] if payload.encoding == "json-attachments"
                 else [len(item.data) for item in content if isinstance(item, BinaryContent)])
        # The canonical user-content payload retains the original encoded length
        # even after attachment bytes move to object storage. Only charge that
        # base64 body against the separate binary budget.
        return text_bytes + payload.size - sum(4 * ((size + 2) // 3) for size in sizes), sum(sizes)

    async def _restore_prompt(
        self, prompt: str, stored: StoredUserInput | None,
    ) -> str | Sequence[UserContent]:
        if stored is None:
            return prompt
        if stored.codec == "text":
            if prompt:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Turn input has two text owners")
            return await self._store.read_scalar_input(stored)
        detached = await self._input_materializer.restore(stored)
        if isinstance(detached, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Detached turn input is not multipart")
        return (*detached, prompt) if prompt else tuple(detached)

    def _instruction_boundary(
        self, turn: _AdmittedTurn, binding: AgentBinding,
    ) -> _RepositoryInstructionBoundary | None:
        resolver = self._resolver_for(binding)
        scopes = turn.instruction_scopes
        initial = turn.repository_instructions
        frontier = turn.instruction_frontier
        overlay = None if frontier is None else frontier.overlay
        if overlay is not None:
            if resolver is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            resolver.validate(_merge_repository_instructions(initial, overlay))

        if resolver is None:
            return None

        async def check(*, initial, overlay, tool_name, tool_call_id, arguments, path_fields):
            nonlocal scopes
            paths = _instruction_paths(arguments, path_fields)
            if not paths:
                return overlay
            discovered = await _discover_repository_overlay(
                resolver, initial, overlay, paths,
                fits_overlay=lambda value: len(canonical_json_bytes(
                    SessionInstructionFrontier(value).to_json(),
                )) <= turn.terminal_bytes,
            )
            if discovered is None:
                return overlay
            # Expose new instructions only after the fenced transaction commits.
            merged = _merging_instruction_scopes(scopes, discovered)
            await self._store.append_instruction_frontier(
                turn.key, generation=turn.generation, max_bytes=turn.terminal_bytes,
                frontier=SessionInstructionFrontier(discovered),
                instruction_scopes=None if merged == scopes else merged,
            )
            scopes = merged
            return discovered

        return _RepositoryInstructionBoundary(check, initial, overlay)

    async def _finish_retained_turn(
        self, turn: _AdmittedTurn, *, state, result=None, transcript=None,
        binding_refs=(), output_ref=None, usage: UsageMetrics | None = None,
    ) -> bool:
        split = turn.split_source
        if split is None:
            options = ({"transcript": transcript, "binding_refs": binding_refs, "output_ref": output_ref, "usage": usage}
                       if state == "succeeded" else {"result": result})
            return await self._store.finish(
                turn.key, generation=turn.generation, turn_no=turn.number, state=state, **options,
            )
        return await split.finish(
            turn.key, generation=turn.generation, turn_no=turn.number, state=state,
            result=result, transcript=transcript, binding_refs=binding_refs, output_ref=output_ref,
        )

    async def settle(
        self, turn: _AdmittedTurn, *, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
    ) -> bool:
        """Execute after durable admission and publish only if this producer still owns the turn."""
        from ..model.messages import canonical_structured_output_reference, encode_model_messages

        captured: _CapturedTurnInteractions | None = None
        capture_trace = None
        if turn.trace_enabled and turn.generation == 1:
            from ._canonical_trace_interactions import _CapturedTurnInteractions

            def capture_trace(staging):
                nonlocal captured
                try:
                    captured = _CapturedTurnInteractions.capture(
                        staging, turn.key.turn_id, turn.key.session_id, settled_only=True,
                    )
                except Exception as error:
                    _logger.warning("Optional Session Trace capture discarded: %s", type(error).__name__)

        def offer_trace(state: str, error_code: str | None = None) -> None:
            if captured is not None:
                from ._lossy_session_trace import SessionTraceBatch
                self._lossy_trace.offer(self._trace_archive, SessionTraceBatch(captured, state, error_code))

        structured = binding.output_binding.mode == "structured"
        model_prompt = turn.prompt if turn.model_prompt is None else turn.model_prompt
        usage = None
        try:
            if turn.completed_checkpoint:
                if turn.continuation is None:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                messages = turn.continuation
                output = reply_from_turn(messages, structured=structured)
                binding.output_binding.validate_payload(output)
            else:
                repository_boundary = self._instruction_boundary(turn, binding)
                native_subagents = nullcontext()
                if turn.native_subagents or turn.recover_native_results is not None:
                    if self._prepare_subagents is None:
                        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                    native_subagents = self._prepare_subagents(
                        turn, binding, context.principal,
                        repository_boundary.current_instructions if repository_boundary is not None
                        else lambda: turn.repository_instructions,
                    )
                async with native_subagents as native_subagent_leaf:
                    deferred_results = turn.deferred_results
                    if turn.recover_native_results is not None:
                        # The live parent owner and renewal loop now fence any
                        # original child first-start. No new delegation is exposed.
                        try:
                            deferred_results = await turn.recover_native_results()
                        except Exception as error:
                            # The child may have crossed its first-start fence.
                            # Missing proof cannot be projected as parent failure.
                            if isinstance(error, AIError) and error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED:
                                if error.safe_details.get("reason") == "native_child_reservation_missing":
                                    raise AIError(error.code, safe_details={
                                        "reason": "native_child_reservation_missing", "generation": turn.generation,
                                    }) from error
                                raise
                            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
                        native_subagent_leaf = None
                    outcome = await self._executor.execute_regenerable(
                        binding=binding, context=context, prompt=model_prompt, history=turn.history, limits=limits,
                        event_sink=event_sink, thinking=thinking, streamed_output_bytes=_CONTEXT_BYTES,
                        workspace=self._workspace, repository_instructions=turn.repository_instructions,
                        repository_instruction_boundary=repository_boundary,
                        canonical_step=None if turn.native_step is None else turn.native_step[0],
                        tool_operations=None if turn.native_step is None else turn.native_step[1],
                        generation=turn.generation, continuation=turn.continuation,
                        deferred_tool_results=deferred_results, deferred_approvals=turn.deferred_approvals,
                        output_retries_used=turn.output_retries_used, mode=turn.mode, planning=turn.planning,
                        plan_store=turn.plan_store, budget=turn.budget,
                        preload_skill_inputs=turn.preload_skill_inputs, split_step=turn.split_source is not None,
                        previous_turn_interrupted=turn.previous_turn_interrupted,
                        native_subagent_leaf=native_subagent_leaf,
                        settled_native_subagents=turn.settled_native_subagents, capture_trace=capture_trace,
                    )
                if isinstance(outcome, DeferredToolRequests):
                    if turn.native_step is None or self._publish_deferred is None:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    capability, tools = turn.native_step
                    if turn.split_source is not None and (
                        turn.generation != 1 or tools._split_approval_ref is None
                        or len(outcome.approvals) != 1 or outcome.calls or any(outcome.metadata.values())
                    ):
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    if not isinstance(capability.store, RuntimeStepStore):
                        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                    checkpoint_id = capability.store.require_interrupted_checkpoint()
                    committed = await self._publish_deferred(
                        turn, checkpoint_id, tools, outcome,
                        split_source=capability.store._canonical_split_source,
                    )
                    if committed:
                        offer_trace("waiting")
                        self._publish(turn.key.session_id, {
                            "type": "waiting", "turn_id": turn.key.turn_id, "turn_no": turn.number,
                            "generation": turn.generation, "revision": turn.revision,
                        })
                        self._notice_lifecycle(turn.key, "waiting", turn.revision)
                    return committed
                messages, output = outcome.messages, outcome.output
                # Resumed checkpoints do not retain accounting for prior attempts.
                if turn.split_source is None and turn.generation == 1 and turn.continuation is None:
                    usage = outcome.usage
            frozen_spec = binding.binding_contract.agent_spec
            context_messages, transcript, binding_refs = prepare_native_turn_transcript(
                model_prompt, turn.history, messages, output,
                instructions=frozen_spec.instructions, system_prompt=frozen_spec.system_prompt, structured=structured,
            )
            # The transcript is published as a native archive root, not in the
            # turn row. Bound its body separately from the terminal receipt.
            if len(encode_model_messages(transcript)) > _CONTEXT_BYTES:
                raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
            output_ref = canonical_structured_output_reference(transcript) if structured else None
            receipt_metadata = {}
            if binding_refs:
                receipt_metadata["binding_refs"] = [list(item) for item in binding_refs]
            if output_ref is not None:
                receipt_metadata["output_ref"] = list(output_ref)
            if usage is not None:
                receipt_metadata["usage"] = asdict(usage)
            # Leave the existing reservation for the root descriptor and fixed
            # receipt fields; only variable metadata competes with that margin.
            if receipt_metadata and len(canonical_json_bytes(receipt_metadata)) > turn.terminal_bytes - _TERMINAL_RESERVE:
                raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
        except Exception as error:
            if (turn.native_step is not None and isinstance(error, AIError)
                    and error.code in {ErrorCode.STORAGE_RECOVERY_REQUIRED, ErrorCode.STORAGE_COMMIT_UNKNOWN,
                                       ErrorCode.TOOL_EFFECT_UNKNOWN}):
                raise
            code = error.code.value if isinstance(error, AIError) else ErrorCode.INTERNAL_ERROR.value
            if code == ErrorCode.INTERNAL_ERROR.value:
                cause = error.__cause__ or error
                frame = cause.__traceback__
                while frame is not None and frame.tb_next is not None:
                    frame = frame.tb_next
                location = ("unknown" if frame is None else
                            f"{frame.tb_frame.f_globals.get('__name__', '?')}:"
                            f"{frame.tb_frame.f_code.co_name}:{frame.tb_lineno}")
                _logger.error(
                    "Canonical turn failed: turn=%s code=%s cause=%s at=%s",
                    turn.key.turn_id, code, type(cause).__name__, location,
                )
            committed = await self._finish_retained_turn(
                turn, state="failed", result={"error_code": code},
            )
            if committed:
                offer_trace("failed", code)
                self._terminal(turn, "failed", error_code=code)
            return committed
        committed = await self._finish_retained_turn(
            turn, state="succeeded",
            transcript=transcript, binding_refs=binding_refs, output_ref=output_ref, usage=usage,
        )
        if committed:
            offer_trace("succeeded")
            self._terminal(
                turn, "succeeded", output=output, messages=context_messages, output_ref=output_ref,
            )
        return committed
