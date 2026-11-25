#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission and settlement for ordinary, regenerable model turns."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

from linktools.core import environ
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from threading import BoundedSemaphore
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy.exc import IntegrityError

from ..agent import AgentBindingContract
from ..capability import AgentContext
from ..core import AuthorizationAction, ExecutionDeltaType, ExecutionEventType, JsonValue, Principal, ResourceKind, ResourceRef, canonical_json_bytes, canonical_sha256, normalize_json_value
from ..errors import AIError, ErrorCode
from ..inputs import stored_input_attachment_views
from ..model import MaterializedUserContent, StoredUserInput, normalize_input_files, validate_model_inputs
from ..persistence.codec import decode_domain, encode_domain
from ..persistence.turns import SessionInstructionFrontier, SessionTurnKey, SessionTurnSegment, decode_session_lineage
from ..session import ToolProcessItem, project_tool_process_event
from ..spec import RepositoryInstructions
from ._agent_executor import DurableBoundary, LiveDelta
from ._regenerable_transcript import prepare_native_turn_transcript, reply_from_turn, restore_turn_prompt
from ._step_store import RuntimeStepStore
from ._repository_instructions import (
    _RepositoryInstructionApproval, _RepositoryInstructionBoundary,
    _discover_repository_overlay, _instruction_paths, _merge_repository_instructions,
)
from ._tool_boundary import is_instruction_file_read

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage, UserContent

    from ..agent import AgentBinding
    from ..capability.steps import AtomicStepPersistence
    from ._tool import RuntimeToolOperationBridge
    from ..core import AuthorizationPolicy, PromptLimits, ThinkingValue
    from ..inputs import ExecutionInputMaterializer
    from ..spec import RepositoryInstructionResolver
    from ..workspace import Workspace
    from ..persistence.turns import CanonicalDeferredFrontier, SqlSessionTurns
    from ..execution import PendingToolContinuation, PendingDeferredCall
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
    model_prompt: Sequence[UserContent] | None = None
    attachments: Sequence[Mapping[str, JsonValue]] = ()
    stored_input: JsonValue = None
    native_step: tuple[AtomicStepPersistence, RuntimeToolOperationBridge] | None = None
    continuation: tuple[ModelMessage, ...] | None = None
    deferred_results: DeferredToolResults | None = None
    deferred_approvals: tuple[PendingDeferredCall, ...] = ()
    instruction_frontier: SessionInstructionFrontier | None = None
    completed_checkpoint: bool = False
    output_retries_used: int = 0


class _RegenerableTurnRunner:
    def __init__(
        self, store: SqlSessionTurns, executor: AgentExecutor, *,
        workspace: Workspace | None,
        input_materializer: ExecutionInputMaterializer,
        instruction_resolver: RepositoryInstructionResolver | None,
        instruction_resolver_for: Callable[[RepositoryInstructions], RepositoryInstructionResolver],
        prepare_step: Callable[[AgentBinding, SessionTurnKey, int], Awaitable[
            tuple[AtomicStepPersistence, RuntimeToolOperationBridge] | None]] | None = None,
    ) -> None:
        self._store = store
        self._executor = executor
        self._workspace = workspace
        self._input_materializer = input_materializer
        self._workspace_access = input_materializer.access
        self._instruction_resolver = instruction_resolver
        self._instruction_resolver_for = instruction_resolver_for
        self._prepare_step = prepare_step
        self._publish_deferred = None
        self._tasks: set[asyncio.Task[None] | asyncio.Task[bool]] = set()
        self._producers: dict[SessionTurnKey, asyncio.Task[bool]] = {}
        self._closed = False
        self._observers: dict[str, set[Callable[[dict[str, JsonValue]], None]]] = {}
        self._lifecycle_observers: set[Callable[[dict[str, JsonValue]], None]] = set()
        self._observed_revisions: dict[str, int] = {}
        self._cancel_notices: dict[SessionTurnKey, asyncio.Task[None]] = {}
        self._observed_cancellation: dict[str, str] = {}

    def bind_deferred_publication(self, publish) -> None:
        self._publish_deferred = publish

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
                if row is not None and row["state"] == "cancelling" and self._producers.get(key) is producer:
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

    def observe(self, session_id: str, callback: Callable[[dict[str, JsonValue]], None]) -> Callable[[], None]:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if not _OBSERVER_SLOTS.acquire(blocking=False):
            raise AIError(ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED)

        def listener(frame: dict[str, JsonValue]) -> None:
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

    def _admitted(self, turn: _AdmittedTurn) -> None:
        self._publish(turn.key.session_id, {
            "type": "admitted", "turn_id": turn.key.turn_id, "turn_no": turn.number,
            "generation": turn.generation, "revision": turn.revision, "prompt": turn.prompt, "attachments": list(turn.attachments),
        })
        self._notice_lifecycle(turn.key, "admitted", turn.revision)

    def _terminal(
        self, turn: _AdmittedTurn, state: str, *, output: JsonValue = None,
        messages: tuple[ModelMessage, ...] = (), error_code: str | None = None,
        activity: dict[str, JsonValue] | None = None,
    ) -> None:
        succeeded = state == "succeeded"
        self._store.note_terminal_context(
            turn.key, turn_no=turn.number, prompt=turn.prompt if succeeded else None,
            stored_input=turn.stored_input if succeeded else None, messages=messages, succeeded=succeeded,
        )
        self._publish(turn.key.session_id, {
            "type": "terminal", "turn_id": turn.key.turn_id, "turn_no": turn.number,
            "generation": turn.generation, "revision": turn.revision + 1,
            "state": state, "prompt": turn.prompt, "attachments": list(turn.attachments), "output": output,
            "error_code": error_code, **(activity or {}),
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
        metadata: Mapping[str, JsonValue], request_key: str | None = None, cwd: str | None = None,
    ) -> dict[str, JsonValue]:
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if any(key.startswith("linktools.ai.") for key in metadata):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        await authorization.authorize(principal, AuthorizationAction.SESSION_CREATE, ResourceRef(
            ResourceKind.SESSION, session_id, tenant_id,
        ))
        cwd = await self.canonicalize_cwd(cwd)
        config: JsonValue = {"metadata": dict(metadata), "create_request_key": request_key, "cwd": cwd}
        try:
            await self._store.create(session_id, config=config)
        except IntegrityError:
            await authorization.authorize(
                principal, AuthorizationAction.SESSION_READ, ResourceRef(ResourceKind.SESSION, session_id, tenant_id),
            )
            existing = await self._store.get_session(session_id)
            if (
                request_key is None or existing is None
                or existing["config_json"] != config
            ):
                raise AIError(ErrorCode.SESSION_CONFLICT) from None
            return {
                "session_id": session_id, "revision": existing["revision"],
                "metadata": dict(metadata), "state": existing["status"], "cwd": cwd,
                "active_turn_id": None if existing["active_turn_id"] is None else existing["active_turn_id"].decode("utf-8"),
            }
        return {
            "session_id": session_id, "revision": 0, "metadata": dict(metadata),
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

    async def start_session(
        self, session_id: str, *, agent_id: str, principal: Principal, namespace: str, tenant_id: str,
        app: object, authorization: AuthorizationPolicy, resolver: _AgentBindingResolver,
        request_key: str, prompt: str, limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
        files: Sequence[str] = (),
    ) -> str:
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
            prior_input = self._stored_input(request)
            prior_files = () if prior_input is None else tuple(
                item["path"] for item in stored_input_attachment_views(prior_input)
            )
            if (
                request["prompt"] != prompt or request["principal"] != asdict(principal)
                or request["thinking"] != thinking or prior_files != normalized_files
                or prior["agent_id"] != agent_id
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return identity

        if session["active_turn_id"] is not None:
            if (identity := await replay()) is not None:
                return identity
            raise AIError(ErrorCode.SESSION_BUSY)

        key = SessionTurnKey(session_id, secrets.token_hex(16))
        await authorization.authorize(
            principal, AuthorizationAction.EXECUTION_RUN, ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id),
        )
        files = await self._input_materializer.canonicalize_files(files) if files else ()
        binding = await resolver.resolve_agent(agent_id)
        config = session["config_json"]
        cwd = config.get("cwd")
        context = AgentContext(
            app, principal, namespace, session_id, key.turn_id, config.get("metadata", {}),
            agent_id=agent_id,
        )
        try:
            if await self.start(
                key, revision=session["revision"], through=session["last_turn_no"],
                request_key=request_key, prompt=prompt, binding=binding, context=context,
                limits=limits, event_sink=event_sink, thinking=thinking, cwd=cwd, files=files, lineage=lineage,
            ):
                return key.turn_id
        except IntegrityError:
            if (identity := await replay()) is not None:
                return identity
            raise
        if (identity := await replay()) is not None:
            return identity
        raise AIError(ErrorCode.SESSION_CONFLICT)

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
        turn = await self._store.get_turn(key)
        if turn is None:
            return False
        await authorization.authorize(
            principal, AuthorizationAction.EXECUTION_CANCEL,
            ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id, turn["request_json"]["principal"]["principal_id"]),
        )
        if not await self._store.request_cancel(key):
            return False
        producer = self._producers.get(key)
        if producer is not None:
            producer.cancel()
        else:
            # A transport-routed cancellation may have no producer on this
            # process to publish its eventual terminal frame. Tell current
            # viewers to reconcile this exact committed turn.
            self._gap(key, cancelling=True)
        self._notice_lifecycle(key, "cancelling", session["revision"])
        return True

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
                if row["state"] not in ("accepted", "cancelling"):
                    return False
            elif row["state"] != "ready" or row["generation"] != ready.generation:
                return False
            request = row["request_json"]
            admitted = Principal(**request["principal"])
            await authorization.authorize(
                principal, AuthorizationAction.EXECUTION_RECOVER,
                ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id, admitted.principal_id),
            )
            generation = row["generation"]
            if ready is not None and admitted != principal:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            if row["state"] == "cancelling":
                if not await self._store.take_over(
                    key, generation=generation, lease_seconds=_LEASE_SECONDS, cancelling=True,
                ):
                    return False
                code = ErrorCode.EXECUTION_CANCELLED.value
                finished = await self._store.finish(
                    key, generation=generation + 1, turn_no=row["turn_no"], state="cancelled", result={"error_code": code},
                )
                if finished:
                    stored = self._stored_input(request)
                    turn = _AdmittedTurn(
                        key, row["turn_no"], request["prompt"], [], 0, session["revision"], None, generation + 1,
                        attachments=() if stored is None else stored_input_attachment_views(stored),
                    )
                    self._terminal(turn, "cancelled", error_code=code)
                return finished

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
            if not 0 <= through - after <= 128:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "The admitted context range is invalid")
            stored = self._stored_input(request)
            remaining = _CONTEXT_BYTES - self._input_bytes(prompt, stored)
            if remaining < 0:
                raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
            history: list[ModelMessage] = []
            positions = await self._store.read_context_range(
                key.session_id, through=through, after=after, lineage=lineage,
                missing=ErrorCode.STORAGE_RECOVERY_REQUIRED,
            )
            for position in positions:
                prior = position.request
                prior_input = self._stored_input(prior)
                remaining -= self._input_bytes(prior["prompt"], prior_input) + position.transcript_bytes
                if remaining < 0:
                    raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
                restored = await self._restore_prompt(prior["prompt"], prior_input)
                history.extend(restore_turn_prompt(restored, position.messages))
            frozen_binding = row["binding_json"]
            if frozen_binding is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED, "The admitted Agent binding is unavailable")
            contract = AgentBindingContract.from_payload(frozen_binding)
            binding = await resolver.restore(contract)
            native_step = None if self._prepare_step is None else await self._prepare_step(binding, key, generation + 1)
            self._executor.require_regenerable(binding, canonical_step=native_step is not None)
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
            context = AgentContext(
                app, admitted, namespace, key.session_id, key.turn_id, request["session_metadata"],
                agent_id=binding.compiled_agent.spec.id,
            )
            model_prompt = await self._restore_prompt(prompt, stored)
            continuation = deferred_results = None
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
                if checkpoint.state != "complete" and not pending:
                    self._gap(key)
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                if ready is None:
                    deferred_results = await tools.resume_canonical_effects(pending)
                else:
                    if (checkpoint.state != "interrupted"
                            or set(pending) != set(settled_results.approvals) | set(settled_results.calls)
                            or set(settled_results.approvals) & set(settled_results.calls)):
                        self._gap(key)
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    for operation in (*settled_continuation.approvals, *settled_continuation.calls):
                        if pending.get(operation.tool_call_id) != operation.tool_name:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    deferred_results = settled_results
                if not pending and checkpoint.state == "complete":
                    events = await steps.list_events(run_id=key.turn_id)
                    completed_checkpoint = any(event.kind == "run_completed" for event in events)
            turn = _AdmittedTurn(
                key, row["turn_no"], prompt, history,
                _TURN_BYTES - len(canonical_json_bytes({**request, "binding": frozen_binding})),
                session["revision"], repository_instructions, generation + 1,
                model_prompt=None if isinstance(model_prompt, str) else model_prompt,
                attachments=() if stored is None else stored_input_attachment_views(stored),
                stored_input=request.get("input"), native_step=native_step,
                continuation=continuation, deferred_results=deferred_results,
                deferred_approvals=() if settled_continuation is None else settled_continuation.approvals,
                instruction_frontier=instruction_frontier,
                completed_checkpoint=completed_checkpoint, output_retries_used=output_retries_used,
            )
            self._admitted(turn)
            try:
                await self._settle_with_lease(
                    turn, binding=binding, context=context, limits=limits, event_sink=event_sink, thinking=request["thinking"],
                )
            except Exception:
                # Recovery already changed the producer generation; the
                # browser must not keep waiting on that generation silently.
                self._gap(key)
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
        prompt: str, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue, cwd: str | None = None,
        files: Sequence[str] = (), lineage: tuple[SessionTurnSegment, ...] = (),
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
                    self._gap(key)
                    _logger.error("Regenerable turn producer stopped: turn=%s error=%s", key.turn_id, type(error).__name__)

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
        for listeners in self._observers.values():
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

    async def _settle_with_lease(
        self, turn: _AdmittedTurn, *, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
    ) -> None:
        process: dict[str, ToolProcessItem] = {}

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
            if isinstance(emission, LiveDelta) and emission.event_type is ExecutionDeltaType.ASSISTANT_TEXT_DELTA:
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
                self._publish(turn.key.session_id, {
                    **fields, **scope, "type": "delta", "delta_kind": "text", "text": emission.content,
                })
            elif isinstance(emission, DurableBoundary):
                item = project_tool_process_event(emission.event_type.value, emission.payload)
                if item is not None:
                    # Model-facing tool feedback can contain private result data.
                    # Keep the typed public status/reason and never forward that text.
                    error_code = item.error_code
                    if item.status == "FAILED" and error_code is None:
                        error_code = ErrorCode.TOOL_EXECUTION_FAILED
                    item = replace(item, error_code=error_code, error_message=None)
                    process[item.key] = item
                    self._publish(turn.key.session_id, {
                        **fields, "type": "tool", "item": asdict(item),
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
            turn, binding=binding, context=context, limits=limits, event_sink=emit, thinking=thinking, process=process,
        ))
        self._producers[turn.key] = producer
        try:
            while not producer.done():
                done, _ = await asyncio.wait((producer,), timeout=_LEASE_SECONDS / 3)
                if done:
                    break
                if not await self._store.renew(turn.key, generation=turn.generation, lease_seconds=_LEASE_SECONDS):
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
                code = ErrorCode.EXECUTION_CANCELLED.value
                if await self._store.finish(
                    turn.key, generation=turn.generation, turn_no=turn.number, state="cancelled",
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

    async def admit(
        self, key: SessionTurnKey, *, revision: int, through: int, request_key: str,
        prompt: str, binding: AgentBinding, context: AgentContext[object], thinking: ThinkingValue,
        cwd: str | None = None, files: Sequence[str] = (), lineage: tuple[SessionTurnSegment, ...] = (),
    ) -> _AdmittedTurn | None:
        """Called by the session owner with authorized identity and execution binding."""
        native_step = None if self._prepare_step is None else await self._prepare_step(binding, key, 1)
        self._executor.require_regenerable(binding, canonical_step=native_step is not None)
        if await self.canonicalize_cwd(cwd) != cwd:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "cwd"})
        instruction_resolver = self._resolver_for(binding)
        repository_instructions = None
        if instruction_resolver is not None:
            repository_instructions = await instruction_resolver.resolve(cwd or ".")
            instruction_resolver.validate(repository_instructions)
        frozen_binding = binding.binding_contract.to_payload()
        request: dict[str, JsonValue] = {
            "prompt": prompt,
            "principal": asdict(context.principal),
            "thinking": thinking,
            "session_metadata": dict(context.session_metadata),
            "cwd": cwd,
            "repository_instructions": None if repository_instructions is None else repository_instructions.to_payload(),
        }
        model_prompt = None
        attachments = ()
        input_size = len(prompt.encode("utf-8"))
        if files:
            materialized = await self._input_materializer.materialize(prompt, files)
            validate_model_inputs(binding.compiled_agent.model, materialized)
            # The materializer prepends attachment parts and retains the text as the last item.
            # Store the text once, separately from replayable file parts.
            model_prompt = tuple(materialized)
            detached = MaterializedUserContent(model_prompt[:-1], {"attachments": materialized.view["attachments"]})
            captured = self._input_materializer.capture(detached)
            input_size += captured.payload.size
            if input_size > _CONTEXT_BYTES:
                raise AIError(ErrorCode.PROMPT_TOO_LARGE)
            stored = await self._input_materializer.materialize_payload(captured, tenant_id=context.principal.tenant_id)
            if not stored.payload.object_refs():
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "Attachment object storage is required")
            request["input"] = encode_domain(stored)
            attachments = stored_input_attachment_views(stored)
        terminal_bytes = _TURN_BYTES - len(canonical_json_bytes({**request, "binding": frozen_binding}))
        if terminal_bytes < _TERMINAL_RESERVE:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)

        remaining = _CONTEXT_BYTES - input_size
        history: list[tuple[ModelMessage, ...]] = []
        context_from = through + 1
        candidates = await self._store.recent_context(key.session_id, through=through, lineage=lineage)
        for candidate in reversed(candidates):
            if not isinstance(candidate.request, dict) or not isinstance(candidate.request.get("prompt"), str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Missing admitted session input")
            candidate_prompt = candidate.request["prompt"]
            stored = self._stored_input(candidate.request)
            size = self._input_bytes(candidate_prompt, stored) + candidate.transcript_bytes
            if size > remaining:
                break
            restored = await self._restore_prompt(candidate_prompt, stored)
            history.append(restore_turn_prompt(restored, candidate.messages))
            remaining -= size
            context_from = candidate.turn_no

        if not await self._store.accept(
            key, expected_revision=revision, context_from=context_from, context_through=through,
            request_key=request_key, request=request, binding=frozen_binding, lease_seconds=_LEASE_SECONDS,
        ):
            return None
        return _AdmittedTurn(
            key, through + 1, prompt,
            [message for messages in reversed(history) for message in messages], terminal_bytes, revision + 1,
            repository_instructions, model_prompt=model_prompt, attachments=attachments,
            stored_input=request.get("input"),
            native_step=native_step,
        )

    @staticmethod
    def _stored_input(request: Mapping[str, JsonValue]) -> StoredUserInput | None:
        return decode_domain(request["input"], StoredUserInput) if "input" in request else None

    @staticmethod
    def _input_bytes(prompt: str, stored: StoredUserInput | None) -> int:
        return len(prompt.encode("utf-8")) + (0 if stored is None else stored.payload.size)

    async def _restore_prompt(
        self, prompt: str, stored: StoredUserInput | None,
    ) -> str | Sequence[UserContent]:
        if stored is None:
            return prompt
        detached = await self._input_materializer.restore(stored)
        if isinstance(detached, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Detached turn input is not multipart")
        return (*detached, prompt)

    def _step_instruction_boundary(
        self, turn: _AdmittedTurn, binding: AgentBinding,
    ) -> _RepositoryInstructionBoundary | None:
        if turn.native_step is None:
            return None
        resolver = self._resolver_for(binding)
        initial = turn.repository_instructions
        frontier = turn.instruction_frontier
        overlay = None if frontier is None else frontier.overlay
        barriers = {} if frontier is None else dict(frontier.barriers)
        if overlay is not None:
            if resolver is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            resolver.validate(_merge_repository_instructions(initial, overlay))

        approved = {}
        for item in turn.deferred_approvals:
            provenance = item.metadata.get("linktools")
            if not isinstance(provenance, Mapping):
                continue
            digest = provenance.get("repository_instruction_frontier")
            if digest is not None:
                if resolver is None or not isinstance(digest, str):
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                approved[item.tool_call_id] = _RepositoryInstructionApproval(item.tool_name, digest)

        if resolver is None:
            return None

        async def check(*, initial, overlay, tool_name, tool_call_id, arguments, path_fields):
            paths = _instruction_paths(arguments, path_fields)
            if not paths:
                return overlay, False
            instruction_read = is_instruction_file_read(tool_name, arguments, path_fields)
            if tool_call_id in barriers:
                if barriers[tool_call_id] != canonical_sha256(normalize_json_value(arguments)):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return overlay, not instruction_read
            discovered = await _discover_repository_overlay(resolver, initial, overlay, paths)
            if discovered is None:
                return overlay, False
            next_barriers = {**barriers, tool_call_id: canonical_sha256(normalize_json_value(arguments))}
            await self._store.append_instruction_frontier(
                turn.key, generation=turn.generation, max_bytes=turn.terminal_bytes,
                frontier=SessionInstructionFrontier(discovered, next_barriers),
            )
            # Expose new instructions only after the fenced transaction commits.
            barriers.update(next_barriers)
            return discovered, not instruction_read

        return _RepositoryInstructionBoundary(check, initial, overlay, approved_frontiers=approved)

    async def settle(
        self, turn: _AdmittedTurn, *, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
        process: Mapping[str, ToolProcessItem] | None = None,
    ) -> bool:
        """Execute after durable admission and publish only if this producer still owns the turn."""
        from ..model.messages import encode_model_messages

        model_prompt = turn.prompt if turn.model_prompt is None else turn.model_prompt
        try:
            if turn.completed_checkpoint:
                if turn.continuation is None:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                messages, output = turn.continuation, reply_from_turn(turn.continuation)
            else:
                repository_boundary = self._step_instruction_boundary(turn, binding)
                outcome = await self._executor.execute_regenerable(
                    binding=binding, context=context, prompt=model_prompt, history=turn.history, limits=limits,
                    event_sink=event_sink, thinking=thinking, streamed_output_bytes=turn.terminal_bytes,
                    workspace=self._workspace, repository_instructions=turn.repository_instructions,
                    repository_instruction_boundary=repository_boundary,
                    canonical_step=None if turn.native_step is None else turn.native_step[0],
                    tool_operations=None if turn.native_step is None else turn.native_step[1],
                    generation=turn.generation, continuation=turn.continuation,
                    deferred_tool_results=turn.deferred_results, deferred_approvals=turn.deferred_approvals,
                    output_retries_used=turn.output_retries_used,
                )
                if isinstance(outcome, DeferredToolRequests):
                    if turn.native_step is None or self._publish_deferred is None:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    capability, tools = turn.native_step
                    if not isinstance(capability.store, RuntimeStepStore):
                        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                    checkpoint_id = capability.store.require_interrupted_checkpoint()
                    if repository_boundary is not None:
                        metadata = {key: dict(value) for key, value in outcome.metadata.items()}
                        for call_id, digest in repository_boundary.approval_frontiers(outcome.approvals).items():
                            item = metadata.setdefault(call_id, {})
                            original = item.get("linktools")
                            evidence = {"repository_instruction_frontier": digest}
                            if isinstance(original, Mapping) and original.get("kind") == "workspace_approval":
                                # Leaf receipt owns authorized arguments. Do not store a second copy.
                                evidence["kind"] = "workspace_approval"
                            elif original is not None:
                                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
                            item["linktools"] = evidence
                        outcome = replace(outcome, metadata=metadata)
                    committed = await self._publish_deferred(turn, checkpoint_id, tools, outcome)
                    if committed:
                        self._publish(turn.key.session_id, {
                            "type": "waiting", "turn_id": turn.key.turn_id, "turn_no": turn.number,
                            "generation": turn.generation, "revision": turn.revision,
                        })
                        self._notice_lifecycle(turn.key, "waiting", turn.revision)
                    return committed
                messages, output = outcome.messages, outcome.output
            frozen_spec = binding.binding_contract.agent_spec
            context_messages, transcript, binding_refs = prepare_native_turn_transcript(
                model_prompt, turn.history, messages, output,
                instructions=frozen_spec.instructions, system_prompt=frozen_spec.system_prompt,
            )
            # Charge actual encoded message bytes and binding ref positions.
            encoded_body = len(encode_model_messages(transcript))
            encoded_refs = len(canonical_json_bytes([list(item) for item in binding_refs])) if binding_refs else 0
            if encoded_body + encoded_refs > turn.terminal_bytes:
                raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
        except Exception as error:
            if (turn.native_step is not None and isinstance(error, AIError)
                    and error.code in {ErrorCode.STORAGE_RECOVERY_REQUIRED, ErrorCode.STORAGE_COMMIT_UNKNOWN,
                                       ErrorCode.TOOL_EFFECT_UNKNOWN}):
                raise
            code = error.code.value if isinstance(error, AIError) else ErrorCode.INTERNAL_ERROR.value
            committed = await self._store.finish(
                turn.key, generation=turn.generation, turn_no=turn.number, state="failed", result={"error_code": code},
            )
            if committed:
                self._terminal(turn, "failed", error_code=code)
            return committed
        committed = await self._store.finish(
            turn.key, generation=turn.generation, turn_no=turn.number, state="succeeded",
            transcript=transcript, binding_refs=binding_refs,
        )
        if committed:
            activity = None
            if process:
                items: list[JsonValue] = []
                for message in messages[len(turn.history):]:
                    if not isinstance(message, ModelResponse):
                        continue
                    for part in message.parts:
                        if isinstance(part, TextPart):
                            items.append({"item_kind": "assistant", "content": part.content})
                        elif isinstance(part, ToolCallPart) and f"tool:{part.tool_call_id}" in process:
                            items.append({"item_kind": "tool_call", "tool_call_id": part.tool_call_id})
                activity = {"items": items, "process": [asdict(item) for item in process.values()]}
            self._terminal(
                turn, "succeeded", output=output, messages=context_messages, activity=activity,
            )
        return committed
