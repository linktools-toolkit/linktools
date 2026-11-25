#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission and settlement for ordinary, regenerable model turns."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

from linktools.core import environ
from typing import TYPE_CHECKING

from sqlalchemy.exc import IntegrityError

from ..agent import AgentBindingContract
from ..capability import AgentContext
from ..core import AuthorizationAction, ExecutionDeltaType, JsonValue, Principal, ResourceKind, ResourceRef, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..inputs import stored_input_attachment_views
from ..model import MaterializedUserContent, StoredUserInput, normalize_input_files, validate_model_inputs
from ..persistence.codec import decode_domain, encode_domain
from ..persistence.turns import SessionTurnKey
from ..spec import RepositoryInstructions
from ._agent_executor import LiveDelta
from ._regenerable_transcript import decode_regenerable_turn, encode_regenerable_turn

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage, UserContent

    from ..agent import AgentBinding
    from ..core import AuthorizationPolicy, PromptLimits, ThinkingValue
    from ..inputs import ExecutionInputMaterializer
    from ..spec import RepositoryInstructionResolver
    from ..workspace import Workspace
    from ..persistence.turns import SqlSessionTurns
    from ._agent_binding_resolver import _AgentBindingResolver
    from ._agent_executor import AgentEmission, AgentExecutor, EventSink

_TURN_BYTES = 64 * 1024
_CONTEXT_BYTES = 1024 * 1024
_TERMINAL_RESERVE = 1024
_LEASE_SECONDS = 60
_MAX_CONCURRENT_TURNS = 4
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


class _RegenerableTurnRunner:
    def __init__(
        self, store: SqlSessionTurns, executor: AgentExecutor, *,
        workspace: Workspace | None,
        input_materializer: ExecutionInputMaterializer,
        instruction_resolver: RepositoryInstructionResolver | None,
        instruction_resolver_for: Callable[[RepositoryInstructions], RepositoryInstructionResolver],
    ) -> None:
        self._store = store
        self._executor = executor
        self._workspace = workspace
        self._input_materializer = input_materializer
        self._workspace_access = input_materializer.access
        self._instruction_resolver = instruction_resolver
        self._instruction_resolver_for = instruction_resolver_for
        self._tasks: set[asyncio.Task[None] | asyncio.Task[bool]] = set()
        self._producers: dict[SessionTurnKey, asyncio.Task[bool]] = {}
        self._closed = False
        self._observers: dict[str, set[Callable[[dict[str, JsonValue]], None]]] = {}
        self._lifecycle_observers: set[Callable[[dict[str, JsonValue]], None]] = set()
        self._observed_revisions: dict[str, int] = {}

    def observe_lifecycle(self, callback: Callable[[dict[str, JsonValue]], None]) -> Callable[[], None]:
        """Offer only committed turn identifiers and revisions to a host carrier."""
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)

        def listener(notice: dict[str, JsonValue]) -> None:
            callback(notice)

        self._lifecycle_observers.add(listener)
        return lambda: self._lifecycle_observers.discard(listener)

    def invalidate_session(self, session_id: str, turn_id: str, *, revision: int) -> None:
        if session_id not in self._observers or revision <= self._observed_revisions.get(session_id, -1):
            return
        self._observed_revisions[session_id] = revision
        self._gap(SessionTurnKey(session_id, turn_id))

    def _notice_lifecycle(self, key: SessionTurnKey, phase: str, revision: int) -> None:
        if key.session_id in self._observers:
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
        listeners = self._observers.setdefault(session_id, set())
        listeners.add(callback)

        def close() -> None:
            listeners.discard(callback)
            if not listeners and self._observers.get(session_id) is listeners:
                self._observers.pop(session_id, None)
                self._observed_revisions.pop(session_id, None)

        return close

    def _publish(self, session_id: str, frame: dict[str, JsonValue]) -> None:
        for callback in tuple(self._observers.get(session_id, ())):
            try:
                callback(frame)
            except Exception as error:
                _logger.error("Session turn observer failed: %s", type(error).__name__)

    def _gap(self, key: SessionTurnKey) -> None:
        self._publish(key.session_id, {"type": "gap", "turn_id": key.turn_id})

    def _admitted(self, turn: _AdmittedTurn) -> None:
        self._publish(turn.key.session_id, {
            "type": "admitted", "turn_id": turn.key.turn_id, "turn_no": turn.number,
            "generation": turn.generation, "revision": turn.revision, "prompt": turn.prompt, "attachments": list(turn.attachments),
        })
        self._notice_lifecycle(turn.key, "admitted", turn.revision)

    def _terminal(self, turn: _AdmittedTurn, state: str, *, output: JsonValue = None, error_code: str | None = None) -> None:
        self._publish(turn.key.session_id, {
            "type": "terminal", "turn_id": turn.key.turn_id, "turn_no": turn.number,
            "generation": turn.generation, "revision": turn.revision + 1,
            "state": state, "prompt": turn.prompt, "attachments": list(turn.attachments), "output": output, "error_code": error_code,
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
                or AgentBindingContract.from_payload(request["binding"]).agent_spec.id != agent_id
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
                limits=limits, event_sink=event_sink, thinking=thinking, cwd=cwd, files=files,
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
        return True

    async def recover_session_turn(
        self, key: SessionTurnKey, *, principal: Principal, namespace: str, tenant_id: str,
        app: object, authorization: AuthorizationPolicy, resolver: _AgentBindingResolver,
        limits: PromptLimits, event_sink: EventSink,
    ) -> bool:
        """Explicit recovery uses the admitted request and does not run on reads."""
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if len(self._tasks) >= _MAX_CONCURRENT_TURNS:
            raise AIError(ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED)

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
            row = await self._store.get_turn(key)
            if row is None or row["state"] not in ("accepted", "cancelling"):
                return False
            request = row["request_json"]
            admitted = Principal(**request["principal"])
            await authorization.authorize(
                principal, AuthorizationAction.EXECUTION_RECOVER,
                ResourceRef(ResourceKind.EXECUTION, key.turn_id, tenant_id, admitted.principal_id),
            )
            generation = row["generation"]
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
            if through > after:
                positions = await self._store.history(key.session_id, through=through, limit=128, after=after)
                if len(positions) != through - after:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED, "The admitted context range is no longer available")
                for position in positions:
                    if position["state"] == "succeeded":
                        prior = position["request_json"]
                        prior_input = self._stored_input(prior)
                        remaining -= self._input_bytes(prior["prompt"], prior_input) + len(canonical_json_bytes(position["result_json"]))
                        if remaining < 0:
                            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
                        restored = await self._restore_prompt(prior["prompt"], prior_input)
                        history.extend(decode_regenerable_turn(restored, position["result_json"]))
            contract = AgentBindingContract.from_payload(request["binding"])
            binding = await resolver.restore(contract)
            self._executor.require_regenerable(binding)
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
            if not await self._store.take_over(key, generation=generation, lease_seconds=_LEASE_SECONDS):
                return False
            turn = _AdmittedTurn(
                key, row["turn_no"], prompt, history,
                _TURN_BYTES - len(canonical_json_bytes(request)), session["revision"], repository_instructions, generation + 1,
                model_prompt=None if isinstance(model_prompt, str) else model_prompt,
                attachments=() if stored is None else stored_input_attachment_views(stored),
            )
            self._admitted(turn)
            await self._settle_with_lease(
                turn, binding=binding, context=context, limits=limits, event_sink=event_sink, thinking=request["thinking"],
            )
            return True

        task = asyncio.create_task(recover(), name="linktools-recover-regenerable-turn")
        self._tasks.add(task)

        def consume(completed: asyncio.Task[bool]) -> None:
            self._tasks.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(consume)
        return await asyncio.shield(task)

    async def start(
        self, key: SessionTurnKey, *, revision: int, through: int, request_key: str,
        prompt: str, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue, cwd: str | None = None,
        files: Sequence[str] = (),
    ) -> bool:
        """Return only after admission commits; a disconnected observer cannot cancel the producer."""
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if len(self._tasks) >= _MAX_CONCURRENT_TURNS:
            raise AIError(ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED)

        accepted: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        async def produce() -> None:
            try:
                turn = await self.admit(
                    key, revision=revision, through=through, request_key=request_key,
                    prompt=prompt, binding=binding, context=context, thinking=thinking, cwd=cwd, files=files,
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

        task = asyncio.create_task(produce(), name="linktools-regenerable-turn")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
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
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._observers.clear()
        self._lifecycle_observers.clear()
        self._observed_revisions.clear()

    async def _settle_with_lease(
        self, turn: _AdmittedTurn, *, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
    ) -> None:
        async def emit(emission: AgentEmission) -> None:
            if isinstance(emission, LiveDelta) and emission.event_type is ExecutionDeltaType.ASSISTANT_TEXT_DELTA:
                self._publish(turn.key.session_id, {
                    "type": "delta", "turn_id": turn.key.turn_id, "turn_no": turn.number,
                    "generation": turn.generation, "revision": turn.revision,
                    "delta_kind": "text", "text": emission.content,
                })
            await event_sink(emission)

        producer = asyncio.create_task(self.settle(
            turn, binding=binding, context=context, limits=limits, event_sink=emit, thinking=thinking,
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
        cwd: str | None = None, files: Sequence[str] = (),
    ) -> _AdmittedTurn | None:
        """Called by the session owner with authorized identity and execution binding."""
        self._executor.require_regenerable(binding)
        if await self.canonicalize_cwd(cwd) != cwd:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "cwd"})
        instruction_resolver = self._resolver_for(binding)
        repository_instructions = None
        if instruction_resolver is not None:
            repository_instructions = await instruction_resolver.resolve(cwd or ".")
            instruction_resolver.validate(repository_instructions)
        request: JsonValue = {
            "prompt": prompt,
            "principal": asdict(context.principal),
            "binding": binding.binding_contract.to_payload(),
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
        terminal_bytes = _TURN_BYTES - len(canonical_json_bytes(request))
        if terminal_bytes < _TERMINAL_RESERVE:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)

        remaining = _CONTEXT_BYTES - input_size
        history: list[tuple[ModelMessage, ...]] = []
        context_from = through + 1
        candidates = await self._store.recent_context(key.session_id, through=through)
        for candidate in reversed(candidates):
            if not isinstance(candidate.request, dict) or not isinstance(candidate.request.get("prompt"), str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Missing admitted session input")
            candidate_prompt = candidate.request["prompt"]
            stored = self._stored_input(candidate.request)
            size = self._input_bytes(candidate_prompt, stored) + len(canonical_json_bytes(candidate.result))
            if size > remaining:
                break
            restored = await self._restore_prompt(candidate_prompt, stored)
            history.append(decode_regenerable_turn(restored, candidate.result))
            remaining -= size
            context_from = candidate.turn_no

        if not await self._store.accept(
            key, expected_revision=revision, context_from=context_from, context_through=through,
            request_key=request_key, request=request, lease_seconds=_LEASE_SECONDS,
        ):
            return None
        return _AdmittedTurn(
            key, through + 1, prompt,
            [message for messages in reversed(history) for message in messages], terminal_bytes, revision + 1,
            repository_instructions, model_prompt=model_prompt, attachments=attachments,
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

    async def settle(
        self, turn: _AdmittedTurn, *, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
    ) -> bool:
        """Execute after durable admission and publish only if this producer still owns the turn."""
        result: JsonValue
        model_prompt = turn.prompt if turn.model_prompt is None else turn.model_prompt
        try:
            outcome = await self._executor.execute_regenerable(
                binding=binding, context=context, prompt=model_prompt, history=turn.history, limits=limits,
                event_sink=event_sink, thinking=thinking, streamed_output_bytes=turn.terminal_bytes,
                workspace=self._workspace, repository_instructions=turn.repository_instructions,
            )
            result = encode_regenerable_turn(model_prompt, turn.history, outcome.messages, outcome.output)
            if len(canonical_json_bytes(result)) > turn.terminal_bytes:
                raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
        except Exception as error:
            code = error.code.value if isinstance(error, AIError) else ErrorCode.INTERNAL_ERROR.value
            committed = await self._store.finish(
                turn.key, generation=turn.generation, turn_no=turn.number, state="failed", result={"error_code": code},
            )
            if committed:
                self._terminal(turn, "failed", error_code=code)
            return committed
        committed = await self._store.finish(
            turn.key, generation=turn.generation, turn_no=turn.number, state="succeeded", result=result,
        )
        if committed:
            self._terminal(turn, "succeeded", output=outcome.output)
        return committed
