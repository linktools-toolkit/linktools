#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission and settlement for ordinary, regenerable model turns."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

from linktools.core import environ
from typing import TYPE_CHECKING

from sqlalchemy.exc import IntegrityError

from ..agent import AgentBindingContract
from ..capability import AgentContext
from ..core import AuthorizationAction, ExecutionDeltaType, JsonValue, Principal, ResourceKind, ResourceRef, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..persistence.turns import SessionTurnKey
from ._agent_executor import LiveDelta
from ._regenerable_transcript import decode_regenerable_turn, encode_regenerable_turn

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from ..agent import AgentBinding
    from ..core import AuthorizationPolicy, PromptLimits, ThinkingValue
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
    generation: int = 1


class _RegenerableTurnRunner:
    def __init__(self, store: SqlSessionTurns, executor: AgentExecutor) -> None:
        self._store = store
        self._executor = executor
        self._tasks: set[asyncio.Task[None] | asyncio.Task[bool]] = set()
        self._producers: dict[SessionTurnKey, asyncio.Task[bool]] = {}
        self._closed = False
        self._observers: dict[str, set[Callable[[dict[str, JsonValue]], None]]] = {}

    def observe(self, session_id: str, callback: Callable[[dict[str, JsonValue]], None]) -> Callable[[], None]:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        listeners = self._observers.setdefault(session_id, set())
        listeners.add(callback)

        def close() -> None:
            listeners.discard(callback)
            if not listeners and self._observers.get(session_id) is listeners:
                self._observers.pop(session_id, None)

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
            "generation": turn.generation, "revision": turn.revision, "prompt": turn.prompt,
        })

    def _terminal(self, turn: _AdmittedTurn, state: str, *, output: JsonValue = None, error_code: str | None = None) -> None:
        self._publish(turn.key.session_id, {
            "type": "terminal", "turn_id": turn.key.turn_id, "turn_no": turn.number,
            "generation": turn.generation, "revision": turn.revision + 1,
            "state": state, "prompt": turn.prompt, "output": output, "error_code": error_code,
        })

    async def create_session(
        self, session_id: str, agent_id: str, *, principal: Principal, tenant_id: str,
        authorization: AuthorizationPolicy, resolver: _AgentBindingResolver,
        metadata: Mapping[str, JsonValue], request_key: str | None = None,
    ) -> dict[str, JsonValue]:
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if any(key.startswith("linktools.ai.") for key in metadata):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        await authorization.authorize(principal, AuthorizationAction.SESSION_CREATE, ResourceRef(
            ResourceKind.SESSION, session_id, tenant_id, principal.principal_id,
        ))
        self._executor.require_regenerable(await resolver.resolve_agent(agent_id))
        config: JsonValue = {"metadata": dict(metadata), "create_request_key": request_key}
        try:
            await self._store.create(session_id, owner_principal_id=principal.principal_id, agent_id=agent_id, config=config)
        except IntegrityError:
            existing = await self._store.get_session(session_id)
            if (
                request_key is None or existing is None
                or existing["owner_principal_id"] != principal.principal_id.encode("utf-8")
                or existing["agent_id"] != agent_id.encode("utf-8")
                or existing["config_json"] != config
            ):
                raise AIError(ErrorCode.SESSION_CONFLICT) from None
            return {
                "session_id": session_id, "agent_id": agent_id, "revision": existing["revision"],
                "metadata": dict(metadata), "state": existing["status"],
                "active_turn_id": None if existing["active_turn_id"] is None else existing["active_turn_id"].decode("utf-8"),
            }
        return {
            "session_id": session_id, "agent_id": agent_id, "revision": 0, "metadata": dict(metadata),
            "state": "open", "active_turn_id": None,
        }

    async def start_session(
        self, session_id: str, *, principal: Principal, namespace: str, tenant_id: str,
        app: object, authorization: AuthorizationPolicy, resolver: _AgentBindingResolver,
        request_key: str, prompt: str, limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
    ) -> str:
        """Start on the authoritative SQL session after session and execution authorization."""
        if principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        session = await self._store.get_session(session_id)
        if session is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, session_id, tenant_id, session["owner_principal_id"].decode("utf-8")),
        )
        if session["status"] != "open":
            raise AIError(ErrorCode.SESSION_CONFLICT)

        async def replay() -> str | None:
            prior = await self._store.find_request(session_id, request_key)
            if prior is None:
                return None
            identity = prior["turn_id"].decode("utf-8")
            await authorization.authorize(
                principal, AuthorizationAction.EXECUTION_RUN,
                ResourceRef(ResourceKind.EXECUTION, identity, tenant_id),
            )
            request = prior["request_json"]
            if (
                request["prompt"] != prompt or request["principal"] != asdict(principal)
                or request["thinking"] != thinking
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
        agent_id = session["agent_id"].decode("utf-8")
        binding = await resolver.resolve_agent(agent_id)
        config = session["config_json"]
        if config.get("cwd") is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "This turn path has no workspace binding")
        context = AgentContext(
            app, principal, namespace, session_id, key.turn_id, config.get("metadata", {}),
            agent_id=agent_id,
        )
        try:
            if await self.start(
                key, revision=session["revision"], through=session["last_turn_no"],
                request_key=request_key, prompt=prompt, binding=binding, context=context,
                limits=limits, event_sink=event_sink, thinking=thinking,
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
        owner = session["owner_principal_id"].decode("utf-8")
        await authorization.authorize(
            principal, AuthorizationAction.SESSION_READ, ResourceRef(ResourceKind.SESSION, key.session_id, tenant_id, owner),
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
                ResourceRef(ResourceKind.SESSION, key.session_id, tenant_id, session["owner_principal_id"].decode("utf-8")),
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
                    turn = _AdmittedTurn(key, row["turn_no"], request["prompt"], [], 0, session["revision"], generation + 1)
                    self._terminal(turn, "cancelled", error_code=code)
                return finished

            prompt = request["prompt"]
            after = row["context_from"] - 1
            through = row["context_through"]
            if not 0 <= through - after <= 128:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "The admitted context range is invalid")
            history: list[ModelMessage] = []
            if through > after:
                positions = await self._store.history(key.session_id, through=through, limit=128, after=after)
                if len(positions) != through - after:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED, "The admitted context range is no longer available")
                for position in positions:
                    if position["state"] == "succeeded":
                        history.extend(decode_regenerable_turn(
                            position["request_json"]["prompt"], position["result_json"],
                        ))
            contract = AgentBindingContract.from_payload(request["binding"])
            binding = await resolver.restore(contract)
            self._executor.require_regenerable(binding)
            context = AgentContext(
                app, admitted, namespace, key.session_id, key.turn_id, request["session_metadata"],
                agent_id=binding.compiled_agent.spec.id,
            )
            if not await self._store.take_over(key, generation=generation, lease_seconds=_LEASE_SECONDS):
                return False
            turn = _AdmittedTurn(
                key, row["turn_no"], prompt, history,
                _TURN_BYTES - len(canonical_json_bytes(request)), session["revision"], generation + 1,
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
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
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
                    prompt=prompt, binding=binding, context=context, thinking=thinking,
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
    ) -> _AdmittedTurn | None:
        """Called by the session owner with authorized identity and execution binding."""
        self._executor.require_regenerable(binding)
        request: JsonValue = {
            "prompt": prompt,
            "principal": asdict(context.principal),
            "binding": binding.binding_contract.to_payload(),
            "thinking": thinking,
            "session_metadata": dict(context.session_metadata),
        }
        terminal_bytes = _TURN_BYTES - len(canonical_json_bytes(request))
        if terminal_bytes < _TERMINAL_RESERVE:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)

        remaining = _CONTEXT_BYTES - len(prompt.encode("utf-8"))
        history: list[tuple[ModelMessage, ...]] = []
        context_from = through + 1
        candidates = await self._store.recent_context(key.session_id, through=through)
        for candidate in reversed(candidates):
            if not isinstance(candidate.request, dict) or not isinstance(candidate.request.get("prompt"), str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Missing admitted session input")
            candidate_prompt = candidate.request["prompt"]
            size = len(candidate_prompt.encode("utf-8")) + len(canonical_json_bytes(candidate.result))
            if size > remaining:
                break
            history.append(decode_regenerable_turn(candidate_prompt, candidate.result))
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
        )

    async def settle(
        self, turn: _AdmittedTurn, *, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
    ) -> bool:
        """Execute after durable admission and publish only if this producer still owns the turn."""
        result: JsonValue
        try:
            outcome = await self._executor.execute_regenerable(
                binding=binding, context=context, prompt=turn.prompt, history=turn.history, limits=limits,
                event_sink=event_sink, thinking=thinking, streamed_output_bytes=turn.terminal_bytes,
            )
            result = encode_regenerable_turn(turn.prompt, turn.history, outcome.messages, outcome.output)
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
