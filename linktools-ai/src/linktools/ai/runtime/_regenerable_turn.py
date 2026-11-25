#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission and settlement for ordinary, regenerable model turns."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from linktools.core import environ
from typing import TYPE_CHECKING

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..persistence.turns import SessionTurnKey
from ._regenerable_transcript import decode_regenerable_turn, encode_regenerable_turn

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from ..agent import AgentBinding
    from ..capability import AgentContext
    from ..core import PromptLimits, ThinkingValue
    from ..persistence.turns import SqlSessionTurns
    from ._agent_executor import AgentExecutor, EventSink

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


class _RegenerableTurnRunner:
    def __init__(self, store: SqlSessionTurns, executor: AgentExecutor) -> None:
        self._store = store
        self._executor = executor
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

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

    async def _settle_with_lease(
        self, turn: _AdmittedTurn, *, binding: AgentBinding, context: AgentContext[object],
        limits: PromptLimits, event_sink: EventSink, thinking: ThinkingValue,
    ) -> None:
        producer = asyncio.create_task(self.settle(
            turn, binding=binding, context=context, limits=limits, event_sink=event_sink, thinking=thinking,
        ))
        try:
            while not producer.done():
                done, _ = await asyncio.wait((producer,), timeout=_LEASE_SECONDS / 3)
                if done:
                    break
                if not await self._store.renew(turn.key, generation=1, lease_seconds=_LEASE_SECONDS):
                    return
            await producer
        finally:
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
            [message for messages in reversed(history) for message in messages], terminal_bytes,
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
        except AIError as error:
            return await self._store.finish(
                turn.key, generation=1, turn_no=turn.number, state="failed", result={"error_code": error.code.value},
            )
        return await self._store.finish(turn.key, generation=1, turn_no=turn.number, state="succeeded", result=result)
