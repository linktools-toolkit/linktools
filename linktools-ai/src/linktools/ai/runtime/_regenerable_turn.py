#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission and settlement for ordinary, regenerable model turns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
