#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session admission and observation support for built-in clients."""

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pydantic import BaseModel

    from ..core import JsonValue, Principal, ThinkingValue
    from ..execution import Execution
    from ..model import UserPromptInput
    from ._runtime_service import Runtime


_PREVIEW_BYTES = 256 * 1024
_PREVIEW_TYPES = ("delta", "tool", "progress", "preview_reset", "thinking_completed", "output_preview")
_WAITING_STATES = {"WAITING_DEFERRED", "WAITING_RETRY"}
_FINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED", "RECOVERY_REQUIRED"}
_END_STATES = _FINAL_STATES | _WAITING_STATES


AppT = TypeVar("AppT")


async def start_bound_execution(
    runtime: "Runtime[AppT]", session_id: str, user_prompt: "UserPromptInput", *, principal: "Principal",
    idempotency_key: str | None = None, output: "type[BaseModel] | None" = None,
    correlation: "Mapping[str, object] | None" = None, files: "Sequence[str]" = (),
    planning: bool | None = None, thinking: "ThinkingValue | None" = None,
) -> "Execution[AppT]":
    """Admit a definition-bound client request without a separate display read."""
    from ..core import validate_idempotency_key, validate_resource_id
    from ..storage import sql_observation_scope
    from ..model import validate_user_input

    runtime._ensure_open()
    validate_resource_id(session_id)
    prompt = validate_user_input(user_prompt)
    key = secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key
    validate_idempotency_key(key)

    async def start(agent_id, resume):
        return await runtime.agents.get(agent_id).execution_port.start(
            prompt, files=files, output=output, principal=principal, session_id=session_id,
            idempotency_key=key, mode="run", planning=planning, thinking=thinking,
            correlation=correlation, _resume=resume)

    with sql_observation_scope("runtime.session.start"):
        return await runtime.sessions._agent_session_service().start_existing(session_id, principal, key, start)


def uses_turn_sessions(runtime: "Runtime") -> bool:
    """Select the configured topology, not the requesting principal's permission."""
    return getattr(getattr(runtime, "sessions", None), "_uses_turn_sessions", False) is True


class TurnObserver:
    """Subscribe before admission; forward only the admitted turn's frames."""

    def __init__(
        self, runtime: "Runtime", session_id: str, principal: "Principal", *, local_workspace: bool, preview: bool = True,
    ) -> None:
        self._runtime = runtime
        self._session_id = session_id
        self._principal = principal
        self._local_workspace = local_workspace
        self._preview = preview
        self._queue: asyncio.Queue[tuple[dict[str, "JsonValue"], int]] = asyncio.Queue(maxsize=256)
        self._bytes = 0
        self._turn_id: str | None = None
        self._terminal: dict[str, "JsonValue"] | None = None
        self._stopped: dict[str, "JsonValue"] | None = None
        self.incomplete = False

    def __enter__(self) -> "TurnObserver":
        self._unsubscribe = self._runtime.sessions.observe_turn_session(self._session_id, self._receive)
        return self

    def __exit__(self, *exc: object) -> None:
        self._unsubscribe()

    def _receive(self, frame: dict[str, "JsonValue"]) -> None:
        turn_id = frame.get("turn_id")
        if self._turn_id is not None and turn_id not in (None, self._turn_id):
            return
        kind = frame.get("type")
        if kind not in ("gap", "terminal", "waiting") and (
            not self._preview or kind not in _PREVIEW_TYPES
        ):
            return
        if kind == "gap" and frame.get("phase") == "cancelling":
            return
        terminal = kind in ("terminal", "waiting")
        stopped = kind == "gap" and frame.get("phase") == "producer_stopped"
        if stopped:
            self._stopped = frame
        if terminal:
            self._terminal = frame
        elif self.incomplete and kind not in ("gap", "tool"):
            return
        size = 0 if terminal else len(json.dumps(frame, ensure_ascii=False).encode("utf-8"))
        if self._queue.full() or self._bytes + size > _PREVIEW_BYTES:
            self.incomplete = True
            while not self._queue.empty():
                self._queue.get_nowait()
            self._bytes = 0
            self._queue.put_nowait(({"type": "gap"}, 0))
            if not terminal:
                return
        self._queue.put_nowait((frame, size))
        self._bytes += size

    async def events(
        self, turn_id: str, *, final_only: bool = False, append_only: bool = True,
    ) -> "AsyncIterator[dict[str, JsonValue]]":
        self._turn_id = turn_id
        end_states = _FINAL_STATES if final_only else _END_STATES
        while True:
            frame, size = await self._queue.get()
            self._bytes -= size
            if frame.get("turn_id") not in (None, turn_id):
                continue
            kind = frame["type"]
            if kind == "terminal":
                if final_only and str(frame.get("state", "")).upper() in _WAITING_STATES:
                    continue
                yield frame
                return
            if kind == "waiting":
                if final_only:
                    continue
                yield {**frame, "type": "terminal", "state": "waiting_deferred"}
                return
            if kind == "gap":
                if final_only and frame.get("phase") == "waiting":
                    continue
                self.incomplete = True
                yield frame
                cached = self._terminal
                if cached is not None and cached.get("turn_id") == turn_id:
                    if not final_only or (
                        cached["type"] == "terminal" and str(cached.get("state", "")).upper() not in _WAITING_STATES
                    ):
                        yield cached if cached["type"] == "terminal" else {
                            **cached, "type": "terminal", "state": "waiting_deferred",
                        }
                        return
                stopped = self._stopped
                stop_context = (
                    {"producer_stopped_generation": stopped["generation"]}
                    if stopped is not None and stopped.get("turn_id") == turn_id
                    and type(stopped.get("generation")) is int else {}
                )
                result = await self._runtime.history.turn_result(
                    self._session_id, turn_id, principal=self._principal, local_workspace=self._local_workspace,
                    **stop_context,
                )
                status = result.get("status")
                if isinstance(status, str) and status in end_states:
                    yield {**result, "type": "terminal", "state": status.lower()}
                    return
            elif kind == "preview_reset":
                if append_only:
                    # Text sinks cannot retract text from the replaced attempt.
                    self.incomplete = True
                yield frame
            elif not self.incomplete or kind == "tool":
                yield frame



@dataclass(frozen=True, slots=True)
class _SessionOccupancy:
    active_turn_id: str | None
    last_turn_no: int


async def _watch_turn_occupancy(
    runtime: "Runtime", session_id: str, principal: "Principal", *, local_workspace: bool = False,
) -> "AsyncIterator[_SessionOccupancy]":
    """Observe admitted/released occupancy, never infer it from transcript delivery."""
    queue: asyncio.Queue[tuple[str, str, int] | None] = asyncio.Queue(maxsize=64)
    resync_pending = False

    def request_snapshot() -> None:
        nonlocal resync_pending
        if resync_pending:
            return
        resync_pending = True
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(None)

    def receive(frame: dict[str, "JsonValue"]) -> None:
        kind = frame.get("type")
        if kind == "gap":
            # These phases do not change the occupied turn.
            if frame.get("phase") not in ("waiting", "cancelling", "producer_stopped"):
                request_snapshot()
            return
        if kind not in ("admitted", "terminal") or resync_pending:
            return
        identity, number = frame.get("turn_id"), frame.get("turn_no")
        if not isinstance(identity, str) or type(number) is not int or number <= 0 or queue.full():
            request_snapshot()
            return
        queue.put_nowait((kind, identity, number))

    async def snapshot() -> _SessionOccupancy:
        history = runtime.history
        if history is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        value = await history.turn_session_state(
            session_id, principal=principal, local_workspace=local_workspace,
        )
        identity, number = value["active_turn_id"], value["last_turn_no"]
        if (identity is not None and not isinstance(identity, str)) or type(number) is not int:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _SessionOccupancy(identity, number)

    unsubscribe = runtime.sessions.observe_turn_session(session_id, receive)
    try:
        current = await snapshot()
        yield current
        while True:
            notice = await queue.get()
            if notice is None:
                resync_pending = False
                next_value = await snapshot()
            else:
                kind, identity, number = notice
                if number <= current.last_turn_no:
                    continue
                if (number != current.last_turn_no + 1
                        or current.active_turn_id not in (None, identity)):
                    # A skipped position needs authority; do not turn a late
                    # terminal message into release of a different active ID.
                    request_snapshot()
                    continue
                next_value = (_SessionOccupancy(identity, current.last_turn_no)
                              if kind == "admitted" else _SessionOccupancy(None, number))
            if next_value != current:
                current = next_value
                yield current
    finally:
        unsubscribe()


__all__ = ["TurnObserver", "start_bound_execution", "uses_turn_sessions"]
