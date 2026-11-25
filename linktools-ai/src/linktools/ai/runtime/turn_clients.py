#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-process turn observation shared by terminal and ACP adapters."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from ..core import JsonValue, Principal
    from ._runtime_service import Runtime


_PREVIEW_BYTES = 256 * 1024
_END_STATES = {"SUCCEEDED", "FAILED", "CANCELLED", "RECOVERY_REQUIRED", "WAITING_DEFERRED", "WAITING_RETRY"}


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
        if kind not in ("delta", "tool", "gap", "terminal", "waiting", "preview_reset"):
            return
        if kind == "gap" and frame.get("phase") == "cancelling":
            return
        if not self._preview and kind in ("delta", "tool", "preview_reset"):
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

    async def events(self, turn_id: str) -> "AsyncIterator[dict[str, JsonValue]]":
        self._turn_id = turn_id
        while True:
            frame, size = await self._queue.get()
            self._bytes -= size
            if frame.get("turn_id") not in (None, turn_id):
                continue
            kind = frame["type"]
            if kind == "terminal":
                yield frame
                return
            if kind == "waiting":
                yield {**frame, "type": "terminal", "state": "waiting_deferred"}
                return
            if kind == "gap":
                self.incomplete = True
                yield frame
                if self._terminal is not None and self._terminal.get("turn_id") == turn_id:
                    yield self._terminal if self._terminal["type"] == "terminal" else {
                        **self._terminal, "type": "terminal", "state": "waiting_deferred",
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
                if isinstance(status, str) and status in _END_STATES:
                    yield {**result, "type": "terminal", "state": status.lower()}
                    return
            elif kind == "preview_reset":
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


__all__ = ["TurnObserver", "uses_turn_sessions"]
