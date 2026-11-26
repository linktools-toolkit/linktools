#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native conversation handles, separate from independently stored executions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing, AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast

from ..core import ExecutionEventType, ExecutionLineageKind
from ..errors import AIError, ErrorCode, ObservationError
from ..execution import ExecutionStreamEvent, ExecutionTreeEvent
from ..observe import WaitResult
from ..observe.stream import ObservationSession, call_observer, validate_timeout
from ..persistence.turns import SessionTurnKey
from .turn_clients import TurnObserver, _END_STATES, _FINAL_STATES, _SessionOccupancy, _watch_turn_occupancy

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..core import ExecutionMode, ExecutionStatus, JsonValue, Principal, RunBudget, UsageMetrics
    from ..history import History
    from ..model import UserPromptInput
    from ..session import SessionContext
    from ._agent import Agent
    from ._domains import RuntimeSessions
    from ._runtime_service import Runtime

AppT = TypeVar("AppT")


class _NativeMetadataObserver(TurnObserver):
    def __init__(self, runtime: Runtime, session_id: str, principal: Principal, *, local_workspace: bool) -> None:
        super().__init__(runtime, session_id, principal, local_workspace=local_workspace)
        # Reuse the same low-level subscription, not the callback's queue.
        # Slow user callbacks cannot prevent the authority reader from waking.
        self.authority = TurnObserver(runtime, session_id, principal, local_workspace=local_workspace, preview=False)

    def __enter__(self) -> _NativeMetadataObserver:
        self._unsubscribe = self._runtime.sessions.observe_turn_session(self._session_id, self._receive_both)
        return self

    def _receive_both(self, frame: dict[str, JsonValue]) -> None:
        self.authority._receive(frame)
        self._receive(frame)

    def _receive(self, frame: dict[str, JsonValue]) -> None:
        # Preview frames are not part of the metadata callback and must
        # not evict tool events from its bounded observer queue.
        if frame.get("type") in ("tool", "gap", "terminal", "waiting"):
            super()._receive(frame)

    def finish_from_authority(self, turn_id: str, result: _NativeTurnResult) -> None:
        live = self._terminal
        if (live is not None and live.get("turn_id") == turn_id and live.get("type") == "terminal"
                and str(live.get("state", "")).upper() in _FINAL_STATES):
            return
        # Use the same queue to preserve any tool frames already received.
        # Only the terminal is proven by the result; live coverage is unknown.
        super()._receive({
            **result.view, "type": "terminal", "turn_id": turn_id,
            "state": result.status.value.lower(), "_authority_fallback": True,
        })


class _NativeTurnResult(Protocol):
    @property
    def key(self) -> SessionTurnKey: ...

    @property
    def usage(self) -> UsageMetrics | None: ...

    @property
    def status(self) -> ExecutionStatus: ...

    @property
    def view(self) -> Mapping[str, JsonValue]: ...


ResultT = TypeVar("ResultT", bound=_NativeTurnResult)
_NativeTurnRead = Callable[[SessionTurnKey, "Principal", bool], Awaitable[_NativeTurnResult]]


@dataclass(frozen=True, slots=True)
class _NativeSessionTurn(Generic[AppT, ResultT]):
    key: SessionTurnKey
    principal: Principal
    _sessions: RuntimeSessions[AppT]
    _read_result: Callable[[SessionTurnKey, Principal], Awaitable[ResultT]]
    _runtime: Runtime[AppT]
    _local_workspace: bool
    _replayed: bool = False

    async def result(self) -> ResultT:
        return await self._read_result(self.key, self.principal)

    async def cancel(self) -> bool:
        return await self._sessions.cancel_turn(self.key.session_id, self.key.turn_id, principal=self.principal)

    async def recover(self) -> bool:
        return await self._sessions.recover_turn(self.key.session_id, self.key.turn_id, principal=self.principal)

    async def observe(self, *, preview: bool = True) -> AsyncIterator[dict[str, JsonValue]]:
        """Observe live previews (including thinking timing and model status) and settlement.

        With preview disabled, observe settlement and delivery gaps only.
        Earlier previews are not replayed.
        """
        with TurnObserver(
            self._runtime, self.key.session_id, self.principal,
            local_workspace=self._local_workspace, preview=preview,
        ) as observer:
            # Subscribe before the authorized read to cover completion between
            # opening this iterator and obtaining its durable current state.
            initial = await self.result()
            if initial.status.value in _END_STATES:
                yield {**initial.view, "type": "terminal", "state": initial.status.value.lower()}
                return
            events = observer.events(self.key.turn_id)
            try:
                async for event in events:
                    yield event
            finally:
                await events.aclose()

    async def wait(self) -> ResultT:
        """Wait for this attempt to settle and read its authorized, typed result."""
        return await self._wait(final_only=False)

    async def wait_final(self, *, timeout_seconds: float | None = None) -> ResultT:
        """Wait across approval and retry pauses without starting or resuming work."""
        validate_timeout(timeout_seconds)
        async with asyncio.timeout(timeout_seconds):
            return await self._wait(final_only=True)

    async def _wait(
        self, *, final_only: bool, on_initial: Callable[[ResultT], None] | None = None,
        subscribed: TurnObserver | None = None,
    ) -> ResultT:
        end_states = _FINAL_STATES if final_only else _END_STATES
        scope = (TurnObserver(
            self._runtime, self.key.session_id, self.principal,
            local_workspace=self._local_workspace, preview=False,
        ) if subscribed is None else nullcontext(subscribed))
        with scope as observer:
            initial = await self.result()
            if on_initial is not None:
                on_initial(initial)
            if initial.status.value in end_states:
                return initial
            events = observer.events(self.key.turn_id, final_only=final_only)
            try:
                async for event in events:
                    if event["type"] == "terminal":
                        return await self.result()
            finally:
                await events.aclose()
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    async def _wait_observed(
        self, observer: _NativeMetadataObserver, *, agent_id: str,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]], timeout_seconds: float | None,
        close_timeout_seconds: float, raise_on_stream_error: bool,
    ) -> WaitResult[ResultT]:
        """Deliver only authorized, live root metadata; no tree cursor or raw body."""
        runtime = self._runtime
        turn_id = self.key.turn_id
        loop = asyncio.get_running_loop()
        initial: asyncio.Future[ResultT] = loop.create_future()
        drained: asyncio.Future[ResultT] = loop.create_future()
        observation = ObservationSession(
            "session-turn", turn_id, None, close_timeout_seconds, runtime._release_observation,
            raise_on_stream_error=raise_on_stream_error,
        )
        runtime._register_observation(observation)

        async def emit(frame: Mapping[str, JsonValue]) -> None:
            kind = frame.get("type")
            if kind == "gap":
                observation.record_error(ObservationError(
                    "stream", cursor=None, safe_details={"coverage_complete": False, "resource_id": turn_id},
                ))
                return
            if kind == "tool" and isinstance(item := frame.get("item"), Mapping):
                name, status, key = item.get("tool_name"), item.get("status"), item.get("key")
                if not isinstance(name, str) or not isinstance(status, str) or not isinstance(key, str):
                    return
                event_type = (ExecutionEventType.TOOL_CALL_STARTED if status == "RUNNING"
                              else ExecutionEventType.TOOL_CALL_FINISHED)
                payload = {"tool_name": name, "status": status, "tool_call_id": key.removeprefix("tool:")}
            elif kind == "terminal":
                status = frame.get("state")
                event_type = {
                    "succeeded": ExecutionEventType.EXECUTION_SUCCEEDED,
                    "failed": ExecutionEventType.EXECUTION_FAILED,
                    "cancelled": ExecutionEventType.EXECUTION_CANCELLED,
                    "recovery_required": ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
                }.get(status if isinstance(status, str) else "")
                if event_type is None:
                    return
                payload = {"status": status}
                if isinstance(code := frame.get("error_code"), str):
                    payload["error_code"] = code
            else:
                return
            await call_observer(on_event, ExecutionTreeEvent(
                execution_id=turn_id, agent_id=agent_id, lineage_kind=ExecutionLineageKind.SESSION_RESUME,
                parent_execution_id=None, root_execution_id=turn_id, parent_invocation_id=None, depth=0,
                event=ExecutionStreamEvent(turn_id, None, event_type.value, payload),
            ), cursor=None)

        async def watch() -> None:
            result = await initial
            if self._replayed and result.status.value in _FINAL_STATES:
                await emit({**result.view, "type": "terminal", "state": result.status.value.lower()})
                return

            async def unblock_terminal() -> None:
                confirmed = await asyncio.shield(drained)
                observer.finish_from_authority(turn_id, confirmed)

            unblock = asyncio.create_task(unblock_terminal(), name=f"native-turn-terminal-{turn_id}")
            try:
                async with aclosing(observer.events(turn_id, final_only=True)) as frames:
                    async for frame in frames:
                        await emit(frame)
                        if frame.get("_authority_fallback") is True:
                            # Report incomplete live coverage only after delivering
                            # the already-authoritative terminal callback.
                            await emit({"type": "gap"})
            finally:
                unblock.cancel()
                await asyncio.gather(unblock, return_exceptions=True)

        result, stream_error = await observation.wait(
            watch(), self._wait(final_only=True, on_initial=initial.set_result, subscribed=observer.authority),
            timeout_seconds, drained,
        )
        return WaitResult(result, None, stream_error)

    async def _replay_events(self) -> AsyncIterator[dict[str, JsonValue]]:
        result = await self.result()
        # Replays have no newly started producer to drain. Current means a
        # point-in-time authoritative observation, not a terminal or takeover.
        kind = "terminal" if result.status.value in _END_STATES else "current"
        yield {**result.view, "type": kind, "state": result.status.value.lower()}


@dataclass(frozen=True, slots=True)
class _NativeSession(Generic[AppT, ResultT]):
    session_id: str
    principal: Principal
    _runtime: Runtime[AppT]
    _sessions: RuntimeSessions[AppT]
    _read_result: Callable[[SessionTurnKey, Principal], Awaitable[ResultT]]
    _local_workspace: bool

    def _history(self) -> History:
        history = self._runtime.history
        if history is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return history

    async def inspect(self) -> dict[str, JsonValue]:
        await self._sessions._require_session_available()
        return await self._history().turn_session_state(
            self.session_id, principal=self.principal, local_workspace=self._local_workspace,
        )

    async def watch(self) -> AsyncIterator[_SessionOccupancy]:
        await self._sessions._require_session_available()
        async with aclosing(_watch_turn_occupancy(
            self._runtime, self.session_id, self.principal, local_workspace=self._local_workspace,
        )) as frames:
            async for frame in frames:
                yield frame

    async def history(self, *, cursor: str | None = None, limit: int = 100) -> dict[str, JsonValue]:
        """Read one signed native history page, including its session state."""
        await self._sessions._require_session_available()
        return await self._history().turn_session(
            self.session_id, principal=self.principal, local_workspace=self._local_workspace,
            cursor=cursor, limit=limit,
        )

    async def timeline(self, *, cursor: str | None = None, limit: int = 100) -> dict[str, JsonValue]:
        """Read the timeline page without flattening or rereading older turns."""
        page = await self.history(cursor=cursor, limit=limit)
        return cast("dict[str, JsonValue]", page["timeline"])

    async def fork(
        self, new_session_id: str, *, expected_revision: int, idempotency_key: str,
    ) -> _NativeSession[AppT, ResultT] | None:
        created = await self._sessions.fork_turn_session(
            self.session_id, new_session_id, principal=self.principal,
            expected_revision=expected_revision, idempotency_key=idempotency_key,
        )
        return replace(self, session_id=new_session_id) if created else None

    async def update(
        self, *, expected_revision: int, metadata: Mapping[str, JsonValue],
        cwd: str | None = None, replace_cwd: bool = False,
    ) -> bool:
        return await self._sessions.update_turn_session(
            self.session_id, principal=self.principal, expected_revision=expected_revision,
            metadata=metadata, cwd=cwd, replace_cwd=replace_cwd,
        )

    async def close(self, *, expected_revision: int) -> bool:
        return await self._sessions.close_turn_session(
            self.session_id, principal=self.principal, expected_revision=expected_revision,
        )

    def plan(
        self, prompt: UserPromptInput, *, agent: str | Agent[AppT], idempotency_key: str,
        files: Sequence[str] = (), session_context: SessionContext = "all",
        thinking: JsonValue = None, budget: RunBudget | None = None,
        output: type[BaseModel] | None = None, correlation: Mapping[str, object] | None = None,
        preview: bool = True, principal: Principal | None = None,
    ) -> AbstractAsyncContextManager[tuple[_NativeSessionTurn[AppT, ResultT], AsyncIterator[dict[str, JsonValue]]]]:
        return self.start(
            prompt, agent=agent, idempotency_key=idempotency_key, files=files, session_context=session_context,
            mode="plan", thinking=thinking, budget=budget, output=output, correlation=correlation, preview=preview, principal=principal,
        )

    def _turn(self, turn_id: str, actor: Principal) -> _NativeSessionTurn[AppT, ResultT]:
        return _NativeSessionTurn(
            SessionTurnKey(self.session_id, turn_id), actor, self._sessions, self._read_result,
            self._runtime, self._local_workspace,
        )

    @asynccontextmanager
    async def start(
        self, prompt: UserPromptInput, *, agent: str | Agent[AppT], idempotency_key: str,
        files: Sequence[str] = (), session_context: SessionContext = "all", mode: ExecutionMode = "run",
        planning: bool | None = None, thinking: JsonValue = None, budget: RunBudget | None = None,
        output: type[BaseModel] | None = None, correlation: Mapping[str, object] | None = None,
        preview: bool = True, principal: Principal | None = None,
    ) -> AsyncIterator[tuple[_NativeSessionTurn[AppT, ResultT], AsyncIterator[dict[str, JsonValue]]]]:
        actor = self.principal if principal is None else principal
        # Subscribe before admission so an immediate result cannot outrun an observer.
        with TurnObserver(
            self._runtime, self.session_id, actor, local_workspace=self._local_workspace, preview=preview,
        ) as observer:
            receipt = await self._sessions._start_turn_receipt(
                self.session_id, prompt, agent=agent, principal=actor, idempotency_key=idempotency_key,
                files=files, session_context=session_context, mode=mode, planning=planning, thinking=thinking,
                budget=budget, output=output, correlation=correlation,
            )
            turn = self._turn(receipt.turn_id, actor)
            events = turn._replay_events() if receipt.replayed else observer.events(receipt.turn_id)
            try:
                yield turn, events
            finally:
                await events.aclose()
