#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native conversation handles, separate from independently stored executions."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast

from ..errors import AIError, ErrorCode
from ..persistence.turns import SessionTurnKey
from .turn_clients import TurnObserver, _END_STATES, _SessionOccupancy, _watch_turn_occupancy

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

    async def result(self) -> ResultT:
        return await self._read_result(self.key, self.principal)

    async def cancel(self) -> bool:
        return await self._sessions.cancel_turn(self.key.session_id, self.key.turn_id, principal=self.principal)

    async def recover(self) -> bool:
        return await self._sessions.recover_turn(self.key.session_id, self.key.turn_id, principal=self.principal)

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
        return await self._history().turn_session_state(
            self.session_id, principal=self.principal, local_workspace=self._local_workspace,
        )

    def watch(self) -> AsyncIterator[_SessionOccupancy]:
        return _watch_turn_occupancy(
            self._runtime, self.session_id, self.principal, local_workspace=self._local_workspace,
        )

    async def history(self, *, cursor: str | None = None, limit: int = 100) -> dict[str, JsonValue]:
        """Read one signed native history page, including its session state."""
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
        return _NativeSessionTurn(SessionTurnKey(self.session_id, turn_id), actor, self._sessions, self._read_result)

    @asynccontextmanager
    async def start(
        self, prompt: UserPromptInput, *, agent: str | Agent[AppT], idempotency_key: str,
        files: Sequence[str] = (), session_context: SessionContext = "all", mode: ExecutionMode = "run",
        planning: bool = False, thinking: JsonValue = None, budget: RunBudget | None = None,
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
