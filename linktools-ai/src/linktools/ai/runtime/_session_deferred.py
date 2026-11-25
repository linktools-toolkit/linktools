#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorize and resolve only the exact deferred frontier of a canonical turn."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from ..core import AuthorizationAction, AuthorizationPolicy, Principal, ResourceKind, ResourceRef
from ..errors import AIError, ErrorCode
from ..execution import PendingToolContinuation
from ..persistence.turns import CanonicalDeferredFrontier, SessionTurnKey, SqlSessionTurns
from ..persistence.state import StateStore

_Result = TypeVar("_Result")


class SessionDeferredCommands:
    def __init__(
        self, turns: SqlSessionTurns, authorization: AuthorizationPolicy, *, tenant_id: str,
        load: Callable[[CanonicalDeferredFrontier], Awaitable[PendingToolContinuation]],
        resume: Callable[[CanonicalDeferredFrontier], Awaitable[None]],
        is_ready: Callable[[CanonicalDeferredFrontier], Awaitable[bool]],
    ) -> None:
        self._turns = turns
        self._authorization = authorization
        self._tenant_id = tenant_id
        self._load = load
        self._resume = resume
        self._is_ready = is_ready

    async def read(
        self, key: SessionTurnKey, *, principal: Principal, action: AuthorizationAction,
    ) -> tuple[CanonicalDeferredFrontier, PendingToolContinuation] | None:
        if principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        # The host owns session ACLs; the turn ID is also the exact resource
        # for the action-specific approval/external-result permission.
        await self._authorization.authorize(
            principal, AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, key.session_id, self._tenant_id),
        )
        await self._authorization.authorize(
            principal, action, ResourceRef(ResourceKind.EXECUTION, key.turn_id, self._tenant_id),
        )
        frontier = await self._turns.get_waiting_turn(key)
        if frontier is None:
            return None
        continuation = await self._load(frontier)
        frontier.require_continuation(continuation)
        return frontier, continuation

    async def settle(
        self, frontier: CanonicalDeferredFrontier, receipt_store: StateStore,
        operation: Callable[[], Awaitable[_Result]],
    ) -> _Result:
        return await self._turns.settle_waiting_turn(
            frontier, receipt_store, operation, is_ready=lambda: self._is_ready(frontier),
        )

    async def resume(self, frontier: CanonicalDeferredFrontier) -> None:
        await self._resume(frontier)
