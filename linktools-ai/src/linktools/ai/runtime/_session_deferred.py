#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorize and resolve only the exact deferred frontier of a canonical turn."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from ..core import AuthorizationAction, AuthorizationPolicy, Principal, ResourceKind, ResourceRef
from ..errors import AIError, ErrorCode
from ..execution import PendingToolContinuation
from ..persistence.turns import CanonicalDeferredFrontier, SessionTurnKey, SqlSessionTurns
from ..persistence.state import StateStore

if TYPE_CHECKING:
    from ._canonical_split_step import _CanonicalSplitStep

_Result = TypeVar("_Result")


class SessionDeferredCommands:
    def __init__(
        self, turns: SqlSessionTurns, authorization: AuthorizationPolicy, *, tenant_id: str,
        load: Callable[[CanonicalDeferredFrontier], Awaitable[PendingToolContinuation]],
        resume: Callable[[CanonicalDeferredFrontier], Awaitable[None]],
        is_ready: Callable[[CanonicalDeferredFrontier], Awaitable[bool]],
        split: tuple[Callable[[], Awaitable[_CanonicalSplitStep]], StateStore, StateStore] | None = None,
    ) -> None:
        self._turns = turns
        self._authorization = authorization
        self._tenant_id = tenant_id
        self._load = load
        self._resume = resume
        self._is_ready = is_ready
        self._split = split

    def uses_split_receipts(self, receipt_store: StateStore) -> bool:
        return self._split is not None and receipt_store.storage_group is not self._split[1].storage_group

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
        operation: Callable[[], Awaitable[_Result]], *,
        read_exact: Callable[[], Awaitable[_Result | None]],
    ) -> _Result:
        if self._split is not None and self.uses_split_receipts(receipt_store):
            open_source, _, tool_store = self._split
            source = await open_source()
            return await source.settle_waiting(
                frontier, receipt_store=receipt_store, tool_store=tool_store, settle=operation,
                is_ready=lambda: self._is_ready(frontier), read_exact=read_exact,
            )
        return await self._turns.settle_waiting_turn(
            frontier, receipt_store, operation, is_ready=lambda: self._is_ready(frontier),
        )

    async def resume(self, frontier: CanonicalDeferredFrontier, *, receipt_store: StateStore) -> None:
        # Split receipt settlement stops at READY; it grants no producer.
        if not self.uses_split_receipts(receipt_store):
            await self._resume(frontier)
