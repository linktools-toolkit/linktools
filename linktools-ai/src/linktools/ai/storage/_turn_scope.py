#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference lease policy for backend-owned live turn lifetimes."""

from __future__ import annotations

import asyncio
import math
from types import TracebackType
from typing import Literal

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._turn import TurnCoordination, TurnCoordinationBackend, TurnScopeIdentity

_logger = environ.get_logger("ai.storage.turn")


class RenewingTurnScope:
    """A single-use turn scope over atomic coordination primitives.

    Backends may return this adapter from ``enter`` or supply their own policy.
    The deadline starts before each mutation is sent, so transport delay never
    creates extra local authority. Cleanup errors are logged without changing
    an established body outcome; uncertain claim failures always propagate.
    Renewal failures revoke authority and retain their original exception for
    validity checks and signal waiters.
    """

    def __init__(
        self, backend: TurnCoordinationBackend, scope: TurnScopeIdentity, *, session_key: str,
        request_key: str, request_digest: str, execution_id: str, owner_id: str,
        lease_seconds: float = 30.0, renew_seconds: float = 5.0,
    ) -> None:
        if (isinstance(lease_seconds, bool) or isinstance(renew_seconds, bool)
                or not math.isfinite(lease_seconds) or not math.isfinite(renew_seconds)
                or not 0 < renew_seconds < lease_seconds):
            raise ValueError("renewal period must be positive and shorter than the finite lease")
        if not isinstance(scope, TurnScopeIdentity):
            raise TypeError("turn scope must be a TurnScopeIdentity")
        self._backend = backend
        self._scope = scope
        self._identity = dict(session_key=session_key, request_key=request_key,
                              request_digest=request_digest, execution_id=execution_id, owner_id=owner_id)
        self._lease_seconds = lease_seconds
        self._renew_seconds = renew_seconds
        self._owner: asyncio.Task[object] | None = None
        self._coordination: TurnCoordination | None = None
        self._renewal: asyncio.Task[None] | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._signal = asyncio.Event()
        self._deadline = 0.0
        self._lost = False
        self._failure: Exception | None = None
        self._cancel_delivered = False
        self._cancel_requested = False
        self._closed = False

    async def __aenter__(self) -> TurnCoordination | None:
        if self._owner is not None:
            raise RuntimeError("turn scope may only be entered once")
        self._owner = asyncio.current_task()
        loop = asyncio.get_running_loop()
        self._deadline = loop.time() + self._lease_seconds
        # A failure here remains an unknown claim outcome, never a busy result.
        claim = asyncio.create_task(self._backend.claim_turn(
            self._scope, **self._identity, lease_seconds=self._lease_seconds,
        ))
        cancelled = False
        while not claim.done():
            try:
                await asyncio.shield(claim)
            except asyncio.CancelledError:
                cancelled = True
        result = claim.result()
        self._coordination = result
        if cancelled:
            await self.__aexit__(asyncio.CancelledError, asyncio.CancelledError(), None)
            raise asyncio.CancelledError
        if result is not None and result.lease is not None:
            if (result.execution_id != self._identity["execution_id"]
                    or result.owner_id != self._identity["owner_id"]):
                error = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self.__aexit__(type(error), error, None)
                raise error
            self._cancel_requested = result.cancel_requested
            if self._cancel_requested:
                self._signal.set()
            if loop.time() >= self._deadline:
                self._lose()
            else:
                self._timer = loop.call_at(self._deadline, self._lose)
                self._renewal = asyncio.create_task(self._renew())
        return result

    def _require_owned(self) -> TurnCoordination:
        result = self._coordination
        if self._closed or result is None or result.lease is None:
            raise RuntimeError("turn scope is not an entered owned lifetime")
        return result

    def require_active(self) -> None:
        self._require_owned()
        if asyncio.get_running_loop().time() >= self._deadline:
            self._lose()
        if self._lost:
            if self._failure is not None:
                raise self._failure
            raise AIError(ErrorCode.STORAGE_CONFLICT, "turn ownership was lost")

    async def wait_signal(self) -> Literal["owner_lost", "cancel_requested"]:
        while True:
            self._require_owned()
            if asyncio.get_running_loop().time() >= self._deadline:
                self._lose()
            if self._lost:
                if self._failure is not None:
                    raise self._failure
                return "owner_lost"
            if self._cancel_requested and not self._cancel_delivered:
                self._cancel_delivered = True
                return "cancel_requested"
            self._signal.clear()
            await self._signal.wait()

    def _lose(self) -> None:
        if not self._closed:
            self._lost = True
            self._signal.set()

    async def _renew(self) -> None:
        owned = self._coordination
        assert owned is not None and owned.lease is not None
        loop = asyncio.get_running_loop()
        try:
            while not self._lost and not self._closed:
                await asyncio.sleep(self._renew_seconds)
                if self._closed or self._lost or loop.time() >= self._deadline:
                    self._lose()
                    return
                deadline = loop.time() + self._lease_seconds
                renewed = await self._backend.renew_turn(owned.lease, lease_seconds=self._lease_seconds)
                # A late response must not resurrect expired local authority.
                if self._closed or self._lost or loop.time() >= self._deadline:
                    self._lose()
                    return
                if (renewed is None or renewed.lease != owned.lease
                        or renewed.execution_id != owned.execution_id or renewed.owner_id != owned.owner_id
                        or loop.time() >= deadline):
                    self._lose()
                    return
                self._deadline = deadline
                if self._timer is not None:
                    self._timer.cancel()
                self._timer = loop.call_at(deadline, self._lose)
                if renewed.cancel_requested:
                    self._cancel_requested = True
                    self._signal.set()
        except Exception as error:
            if not self._closed:
                self._failure = error
                self._lose()
        except asyncio.CancelledError:
            # A backend cancelling itself is owner loss, not cancellation of
            # the caller's control monitor. Normal exit has already closed us.
            self._lose()

    async def _cleanup(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self._renewal is not None:
            self._renewal.cancel()
            try:
                await self._renewal
            except asyncio.CancelledError:
                pass
        owned = self._coordination
        if owned is not None and owned.lease is not None:
            try:
                await self._backend.release_turn(owned.lease)
            except (Exception, asyncio.CancelledError):
                _logger.warning("Turn lease release failed; expiry retains the cleanup bound", exc_info=True)

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._owner is None or asyncio.current_task() is not self._owner or self._closed:
            raise RuntimeError("turn scope must exit once in its entering task")
        self._closed = True
        self._signal.set()
        cleanup = asyncio.create_task(self._cleanup())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled and exc is None:
            raise asyncio.CancelledError


__all__ = ["RenewingTurnScope"]
