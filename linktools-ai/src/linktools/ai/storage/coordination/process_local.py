#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProcessLocalLeaseCoordinator: the in-process reference LeaseCoordinator.

This is the process-local reference implementation of the
:class:`linktools.ai.storage.protocols.LeaseCoordinator` Protocol -- the
"process-local coordination reference implementation" the plan (§4.4, §4.11)
requires the core to ship. It is correct within a single process: acquires are
mutually exclusive per key, the fencing token is monotonic across
(re)acquisitions, and an expired lease can be reclaimed. It is NOT correct
across processes or hosts; a deployment that needs that must inject a
distributed coordinator (Redis, etcd, a DB-backed lease table) implementing the
same Protocol. The RuntimeBuilder capability-gates on
``StorageFeatures.coordination`` so a multi-worker Job or multi-process Swarm
configured against process-local coordination fails fast at build time rather
than silently racing.

Fencing monotonicity is the defining guarantee: the integer counter strictly
increases on every grant, so a JobStore state commit that records the token it
observed can reject a stale write from a holder whose lease expired and was
reclaimed -- even though the stale holder still believes it owns the lock.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from ...errors import StorageConcurrencyNotSupportedError
from ..protocols import LeaseToken


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProcessLocalLeaseCoordinator:
    """Process-local LeaseCoordinator with monotonic fencing.

    All state lives in one process's memory; a single ``asyncio`` lock guards
    the holder map so concurrent coroutines serialize on acquire/renew/release.
    Acquire on a key whose current lease has passed its expiry reclaims it (and
    mints a strictly larger fencing token); acquire on a live lease held by
    another owner returns ``None``.
    """

    def __init__(self) -> None:
        # key -> the LeaseToken currently held (or None after release/expiry).
        self._holders: "dict[str, LeaseToken]" = {}
        # Monotonic grant counter -- the source of fencing tokens.
        self._counter = 0
        self._lock = asyncio.Lock()

    def _is_live(self, token: "LeaseToken | None", now: datetime) -> bool:
        return token is not None and token.expires_at > now

    async def acquire(
        self, *, key: str, owner_id: str, ttl: timedelta
    ) -> "LeaseToken | None":
        async with self._lock:
            now = _utcnow()
            current = self._holders.get(key)
            if self._is_live(current, now):
                # Held by someone else (a live lease is never anonymous).
                return None
            # Free, or the previous lease expired -> grant a fresh one with a
            # strictly larger fencing token (re-acquire after expiry must
            # increase the token so stale holders are detectable).
            self._counter += 1
            token = LeaseToken(
                lease_id=f"local-{self._counter}",
                owner_id=owner_id,
                fencing_token=self._counter,
                expires_at=now + ttl,
                key=key,
            )
            self._holders[key] = token
            return token

    async def renew(self, *, token: LeaseToken, ttl: timedelta) -> LeaseToken:
        async with self._lock:
            now = _utcnow()
            current = self._holders.get(token.key)
            # Renew only succeeds for the current holder of a still-live lease
            # whose token matches. Renewing MUST NOT change the fencing token;
            # only the expiry moves forward.
            if current is None or current.lease_id != token.lease_id:
                raise StorageConcurrencyNotSupportedError(
                    "renew refused: lease no longer held by this token"
                )
            if current.expires_at <= now:
                raise StorageConcurrencyNotSupportedError(
                    "renew refused: lease already expired"
                )
            renewed = LeaseToken(
                lease_id=current.lease_id,
                owner_id=current.owner_id,
                fencing_token=current.fencing_token,
                expires_at=now + ttl,
                key=current.key,
            )
            self._holders[token.key] = renewed
            return renewed

    async def release(self, *, token: LeaseToken) -> None:
        async with self._lock:
            current = self._holders.get(token.key)
            # Release is idempotent: releasing a token that no longer matches
            # the holder (already released / reclaimed) is a no-op, not an
            # error -- the resource is already free.
            if current is not None and current.lease_id == token.lease_id:
                self._holders.pop(token.key, None)


__all__: "list[str]" = ["ProcessLocalLeaseCoordinator"]
