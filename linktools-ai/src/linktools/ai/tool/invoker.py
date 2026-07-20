#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolInvoker: raw handler invocation only -- timeout, heartbeat-during-
execution (idempotency lease renewal), and transient-error retry. No policy
decision, approval gate, or idempotency-claim lifecycle lives here; those are
GovernedToolInvoker's (``tool.executor.GovernedToolInvoker``) responsibility. It
constructs a ToolInvoker internally and delegates the actual handler run to
it, so "how a handler is safely called" stays separate from "whether/under
what governance it may be called at all"."""

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from .idempotency import IdempotencyStore, ToolIdempotencyOptions
    from .models import ToolDescriptor
    from .policy import EffectiveToolPolicy
    from .retry import RetryPolicy


class ToolInvoker:
    def __init__(
        self,
        *,
        idempotency_store: "IdempotencyStore | None" = None,
        idempotency_options: "ToolIdempotencyOptions | None" = None,
        retry_policy: "RetryPolicy | None" = None,
        metrics: Any = None,
    ) -> None:
        self._idempotency_store = idempotency_store
        # When set, a long-running idempotent Handler is kept alive by a
        # background renew loop (the lease heartbeat); None disables the
        # heartbeat (the claim's initial lease from claim() governs).
        self._idempotency_options = idempotency_options
        self._metrics = metrics
        from .retry import DefaultRetryPolicy

        self._retry_policy = retry_policy or DefaultRetryPolicy()

    async def _invoke_handler(self, handler, arguments, timeout, claim):
        """Invoke the handler once with timeout and -- when idempotency options
        are configured and a claim is held -- a background lease-renew
        heartbeat. Returns the result or raises."""
        if self._idempotency_options is not None and claim is not None:
            return await self._run_with_heartbeat(
                handler(**arguments), claim, timeout
            )
        if timeout is not None:
            return await asyncio.wait_for(handler(**arguments), timeout=timeout)
        return await handler(**arguments)

    async def _run_with_heartbeat(self, coro, claim, timeout):
        """Run ``coro`` as a task while a background loop renews the claim's
        lease. If a renew fails the claim was stolen -- the handler task is
        cancelled (it must not keep producing side effects under a lost claim)
        and LostIdempotencyClaimError is raised."""
        from datetime import datetime, timezone

        from ..errors import LostIdempotencyClaimError

        options = self._idempotency_options
        handler_task = asyncio.ensure_future(coro)
        lease_lost = []
        # The claim's persisted expiry is the last point at which this worker
        # is allowed to produce side effects. Transient renew failures cannot
        # extend that deadline locally.
        confirmed_deadline = getattr(claim, "lease_expires_at", None)

        async def _heartbeat():
            nonlocal confirmed_deadline
            while True:
                await asyncio.sleep(options.heartbeat_seconds)
                try:
                    renewed = await self._idempotency_store.renew(
                        claim,
                        now=datetime.now(timezone.utc),
                        lease_seconds=options.lease_seconds,
                    )
                    confirmed_deadline = renewed.lease_expires_at
                except LostIdempotencyClaimError as exc:
                    if self._metrics is not None:
                        self._metrics.counter("tool_idempotency_lease_lost_total")
                    # Claim stolen / superseded: stop the handler so it does not
                    # keep producing side effects under a lost claim.
                    lease_lost.append(exc)
                    handler_task.cancel()
                    return
                except Exception:
                    # Transient store error: retry only while the last
                    # confirmed lease is still valid. Once it expires, stop
                    # the handler fail-closed; another worker may reclaim it.
                    if confirmed_deadline is not None and datetime.now(timezone.utc) >= confirmed_deadline:
                        exc = LostIdempotencyClaimError(
                            f"idempotent claim ({claim.scope}, {claim.key}) lease expired"
                        )
                        lease_lost.append(exc)
                        handler_task.cancel()
                        return
                    continue

        heartbeat = asyncio.create_task(_heartbeat())
        try:
            if timeout is not None:
                return await asyncio.wait_for(handler_task, timeout=timeout)
            return await handler_task
        except asyncio.CancelledError:
            # Distinguish a lease-loss cancellation (heartbeat cancelled the
            # handler) from an external cancellation.
            if lease_lost:
                raise LostIdempotencyClaimError(
                    f"idempotent claim ({claim.scope}, {claim.key}) "
                    f"was stolen mid-execution"
                ) from lease_lost[0]
            raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def run_with_retry(
        self,
        handler: "Callable[..., Awaitable[Any]]",
        arguments: "dict[str, Any]",
        *,
        timeout: "float | None",
        claim: Any,
        max_retries: int,
        effective_policy: "EffectiveToolPolicy",
        descriptor: "ToolDescriptor",
    ) -> Any:
        """Invoke ``handler`` (via :meth:`_invoke_handler`), retrying transient
        HANDLER errors only, up to ``max_retries`` additional attempts.

        A failure caught here means the side effect may NOT have happened yet,
        so retrying the Handler is safe. The fenced idempotency commit is
        GovernedToolInvoker's concern, intentionally out of this loop, so a
        commit failure can never resolve by re-invoking the Handler.

        Raises the last error once attempts are exhausted or the retry policy
        declines. ``LostIdempotencyClaimError`` is never retried -- the claim
        was stolen mid-execution and must propagate immediately."""
        from ..errors import LostIdempotencyClaimError
        from .retry import backoff_delay

        last_error: "Exception | None" = None
        attempt = 0
        while True:
            try:
                return await self._invoke_handler(handler, arguments, timeout, claim)
            except Exception as exc:  # noqa: BLE001 - decide via retry policy
                if isinstance(exc, LostIdempotencyClaimError):
                    raise
                last_error = exc
                if attempt >= max_retries:
                    break
                if not self._retry_policy.should_retry(
                    error=exc,
                    attempt=attempt,
                    policy=effective_policy,
                    descriptor=descriptor,
                ):
                    break
                # Transient: back off (cancellable) and retry.
                await asyncio.sleep(backoff_delay(attempt + 1))
                attempt += 1
        raise last_error  # type: ignore[misc]
