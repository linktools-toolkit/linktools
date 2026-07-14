#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolExecutor: consults PolicyEngine before a tool executes, translating
its decision into the corresponding domain error.

Approval flow: the executor mints an ``approval_id`` and raises ``RunPaused``
carrying every field needed to build the ApprovalRequest. AgentRunner's pause
handler then persists the ApprovalRequest (deduping on
``(run_id, tool_call_id)``) alongside the checkpoint save + WAITING_APPROVAL
transition + pause events -- on a cross-store-transactional Storage that
happens in one UnitOfWork, so a crash between "approval persisted" and "run
paused" is impossible. ``Runtime.resume`` completes the flow. The executor only
emits the domain signal (RunPaused); it never persists approval state itself,
so there is a single approval path."""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..agent.approval import ApprovalStatus
from ..errors import (
    IdempotencyConflictError,
    IdempotencyInProgressError,
    RunPaused,
    ToolDeniedError,
)
from ..policy.engine import PolicyDecisionKind, PolicyEngine, ToolContext, ToolRequest
from .idempotency import IdempotencyStore, compute_request_hash

if TYPE_CHECKING:
    from ..agent.approval import ApprovalStore
    from .models import ToolDescriptor
    from .policy import EffectiveToolPolicy

_LOGGER = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(
        self,
        *,
        policy: PolicyEngine,
        approval_store: "ApprovalStore | None" = None,
        run_id_resolver: "Callable[[ToolContext], str] | None" = None,
        idempotency_store: "IdempotencyStore | None" = None,
        retry_policy: Any = None,
    ) -> None:
        self._policy = policy
        self._approval_store = approval_store
        self._run_id_resolver = run_id_resolver
        self._idempotency_store = idempotency_store
        # Retry is policy-driven: only clearly-transient errors, and never a
        # mutating non-idempotent tool. DefaultRetryPolicy when none supplied.
        from .retry import DefaultRetryPolicy

        self._retry_policy = retry_policy or DefaultRetryPolicy()

    async def check(self, request: ToolRequest, context: ToolContext) -> None:
        decision = await self._policy.evaluate(request, context)
        if decision.kind == PolicyDecisionKind.DENY:
            raise ToolDeniedError(
                decision.reason or f"tool denied: {request.tool_name}"
            )
        if decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL:
            # Resume gate: if the approval_store already holds an APPROVED
            # request matching (run_id, tool_call_id), the call was approved
            # externally (e.g. via the pause UI) and is now being re-driven by
            # the model with the same tool_call_id -- let it through instead
            # of re-raising RunPaused.
            if await self._already_approved(request, context):
                return
            run_id = (
                self._run_id_resolver(context)
                if self._run_id_resolver is not None
                else context.run_id
            )
            # The executor only emits the domain signal (RunPaused); it does
            # NOT persist approval state. AgentRunner's pause handler persists
            # the ApprovalRequest atomically with the checkpoint/transition/
            # event writes, so there is a single approval path.
            tool_call_id = context.tool_call_id or str(uuid.uuid4())
            raise RunPaused(
                run_id=run_id,
                approval_id=str(uuid.uuid4()),
                tool_call_id=tool_call_id,
                tool_name=request.tool_name,
                reason=decision.reason,
                arguments=dict(request.arguments),
            )

    async def _already_approved(
        self, request: ToolRequest, context: ToolContext
    ) -> bool:
        """True iff the approval_store has an APPROVED request matching
        ``(run_id, tool_call_id)`` -- the resume case. False when no store is
        wired, when the context carries no ``tool_call_id`` (no stable key to
        match on -- the uuid fallback path can't recognize a re-drive), or
        when no matching APPROVED request exists.

        Consults ``list_for_run`` (status-agnostic) rather than
        ``list_pending`` because the matching request is APPROVED, not PENDING
        -- ``list_pending`` would filter it out and the gate would never fire.
        """
        if self._approval_store is None or context.tool_call_id is None:
            return False
        run_id = (
            self._run_id_resolver(context)
            if self._run_id_resolver is not None
            else context.run_id
        )
        return await self.is_approved(run_id, context.tool_call_id)

    async def is_approved(
        self, run_id: "str | None", tool_call_id: "str | None"
    ) -> bool:
        """Public resume gate: True iff the approval_store holds an APPROVED
        request matching ``(run_id, tool_call_id)``. Used by managed-path
        approval (pipeline REQUIRE_APPROVAL / policy.require_approval) to
        recognize a re-driven call after external approval -- without it, a
        stateless pipeline would re-raise RunPaused on every resume drive and
        the run could never complete. False when no store/tool_call_id wired."""
        if self._approval_store is None or run_id is None or tool_call_id is None:
            return False
        requests = await self._approval_store.list_for_run(run_id)
        return any(
            r.tool_call_id == tool_call_id and r.status is ApprovalStatus.APPROVED
            for r in requests
        )

    async def execute(
        self,
        request: ToolRequest,
        context: ToolContext,
        handler: "Callable[..., Awaitable[Any]]",
        *,
        descriptor: "ToolDescriptor",
        effective_policy: "EffectiveToolPolicy",
        timeout: "float | None" = None,
        max_retries: int = 0,
        idempotency_key: "str | None" = None,
        schema_version: str = "1",
        idempotency_scope: "str | None" = None,
    ) -> Any:
        """Policy-check then run ``handler``, optionally with timeout/retry.

        ``descriptor`` and ``effective_policy`` are required: the retry decision
        reads ``descriptor.mutating`` and ``effective_policy.idempotent`` (a
        mutating non-idempotent tool is never retried), so a caller that omits
        them would let the executor invent a non-mutating default policy and
        retry a write it must not. Every real call site
        (ManagedToolAdapter) passes the finalized descriptor and policy it just
        resolved; a direct executor call with no descriptor/policy is a
        programming error, surfaced as ``TypeError`` by the signature itself.

        ``timeout`` wraps each handler call in :func:`asyncio.wait_for`; an
        :class:`asyncio.TimeoutError` is caught like any other exception
        (retried if attempts remain, raised otherwise). ``max_retries`` is the
        number of additional attempts after the first -- on any exception the
        call is retried up to ``max_retries`` times, after which the last error
        is re-raised. The defaults (``timeout=None``, ``max_retries=0``) mean a
        single handler call with no timeout wrap.

        ``idempotency_key`` enables persistent tool-call idempotency. When the
        executor was constructed with an
        ``idempotency_store`` AND ``idempotency_key`` is provided, the
        ``(scope, key)`` is ``(context.run_id, idempotency_key)``. The store
        survives process restart (dict-only idempotency is forbidden).
        Behavior per ``IdempotencyStatus``:

        - COMPLETED + same request hash -> cached ``result`` returned, the
          handler is NOT invoked.
        - RESERVED  + same request hash -> ``IdempotencyInProgressError``
          (another in-flight call owns the reservation).
        - FAILED    + same request hash -> retry: the handler runs again and
          the terminal transition overwrites the FAILED record.
        - Same (scope, key) with a different request hash -> reserve raises
          ``IdempotencyConflictError`` (same key reused for different args).
        - Fresh reservation -> handler runs; ``complete`` on success,
          ``fail`` on exception.

        ``schema_version`` (default ``"1"``) is folded into the request hash
        so a ToolSpec whose input contract changed shape bumps its
        schema_version and gets a fresh hash -- a stale idempotency record
        from before the change is never mistaken for a match. Callers that
        have a ``ToolSpec`` in hand should pass ``spec.schema_version`` here.

        With no ``idempotency_store`` (or no ``idempotency_key``), idempotency
        is off: the handler runs every call and nothing is persisted."""
        await self.check(request, context)
        # Idempotency is scoped to context.run_id so the same idempotency_key
        # can be reused across different runs without colliding. ``scope`` is
        # part of the request_hash too, so a hash mismatch always reflects a
        # genuine difference in (tool, args, scope) -- not just (tool, args).
        # Fail closed: a tool declared idempotent (so a key was provided) but
        # run against a Storage with no IdempotencyStore must not silently run
        # non-idempotently -- that would let a replayed call execute twice.
        if idempotency_key is not None and self._idempotency_store is None:
            from ..errors import StorageCapabilityError

            raise StorageCapabilityError(
                f"tool {request.tool_name!r} is idempotent but no IdempotencyStore "
                f"is wired; refusing to run non-idempotently"
            )
        use_idempotency = (
            self._idempotency_store is not None and idempotency_key is not None
        )
        claim = None
        if use_idempotency:
            import uuid as _uuid

            from .idempotency import ClaimDisposition

            scope = idempotency_scope or context.run_id
            request_hash = compute_request_hash(
                request.tool_name,
                request.arguments,
                scope,
                schema_version=schema_version,
            )
            owner_id = _uuid.uuid4().hex
            claim_result = await self._idempotency_store.claim(
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                owner_id=owner_id,
            )
            disposition = claim_result.disposition
            if disposition is ClaimDisposition.REPLAY:
                return claim_result.record.result
            if disposition is ClaimDisposition.IN_PROGRESS:
                raise IdempotencyInProgressError(
                    f"idempotent request in progress: key={idempotency_key!r}"
                )
            if disposition is ClaimDisposition.CONFLICT:
                raise IdempotencyConflictError(
                    f"idempotency key {idempotency_key!r} reused with a different request"
                )
            # ACQUIRED: run the handler, then complete/fail with the fenced claim.
            claim = claim_result.claim
        arguments = dict(request.arguments)
        last_error: "Exception | None" = None
        from .retry import backoff_delay

        attempt = 0
        while True:
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(
                        handler(**arguments), timeout=timeout
                    )
                else:
                    result = await handler(**arguments)
                if use_idempotency:
                    await self._idempotency_store.complete(claim, result)
                return result
            except Exception as exc:  # noqa: BLE001 - decide via retry policy
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
        if use_idempotency:
            from ..security.redact import redact_exception

            safe_error = (
                redact_exception(last_error)
                if last_error is not None
                else "unknown error"
            )
            await self._idempotency_store.fail(claim, safe_error)
        raise last_error  # type: ignore[misc]
