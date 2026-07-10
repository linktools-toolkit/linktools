#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolExecutor: consults PolicyEngine before a tool executes, translating
its decision into the corresponding domain error.

Canonical approval flow (``pause_on_approval=True``): the executor mints an
``approval_id`` and raises ``RunPaused`` carrying every field needed to build
the ApprovalRequest. AgentRunner's pause handler then persists the
ApprovalRequest (deduping on ``(run_id, tool_call_id)``) alongside the
checkpoint save + WAITING_APPROVAL transition + pause events -- on a
cross-store-transactional Storage that happens in one UnitOfWork, so a crash
between "approval persisted" and "run paused" is impossible. ``Runtime.resume``
completes the flow.

Fallback (``pause_on_approval=False``): for runtimes without the pause/resume
path, the executor persists a PENDING ApprovalRequest (when an approval store
is wired) and emits an ApprovalRequested event (when an event store is wired)
before raising ``ToolApprovalRequiredError``. This is the simpler, non-pausing
mode; the pause/resume flow above is the recommended canonical path."""
import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..agent.approval import ApprovalStatus, build_approval_request
from ..errors import IdempotencyInProgressError, RunPaused, ToolApprovalRequiredError, ToolDeniedError
from ..events.payloads import ApprovalRequested
from ..policy.engine import PolicyDecisionKind, PolicyEngine, ToolContext, ToolRequest
from .idempotency import IdempotencyStatus, IdempotencyStore, compute_request_hash

if TYPE_CHECKING:
    from ..agent.approval import ApprovalRequest, ApprovalStore
    from ..events.store import EventStore

_LOGGER = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(
        self,
        *,
        policy: PolicyEngine,
        approval_store: "ApprovalStore | None" = None,
        event_store: "EventStore | None" = None,
        run_id_resolver: "Callable[[ToolContext], str] | None" = None,
        pause_on_approval: bool = False,
        idempotency_store: "IdempotencyStore | None" = None,
    ) -> None:
        self._policy = policy
        self._approval_store = approval_store
        self._event_store = event_store
        self._run_id_resolver = run_id_resolver
        self._idempotency_store = idempotency_store
        # When True, the require-approval branch persists the request (via
        # _record_approval) and then raises RunPaused(run_id, approval_id)
        # INSTEAD of ToolApprovalRequiredError. RunPaused is a RunError (not a
        # ToolError) so PolicyCapability's catch list does not translate it
        # into SkipToolExecution -- it propagates out of pydantic-ai's
        # tool-execution stack to AgentRunner ( catch it). Default
        # False preserves today's behavior byte-for-byte.
        self._pause_on_approval = pause_on_approval

    async def check(self, request: ToolRequest, context: ToolContext) -> None:
        decision = await self._policy.evaluate(request, context)
        if decision.kind == PolicyDecisionKind.DENY:
            raise ToolDeniedError(decision.reason or f"tool denied: {request.tool_name}")
        if decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL:
            # Resume gate: if the approval_store already holds an APPROVED
            # request matching (run_id, tool_call_id), the call was approved
            # externally (e.g. via the pause UI) and is now being re-driven by
            # the model with the same tool_call_id -- let it through instead
            # of re-persisting a PENDING duplicate and re-raising.
            if await self._already_approved(request, context):
                return
            run_id = (
                self._run_id_resolver(context)
                if self._run_id_resolver is not None
                else context.run_id
            )
            if self._pause_on_approval:
                # do NOT persist here. Mint the id and hand every
                # field the suspension handler needs to AgentRunner via
                # RunPaused -- the actual ApprovalStore write happens there,
                # atomically with the checkpoint/transition/event writes.
                tool_call_id = context.tool_call_id or str(uuid.uuid4())
                raise RunPaused(
                    run_id=run_id,
                    approval_id=str(uuid.uuid4()),
                    tool_call_id=tool_call_id,
                    tool_name=request.tool_name,
                    reason=decision.reason,
                    arguments=dict(request.arguments),
                )
            # Legacy direct-raise path: persist immediately (unchanged).
            await self._record_approval(request, context, decision.reason)
            raise ToolApprovalRequiredError(
                decision.reason or f"tool requires approval: {request.tool_name}"
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
            self._run_id_resolver(context) if self._run_id_resolver is not None
            else context.run_id
        )
        requests = await self._approval_store.list_for_run(run_id)
        return any(
            r.tool_call_id == context.tool_call_id
            and r.status is ApprovalStatus.APPROVED
            for r in requests
        )

    async def _record_approval(
        self,
        request: ToolRequest,
        context: ToolContext,
        reason: "str | None",
    ) -> "ApprovalRequest | None":
        """Persist a PENDING ApprovalRequest (and emit ApprovalRequested) when
        an approval_store is wired, returning the persisted request. Returns
        None when no store is wired. Best-effort audit: event-emission failures
        are logged and swallowed (the approval record is the source of truth);
        the caller (check) still raises afterward regardless of outcome here.
        With approval_store=None this is a no-op (default-None path: behavior
        identical to today).

        Only used by the legacy ``pause_on_approval=False`` direct-raise path
        -- the pause path does NOT call this; it defers persistence
        to AgentRunner's suspension handler instead (see ``check``).

        Event emission is best-effort: the EventStore assigns the sequence
        itself, so a failure here is logged and swallowed --
        the approval record remains the source of truth. The store's
        per-stream sequence counter naturally interleaves this event with any
        others emitted for the same run (e.g. AgentRunner's RunStarted)."""
        if self._approval_store is None:
            return None

        run_id = (
            self._run_id_resolver(context) if self._run_id_resolver is not None
            else context.run_id
        )
        # Prefer the pydantic-ai ToolCallPart id threaded through ToolContext
        # by PolicyCapability (the linchpin of resume: a re-driven call after
        # approve() must find the matching approval). Fall back to a fresh uuid
        # when unset (e.g. tests that construct ToolContext directly without a
        # tool_call_id -- test_executor_approval.py relies on this path).
        tool_call_id = context.tool_call_id or str(uuid.uuid4())
        approval = build_approval_request(
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=request.tool_name,
            reason=reason,
            arguments=request.arguments,
        )
        await self._approval_store.create(approval)

        if self._event_store is not None:
            try:
                await self._event_store.append(
                    stream_id=approval.run_id,
                    run_id=approval.run_id,
                    root_run_id=approval.run_id,
                    parent_run_id=None,
                    session_id=context.session_id,
                    runnable_id=request.tool_name,
                    payload=ApprovalRequested(
                        approval_id=approval.id,
                        tool_name=request.tool_name,
                        reason=reason or "",
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - best-effort audit
                _LOGGER.warning(
                    "failed to append ApprovalRequested event for approval %s: %s",
                    approval.id,
                    exc,
                )

        return approval

    async def execute(
        self,
        request: ToolRequest,
        context: ToolContext,
        handler: "Callable[..., Awaitable[Any]]",
        *,
        timeout: "float | None" = None,
        max_retries: int = 0,
        idempotency_key: "str | None" = None,
        schema_version: str = "1",
    ) -> Any:
        """Policy-check then run ``handler``, optionally with timeout/retry.

        ``timeout`` wraps each handler call in :func:`asyncio.wait_for`; an
        :class:`asyncio.TimeoutError` is caught like any other exception
        (retried if attempts remain, raised otherwise). ``max_retries`` is the
        number of additional attempts after the first -- on any exception the
        call is retried up to ``max_retries`` times, after which the last error
        is re-raised. Defaults (``timeout=None``, ``max_retries=0``) preserve
        the legacy single-call-no-timeout behavior exactly.

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
          the terminal transition overwrites the prior FAILED record.
        - Same (scope, key) with a different request hash -> reserve raises
          ``IdempotencyConflictError`` (same key reused for different args).
        - Fresh reservation -> handler runs; ``complete`` on success,
          ``fail`` on exception.

        ``schema_version`` (default ``"1"``) is folded into the request hash
        so a ToolSpec whose input contract changed shape bumps its
        schema_version and gets a fresh hash -- a stale idempotency record
        from before the change is never mistaken for a match. Callers that
        have a ``ToolSpec`` in hand should pass ``spec.schema_version`` here;
        omitting it preserves the prior hash formula's default.

        Default ``idempotency_store=None`` (or ``idempotency_key=None``)
        preserves today's no-idempotency behavior: the handler runs every
        call and nothing is persisted."""
        await self.check(request, context)
        # Idempotency is scoped to context.run_id so the same idempotency_key
        # can be reused across different runs without colliding. ``scope`` is
        # part of the request_hash too, so a hash mismatch always reflects a
        # genuine difference in (tool, args, scope) -- not just (tool, args).
        use_idempotency = self._idempotency_store is not None and idempotency_key is not None
        if use_idempotency:
            scope = context.run_id
            request_hash = compute_request_hash(
                request.tool_name, request.arguments, scope,
                schema_version=schema_version,
            )
            existing = await self._idempotency_store.reserve(scope, idempotency_key, request_hash)
            if existing is not None:
                if existing.status is IdempotencyStatus.COMPLETED:
                    return existing.result
                if existing.status is IdempotencyStatus.RESERVED:
                    raise IdempotencyInProgressError(
                        f"idempotent request in progress: key={idempotency_key!r}"
                    )
                # FAILED: fall through to re-execute. The handler runs again
                # and complete()/fail() overwrites the prior terminal record
                # FAILED allows retry per policy.
        arguments = dict(request.arguments)
        last_error: "Exception | None" = None
        for attempt in range(max_retries + 1):
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(handler(**arguments), timeout=timeout)
                else:
                    result = await handler(**arguments)
                if use_idempotency:
                    await self._idempotency_store.complete(scope, idempotency_key, result)
                return result
            except Exception as exc:  # noqa: BLE001 - retry then re-raise
                last_error = exc
                if attempt < max_retries:
                    continue
        if use_idempotency:
            await self._idempotency_store.fail(scope, idempotency_key, str(last_error))
        raise last_error  # type: ignore[misc]
