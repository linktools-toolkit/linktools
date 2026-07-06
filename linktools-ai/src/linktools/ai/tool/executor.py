#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolExecutor: consults PolicyEngine before a tool executes, translating
its decision into the corresponding domain error.

When ``approval_store`` is wired and policy returns REQUIRE_APPROVAL, the
executor persists a PENDING ApprovalRequest and (if ``event_store`` is
also wired) emits an ApprovalRequested event -- both BEFORE the
ToolApprovalRequiredError is raised, so PolicyCapability still translates
the raise into SkipToolExecution and the model sees the "approval needed"
tool result. Default-None (no stores wired) preserves today's behavior
identically: just raise, no persistence, no event."""
import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..agent.approval import ApprovalStatus, build_approval_request
from ..errors import RunPaused, ToolApprovalRequiredError, ToolDeniedError
from ..events.payloads import ApprovalRequested
from ..policy.engine import PolicyDecisionKind, PolicyEngine, ToolContext, ToolRequest

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
        idempotency_cache: "dict[tuple[str, str], Any] | None" = None,
    ) -> None:
        self._policy = policy
        self._approval_store = approval_store
        self._event_store = event_store
        self._run_id_resolver = run_id_resolver
        self._idempotency_cache = idempotency_cache
        # When True, the require-approval branch persists the request (via
        # _record_approval) and then raises RunPaused(run_id, approval_id)
        # INSTEAD of ToolApprovalRequiredError. RunPaused is a RunError (not a
        # ToolError) so PolicyCapability's catch list does not translate it
        # into SkipToolExecution -- it propagates out of pydantic-ai's
        # tool-execution stack to AgentRunner (Tasks 6-7 catch it). Default
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
            approval = await self._record_approval(request, context, decision.reason)
            if self._pause_on_approval and approval is not None:
                run_id = (
                    self._run_id_resolver(context)
                    if self._run_id_resolver is not None
                    else context.run_id
                )
                raise RunPaused(run_id=run_id, approval_id=approval.id)
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

        The return value lets the ``pause_on_approval=True`` branch reach the
        persisted ``approval.id`` for ``RunPaused(approval_id=...)`` without
        re-doing the run_id resolution / build work.

        Event emission is best-effort: the EventStore assigns the sequence
        itself (review doc §8.1), so a failure here is logged and swallowed --
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
    ) -> Any:
        """Policy-check then run ``handler``, optionally with timeout/retry.

        ``timeout`` wraps each handler call in :func:`asyncio.wait_for`; an
        :class:`asyncio.TimeoutError` is caught like any other exception
        (retried if attempts remain, raised otherwise). ``max_retries`` is the
        number of additional attempts after the first -- on any exception the
        call is retried up to ``max_retries`` times, after which the last error
        is re-raised. Defaults (``timeout=None``, ``max_retries=0``) preserve
        the legacy single-call-no-timeout behavior exactly.

        ``idempotency_key`` enables tool-level idempotency (spec section 27,
        basic form): when the executor was constructed with an
        ``idempotency_cache`` dict AND ``idempotency_key`` is provided, the
        cache is keyed by ``(request.tool_name, idempotency_key)``. A repeat
        call with the same key returns the cached result without invoking the
        handler; the first successful call stores its result. A failed
        (exception-raising) call is NOT cached, so a retry with the same key
        re-invokes the handler. This is deliberately MINIMAL -- an in-process
        dict, no hash-based conflict detection, no persistence: the same key
        with different arguments simply returns the cached result. The full
        spec (hash-based conflict, persistent IdempotencyStore, TTL) is a
        future enhancement. For production the caller may supply a shared or
        TTL-backed dict. Default ``idempotency_cache=None`` (or
        ``idempotency_key=None``) preserves today's no-cache behavior."""
        await self.check(request, context)
        cache_key: "tuple[str, str] | None" = None
        if self._idempotency_cache is not None and idempotency_key is not None:
            cache_key = (request.tool_name, idempotency_key)
            if cache_key in self._idempotency_cache:
                return self._idempotency_cache[cache_key]
        arguments = dict(request.arguments)
        last_error: "Exception | None" = None
        for attempt in range(max_retries + 1):
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(handler(**arguments), timeout=timeout)
                else:
                    result = await handler(**arguments)
                if cache_key is not None:
                    self._idempotency_cache[cache_key] = result
                return result
            except Exception as exc:  # noqa: BLE001 - retry then re-raise
                last_error = exc
                if attempt < max_retries:
                    continue
        raise last_error  # type: ignore[misc]
