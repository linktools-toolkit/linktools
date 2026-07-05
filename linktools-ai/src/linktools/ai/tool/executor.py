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
import itertools
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..agent_runtime.approval import build_approval_request
from ..errors import ToolApprovalRequiredError, ToolDeniedError
from ..events.envelope import EventEnvelope
from ..events.payloads import ApprovalRequested
from ..policy.engine import PolicyDecisionKind, PolicyEngine, ToolContext, ToolRequest

if TYPE_CHECKING:
    from ..agent_runtime.approval import ApprovalStore
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
    ) -> None:
        self._policy = policy
        self._approval_store = approval_store
        self._event_store = event_store
        self._run_id_resolver = run_id_resolver
        # Per-executor monotonic counter for event sequences. The executor is
        # compiled once and reused across many runs; sequence uniqueness is
        # enforced per-run at the storage layer (paths/rows are keyed by
        # (run_id, sequence)), so a single counter advancing across runs is
        # safe -- each run sees only its own events.
        self._event_sequence = itertools.count(1)

    async def check(self, request: ToolRequest, context: ToolContext) -> None:
        decision = await self._policy.evaluate(request, context)
        if decision.kind == PolicyDecisionKind.DENY:
            raise ToolDeniedError(decision.reason or f"tool denied: {request.tool_name}")
        if decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL:
            await self._record_approval(request, context, decision.reason)
            raise ToolApprovalRequiredError(
                decision.reason or f"tool requires approval: {request.tool_name}"
            )

    async def _record_approval(
        self,
        request: ToolRequest,
        context: ToolContext,
        reason: "str | None",
    ) -> None:
        """Persist a PENDING ApprovalRequest (and emit ApprovalRequested) when
        an approval_store is wired. Best-effort audit: event-emission failures
        are logged and swallowed (the approval record is the source of truth);
        the caller (check) still raises ToolApprovalRequiredError afterward
        regardless of outcome here. With approval_store=None this is a no-op
        (default-None path: behavior identical to today)."""
        if self._approval_store is None:
            return

        run_id = (
            self._run_id_resolver(context) if self._run_id_resolver is not None
            else context.run_id
        )
        # ToolContext carries no per-call id, so we mint a fresh uuid4 for
        # tool_call_id. If a future ToolContext gains a call_id field, prefer
        # that; for now a fresh uuid is the documented behavior.
        tool_call_id = str(uuid.uuid4())
        approval = build_approval_request(
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=request.tool_name,
            reason=reason,
            arguments=request.arguments,
        )
        await self._approval_store.create(approval)

        if self._event_store is not None:
            envelope = EventEnvelope(
                event_id=f"{approval.id}-requested",
                sequence=next(self._event_sequence),
                occurred_at=datetime.now(timezone.utc),
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
            try:
                await self._event_store.append(envelope)
            except Exception as exc:  # noqa: BLE001 - best-effort audit
                _LOGGER.warning(
                    "failed to append ApprovalRequested event for approval %s: %s",
                    approval.id,
                    exc,
                )

    async def execute(
        self,
        request: ToolRequest,
        context: ToolContext,
        handler: "Callable[..., Awaitable[Any]]",
    ) -> Any:
        await self.check(request, context)
        return await handler(**dict(request.arguments))
