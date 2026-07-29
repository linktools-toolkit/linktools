"""Public runtime facade over execution orchestration and authorized queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ..agent.spec import AgentSpec
from ..execution.domain import ApprovalDecision, RunRecord
from ..execution.query import ExecutionQueryService, ExecutionDetailView
from ..execution.service import ExecutionService
from ..governance.identity import PrincipalContext


@dataclass(frozen=True, slots=True)
class Runtime:
    execution: ExecutionService
    query: ExecutionQueryService

    async def run(self, spec: AgentSpec, prompt: str, *, session_id: str | None = None, **kwargs: Any) -> object:
        # A session is never implicitly shared: omitting session_id starts an
        # isolated single-turn session rather than collapsing every anonymous
        # run onto a fixed "session" id. The streaming facade method was removed: there is no real
        # event-subscription infrastructure yet, so the facade exposes only the
        # scalar run/resume/cancel surface.
        return await self.execution.run(spec, prompt, session_id=session_id or uuid4().hex, **kwargs)

    async def resume(self, run_id: str, *, user_id: str | None = None, tenant_id: str | None = None) -> object:
        return await self.execution.resume(run_id, user_id=user_id, tenant_id=tenant_id)

    async def cancel(self, run_id: str, *, user_id: str | None = None, tenant_id: str | None = None) -> None:
        await self.execution.cancel(run_id, user_id=user_id, tenant_id=tenant_id)

    async def decide_approval(self, run_id: str, *, approval_id: str, decision: ApprovalDecision, decided_by: str, user_id: str | None = None, tenant_id: str | None = None) -> RunRecord:
        return await self.execution.decide_approval(run_id, approval_id=approval_id, decision=decision, decided_by=decided_by, user_id=user_id, tenant_id=tenant_id)

    async def inspect(self, *, run_id: str, principal: PrincipalContext) -> ExecutionDetailView:
        return await self.query.get_run_detail(run_id=run_id, principal=principal)

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


__all__ = ["Runtime"]
