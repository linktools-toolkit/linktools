"""Public runtime facade over execution orchestration and authorized queries."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from ..agent.spec import AgentSpec
from ..agent.assembly.assembler import AgentAssembler
from ..agent.sandbox.protocols import Sandbox
from ..agent.mcp.connection import MCPConnectionPool
from ..errors import PrincipalAccessDeniedError
from ..execution.domain import ApprovalDecision, RunRecord
from ..execution.query import ExecutionQueryService, ExecutionDetailView
from ..execution.service import ExecutionService
from ..governance.identity import PrincipalContext


@dataclass(frozen=True, slots=True)
class Runtime:
    execution: ExecutionService
    query: ExecutionQueryService
    assembler: AgentAssembler
    tool_execution_ready: bool
    sandbox: Sandbox | None = None
    mcp_connections: MCPConnectionPool | None = None

    async def run(
        self,
        spec: AgentSpec,
        prompt: str,
        *,
        principal: PrincipalContext,
        session_id: str | None = None,
        execution_id: str | None = None,
    ) -> object:
        # A session is never implicitly shared: omitting session_id starts an
        # isolated single-turn session rather than collapsing every anonymous
        # run onto a fixed "session" id. The streaming facade method was removed: there is no real
        # event-subscription infrastructure yet, so the facade exposes only the
        # scalar run/resume/cancel surface.
        if not isinstance(principal, PrincipalContext):
            raise PrincipalAccessDeniedError(
                "a valid PrincipalContext is required"
            )
        resolved_session_id = session_id or uuid4().hex
        resolved_execution_id = execution_id or uuid4().hex
        return await self.execution.run(
            spec,
            prompt,
            principal=principal,
            session_id=resolved_session_id,
            execution_id=resolved_execution_id,
        )

    async def resume(self, execution_id: str, *, principal: PrincipalContext) -> object:
        return await self.execution.resume(execution_id, principal=principal)

    async def cancel(self, execution_id: str, *, principal: PrincipalContext) -> None:
        await self.execution.cancel(execution_id, principal=principal)

    async def decide_approval(
        self,
        execution_id: str,
        *,
        approval_id: str,
        decision: ApprovalDecision,
        principal: PrincipalContext,
    ) -> RunRecord:
        return await self.execution.decide_approval(
            execution_id,
            approval_id=approval_id,
            decision=decision,
            principal=principal,
        )

    async def inspect(self, *, run_id: str, principal: PrincipalContext) -> ExecutionDetailView:
        return await self.query.get_run_detail(run_id=run_id, principal=principal)

    async def aclose(self) -> None:
        if self.mcp_connections is not None:
            await self.mcp_connections.close()

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


__all__ = ["Runtime"]
