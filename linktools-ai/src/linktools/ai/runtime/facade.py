"""Public runtime facade over execution orchestration and authorized queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Any

from ..agent.spec import AgentSpec
from ..execution.query import ExecutionQueryService, RunDetailView
from .executor import RuntimeExecutor
from ..governance.identity import PrincipalContext


@dataclass(frozen=True, slots=True)
class Runtime:
    execution: RuntimeExecutor
    query: ExecutionQueryService

    async def run(self, spec: AgentSpec, prompt: str, *, session_id: str | None = None, **kwargs: Any) -> object:
        return await self.execution.run(spec, prompt, session_id=session_id or "session", **kwargs)

    async def run_stream(self, spec: AgentSpec, prompt: str, *, session_id: str | None = None, **kwargs: Any) -> AsyncIterator[dict[str, object]]:
        result = await self.run(spec, prompt, session_id=session_id, **kwargs)
        yield {"type": "run.completed", "output": result}

    async def resume(self, run_id: str) -> object:
        return await self.execution.resume(run_id)

    async def cancel(self, run_id: str) -> None:
        await self.execution.cancel(run_id)

    async def inspect(self, *, run_id: str, principal: PrincipalContext) -> RunDetailView:
        return await self.query.get_run_detail(run_id=run_id, principal=principal)

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


__all__ = ["Runtime"]
