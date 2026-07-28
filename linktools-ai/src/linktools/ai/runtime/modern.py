"""Runtime path backed by the v4 storage contracts."""

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic_ai import Agent as PydanticAgent
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage

from ..agent.spec import AgentSpec
from ..execution.models import RunDefinitionSnapshot, RunKind, RunStatus
from ..execution.store import ExecutionStore
from ..identity.principal import PrincipalContext
from ..model.resolver import ModelResolver
from ..run.cancellation import CancellationToken
from ..run.context import RunContext
from ..run.models import RunnableType
from .storage import RuntimeStorage


class _Events:
    async def emit(self, event: Any) -> None:
        return None

    async def publish(self, event: Any) -> None:
        return None


@dataclass(frozen=True, slots=True)
class ModernRuntime:
    storage: RuntimeStorage
    model_resolver: ModelResolver

    async def run(self, spec: AgentSpec, prompt: str, *, session_id: str | None = None, run_id: str | None = None, user_id: str | None = None, tenant_id: str | None = None, **_: Any) -> Any:
        from ..agent.dependencies import AgentDependencies
        from ..agent.engine import AgentEngine
        from ..agent.models import AgentCompleted, AgentInput, CompiledAgent

        if session_id is None:
            session_id = run_id or "session-" + prompt[:16]
        if await self.storage.execution.get_session(session_id) is None:
            await self.storage.execution.create_session(session_id=session_id, user_id=user_id, tenant_id=tenant_id)
        raw_messages = await self.storage.execution.load_session_context(session_id)
        messages = tuple(TypeAdapter(ModelMessage).validate_python(message) for message in raw_messages)
        run_id = run_id or uuid4().hex
        record = await self.storage.execution.start_run(run_id=run_id, session_id=session_id, kind=RunKind.USER_TURN, definition=RunDefinitionSnapshot(spec.id, spec.model.primary), user_prompt=prompt)
        claimed = await self.storage.execution.claim_run(run_id, owner="runtime")
        resolved = self.model_resolver.resolve(spec.model)
        agent = PydanticAgent(resolved.model, output_type=spec.output_schema or str, deps_type=AgentDependencies, instructions=spec.instructions.instructions)
        compiled = CompiledAgent(spec=spec, pydantic_agent=agent, model_bundle=resolved, policy_capability=None)
        context = RunContext(run_id=run_id, root_run_id=run_id, parent_run_id=None, session_id=session_id, runnable_id=spec.id, runnable_type=RunnableType.AGENT, user_id=user_id, tenant_id=tenant_id, workspace=None)
        engine = AgentEngine(trace_store=self.storage.execution)
        outcome = await engine.execute_pure(compiled, AgentInput(prompt=prompt, message_history=tuple(messages)), context, cancellation=CancellationToken(), live_events=_Events(), security_events=_Events())
        snapshot = getattr(outcome, "snapshot", None)
        if snapshot is None:
            raise RuntimeError("agent outcome did not produce a run snapshot")
        if isinstance(outcome, AgentCompleted):
            await self.storage.execution.complete_run(run_id, owner="runtime", fence=claimed.execution_fence, snapshot=snapshot)
            return outcome.result.output
        if outcome.__class__.__name__ == "AgentCancelled":
            await self.storage.execution.acknowledge_cancel(run_id, owner="runtime", fence=claimed.execution_fence, snapshot=snapshot)
            return None
        await self.storage.execution.fail_run(run_id, owner="runtime", fence=claimed.execution_fence, snapshot=snapshot)
        raise RuntimeError(getattr(getattr(outcome, "error", None), "message", "agent run failed"))

    async def aclose(self) -> None:
        return None

    async def run_stream(self, spec: AgentSpec, prompt: str, **kwargs: Any):
        result = await self.run(spec, prompt, **kwargs)
        yield {"type": "run.completed", "output": result}

    async def resume(self, run_id: str):
        raise RuntimeError(f"run {run_id} cannot be resumed without its AgentSpec")

    async def cancel(self, run_id: str) -> None:
        record = await self.storage.execution.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        await self.storage.execution.request_cancel(
            run_id, owner="runtime", fence=record.execution_fence
        )

    async def inspect(self, spec: AgentSpec) -> dict[str, Any]:
        return {"agent_id": spec.id, "model": spec.model.primary}

    async def __aenter__(self) -> "ModernRuntime":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
