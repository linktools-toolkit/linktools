"""Feature provider contract and per-execution context."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..sandbox.protocols import Sandbox
from .models import AgentContribution, AgentFeatureRef


@runtime_checkable
class AgentAssemblyEventSink(Protocol):
    async def emit(self, event: object) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentFeatureContext:
    agent_id: str
    execution_id: str
    root_execution_id: str
    parent_execution_id: str | None
    session_id: str
    tenant_id: str | None
    user_id: str | None
    workspace: str | None
    sandbox: Sandbox | None
    events: AgentAssemblyEventSink | None = None


@runtime_checkable
class AgentFeatureProvider(Protocol):
    supported_kinds: tuple[str, ...]

    async def resolve(
        self,
        ref: AgentFeatureRef,
        context: AgentFeatureContext,
    ) -> AgentContribution: ...


def provider_kinds(provider: AgentFeatureProvider) -> frozenset[str]:
    kinds = provider.supported_kinds
    if not kinds or any(not kind.strip() for kind in kinds):
        from ...errors import AgentAssemblyError

        raise AgentAssemblyError(
            "AgentFeatureProvider.supported_kinds must contain non-empty strings"
        )
    return frozenset(kinds)
