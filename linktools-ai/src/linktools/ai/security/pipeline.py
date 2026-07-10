#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SecurityPipeline: the formal extension point for downstream safety audit,
risk assessment, and decision-making on model invocations and tool calls."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class PipelineAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    MODIFY = "modify"
    AUDIT_ONLY = "audit_only"


@dataclass(frozen=True)
class PipelineDecision:
    action: PipelineAction
    reason: "str | None" = None
    modified_payload: Any = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


# Event types passed to pipeline hooks.
@dataclass(frozen=True)
class ModelInvocationEvent:
    prompt: str
    run_id: "str | None" = None
    agent_id: "str | None" = None


@dataclass(frozen=True)
class ModelResultEvent:
    output: Any
    run_id: "str | None" = None


@dataclass(frozen=True)
class ToolInvocationEvent:
    tool_name: str
    arguments: "Mapping[str, Any]"
    run_id: "str | None" = None


@dataclass(frozen=True)
class ToolResultEvent:
    tool_name: str
    result: Any
    success: bool = True
    run_id: "str | None" = None


@dataclass(frozen=True)
class SecurityEvent:
    kind: str
    detail: str
    run_id: "str | None" = None


@runtime_checkable
class SecurityPipeline(Protocol):
    """Extension point for downstream safety audit/decision. Each hook returns a
    PipelineDecision; the Runtime/ToolExecutor honors DENY > REQUIRE_APPROVAL >
    MODIFY/ALLOW."""

    async def before_model(self, event: ModelInvocationEvent) -> PipelineDecision: ...
    async def after_model(self, event: ModelResultEvent) -> PipelineDecision: ...
    async def before_tool(self, event: ToolInvocationEvent) -> PipelineDecision: ...
    async def after_tool(self, event: ToolResultEvent) -> PipelineDecision: ...
    async def on_security_event(self, event: SecurityEvent) -> PipelineDecision: ...


class CompositeSecurityPipeline:
    """Composes multiple SecurityPipelines. Decision precedence:
    DENY > REQUIRE_APPROVAL > MODIFY (in order) > ALLOW. AUDIT_ONLY never
    changes the outcome."""

    def __init__(self, pipelines: "Sequence[SecurityPipeline]") -> None:
        self._pipelines = tuple(pipelines)

    async def _evaluate(self, hook_name: str, event: Any) -> PipelineDecision:
        modifications: "list[Any]" = []
        for p in self._pipelines:
            decision = await getattr(p, hook_name)(event)
            if decision.action == PipelineAction.DENY:
                return decision
            if decision.action == PipelineAction.REQUIRE_APPROVAL:
                return decision
            if decision.action == PipelineAction.MODIFY and decision.modified_payload is not None:
                modifications.append(decision.modified_payload)
        final_payload = modifications[-1] if modifications else None
        return PipelineDecision(action=PipelineAction.ALLOW, modified_payload=final_payload)

    async def before_model(self, event: ModelInvocationEvent) -> PipelineDecision:
        return await self._evaluate("before_model", event)

    async def after_model(self, event: ModelResultEvent) -> PipelineDecision:
        return await self._evaluate("after_model", event)

    async def before_tool(self, event: ToolInvocationEvent) -> PipelineDecision:
        return await self._evaluate("before_tool", event)

    async def after_tool(self, event: ToolResultEvent) -> PipelineDecision:
        return await self._evaluate("after_tool", event)

    async def on_security_event(self, event: SecurityEvent) -> PipelineDecision:
        return await self._evaluate("on_security_event", event)
