#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SecurityPipeline: the formal extension point for downstream safety audit,
risk assessment, and decision-making on model invocations and tool calls."""

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
from ..utils.freeze import freeze_value
from ..errors import PipelineExecutionError


class PipelineAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    MODIFY = "modify"
    AUDIT_ONLY = "audit_only"
    # after_tool-specific actions: distinct names so a result-level decision is
    # not confused with a before_tool call-level DENY/MODIFY. DENY_RESULT
    # discards the tool's result; MODIFY_RESULT replaces it. Treated as
    # DENY/MODIFY respectively for precedence in CompositeSecurityPipeline.
    DENY_RESULT = "deny_result"
    MODIFY_RESULT = "modify_result"


@dataclass(frozen=True)
class PipelineDecision:
    action: PipelineAction
    reason: "str | None" = None
    modified_payload: Any = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


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
    """Context for a tool-call governance decision. Carries the full identity
    chain a downstream SecurityPipeline needs to audit/decide: the call id, the
    run lineage (root/parent), session, agent, principal (user/tenant/workspace),
    and the capability that contributed the tool. Fields default to None so
    existing constructions keep working when only a subset is available."""

    tool_name: str
    arguments: "Mapping[str, Any]"
    run_id: "str | None" = None
    call_id: "str | None" = None
    root_run_id: "str | None" = None
    parent_run_id: "str | None" = None
    session_id: "str | None" = None
    agent_id: "str | None" = None
    user_id: "str | None" = None
    tenant_id: "str | None" = None
    workspace: Any = None
    capability_kind: "str | None" = None
    capability_name: "str | None" = None
    risk: "str | None" = None
    mutating: "bool | None" = None
    # The tool's parameter JSON schema, so a CompositeSecurityPipeline can
    # re-validate arguments after EACH MODIFY between pipelines (not only the
    # final one). None when no schema is available (validation is then a no-op).
    parameter_schema: "Mapping[str, Any] | None" = None


@dataclass(frozen=True)
class ToolResultEvent:
    tool_name: str
    result: Any
    success: bool = True
    run_id: "str | None" = None
    call_id: "str | None" = None


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


# Maps each event type to the field a MODIFY decision's modified_payload
# replaces when threading it into the next pipeline in the chain.
_MODIFY_FIELD: "dict[type, str]" = {
    ModelInvocationEvent: "prompt",
    ModelResultEvent: "output",
    ToolInvocationEvent: "arguments",
    ToolResultEvent: "result",
}


def _with_modified_payload(event: Any, payload: Any) -> Any:
    field_name = _MODIFY_FIELD.get(type(event))
    if field_name is None:
        return event
    return dataclasses.replace(event, **{field_name: payload})


def validate_tool_decision(decision: PipelineDecision, *, stage: str) -> None:
    before = {
        PipelineAction.ALLOW,
        PipelineAction.DENY,
        PipelineAction.REQUIRE_APPROVAL,
        PipelineAction.MODIFY,
        PipelineAction.AUDIT_ONLY,
    }
    after = {
        PipelineAction.ALLOW,
        PipelineAction.DENY_RESULT,
        PipelineAction.MODIFY_RESULT,
        PipelineAction.AUDIT_ONLY,
    }
    allowed = before if stage == "before" else after
    if decision.action not in allowed:
        raise PipelineExecutionError(
            f"pipeline action {decision.action.value!r} is invalid at {stage}_tool"
        )


class CompositeSecurityPipeline:
    """Composes multiple SecurityPipelines. Decision precedence:
    DENY > REQUIRE_APPROVAL > MODIFY (in order) > ALLOW. AUDIT_ONLY never
    changes the outcome. Only DENY short-circuits the pipeline chain -- every
    pipeline is still consulted after a REQUIRE_APPROVAL so a later pipeline's
    DENY is never masked by an earlier approval requirement."""

    def __init__(self, pipelines: "Sequence[SecurityPipeline]") -> None:
        self._pipelines = tuple(pipelines)

    async def _evaluate(self, hook_name: str, event: Any) -> PipelineDecision:
        # Sequential MODIFY: each pipeline receives the PREVIOUS pipeline's
        # modified payload (not the original event unmodified) -- pipeline B
        # sees pipeline A's edits, not just its own view of the original call.
        # After each MODIFY the payload is re-validated against the tool's
        # parameter schema (carried on the event) so a misbehaving intermediate
        # pipeline is caught before the next one observes an invalid payload.
        current_event = event
        schema = getattr(event, "parameter_schema", None)
        last_modified_payload: Any = None
        require_approval: "PipelineDecision | None" = None
        for p in self._pipelines:
            decision = await getattr(p, hook_name)(current_event)
            if hook_name == "before_tool":
                validate_tool_decision(decision, stage="before")
            elif hook_name == "after_tool":
                validate_tool_decision(decision, stage="after")
            if decision.action in (PipelineAction.DENY, PipelineAction.DENY_RESULT):
                return decision
            if decision.action == PipelineAction.REQUIRE_APPROVAL:
                require_approval = decision
            elif (
                decision.action in (PipelineAction.MODIFY, PipelineAction.MODIFY_RESULT)
                and decision.modified_payload is not None
            ):
                last_modified_payload = decision.modified_payload
                if schema is not None:
                    from ..tool.schema import validate_arguments

                    tool_name = getattr(event, "tool_name", "")
                    # Raise as a DENY so the bad MODIFY never propagates.
                    try:
                        validate_arguments(
                            last_modified_payload, schema, tool_name=tool_name
                        )
                    except Exception as exc:
                        return PipelineDecision(
                            action=PipelineAction.DENY,
                            reason=f"pipeline MODIFY failed schema validation: {exc}",
                        )
                current_event = _with_modified_payload(
                    current_event, last_modified_payload
                )
        if require_approval is not None:
            return require_approval
        return PipelineDecision(
            action=PipelineAction.ALLOW, modified_payload=last_modified_payload
        )

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
