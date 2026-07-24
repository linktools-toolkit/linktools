#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CompiledAgent: the stateless output of AgentCompiler.compile(). Reusable
across many Runs -- no Session, no Run, no Checkpoint, no Workspace, and no
mutable per-Run fields anywhere on its
capabilities. policy_capability/middleware_capability are the SAME instances
already inside pydantic_agent's capabilities=[...] list; the per-Run
ToolContext reaches them via pydantic-ai dependency injection
(``deps=AgentDependencies(...)`` -> ``ctx.deps.tool_context``), so one
CompiledAgent is safe to share across concurrent Runs."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelMessage

from ..model.resolver import ResolvedModel
from ..run.models import RunErrorInfo, RunResult
from ..tool.pydantic import PolicyCapability
from .spec import AgentSpec

if TYPE_CHECKING:
    from ..middleware.capability import MiddlewareCapability


@dataclass(frozen=True, slots=True)
class CompiledAgent:
    spec: AgentSpec
    pydantic_agent: PydanticAgent
    model_bundle: ResolvedModel
    policy_capability: PolicyCapability
    middleware_capability: "MiddlewareCapability | None" = None


@dataclass(frozen=True, slots=True)
class AgentInput:
    """The AgentEngine-facing execution request (spec 12.4's ``input:
    AgentInput``) -- a dedicated type rather than reusing ``run.models.RunInput``
    directly, so AgentEngine's public surface does not couple to the Run
    domain's own input shape as that shape evolves independently."""

    prompt: str
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunUsage:
    """Token/cost usage summary an execution produced. A typed replacement
    for the free-form ``token_usage: Mapping`` carried on ``run.models.RunResult``
    -- ``AgentExecutionOutcome.usage`` reports this directly rather than via an
    untyped mapping."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: "float | None" = None


@dataclass(frozen=True, slots=True)
class PauseRequest:
    """Everything RunCoordinator needs to persist an ApprovalRequest and
    checkpoint on a PAUSED outcome, without AgentEngine touching
    ApprovalStore/CheckpointStore itself. Mirrors the fields
    ``errors.RunPaused`` already carries (see errors.py) -- this is the
    typed, Store-free equivalent surfaced on ``AgentExecutionOutcome``
    instead of the exception, once AgentEngine stops raising it as control
    flow."""

    approval_id: str
    tool_call_id: "str | None" = None
    tool_name: "str | None" = None
    reason: "str | None" = None
    arguments: "Mapping[str, Any]" = field(default_factory=dict)
    idempotency_key: "str | None" = None
    binding: "Mapping[str, Any]" = field(default_factory=dict)


class AgentExecutionStatus(str, Enum):
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AgentExecutionOutcome:
    """The sole return shape of ``AgentEngine.execute()`` under the section
    12 API (spec 12.3/12.4): a single awaited value replacing today's
    async-generator-of-dict-events shape, so RunCoordinator can converge run
    lifecycle (transition/checkpoint/session/event writes) from ONE outcome
    object instead of iterating a stream and inferring state from event
    shapes. Field combinations are constrained by ``status``:

    - COMPLETED: ``result`` is set, ``pause_request``/``error`` are None.
    - PAUSED: ``pause_request`` is set, ``result``/``error`` are None.
    - FAILED: ``error`` is set, ``result``/``pause_request`` are None.
    - CANCELLED: ``result``/``pause_request``/``error`` are all typically
      None (a cancellation has no output to report).

    This dataclass does not itself enforce that constraint (no validating
    ``__post_init__`` yet) -- RunCoordinator's convergence step is what
    switches on ``status``; a future increment may add the check once that
    consumer exists."""

    status: AgentExecutionStatus
    result: "RunResult | None" = None
    pause_request: "PauseRequest | None" = None
    error: "RunErrorInfo | None" = None
    messages: "tuple[ModelMessage, ...]" = ()
    usage: "RunUsage | None" = None
