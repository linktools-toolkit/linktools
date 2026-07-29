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
from typing import TYPE_CHECKING, Any, Mapping, Protocol, TypeAlias, runtime_checkable

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelMessage

from ..model.resolver import ResolvedModel
from ..execution.run import RunErrorInfo, RunResult
from ..tool.pydantic import PolicyCapability
from .spec import AgentSpec


@runtime_checkable
class ModelStreamCapability(Protocol):
    @property
    def supports_streaming(self) -> bool:
        ...


def model_supports_streaming(model: object) -> bool:
    # Explicit flag for custom model wrappers that declare their capability.
    if getattr(model, "supports_streaming", False) is True:
        return True
    # FunctionModel: stream only when a stream_function was provided.
    if hasattr(model, "stream_function"):
        return getattr(model, "stream_function") is not None
    # Real models (OpenAI, Anthropic, etc.) support streaming natively.
    return True

if TYPE_CHECKING:
    from .middleware.capability import MiddlewareCapability


@dataclass(frozen=True, slots=True)
class CompiledAgent:
    spec: AgentSpec
    pydantic_agent: PydanticAgent
    model_bundle: ResolvedModel
    policy_capability: "PolicyCapability | None"
    middleware_capability: "MiddlewareCapability | None" = None


@dataclass(frozen=True, slots=True)
class AgentInput:
    """The AgentEngine-facing execution request (the ``input:
    AgentInput``) -- a dedicated type rather than reusing ``run.models.RunInput``
    directly, so AgentEngine's public surface does not couple to the Run
    domain's own input shape as that shape evolves independently."""

    prompt: str
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
    # Session history RunCoordinator already loaded (via SessionReader) and
    # converted to pydantic-ai's message shape -- AgentEngine folds it into
    # the model prompt but never reads persistence itself. Empty (default)
    # means a new run with no prior turns.
    message_history: "tuple[ModelMessage, ...]" = ()
    # True when this execution resumes a paused run from a checkpoint: the
    # prompt is already baked into ``message_history`` and must not be
    # re-fed alongside it.
    resuming: bool = False


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
    persistence itself. Mirrors the fields
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


@dataclass(frozen=True, slots=True)
class AgentCompleted:
    """AgentEngine ran to a successful final output. The serialized message
    history at completion travels on ``snapshot`` (the trace collector's
    resume snapshot), which the execution service persists alongside the COMPLETED
    transition so a later resume/replay has the same message state a paused
    run would have checkpointed."""

    result: RunResult
    usage: RunUsage
    snapshot: Any = None


@dataclass(frozen=True, slots=True)
class AgentPaused:
    """AgentEngine suspended on a tool call awaiting approval. Carries
    everything RunCoordinator needs to persist the ApprovalRequest and
    snapshot without AgentEngine touching persistence
    itself."""

    request: PauseRequest
    usage: RunUsage
    snapshot: Any = None


@dataclass(frozen=True, slots=True)
class AgentFailed:
    """AgentEngine caught an expected provider/model/tool failure. ``error``
    is already redacted -- callers never need to sanitize it further.
    Configuration/invariant/protocol violations and unknown programming
    errors are NOT reported this way; they propagate as raised exceptions
    instead of becoming an ``AgentFailed``."""

    error: RunErrorInfo
    retryable: bool
    usage: RunUsage
    snapshot: Any = None


@dataclass(frozen=True, slots=True)
class AgentCancelled:
    """An explicit Run cancellation converged cleanly (as opposed to an
    external ``asyncio.CancelledError``, which the engine re-raises after
    cleanup rather than reporting as an outcome)."""

    reason: "str | None"
    usage: RunUsage
    snapshot: Any = None


AgentExecutionOutcome: TypeAlias = (
    AgentCompleted | AgentPaused | AgentFailed | AgentCancelled
)
"""The sole return shape of ``AgentEngine.execute_pure()``: a single awaited
discriminated union, so RunCoordinator can converge run lifecycle
(transition/checkpoint/session/event writes) from ONE outcome object instead
of iterating a stream and inferring state from event shapes. Unlike a flat
dataclass with nullable fields per status, invalid combinations (e.g. a
completed run carrying a pause request) are not constructible -- callers
dispatch with ``isinstance()``."""
