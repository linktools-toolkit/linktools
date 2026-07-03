#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared base class for all agent types."""

import abc
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.exceptions import (
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)

from .support.utils import model_type as _model_type, call_id as _call_id
from .core.registry import AgentSpec
from .mcp.registry import MCPServerSpec
from .skill.registry import SkillSpec
from .subagent.registry import SubagentSpec
from .session.types import Session, SessionTurn, build_input_display_entries
from .core.model_runtime import (
    ModelOutputError,
    ModelTurnLimitExceeded,
    RuntimeModelConfig,
    build_model,
)
from .mcp.client import build_mcp_toolset
from .core.prompt import PromptContext, build_prompt, unavailable_adapters_note
from .core.runtime import AgentExecutionContext, AgentKernel
from .core.run import RuntimeRunCapability
from .skill.view import view_skill, view_available_skills
from .execution.local import LocalExecutionBackend
from .execution.toolset import BuiltinToolContext, HookedBuiltinToolset, build_builtin_toolset
from pydantic_ai.capabilities import MCP
from .mcp.capability import HookedMCPCapability
from .skill.capability import SkillCapability
from .subagent.capability import SubagentCapability
from .security.hook import SecurityCapability
from .stuck_loop.capability import StuckLoopCapability
from .periodic_reminder.capability import PeriodicReminderCapability
from .budget.hook import BudgetCapability
from .budget.tracker import BudgetTracker
from .tool_search.capability import ToolSearchCapability
from .plan.capability import PlanCapability
from .memory.capability import MemoryCapability
from .swarm.capability import SwarmCapability

if TYPE_CHECKING:
    from .checkpoint.protocols import CheckpointStore
    from .session.artifact import AgentArtifactStore
    from .swarm.protocols import TaskQueue


class BaseAgent(metaclass=abc.ABCMeta):
    """Abstract agent base.

    The `Session` is the canonical execution context for this agent instance.
    Pipeline wrappers may rebind `self.session` before each invocation when the
    same agent instance is reused across traces.
    """

    spec: AgentSpec
    session: Session
    execution_context: AgentExecutionContext
    kernel: AgentKernel
    model_config_resolver: "Callable[[str], RuntimeModelConfig]"

    def __init__(
        self,
        spec: AgentSpec,
        session: Session,
        execution_context: AgentExecutionContext,
        *,
        model_config_resolver: "Callable[[str], RuntimeModelConfig]",
        enable_stuck_loop_detection: bool = False,
        enable_periodic_reminders: bool = False,
        enable_tool_search: bool = False,
        enable_security_preset: bool = True,
        budget_usd: "float | None" = None,
        task_queue: "TaskQueue | None" = None,
        checkpoint_store: "CheckpointStore | None" = None,
        enable_checkpointing: bool = False,
        enable_plan_mode: bool = False,
        enable_memory: bool = False,
        # Accepted per the spec's full constructor signature but still genuinely
        # inert -- nothing in this codebase reads self.fallback_models or
        # self.context_files beyond storing them here.
        fallback_models: "tuple[str, ...]" = (),
        context_files: "tuple[str, ...]" = ("AGENTS.md", "CLAUDE.md"),
    ) -> None:
        self.spec = spec
        self.session = session
        self.execution_context = execution_context
        self.kernel = execution_context.kernel
        self.model_config_resolver = model_config_resolver
        self.enable_stuck_loop_detection = enable_stuck_loop_detection
        self.enable_periodic_reminders = enable_periodic_reminders
        self.enable_tool_search = enable_tool_search
        self.enable_security_preset = enable_security_preset
        self.budget_usd = budget_usd
        self.task_queue = task_queue
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_store = checkpoint_store
        self.enable_plan_mode = enable_plan_mode
        self.enable_memory = enable_memory
        self.fallback_models = fallback_models
        self.context_files = context_files

    @property
    def agent_id(self) -> str:
        return self.spec.name

    @property
    def agent_model(self) -> str:
        return self.spec.model

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @abc.abstractmethod
    async def generate(
        self,
        inputs: Any,
        *,
        image_inputs: "list[str] | None" = None,
        call_id: "str | None" = None,
    ) -> "dict[str, Any]":
        """Generate a complete result for `inputs` within `self.session`."""
        ...

    @abc.abstractmethod
    async def stream(
        self,
        inputs: Any,
        *,
        image_inputs: "list[str] | None" = None,
        call_id: "str | None" = None,
    ) -> "AsyncIterator[dict[str, Any]]":
        """Stream incremental events for `inputs` when supported by the agent."""
        ...

    @abc.abstractmethod
    def snapshot(self) -> "dict[str, object]":
        """Runtime capability summary for web display and trace reports."""
        ...


@dataclass(slots=True)
class LlmCallContext:
    agent: Agent
    bundle: Any
    prompt: str
    history: "list[Any]"
    model_type: str
    system_prompt: str
    llm_call_id: str
    parent_call_id: "str | None"
    mcp_specs: "list[MCPServerSpec]"
    capability: "RuntimeRunCapability"


@dataclass(frozen=True, slots=True)
class LlmCallRequest:
    model_type: str
    prompt: str
    parent_call_id: "str | None"
    system_prompt: "str | None"
    skills: "list[SkillSpec]"
    subagents: "list[SubagentSpec]"
    output_type: Any
    call_id: "str | None" = None


@dataclass(frozen=True, slots=True)
class LlmRunResult:
    messages: "list[Any]"
    token_usage: "dict[str, Any]"
    usage: Any
    display_entries: "list[dict[str, Any]] | None" = None


@dataclass(frozen=True, slots=True)
class BuiltLlmAgent:
    agent: Agent
    bundle: Any
    prompt: str
    mcp_specs: "list[MCPServerSpec]"
    capability: "RuntimeRunCapability"


@dataclass(frozen=True, slots=True)
class LlmCallOutcome:
    started_at: float
    success: bool
    response: Any
    error: "str | None"
    token_usage: "dict[str, Any]"
    error_detail: "dict[str, Any] | None" = None


@dataclass(frozen=True, slots=True)
class LlmCallRecord:
    call_id: str
    model_type: str
    model: "str | None"
    token_usage: "dict[str, Any]"
    requests: int

    def as_dict(self) -> "dict[str, Any]":
        return {
            "call_id": self.call_id,
            "model_type": self.model_type,
            "model": self.model,
            "token_usage": self.token_usage,
            "requests": self.requests,
        }


class LlmAgent(BaseAgent):
    """Base LLM agent: prompt + skills + MCP + subagents, with structured generation.

    This is the shared LLM base for *every* agent — pipeline workers/stages and report
    agents all drive its structured `generate()` path; it is not "conversation-only". It
    also provides `stream()` for streamed conversational turns. It carries no execution
    environment (no file/terminal tools, no runtime working directory in the prompt).
    `RuntimeAgent` specializes it with the execution env.
    """

    #: Structured output model for this agent type. `None` → free-form dict output.
    OUTPUT_MODEL: "type[BaseModel] | None" = None

    async def generate(
        self,
        inputs: Any,
        *,
        image_inputs: "list[str] | None" = None,
        call_id: "str | None" = None,
    ) -> "dict[str, Any]":
        prompt, skills, subagents, agent_call_id = self._build_prompt(
            self,
            inputs,
            call_id=call_id,
        )
        # Agent prompts (capabilities/**/agent.md) instruct the model to emit the
        # AgentResult as plain JSON text, not a tool call. PromptedOutput keeps that
        # contract (allow_text_output=True -> tool_choice stays "auto"); the default
        # ToolOutput mode would force a dedicated output tool call, and models that
        # follow the prompt and reply with plain JSON instead would loop on output
        # retries until UsageLimits.request_limit is exhausted.
        ctx = await self._build_call_context(
            LlmCallRequest(
                model_type=_model_type(self.agent_model),
                prompt=prompt,
                call_id=call_id,
                system_prompt=self.spec.system_prompt or None,
                parent_call_id=agent_call_id,
                skills=skills,
                subagents=subagents,
                output_type=self._default_output_type(),
            )
        )
        t = time.monotonic()
        success = True
        payload: "dict[str, Any] | None" = None
        error: "str | None" = None
        error_detail: "dict[str, Any] | None" = None
        token_usage: "dict[str, Any]" = {}
        await self.session.set_status("busy")
        self._emit_call_started(ctx, image_inputs=image_inputs)
        try:
            result = await ctx.agent.run(
                self._build_user_prompt(ctx.prompt, image_inputs),
                message_history=ctx.history,
                usage_limits=ctx.bundle.usage_limits,
            )
            payload = self._coerce_output(result.output)
            usage = result.usage
            token_usage = self._usage_summary(usage)
            await self._save_call(
                ctx,
                LlmRunResult(
                    messages=result.all_messages(),
                    token_usage=token_usage,
                    usage=usage,
                ),
            )
            return payload
        except asyncio.CancelledError:
            success = False
            error = "cancelled"
            error_detail = ctx.capability.last_error_detail
            raise
        except UsageLimitExceeded as exc:
            success = False
            error = str(exc)
            raise ModelTurnLimitExceeded(f"{ctx.model_type}: usage/turn limit exceeded: {exc}") from exc
        except (UnexpectedModelBehavior, ModelHTTPError) as exc:
            success = False
            error = str(exc)
            error_detail = ctx.capability.last_error_detail
            raise ModelOutputError(f"{ctx.model_type}: model loop failed: {exc}") from exc
        except Exception as exc:
            success = False
            error = str(exc)
            error_detail = ctx.capability.last_error_detail
            raise
        finally:
            self._emit_call_finished(
                ctx,
                LlmCallOutcome(
                    started_at=t,
                    success=success,
                    response=payload,
                    error=error,
                    token_usage=token_usage,
                    error_detail=error_detail,
                ),
            )
            await self.session.set_status("idle" if success else "error", message=error)

    async def stream(
        self,
        inputs: Any,
        *,
        image_inputs: "list[str] | None" = None,
        call_id: "str | None" = None,
    ) -> "AsyncIterator[dict[str, Any]]":
        """Stream a conversational turn as structured events, persisting the completed turn.

        Mirrors `generate()` setup (prompt/skills/subagents/MCP toolsets/history) but drives the
        agent with `agent.iter()` so the graph runs to completion while we stream both text
        and tool activity. Yields dict events:

        - ``{"type": "text", "text": <delta>}`` — incremental answer text.
        - ``{"type": "tool", "name": <tool>, "phase": "start"|"end", "ok": <bool|None>}`` —
          a tool call beginning / finishing (``ok`` set on ``end``).

        Unlike `run_stream(output_type=str)` — which treats the first text output as the
        final result and neither continues the tool loop nor exposes tool events — `iter()`
        runs every tool turn and lets us surface them, so a follow-up that consults data
        adapters streams progress instead of appearing to hang.
        """
        from pydantic_ai.messages import (
            FunctionToolCallEvent,
            FunctionToolResultEvent,
            PartDeltaEvent,
            PartStartEvent,
            TextPart,
            TextPartDelta,
            ToolReturnPart,
        )

        prompt, skills, subagents, agent_call_id = self._build_prompt(
            self,
            inputs,
            call_id=call_id,
        )
        ctx = await self._build_call_context(
            LlmCallRequest(
                model_type=_model_type(self.agent_model),
                prompt=prompt,
                parent_call_id=agent_call_id,
                system_prompt=self.spec.system_prompt or None,
                skills=skills,
                subagents=subagents,
                output_type=str,
            )
        )

        t = time.monotonic()
        success = True
        error: "str | None" = None
        accumulated = ""
        token_usage: "dict[str, Any]" = {}
        await self.session.set_status("busy")
        self._emit_call_started(ctx, image_inputs=image_inputs)
        try:
            async with ctx.agent.iter(
                self._build_user_prompt(ctx.prompt, image_inputs),
                message_history=ctx.history,
                usage_limits=ctx.bundle.usage_limits,
            ) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                                    text = event.part.content
                                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                    text = event.delta.content_delta
                                else:
                                    continue
                                if text:
                                    accumulated += text
                                    yield {"type": "text", "text": text}
                    elif Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as tool_stream:
                            async for event in tool_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    yield {"type": "tool", "name": event.part.tool_name, "phase": "start", "ok": None}
                                elif isinstance(event, FunctionToolResultEvent):
                                    yield {
                                        "type": "tool",
                                        "name": event.part.tool_name,
                                        "phase": "end",
                                        "ok": isinstance(event.part, ToolReturnPart),
                                    }
                result = run.result
                usage = run.usage
                token_usage = self._usage_summary(usage)
                messages = result.all_messages() if result is not None else ctx.history
            await self._save_call(
                ctx,
                LlmRunResult(
                    messages=messages,
                    token_usage=token_usage,
                    usage=usage,
                    display_entries=build_input_display_entries(inputs),
                ),
            )
        except UsageLimitExceeded as exc:
            success = False
            error = str(exc)
            raise ModelTurnLimitExceeded(f"{ctx.model_type}: usage/turn limit exceeded: {exc}") from exc
        except (UnexpectedModelBehavior, ModelHTTPError) as exc:
            success = False
            error = str(exc)
            raise ModelOutputError(f"{ctx.model_type}: model loop failed: {exc}") from exc
        except Exception as exc:
            success = False
            error = str(exc)
            raise
        finally:
            self._emit_call_finished(
                ctx,
                LlmCallOutcome(
                    started_at=t,
                    success=success,
                    response=accumulated,
                    error=error,
                    token_usage=token_usage,
                    error_detail=ctx.capability.last_error_detail,
                ),
            )
            await self.session.set_status("idle" if success else "error", message=error)

    # ------------------------------------------------------------------
    # LLM call (pydantic-ai)
    # ------------------------------------------------------------------

    @classmethod
    def _build_prompt(
        cls,
        agent: "LlmAgent",
        inputs: Any,
        *,
        call_id: "str | None",
    ) -> "tuple[str, list[SkillSpec], list[SubagentSpec], str]":
        skills = agent.skills
        subagents = agent.subagents
        agent_call_id = call_id or _call_id("agent", agent.agent_id)
        prompt = build_prompt(
            PromptContext(
                spec=agent.spec,
                input_data=inputs,
                skills=skills,
                subagents=subagents,
                runtime_dir=agent._runtime_prompt_dir,
            )
        )
        return prompt, skills, subagents, agent_call_id

    @classmethod
    def _default_output_type(cls) -> Any:
        return PromptedOutput(cls.OUTPUT_MODEL, template=False) if cls.OUTPUT_MODEL else dict[str, Any]

    async def _build_call_context(self, request: LlmCallRequest) -> LlmCallContext:
        built = self._build_model_agent(request)
        history = await self.session.load_history()
        return LlmCallContext(
            agent=built.agent,
            bundle=built.bundle,
            prompt=built.prompt,
            history=history,
            model_type=built.bundle.config.model_type,
            system_prompt=request.system_prompt or "",
            llm_call_id=request.call_id or _call_id(request.parent_call_id or self.agent_id, "llm", str(self._next_call_index(history))),
            parent_call_id=request.parent_call_id,
            mcp_specs=built.mcp_specs,
            capability=built.capability,
        )

    def _emit_call_started(self, ctx: LlmCallContext, *, image_inputs: "list[str] | None" = None) -> None:
        from .session.types import FileSession

        kernel = self.kernel
        mcp_servers = sorted({s.server_name or s.name for s in ctx.mcp_specs}) or None
        kernel.trigger(
            "llm_call_start",
            **self.execution_context.context,
            agent_id=self.agent_id,
            model_type=ctx.model_type,
            session_id=self.session.session_id,
            system_prompt=ctx.system_prompt,
            prompt=ctx.prompt,
            call_id=ctx.llm_call_id,
            parent_call_id=ctx.parent_call_id,
            **({"session_dir": str(self.session.root)} if isinstance(self.session, FileSession) else {}),
            **({"mcp_servers": mcp_servers} if mcp_servers else {}),
            **({"images": image_inputs} if image_inputs is not None else {}),
        )

    async def _save_call(self, ctx: LlmCallContext, result: LlmRunResult) -> None:
        await self.session.persist(
            SessionTurn(
                history=ctx.history,
                all_messages=result.messages,
                model=ctx.bundle.config,
                token_usage=result.token_usage,
                llm_call=self._build_call_record(
                    call_id=ctx.llm_call_id,
                    model_type=ctx.model_type,
                    model=ctx.bundle.config.model,
                    token_usage=result.token_usage,
                    requests=getattr(result.usage, "requests", 0) or 0,
                ).as_dict(),
                display_entries=result.display_entries,
                system_prompt=ctx.system_prompt,
            )
        )
        await self._maybe_save_checkpoint()

    async def _maybe_save_checkpoint(self) -> None:
        if not self.enable_checkpointing:
            return
        store = self.checkpoint_store or self._default_checkpoint_store()
        # NOTE: _checkpoint_seq is scoped to this agent instance's lifetime, not
        # persisted across reconstructions of an agent for the same session_id --
        # a new agent object for an existing session restarts the counter at 1,
        # which will overwrite that session's prior "1.bin" checkpoint. Fine for
        # now (checkpoint restore orchestration is out of scope for this plan),
        # but worth knowing before building on top of this.
        self._checkpoint_seq = getattr(self, "_checkpoint_seq", 0) + 1
        content = await self._checkpoint_snapshot_bytes()
        await store.save(self.session.session_id, self._checkpoint_seq, content)

    def _default_checkpoint_store(self) -> "CheckpointStore":
        from .checkpoint.local import FileCheckpointStore

        return FileCheckpointStore(root=self.session.root / "checkpoints")

    def _plan_artifact_store(self) -> "AgentArtifactStore":
        from .session.local import LocalAgentArtifactStore

        return LocalAgentArtifactStore(root=self.session.root / "artifacts")

    def _memory_root(self) -> Path:
        return self.session.root / "memory"

    def _build_feature_capabilities(self) -> "list[Any]":
        """Feature-toggle-driven capabilities, in a fixed order, for `_build_model_agent`
        to append to the pydantic_ai `Agent`'s `capabilities=` list. Order matches the
        spec's "特性开关" constructor parameter order: security, stuck-loop detection,
        periodic reminders, budget, plan mode, memory, swarm, tool search."""
        capabilities: "list[Any]" = []
        if self.enable_security_preset:
            capabilities.append(SecurityCapability())
        if self.enable_stuck_loop_detection:
            capabilities.append(StuckLoopCapability())
        if self.enable_periodic_reminders:
            capabilities.append(PeriodicReminderCapability())
        if self.budget_usd is not None:
            capabilities.append(BudgetCapability(BudgetTracker(budget_usd=self.budget_usd)))
        if self.enable_plan_mode:
            capabilities.append(PlanCapability(
                session_id=self.session.session_id,
                artifact_store=self._plan_artifact_store(),
            ))
        if self.enable_memory:
            capabilities.append(MemoryCapability(root=self._memory_root()))
        if self.task_queue is not None:
            capabilities.append(SwarmCapability(task_queue=self.task_queue, agent_id=self.agent_id))
        if self.enable_tool_search:
            capabilities.append(ToolSearchCapability(tool_names=tuple(self.tools)))
        return capabilities

    async def _checkpoint_snapshot_bytes(self) -> bytes:
        """Raw bytes for a checkpoint of the current session state. FileSession-only
        in this plan -- RemoteSession checkpointing needs SessionTranscriptHead
        serialization, deferred to a future plan (see this task's "Scope note")."""
        from .session.types import FileSession

        if isinstance(self.session, FileSession):
            return await asyncio.to_thread((self.session.root / "context.json").read_bytes)
        raise NotImplementedError(
            f"checkpointing is only implemented for FileSession, got {type(self.session).__name__}"
        )

    def _emit_call_finished(self, ctx: LlmCallContext, outcome: LlmCallOutcome) -> None:
        from .session.types import FileSession

        self.kernel.trigger(
            "post_llm_call",
            **self.execution_context.context,
            agent_id=self.agent_id,
            model_type=ctx.model_type,
            session_id=self.session.session_id,
            **({"session_dir": str(self.session.root)} if isinstance(self.session, FileSession) else {}),
            duration_ms=round((time.monotonic() - outcome.started_at) * 1000, 2),
            success=outcome.success,
            response=outcome.response,
            error=outcome.error,
            error_detail=outcome.error_detail,
            token_usage=outcome.token_usage,
            call_id=ctx.llm_call_id,
            parent_call_id=ctx.parent_call_id,
        )

    def _build_model_agent(self, request: LlmCallRequest) -> BuiltLlmAgent:
        """Construct the pydantic-ai Agent + capabilities shared by the structured and
        streaming call paths. Returns BuiltLlmAgent(agent, bundle, prompt, mcp_specs,
        capability); `prompt` may gain an unavailable-adapters note. `capability` is
        always the RuntimeRunCapability instance (file/terminal + instructions/settings/
        on_run_error) — callers read `.last_error_detail` off it; skill/subagent/MCP each
        get their own capability instance appended to the list, not tracked individually."""
        bundle = build_model(self.model_config_resolver(request.model_type))
        builtin_toolset = build_builtin_toolset(
            BuiltinToolContext(
                backend=LocalExecutionBackend(
                    runtime_dir=getattr(self, "workdir", None) or Path.cwd(),
                    base_dirs=[self.spec.base_dir] if self.spec.base_dir else [],
                ),
                enabled_tools=set(self.tools),
            )
        )
        kernel = self.kernel
        hook_context = self.execution_context.context
        builtin_toolset = HookedBuiltinToolset(builtin_toolset, kernel, hook_context, request.parent_call_id)
        mcp_specs = self.execution_context.capabilities.mcp_servers
        prompt = request.prompt
        if note := unavailable_adapters_note(self.execution_context.capabilities.missing_mcp_sources):
            prompt = f"{prompt}\n\n{note}"

        runtime_capability = RuntimeRunCapability(
            instructions=request.system_prompt or "",
            toolset=builtin_toolset,
            model_settings=bundle.settings,
        )
        capabilities: "list[Any]" = [runtime_capability, *self._build_feature_capabilities()]
        if request.skills:
            capabilities.append(SkillCapability(
                skill_view_fn=lambda arguments: self._read_skill(request.skills, arguments),
                kernel=kernel, context=hook_context, parent_call_id=request.parent_call_id,
            ))
        if request.subagents:
            capabilities.append(SubagentCapability(
                run_subagent_fn=self._invoke_subagent,
                allowed_subagents={s.name for s in request.subagents},
                kernel=kernel, context=hook_context, parent_call_id=request.parent_call_id,
            ))
        for spec in mcp_specs:
            inner_mcp = MCP(
                url=spec.url or f"local://{spec.name}",
                native=False,
                local=build_mcp_toolset(spec),
                id=spec.server_name or spec.name,
            )
            capabilities.append(HookedMCPCapability(
                wrapped=inner_mcp, server_name=spec.server_name or spec.name,
                kernel=kernel, context=hook_context, parent_call_id=request.parent_call_id,
            ))

        agent = Agent(
            bundle.model,
            output_type=request.output_type,
            capabilities=capabilities,
            retries=max(0, int(bundle.config.raw.get("max_retries", 1))),
        )
        return BuiltLlmAgent(agent=agent, bundle=bundle, prompt=prompt, mcp_specs=mcp_specs, capability=runtime_capability)

    # ------------------------------------------------------------------
    # Output / usage helpers
    # ------------------------------------------------------------------

    @classmethod
    def _coerce_output(cls, output: Any) -> "dict[str, Any]":
        if isinstance(output, BaseModel):
            # Output models expose as_payload(); fall back to model_dump otherwise.
            as_payload = getattr(output, "as_payload", None)
            return as_payload() if callable(as_payload) else output.model_dump(mode="json")
        if isinstance(output, dict):
            return output
        return {"result": output}

    @classmethod
    def _usage_summary(cls, usage: Any) -> "dict[str, Any]":
        # Shape matches the legacy contract consumed by observability/monitoring/console
        # (engine/infra/observability.py, engine/secops/trace_metrics.py,
        # engine/server/static/{console,}/js): {"usage": {input/output/cache_*}, "num_turns"}.
        return {
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(usage, "cache_read_tokens", 0) or 0,
                "cache_miss_input_tokens": getattr(usage, "cache_write_tokens", 0) or 0,
            },
            "num_turns": getattr(usage, "requests", 0) or 0,
        }

    @classmethod
    def _build_user_prompt(cls, prompt: str, image_inputs: "list[str] | None") -> Any:
        if not image_inputs:
            return prompt
        from pydantic_ai.messages import BinaryContent, ImageUrl
        parts: "list[Any]" = [prompt]
        for image in image_inputs:
            if str(image).startswith(("http://", "https://")):
                parts.append(ImageUrl(url=image))
            else:
                import base64
                parts.append(BinaryContent(data=base64.b64decode(image), media_type="image/png"))
        return parts

    # ------------------------------------------------------------------
    # Tool plumbing — native MCP
    # ------------------------------------------------------------------

    async def _invoke_subagent(
        self,
        subagent_id: str,
        input_data: Any,
        *,
        call_id: str,
    ) -> "dict[str, Any]":
        hook_context = self.execution_context.context
        t = time.monotonic()
        self.kernel.trigger("subagent_start", **hook_context, parent_agent_id=self.agent_id, subagent_id=subagent_id, call_id=call_id)
        success = True
        try:
            child_spec = next(
                subagent for subagent in self.execution_context.capabilities.subagents
                if subagent.name == subagent_id
            )
            child_session = self.session.copy(child_session_id=f"subagent_{child_spec.name}_{uuid.uuid4().hex[:12]}")
            child_context = self.kernel.build_context(
                child_spec,
                child_session,
                builtin_tool_names=SubAgent._BUILTIN_TOOL_NAMES,
                context=hook_context,
            )
            child_agent = SubAgent(
                child_spec,
                child_session,
                execution_context=child_context,
                model_config_resolver=self.model_config_resolver,
                workdir=getattr(self, "workdir", None),
            )
            return await child_agent.generate(input_data, call_id=call_id)
        except Exception:
            success = False
            raise
        finally:
            self.kernel.trigger("subagent_end", **hook_context, parent_agent_id=self.agent_id, subagent_id=subagent_id, duration_ms=round((time.monotonic() - t) * 1000, 2), status="completed" if success else "failed", call_id=call_id)

    # ------------------------------------------------------------------
    # Supported capabilities
    # ------------------------------------------------------------------

    #: Builtin (execution-env) tools this agent class exposes. Empty for the
    #: conversational base; RuntimeAgent enables file/terminal.
    _BUILTIN_TOOL_NAMES: "frozenset[str]" = frozenset()

    @property
    def _runtime_prompt_dir(self) -> "Path | None":
        """Working directory advertised to the model in the prompt; None = no execution
        environment (conversational). RuntimeAgent returns the agent's workdir."""
        return None

    def snapshot(self) -> "dict[str, object]":
        return {
            "agent_id": self.spec.name,
            "description": self.spec.description,
            "model": self.spec.model,
            "tools": self.tools,
            "mcps": [spec.name for spec in self.mcp_servers],
            "skills": [s.name for s in self.skills],
            "subagents": [s.name for s in self.subagents],
        }

    @property
    def tools(self) -> "list[str]":
        return list(self.execution_context.capabilities.builtin_tools)

    @property
    def skills(self) -> "list[SkillSpec]":
        return list(self.execution_context.capabilities.skills)

    @property
    def subagents(self) -> "list[SubagentSpec]":
        return list(self.execution_context.capabilities.subagents)

    @property
    def mcp_servers(self) -> "list[MCPServerSpec]":
        return list(self.execution_context.capabilities.mcp_servers)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    @classmethod
    def _build_call_record(
        cls,
        *,
        call_id: str,
        model_type: str,
        model: "str | None",
        token_usage: "dict[str, Any]",
        requests: int,
    ) -> LlmCallRecord:
        """Per-call summary persisted alongside the session history."""
        return LlmCallRecord(
            call_id=call_id,
            model_type=model_type,
            model=model,
            token_usage=token_usage,
            requests=requests,
        )

    @classmethod
    def _next_call_index(cls, history: "list[Any]") -> int:
        """Next LLM call sequence number, derived from prior history regardless of
        persistence backend (one ModelResponse ≈ one prior call). Used only to make the
        call_id readable; uniqueness within a turn is provided by the agent call prefix."""
        from pydantic_ai.messages import ModelResponse
        return sum(1 for m in history if isinstance(m, ModelResponse)) + 1

    # Skill helpers
    # ------------------------------------------------------------------

    def _read_skill(self, skills: "list[SkillSpec]", arguments: "dict[str, Any]") -> "dict[str, Any]":
        sid = arguments.get("skill_id")
        fp = arguments.get("file_path")
        skill_id = str(sid).strip() if sid else None
        file_path = str(fp).strip() if fp else None
        if skill_id:
            return view_skill(skills, skill_id, file_path=file_path)
        return view_available_skills(skills, file_path=file_path)


class RuntimeAgent(LlmAgent):
    """LLM agent + execution environment.

    Adds file/terminal builtin tools (gated by `allowed_tools`) and advertises a runtime
    working directory in the prompt, plus `_write_runtime_json` for trace-file output.
    Used by workers and stage agents that read/write trace files or run commands.

    `workdir` belongs to the agent, not the session -- it's the directory
    file/bash tools execute in, decoupled from wherever the session's own
    history/artifacts are stored (`FileSession.root`, if any). Defaults to
    the process's current working directory."""

    _BUILTIN_TOOL_NAMES: "frozenset[str]" = frozenset({"file", "terminal"})

    def __init__(
        self,
        spec: AgentSpec,
        session: Session,
        execution_context: AgentExecutionContext,
        *,
        workdir: "Path | None" = None,
        **feature_toggles: Any,
    ) -> None:
        super().__init__(spec, session, execution_context, **feature_toggles)
        self.workdir = workdir or Path.cwd()

    @property
    def _runtime_prompt_dir(self) -> "Path | None":
        return self.workdir

    def _write_runtime_json(self, rel_path: str, payload: object) -> None:
        path: Path = self.workdir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class SubAgent(RuntimeAgent):
    """Lightweight child agent invoked via `call_subagent`. Recursion stops here."""

    def __init__(
        self,
        spec: SubagentSpec,
        session: Session,
        execution_context: AgentExecutionContext,
        **feature_toggles: Any,
    ) -> None:
        super().__init__(spec, session, execution_context=execution_context, **feature_toggles)

    @property
    def subagents(self) -> "list[SubagentSpec]":
        return []
