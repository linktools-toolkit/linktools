#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single place in the tool domain that imports the pydantic-ai Tool API
(spec §11.5). PolicyCapability adapts ToolExecutor into a pydantic-ai
AbstractCapability; ManagedToolsetWrapper wraps any AbstractToolset so every
call_tool passes through the ManagedToolAdapter governance chain. Keeping these
here means ``from pydantic_ai`` appears in exactly one tool/ file (§23.6)."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.toolsets import WrapperToolset

from ..errors import RunPaused, ToolApprovalRequiredError, ToolDeniedError
from ..policy.engine import ToolRequest
from .executor import ToolExecutor
from .managed import ManagedToolAdapter
from .models import ToolDescriptor

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition

    from ..run.context import RunContext as _RunContext
    from ..security.pipeline import SecurityPipeline
    from .policy import ResolvedToolPolicy, ToolPolicyProvider


@dataclass
class PolicyCapability(AbstractCapability[None]):
    """Adapts ToolExecutor into a pydantic-ai AbstractCapability, converting
    ToolDeniedError/ToolApprovalRequiredError into SkipToolExecution so a
    denied call surfaces as a tool result the model can see.

    The per-Run ToolContext arrives via pydantic-ai dependency injection
    (``ctx.deps.tool_context``); no mutable per-Run field on the capability, so
    a single CompiledAgent/PolicyCapability is safe across concurrent Runs."""

    executor: ToolExecutor

    async def before_tool_execute(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
    ) -> Any:
        lookup = getattr(ctx.deps, "descriptor_lookup", None)
        descriptor = lookup.get(tool_def.name) if lookup else None
        # Managed tools (descriptor in the per-run lookup) are governed by
        # ManagedToolAdapter -> ToolExecutor.execute, which already runs this
        # PolicyEngine. Running check() here too would double-execute every
        # rule. So this hook only governs LEGACY/raw tools (no descriptor).
        if descriptor is not None:
            return args
        request = ToolRequest(
            tool_name=tool_def.name,
            arguments=args,
            category=None,
            risk=None,
            mutating=None,
        )
        base = ctx.deps.tool_context
        context = replace(base, tool_call_id=call.tool_call_id)
        try:
            await self.executor.check(request, context)
        except ToolDeniedError as exc:
            raise SkipToolExecution({"error": str(exc)}) from exc
        except ToolApprovalRequiredError as exc:
            raise SkipToolExecution({"error": str(exc)}) from exc
        return args

    async def after_tool_execute(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
        result: Any,
    ) -> Any:
        return result

    async def on_tool_execute_error(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
        error: BaseException,
    ) -> Any:
        # RunPaused is a run-level control-flow signal (approval/pause raised
        # from inside a managed tool handler), NOT a tool error. Re-raise it so
        # it propagates out of pydantic-ai's tool loop to AgentRunner's pause
        # handler instead of surfacing a skip-result and continuing.
        if isinstance(error, RunPaused):
            raise error
        raise SkipToolExecution(
            {"error": f"{type(error).__name__}: {error}"}
        ) from error


def build_policy_capability(executor: ToolExecutor) -> PolicyCapability:
    return PolicyCapability(executor=executor)


class ManagedToolsetWrapper(WrapperToolset):
    """Wraps an AbstractToolset (e.g. MCPToolset) so every call_tool is
    dispatched through a per-tool ManagedToolAdapter -- descriptor resolution,
    policy, pipeline, and ToolExecutor.execute are ManagedToolAdapter's job,
    not this wrapper's. Pure dispatch: carries no governance logic of its own."""

    def __init__(
        self,
        wrapped: Any,
        *,
        descriptors: "Mapping[str, ToolDescriptor]",
        security_pipeline: "SecurityPipeline | None" = None,
        tool_executor: Any = None,
        policy_provider: "ToolPolicyProvider | None" = None,
        baseline_policy: "ResolvedToolPolicy | None" = None,
        run_context: "_RunContext | None" = None,
        event_store: Any = None,
        security_audit_failure_mode: Any = "fail_closed",
        security_event_emitter: Any = None,
    ) -> None:
        super().__init__(wrapped)
        self._descriptors = dict(descriptors)
        self._pipeline = security_pipeline
        self._tool_executor = tool_executor
        self._policy_provider = policy_provider
        self._baseline = baseline_policy
        self._run_context = run_context
        self._event_store = event_store
        self._security_audit_failure_mode = security_audit_failure_mode
        self._security_event_emitter = security_event_emitter

    async def call_tool(self, name, tool_args, ctx, tool):
        descriptor = self._descriptors.get(name)
        if descriptor is None:
            # No governance metadata for this tool name -- fail closed rather
            # than silently falling through to the raw, ungoverned toolset.
            raise ToolDeniedError(
                f"tool {name!r} has no registered descriptor; refusing to "
                f"call an ungoverned tool"
            )

        async def handler(**arguments: Any) -> Any:
            return await self.wrapped.call_tool(name, arguments, ctx, tool)

        adapter = ManagedToolAdapter(
            descriptor=descriptor,
            handler=handler,
            tool_executor=self._tool_executor,
            policy_provider=self._policy_provider,
            security_pipeline=self._pipeline,
            baseline_policy=self._baseline,
            run_context=self._run_context,
            event_store=self._event_store,
            security_audit_failure_mode=self._security_audit_failure_mode,
            security_event_emitter=self._security_event_emitter,
        )
        tool_call_id = getattr(ctx, "tool_call_id", None)
        parameter_schema = getattr(
            getattr(tool, "tool_def", None), "parameters_json_schema", None
        )
        arguments = dict(tool_args) if isinstance(tool_args, Mapping) else {}
        return await adapter.invoke(
            tool_call_id=tool_call_id, parameter_schema=parameter_schema, **arguments
        )
