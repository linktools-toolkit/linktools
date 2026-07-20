#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single place in the tool domain that imports the pydantic-ai Tool API.
PolicyCapability adapts GovernedToolInvoker into a pydantic-ai
AbstractCapability; ManagedToolsetWrapper wraps any AbstractToolset so every
call_tool passes through the ManagedToolAdapter governance chain. Keeping these
here means ``from pydantic_ai`` appears in exactly one tool module."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ..errors import RunPaused, ToolDeniedError
from .executor import GovernedToolInvoker
from .managed import ManagedToolAdapter
from .models import ManagedToolDefinition, ToolDescriptor

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition

    from ..run.context import RunContext as _RunContext
    from ..governance.security.pipeline import SecurityPipeline
    from .policy import ResolvedToolPolicy, ToolPolicyProvider


@dataclass
class PolicyCapability(AbstractCapability[None]):
    """Adapts GovernedToolInvoker into a pydantic-ai AbstractCapability, converting
    ToolDeniedError into SkipToolExecution so a denied call surfaces as a tool
    result the model can see. RunPaused (the approval signal) is a RunError, so
    it is re-raised (not converted) and propagates to AgentEngine's pause
    handler.

    The per-Run ToolContext arrives via pydantic-ai dependency injection
    (``ctx.deps.tool_context``); no mutable per-Run field on the capability, so
    a single CompiledAgent/PolicyCapability is safe across concurrent Runs."""

    executor: GovernedToolInvoker

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
        if descriptor is None:
            raise ToolDeniedError(f"tool {tool_def.name!r} is not managed")
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
        # it propagates out of pydantic-ai's tool loop to AgentEngine's pause
        # handler instead of surfacing a skip-result and continuing.
        if isinstance(error, RunPaused):
            raise error
        from ..governance.security.redact import redact_exception

        safe_error = redact_exception(error)
        raise SkipToolExecution(
            {"error_type": type(error).__name__, "error": safe_error}
        ) from error


def build_policy_capability(executor: GovernedToolInvoker) -> PolicyCapability:
    return PolicyCapability(executor=executor)


def build_managed_toolset(definition: ManagedToolDefinition) -> AbstractToolset:
    """Adapt one managed definition to the model toolset API."""
    from pydantic_ai.toolsets import FunctionToolset

    toolset = FunctionToolset()
    if definition.parameters_json_schema:
        from pydantic_ai.tools import Tool

        toolset.add_tool(
            Tool.from_schema(
                function=definition.handler,
                name=definition.descriptor.name,
                description=definition.description or definition.descriptor.name,
                json_schema=dict(definition.parameters_json_schema),
            )
        )
    else:
        toolset.add_function(
            definition.handler,
            name=definition.descriptor.name,
            description=definition.description,
        )
    return toolset


class ManagedToolsetWrapper(WrapperToolset):
    """Wraps an AbstractToolset (e.g. MCPToolset) so every call_tool is
    dispatched through a per-tool ManagedToolAdapter -- descriptor resolution,
    policy, pipeline, and GovernedToolInvoker.execute are ManagedToolAdapter's job,
    not this wrapper's. Pure dispatch: carries no governance logic of its own."""

    def __init__(
        self,
        wrapped: Any,
        *,
        descriptors: "Mapping[str, ToolDescriptor]",
        security_pipeline: "SecurityPipeline | None" = None,
        tool_executor: Any,
        policy_provider: "ToolPolicyProvider | None" = None,
        baseline_policy: "ResolvedToolPolicy | None" = None,
        run_context: "_RunContext | None" = None,
        event_store: Any = None,
        security_audit_failure_mode: Any = "fail_closed",
        security_event_emitter: Any = None,
    ) -> None:
        if tool_executor is None:
            raise ToolDeniedError("ManagedToolsetWrapper requires GovernedToolInvoker")
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
