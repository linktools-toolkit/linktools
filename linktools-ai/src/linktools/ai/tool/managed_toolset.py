#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ManagedToolsetWrapper: wraps any pydantic-ai AbstractToolset so that every
call_tool invocation passes through the security governance chain. Used for
toolsets that don't expose individual handlers (e.g. MCPToolset), where
per-function ManagedToolAdapter wrapping at construction time is not possible.

Pure dispatch: the wrapper carries no governance logic of its own. Per call it
looks up the descriptor matching the invoked tool name (never "the first
descriptor for the whole toolset") and builds a ManagedToolAdapter around a
handler that forwards into the wrapped toolset -- the same descriptor ->
policy -> pipeline -> ToolExecutor.execute chain every other tool goes
through."""

from typing import TYPE_CHECKING, Any, Mapping

from pydantic_ai.toolsets import WrapperToolset

from ..errors import ToolDeniedError
from ..security.descriptor import ToolDescriptor
from .managed import ManagedToolAdapter

if TYPE_CHECKING:
    from ..run.context import RunContext
    from ..security.pipeline import SecurityPipeline
    from .policy import ResolvedToolPolicy, ToolPolicyProvider


class ManagedToolsetWrapper(WrapperToolset):
    """Wraps an AbstractToolset (e.g. MCPToolset) so every call_tool is
    dispatched through a per-tool ManagedToolAdapter -- descriptor resolution,
    policy, pipeline, and ToolExecutor.execute are ManagedToolAdapter's job,
    not this wrapper's."""

    def __init__(
        self,
        wrapped: Any,
        *,
        descriptors: "Mapping[str, ToolDescriptor]",
        security_pipeline: "SecurityPipeline | None" = None,
        tool_executor: Any = None,
        policy_provider: "ToolPolicyProvider | None" = None,
        baseline_policy: "ResolvedToolPolicy | None" = None,
        run_context: "RunContext | None" = None,
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
        # Thread the REAL pydantic-ai tool_call_id (on ctx) through to the
        # adapter so a pause it raises keys the ApprovalRequest on the same id
        # the model's message history uses -- the linchpin of resume (a
        # re-driven call after approve() must match the stored approval).
        tool_call_id = getattr(ctx, "tool_call_id", None)
        # The tool's parameter JSON schema, for re-validating arguments after a
        # pipeline MODIFY. Lives on the ToolDefinition.
        parameter_schema = getattr(getattr(tool, "tool_def", None), "parameters_json_schema", None)
        arguments = dict(tool_args) if isinstance(tool_args, Mapping) else {}
        return await adapter.invoke(
            tool_call_id=tool_call_id, parameter_schema=parameter_schema, **arguments)
