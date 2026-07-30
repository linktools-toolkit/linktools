"""The only adapter between internal tools and Pydantic AI."""

from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from .models import ExecuteTool, ToolExecutionContext
from .service import ToolExecutionService
from .models import ToolDefinition


@dataclass
class ToolPolicyCapability(AbstractCapability[None]):
    """SDK capability hook; policy remains owned by ToolExecutionService."""

    service: ToolExecutionService


def build_policy_capability(
    service: ToolExecutionService,
) -> ToolPolicyCapability:
    return ToolPolicyCapability(service=service)


@dataclass(slots=True)
class PydanticAIToolAdapter:
    service: ToolExecutionService

    def build_toolset(
        self,
        definitions: tuple[ToolDefinition, ...],
        *,
        context: ToolExecutionContext,
    ) -> AbstractToolset:
        toolset = FunctionToolset()
        for definition in definitions:
            self._add_definition(toolset, definition, context=context)
        return toolset

    def _add_definition(
        self,
        toolset: FunctionToolset,
        definition: ToolDefinition,
        *,
        context: ToolExecutionContext,
    ) -> None:
        async def invoke(run_context: RunContext[Any], **arguments: Any) -> object:
            invocation_context = ToolExecutionContext(
                execution_id=context.execution_id,
                tool_call_id=run_context.tool_call_id or context.tool_call_id,
                dependencies=context.dependencies,
                run_context=context.run_context,
                approved_tool_call_id=context.approved_tool_call_id,
                approved_binding_fingerprint=context.approved_binding_fingerprint,
                trace_sink=context.trace_sink,
                metadata=context.metadata,
            )
            return await self.service.execute(
                ExecuteTool(
                    definition=definition,
                    arguments=arguments,
                    context=invocation_context,
                )
            )

        if definition.input_schema is not None:
            from pydantic_ai.tools import Tool

            toolset.add_tool(
                Tool.from_schema(
                    function=invoke,
                    name=definition.descriptor.name,
                    description=(
                        definition.description or definition.descriptor.name
                    ),
                    json_schema=dict(definition.input_schema),
                    takes_ctx=True,
                )
            )
        else:
            from pydantic_ai.tools import Tool

            declared = Tool(definition.handler)
            toolset.add_tool(
                Tool.from_schema(
                    function=invoke,
                    name=definition.descriptor.name,
                    description=(
                        definition.description
                        or declared.description
                        or definition.descriptor.name
                    ),
                    json_schema=dict(
                        declared.function_schema.json_schema
                    ),
                    takes_ctx=True,
                )
            )
