"""Invocation-level values consumed by :class:`ToolExecutionService`."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Protocol

from ....json import JsonValue
from ..models import ToolDefinition

if TYPE_CHECKING:
    from ...dependencies import AgentDependencies
    from ....execution.context import RunContext


class ToolTraceSink(Protocol):
    async def tool_result(self, payload: JsonValue) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    execution_id: str
    tool_call_id: str
    dependencies: "AgentDependencies"
    run_context: "RunContext | None" = None
    approved_tool_call_id: str | None = None
    approved_binding_fingerprint: str | None = None
    trace_sink: ToolTraceSink | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecuteTool:
    definition: ToolDefinition
    arguments: Mapping[str, JsonValue]
    context: ToolExecutionContext
