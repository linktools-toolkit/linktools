#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolContribution: the structured output a CapabilityProvider produces for each
tool it exposes. Pairs the pydantic-ai toolset with explicit ToolDescriptors so
the assembler + ManagedToolAdapter never need to guess tool names or categories
via runtime introspection.

The preferred per-tool form is ``ManagedToolDefinition`` (one descriptor + its
raw handler + parameter schema per entry); the ``toolset + descriptors`` form is
the fallback for opaque toolsets (e.g. MCP) whose handlers cannot be extracted
individually."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..security.descriptor import ToolDescriptor


@dataclass(frozen=True, slots=True)
class ManagedToolDefinition:
    """One model-callable tool: its descriptor, the raw async handler that
    actually executes it, and the tool's parameter JSON schema. The schema is
    consumed when the tool is registered with the model (a ``**kwargs`` handler
    -- e.g. an MCP forwarding closure -- has no signature to derive one from, so
    the explicit schema is what tells the model the tool's parameters) and is
    re-used to re-validate arguments after a pipeline MODIFY."""
    descriptor: ToolDescriptor
    handler: Any  # Callable[..., Awaitable[Any]]
    parameters_json_schema: "Mapping[str, Any] | None" = None


@dataclass(frozen=True, slots=True)
class ToolContribution:
    """A capability's exposed tools. ``tools`` is the preferred per-tool form
    (ManagedToolDefinition per tool). ``toolset + descriptors`` is the explicit
    fallback for opaque toolsets (MCP) where handlers cannot be extracted."""
    toolset: Any = None
    descriptors: "tuple[ToolDescriptor, ...]" = ()
    tools: "tuple[ManagedToolDefinition, ...]" = ()
    legacy_adapter: bool = False


def declared_tool_definitions(
    toolset: Any, descriptors: tuple[ToolDescriptor, ...],
) -> tuple[ManagedToolDefinition, ...]:
    """Build explicit definitions at a Provider boundary.

    Providers own the mapping from declared descriptors to handlers; the
    assembler deliberately has no toolset introspection fallback.
    """
    tools = getattr(toolset, "tools", None)
    if not isinstance(tools, dict):
        raise TypeError("declared tool definitions require an introspectable toolset")
    declared = {descriptor.name for descriptor in descriptors}
    actual = {str(name) for name in tools}
    if declared != actual:
        raise ValueError(
            f"tool descriptor mismatch: missing={sorted(declared - actual)}, "
            f"extra={sorted(actual - declared)}")
    return tuple(
        ManagedToolDefinition(
            descriptor=descriptor,
            handler=tools[descriptor.name].function,
            parameters_json_schema=getattr(
                getattr(tools[descriptor.name], "tool_def", None),
                "parameters_json_schema", None),
        )
        for descriptor in descriptors
    )
