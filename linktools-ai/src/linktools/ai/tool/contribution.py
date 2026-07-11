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
    (ManagedToolDefinition per tool). ``toolset + descriptors`` is the fallback
    for opaque toolsets (MCP) where individual handlers cannot be extracted; in
    that form every descriptor must still resolve to a unique handler in the
    toolset (enforced at assembly)."""
    toolset: Any = None
    descriptors: "tuple[ToolDescriptor, ...]" = ()
    tools: "tuple[ManagedToolDefinition, ...]" = ()

