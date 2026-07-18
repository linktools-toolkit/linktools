#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The Tool domain model: the single place the tool data classes
live. ToolDescriptor (structured metadata classifying a tool for policy
decisions) + default_risk_for_category + ManagedToolDefinition + ToolContribution
+ declared_tool_definitions. Moved out of security/descriptor.py + tool/contribution.py
so the tool domain owns its descriptors; keeping the dependency one-way."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..utils.freeze import freeze_value

# Standard category -> default risk mapping. Unknown categories
# default to "high" (conservative).
_CATEGORY_RISK: "dict[str, str]" = {
    "discovery": "low",
    "file-read": "low",
    "network-read": "medium",
    "file-write": "medium",
    "mcp-read": "medium",
    "subagent": "medium",
    "terminal": "high",
    "network-write": "high",
    "mcp-write": "high",
    "package-execute": "high",
    "package-read": "low",
}


def default_risk_for_category(category: str) -> str:
    """Conservative risk for a category. Unknown -> high."""
    return _CATEGORY_RISK.get(category, "high")


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Structured metadata classifying a tool for policy decisions. Avoids
    guessing risk from function names. Each tool exposed to the model must have
    a descriptor so the governance chain (policy, pipeline, baseline) can make
    decisions based on category/risk, not name patterns."""

    name: str
    source: str  # "builtin" | "mcp" | "skill" | "subagent" | "package"
    category: str  # stable security classification
    risk: str  # "low" | "medium" | "high" | "critical"
    mutating: bool
    capability_kind: str = ""  # ToolRef kind that produced this tool
    capability_name: str = ""  # capability instance name
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolDescriptor.name must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("ToolDescriptor.source must be a non-empty string")
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("ToolDescriptor.category must be a non-empty string")
        if not isinstance(self.risk, str) or not self.risk.strip():
            raise ValueError("ToolDescriptor.risk must be a non-empty string")
        if not isinstance(self.mutating, bool):
            raise TypeError("ToolDescriptor.mutating must be a boolean")
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))

    def fingerprint(self) -> str:
        import hashlib
        from ..json import canonical_json
        payload = {"name": self.name, "source": self.source,
                   "category": self.category, "risk": self.risk,
                   "mutating": self.mutating, "capability_kind": self.capability_kind,
                   "capability_name": self.capability_name, "metadata": dict(self.metadata)}
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ManagedToolDefinition:
    """One model-callable tool: its descriptor, the raw async handler that
    actually executes it, and the tool's parameter JSON schema. The schema is
    consumed when the tool is registered with the model (a ``**kwargs`` handler
    -- e.g. an MCP forwarding closure -- has no signature to derive one from, so
    the explicit schema is what tells the model the tool's parameters) and is
    re-used to re-validate arguments after a pipeline MODIFY. ``description`` is
    the human-readable text shown to the model; when absent the descriptor name
    is used as a fallback. ``schema_version`` is the tool's input-contract
    version; a change here invalidates cached idempotency records for the tool."""

    descriptor: ToolDescriptor
    handler: Any  # Callable[..., Awaitable[Any]]
    parameters_json_schema: "Mapping[str, Any] | None" = None
    description: "str | None" = None
    schema_version: str = "1"


@dataclass(frozen=True, slots=True)
class ToolContribution:
    """A capability's exposed tools, one ManagedToolDefinition per tool. This is
    the single contribution form -- unmanaged toolsets/descriptors tuples are not
    accepted; providers build ManagedToolDefinition entries (e.g. via
    :func:`declared_tool_definitions`)."""

    tools: "tuple[ManagedToolDefinition, ...]" = ()


def declared_tool_definitions(
    toolset: Any,
    descriptors: "tuple[ToolDescriptor, ...]",
) -> "tuple[ManagedToolDefinition, ...]":
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
            f"extra={sorted(actual - declared)}"
        )
    return tuple(
        ManagedToolDefinition(
            descriptor=descriptor,
            handler=tools[descriptor.name].function,
            parameters_json_schema=getattr(
                getattr(tools[descriptor.name], "tool_def", None),
                "parameters_json_schema",
                None,
            ),
            description=getattr(
                getattr(tools[descriptor.name], "tool_def", None), "description", None
            ),
        )
        for descriptor in descriptors
    )
