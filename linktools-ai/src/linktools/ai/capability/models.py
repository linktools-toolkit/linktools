#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The Capability domain model: the single place the capability
data classes live. CapabilityRef / CapabilityRuntimeOptions / CapabilityBundle /
CapabilityInspection. The exposure policy lives in :mod:`.exposure`, the
CapabilityProvider Protocol in :mod:`.provider`."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from ..tool.models import ToolDescriptor
from ..utils.freeze import freeze_value
from .exposure import CapabilityToolExposurePolicy


def requires_capability_assembler(*, tools, execution) -> bool:
    # ``None`` and ``()`` both mean no capabilities. Builtins are enabled only
    # by an explicit RuntimeTool/Capability option and are materialized by the
    # caller before reaching this predicate.
    return bool(tools)


if TYPE_CHECKING:
    from ..prompt.window import SessionWindowPolicy


@dataclass(frozen=True, slots=True)
class CapabilityRef:
    """A resolved (kind, name, config) triple targeting one CapabilityProvider.
    ``kind`` selects the provider; ``name`` is interpreted by that provider
    (e.g. builtin:file, skill:sql, mcp:risk-data)."""

    kind: str
    name: str
    config: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("CapabilityRef.kind must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("CapabilityRef.name must be a non-empty string")
        object.__setattr__(self, "config", freeze_value(dict(self.config)))

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class CapabilityRuntimeOptions:
    """The runtime-policy bundle -- distinct from Storage (state) and
    ProviderBundle (declarations). Holds the tool-exposure policy, optional
    prompt/memory/subagent policies, and the MCP wildcard gate."""

    tool_exposure: "CapabilityToolExposurePolicy | None" = None
    # Optional policies. None means "use the runner's default" (which preserves
    # historical behavior) or "not wired" (Noop). Each is substitutable.
    session_window_policy: "SessionWindowPolicy | None" = None
    memory_policy: Any = None
    retrieval_policy: Any = None
    prompt_context_formatter: Any = None
    subagent_context_policy: Any = None
    allow_mcp_wildcard: bool = False
    # Builtins are opt-in. ``None`` is accepted only by legacy callers that
    # construct AgentRunner directly; Runtime-built graphs default closed.
    enable_builtin_tools: bool = False


@dataclass(slots=True)
class CapabilityBundle:
    """The output of resolving one (or many) CapabilityRef(s). A bundle
    contributes zero or more of: prompt sections (injected text) and
    tool_contributions (ToolContribution with explicit descriptors). Raw
    toolsets / middleware / resources / declared pipelines are not accepted --
    providers return explicit ToolContribution entries; the run's security
    pipeline comes from the SecurityBaseline, not from capabilities."""

    prompt_sections: "Mapping[str, str]" = field(default_factory=dict)
    tool_contributions: "tuple[Any, ...]" = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tool_contributions, tuple):
            raise TypeError("CapabilityBundle.tool_contributions must be a tuple")
        self.prompt_sections = dict(self.prompt_sections)

    @classmethod
    def empty(cls) -> "CapabilityBundle":
        return cls()


@dataclass(frozen=True)
class CapabilityInspection:
    """A stable, immutable view of what an AgentSpec resolves to. Returned by
    Runtime.inspect so downstream tooling never depends on the mutable internal
    CapabilityBundle / raw handlers."""

    tools: "tuple[ToolDescriptor, ...]" = ()
    prompt_sections: "Mapping[str, str]" = field(default_factory=dict)
    warnings: "tuple[str, ...]" = ()
    exposure_policy: CapabilityToolExposurePolicy = field(
        default_factory=CapabilityToolExposurePolicy
    )

    @classmethod
    def from_bundle(
        cls,
        bundle: Any,
        *,
        exposure_policy: "CapabilityToolExposurePolicy | None" = None,
    ) -> "CapabilityInspection":
        """Build an inspection from a CapabilityBundle without leaking its
        mutable internals: tools come from the per-tool definitions, prompt
        sections are copied."""
        tools: "list[ToolDescriptor]" = []
        for c in bundle.tool_contributions:
            tools.extend(md.descriptor for md in c.tools)
        # Deduplicate by name while preserving order.
        seen: "set[str]" = set()
        unique: "list[ToolDescriptor]" = []
        for d in tools:
            if d.name not in seen:
                seen.add(d.name)
                unique.append(d)
        return cls(
            tools=tuple(unique),
            prompt_sections=freeze_value(dict(bundle.prompt_sections)),
            exposure_policy=exposure_policy or CapabilityToolExposurePolicy(),
        )
