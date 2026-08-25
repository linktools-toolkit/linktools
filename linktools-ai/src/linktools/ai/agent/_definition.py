#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable executable Agent definitions."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..capability import CapabilityBinding, validate_fingerprint
from ..core import ImmutableJsonMapping, JsonValue
from ..errors import AIError, ErrorCode
from ..model import ModelBinding
from ..spec import AgentSpec

_TRUSTED_TOOL_CLASSES = frozenset(
    {
        "control",
        "filesystem.read",
        "filesystem.write",
        "shell",
        "memory.read",
        "memory.write",
    }
)


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Freeze the output-independent identity of one executable Agent."""

    digest: str
    spec: AgentSpec
    model: ModelBinding
    effective_capabilities: "tuple[CapabilityBinding, ...]"
    local_runtime_capability_descriptors: "tuple[Mapping[str, JsonValue], ...]"
    trusted_tool_classes: "tuple[tuple[str, str], ...]" = ()
    trusted_mcp_selectors: "tuple[str, ...]" = ()
    global_runtime_capability_descriptors: "tuple[Mapping[str, JsonValue], ...]" = ()

    def __post_init__(self) -> None:
        validate_fingerprint(self.digest)
        if any(capability is None for capability in self.effective_capabilities):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        try:
            identities = tuple(
                (capability.provider, capability.id)
                for capability in self.effective_capabilities
            )
            local_descriptors = tuple(
                ImmutableJsonMapping(value)
                for value in self.local_runtime_capability_descriptors
            )
            global_descriptors = tuple(
                ImmutableJsonMapping(value)
                for value in self.global_runtime_capability_descriptors
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        if len(set(identities)) != len(identities):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        object.__setattr__(
            self,
            "local_runtime_capability_descriptors",
            local_descriptors,
        )
        object.__setattr__(
            self,
            "global_runtime_capability_descriptors",
            global_descriptors,
        )
        selectors: set[str] = set()
        previous_selector: str | None = None
        for selector in self.trusted_mcp_selectors:
            if (
                not isinstance(selector, str)
                or not selector.startswith("mcp__")
                or selector == "mcp__"
                or "__" in selector[5:]
                or selector in selectors
                or (previous_selector is not None and selector < previous_selector)
            ):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            selectors.add(selector)
            previous_selector = selector
        names: set[str] = set()
        previous: str | None = None
        for name, tool_class in self.trusted_tool_classes:
            if (
                not isinstance(name, str)
                or not name.strip()
                or tool_class not in _TRUSTED_TOOL_CLASSES
                or name in names
                or (previous is not None and name < previous)
            ):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            names.add(name)
            previous = name


__all__ = ["AgentDefinition"]
