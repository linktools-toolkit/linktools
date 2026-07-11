"""Explicit compatibility wrapper for legacy opaque toolsets."""

import warnings
from dataclasses import dataclass
from typing import Any

from ..errors import CapabilityResolutionError
from ..security.descriptor import ToolDescriptor
from .contribution import ToolContribution


@dataclass(frozen=True, slots=True)
class LegacyToolsetAdapter:
    toolset: Any
    descriptors: tuple[ToolDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.descriptors:
            raise CapabilityResolutionError(
                "legacy toolset adapter requires explicit descriptors")
        names = getattr(self.toolset, "tools", None)
        if isinstance(names, dict):
            declared = {descriptor.name for descriptor in self.descriptors}
            actual = {str(name) for name in names}
            if declared != actual:
                raise CapabilityResolutionError(
                    "legacy toolset adapter descriptors do not match toolset: "
                    f"missing={sorted(declared - actual)}, extra={sorted(actual - declared)}")
        warnings.warn(
            "LegacyToolsetAdapter is deprecated; migrate to ManagedToolDefinition",
            DeprecationWarning, stacklevel=2)

    def contribution(self) -> ToolContribution:
        return ToolContribution(toolset=self.toolset, descriptors=self.descriptors,
                                legacy_adapter=True)
