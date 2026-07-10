#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolContribution: the structured output a CapabilityProvider produces for each
tool it exposes. Pairs the pydantic-ai toolset with explicit ToolDescriptors so
the assembler + ManagedToolAdapter never need to guess tool names or categories
via runtime introspection."""

from dataclasses import dataclass, field
from typing import Any

from ..security.descriptor import ToolDescriptor


@dataclass(frozen=True, slots=True)
class ToolContribution:
    """A toolset paired with its descriptors. The assembler uses descriptors for
    conflict detection + counting; the adapter uses them for policy/pipeline."""
    toolset: Any
    descriptors: "tuple[ToolDescriptor, ...]" = ()
