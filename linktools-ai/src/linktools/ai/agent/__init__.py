#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static Agent definition, compilation, binding, and output contracts."""

from ._binding import AgentBinding, AgentBindingSnapshot, SemanticPin, SubagentRef
from ._catalog import AgentCatalog
from ._compiler import AgentCompiler
from ._definition import AgentDefinition
from ._output import AssistantTextOutput, OutputBinding, OutputMode, bind_output, restore_output

__all__ = [
    "AgentBinding",
    "AgentBindingSnapshot",
    "AgentCatalog",
    "AgentCompiler",
    "AgentDefinition",
    "AssistantTextOutput",
    "OutputBinding",
    "OutputMode",
    "SemanticPin",
    "SubagentRef",
    "bind_output",
    "restore_output",
]
