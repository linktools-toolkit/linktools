#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent feature declarations and the single assembly boundary."""

from .assembler import AgentAssembler
from .inspection import AgentInspection
from .models import (
    AgentAssembly,
    AgentContribution,
    AgentFeatureRef,
    parse_agent_feature_refs,
)
from .provider import AgentAssemblyEventSink, AgentFeatureContext, AgentFeatureProvider
from .registry import AgentFeatureRegistry

__all__ = [
    "AgentAssembler",
    "AgentAssembly",
    "AgentAssemblyEventSink",
    "AgentContribution",
    "AgentFeatureContext",
    "AgentFeatureProvider",
    "AgentFeatureRef",
    "AgentFeatureRegistry",
    "AgentInspection",
    "parse_agent_feature_refs",
]
