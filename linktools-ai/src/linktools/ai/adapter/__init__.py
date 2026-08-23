#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapters for external runtimes and framework boundaries."""

from ._history import StepExecutionHistoryReader, StepSessionHistoryReader
from ._identity import StaticPrincipalProvider
from ._memory import RuntimeMemoryStore
from ._nats import NatsPublisher
from ._provider import ProviderClient

__all__ = [
    "NatsPublisher",
    "ProviderClient",
    "PydanticMCPRuntime",
    "RuntimeMemoryStore",
    "StaticPrincipalProvider",
    "StepExecutionHistoryReader",
    "StepSessionHistoryReader",
]

from ._mcp import PydanticMCPRuntime
