#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunDispatcher: the narrow interface Job/Swarm/Subagent execution depends on
to run a compiled agent, instead of importing AgentEngine or Runtime-builder
internals directly."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .context import RunContext
from .models import RunInput, RunResult

if TYPE_CHECKING:
    from ..agent.models import CompiledAgent


@dataclass(frozen=True, slots=True)
class RunDispatchRequest:
    agent: "CompiledAgent"
    input: RunInput
    context: RunContext


@runtime_checkable
class RunDispatcher(Protocol):
    async def dispatch(self, request: RunDispatchRequest) -> RunResult: ...
