#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Runtime capabilities for optional durable execution evidence."""

from typing import TypeVar

from pydantic_ai.capabilities import AbstractCapability

from ..capability import AgentContext

AppT = TypeVar("AppT")


class RestartCheckpointCapability(AbstractCapability[AgentContext[AppT]]):
    """Retain model-step checkpoints for explicitly recovered execution.

    Register at Runtime scope. Runtime owns checkpoint transactions and producer
    fencing; ordinary native hooks must not commit these facts independently.
    """

    id: str | None = "runtime.restart-checkpoints"


class DurableTraceCapability(AbstractCapability[AgentContext[AppT]]):
    """Retain execution transcripts, model interactions and step observations.

    This does not enable restarting an interrupted execution.
    """

    id: str | None = "runtime.durable-trace"


__all__ = ["RestartCheckpointCapability", "DurableTraceCapability"]
