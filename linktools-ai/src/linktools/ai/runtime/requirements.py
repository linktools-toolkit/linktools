#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runtime component and topology requirements."""

from dataclasses import dataclass
from enum import StrEnum

from ..agent.tool.exposure import ToolExposurePolicy


class RuntimeTopology(StrEnum):
    SINGLE_PROCESS = "single_process"
    MULTI_PROCESS = "multi_process"


@dataclass(frozen=True, slots=True)
class RuntimeRequirements:
    tools: bool = False
    tasks: bool = False
    memory: bool = False
    artifacts: bool = False
    topology: RuntimeTopology = RuntimeTopology.SINGLE_PROCESS
    tool_exposure: ToolExposurePolicy = ToolExposurePolicy()


__all__ = ["RuntimeRequirements", "RuntimeTopology"]
