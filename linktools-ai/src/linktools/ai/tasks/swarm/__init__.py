#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swarm subsystem value types and specs. The task_graph strategy holds no
authoritative swarm-level state: the parent RunRecord and per-node
TaskExecution are the sole authorities. This package owns the AgentRef member
type, the SwarmSpec/policy/limits specs, the aggregation reductions, and the
strategy outcome shapes."""

from .models import AgentRef, SwarmCompleted, SwarmExecutionOutcome, SwarmFailed, SwarmRunView
from .spec import SwarmSpec

__all__ = [
    "AgentRef",
    "SwarmCompleted",
    "SwarmExecutionOutcome",
    "SwarmFailed",
    "SwarmRunView",
    "SwarmSpec",
]
