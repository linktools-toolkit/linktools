#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""swarm: the unified Swarm subsystem's persistence contract and pure
domain models (SwarmRun/SwarmTask, AgentRef, TaskInput, TokenUsage) plus the
SwarmStore Protocol; backends are FilesystemSwarmStore (single-process)
and SqlAlchemySwarmStore (multi-process)."""

from .spec import SwarmSpec

__all__ = ["SwarmSpec"]
