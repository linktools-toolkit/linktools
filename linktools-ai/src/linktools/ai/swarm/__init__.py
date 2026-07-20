#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""swarm: the unified Swarm subsystem's persistence contract and pure
domain models (SwarmRun/SwarmTask, AgentRef, TaskInput, TokenUsage) plus the
SwarmStore Protocol. Backends (FilesystemSwarmStore, SqlAlchemySwarmStore) land in
later phases."""

from .spec import SwarmSpec

__all__ = ["SwarmSpec"]
