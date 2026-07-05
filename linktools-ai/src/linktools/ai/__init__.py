#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai public API (spec section 5). The package root exports:
AgentSpec, Runtime, FileStorage, SqlAlchemyStorage, plus Storage (the composed
type callers receive). Every other name -- Compiler, Runner, Store
implementations, Middleware base classes, PolicyEngine, ToolExecutor, etc. --
is accessed via its own submodule and is not re-exported here.

# SwarmSpec: added in Phase 4 (the swarm plan), alongside SwarmRunner/SwarmStore.
"""

from .agent_runtime.spec import AgentSpec
from .runtime import Runtime
from .storage.facade import FileStorage, SqlAlchemyStorage, Storage

__all__ = ["AgentSpec", "Runtime", "FileStorage", "SqlAlchemyStorage", "Storage"]
