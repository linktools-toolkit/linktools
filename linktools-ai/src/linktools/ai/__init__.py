#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai public API (spec §20.3). The package root re-exports the hot
types so downstream imports stay short:

    from linktools.ai import Runtime, AgentSpec, ToolRef, Storage, FileStorage

SqlAlchemyStorage is intentionally NOT re-exported here -- it depends on the
optional SQLAlchemy extra and is loaded lazily from ``linktools.ai.storage``
(spec §21.7). Every other name (Compiler, Runner, Store implementations,
Middleware, PolicyEngine, ToolExecutor, providers, ...) is reached via its own
submodule.

Importing this package has no heavy side effects: no file scans, no DB/MCP
connections, no Runtime construction."""

from .agent import AgentSpec, MiddlewareRef, PromptSpec, ToolRef
from .model import ModelPolicy, ModelRouter, RuntimeModelConfig
from .runtime import Runtime
from .storage import FileStorage, Storage
from .swarm.spec import SwarmSpec

__all__ = [
    "Runtime",
    "AgentSpec",
    "PromptSpec",
    "ToolRef",
    "MiddlewareRef",
    "ModelPolicy",
    "ModelRouter",
    "RuntimeModelConfig",
    "Storage",
    "FileStorage",
    "SwarmSpec",
]
