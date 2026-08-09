#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process composition roots for Runtime, workspace and transports."""

from ._acp import ACPAgent, ACPApplication, ACPConnection, run_stdio, serve_stdio
from ._assembly import (
    AppServices,
    RuntimePersistenceConfig,
    RuntimeResources,
    build_app_services,
    build_asset_codecs,
    build_runtime_services,
    namespace_scoped_step_db_path,
    open_runtime_services,
    open_runtime_resources,
)
from ._facade import LocalRuntimeDependencies, Runtime, RuntimeDependencies, build_local_runtime, build_runtime, build_runtime_access
from ._fields import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, REMOTE_RUNTIME_URL
from ._workbench import (
    BindingExecutionLauncher,
    EventHandler,
    TextHandler,
    WorkspaceAgentRuntime,
    WorkspaceExecutionLauncher,
    WorkspaceRunResult,
    WorkspaceSession,
    open_workspace_runtime,
)

__all__ = [
    "ACPAgent", "ACPApplication", "ACPConnection", "AppServices", "BindingExecutionLauncher", "EventHandler", "OPENAI_API_KEY",
    "OPENAI_BASE_URL", "OPENAI_MODEL", "REMOTE_RUNTIME_URL", "LocalRuntimeDependencies", "Runtime", "RuntimeDependencies",
    "RuntimePersistenceConfig", "RuntimeResources", "TextHandler", "WorkspaceAgentRuntime", "WorkspaceExecutionLauncher",
    "WorkspaceRunResult", "WorkspaceSession", "build_app_services",
    "build_asset_codecs", "build_local_runtime", "build_runtime", "build_runtime_access", "build_runtime_services",
    "namespace_scoped_step_db_path", "open_runtime_services", "open_runtime_resources", "open_workspace_runtime",
    "run_stdio", "serve_stdio",
]
