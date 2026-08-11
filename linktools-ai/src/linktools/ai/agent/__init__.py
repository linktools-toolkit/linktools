#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent declaration binding and execution boundaries."""

from ._binding import (
    AgentBinder,
    AgentBinding,
    AgentBindingManifest,
    AgentBindingRegistry,
)
from ._capabilities import (
    AgentRunScope,
    EmptyAgentCatalog,
    EmptySkillCatalog,
    compose_platform_capabilities,
)
from ._catalog import AgentCatalogItem, AgentCatalogSnapshot, AgentCatalogView
from ._deps import AgentDeps
from ._output import (
    ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
    ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
    AssistantTextOutput,
    OutputSchemaManifest,
    OutputSchemaManifestEntry,
    OutputTypeRegistry,
)
from ._runner import (
    AgentRunner,
    BoundAgentRunner,
    EventHandler,
    WorkspaceAgentResult,
    WorkspaceAgentRunner,
)

__all__ = [
    "AgentBinder", "AgentBinding", "AgentBindingManifest", "AgentBindingRegistry", "AgentCatalogItem",
    "AgentCatalogSnapshot", "AgentCatalogView", "AgentDeps", "AgentRunScope", "AgentRunner", "BoundAgentRunner", "EventHandler",
    "ASSISTANT_TEXT_OUTPUT_SCHEMA_ID", "ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION", "AssistantTextOutput", "EmptyAgentCatalog",
    "EmptySkillCatalog", "OutputSchemaManifest", "OutputSchemaManifestEntry", "OutputTypeRegistry", "WorkspaceAgentResult",
    "WorkspaceAgentRunner", "compose_platform_capabilities",
]
