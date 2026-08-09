#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Static Agent runtime support API."""

from ._binding import AgentBinding
from ._capabilities import (
    AgentCapabilityScope,
    AgentCatalogItem,
    AgentCatalogSnapshot,
    AgentCatalogView,
    EmptyAgentCatalog,
    EmptySkillCatalog,
    compose_parent_capabilities,
)
from ._deps import AgentDeps
from ._runner import AgentRunner, WorkspaceAgentResult, WorkspaceAgentRunner
from ..capability import SkillCatalogView

__all__ = [
    "AgentBinding", "AgentCapabilityScope", "AgentCatalogItem", "AgentCatalogSnapshot", "AgentCatalogView", "AgentDeps",
    "AgentRunner", "EmptyAgentCatalog", "EmptySkillCatalog", "SkillCatalogView", "WorkspaceAgentResult", "WorkspaceAgentRunner",
    "compose_parent_capabilities",
]
