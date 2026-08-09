#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Static Agent runtime support API."""

from ._binding import AgentBinding, BindingDependencies, BindingExecutionPlan, BindingExecutionRegistry, build_binding_plan
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
from ._runner import AgentRunner, BindingAgentRunner, ModelMaterializer, WorkspaceAgentResult, WorkspaceAgentRunner
from ..capability import SkillCatalogView

__all__ = [
    "AgentBinding", "AgentCapabilityScope", "AgentCatalogItem", "AgentCatalogSnapshot", "AgentCatalogView", "AgentDeps",
    "AgentRunner", "BindingAgentRunner", "BindingDependencies", "BindingExecutionPlan", "BindingExecutionRegistry", "EmptyAgentCatalog", "EmptySkillCatalog", "ModelMaterializer", "SkillCatalogView", "WorkspaceAgentResult", "WorkspaceAgentRunner", "build_binding_plan",
    "compose_parent_capabilities",
]
