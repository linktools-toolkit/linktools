#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-scoped Harness and platform capability composition."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from linktools.core import environ
from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability as PydanticAgentCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    DeduplicateFileReads,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.conversation_search import (
    ConversationSearch,
    SnapshotHistorySource,
)
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
from pydantic_ai_harness.memory import Memory, SearchableMemoryStore
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.step_persistence import StepPersistence, StepStore
from pydantic_ai_harness.subagents import SubAgent, SubAgents

from ..capability import SkillCatalogView, SkillDescriptor
from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import SkillSpec
from ._catalog import AgentCatalogItem, AgentCatalogSnapshot, AgentCatalogView

_logger = environ.get_logger("ai.agent.capabilities")


@dataclass(frozen=True, slots=True)
class AgentRunScope:
    root: Path
    agent_name: str
    conversation_id: "str | None"
    step_run_id: str
    segment_sequence: "int | None"
    memory_namespace: "str | None"
    step_store: StepStore
    inherited_capabilities: "tuple[PydanticAgentCapability[None], ...]"
    agent_catalog: AgentCatalogView
    memory_store: "SearchableMemoryStore | None"
    context_target_tokens: "int | None" = None
    parent_step_run_id: "str | None" = None


async def compose_platform_capabilities(
    scope: AgentRunScope,
    *,
    model_factory: Callable[["str | Model | None"], "str | Model"],
    parent_model: "str | Model",
) -> "tuple[PydanticAgentCapability[None], ...]":
    _validate_compaction_target(scope.context_target_tokens)
    capabilities: "list[PydanticAgentCapability[None]]" = [
        StepPersistence(
            store=scope.step_store,
            agent_name=scope.agent_name,
            run_id=scope.step_run_id,
            parent_run_id=scope.parent_step_run_id,
            metadata={
                "capability_scope": "parent",
                "agent_name": scope.agent_name,
                **({} if scope.segment_sequence is None else {"segment_sequence": str(scope.segment_sequence)}),
            },
        )
    ]
    if scope.memory_namespace is not None:
        if scope.memory_store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(Memory(store=scope.memory_store, namespace="", agent_name="memory", inject_memory=False))
    capabilities.append(Planning())
    if scope.conversation_id:
        capabilities.append(ConversationSearch(SnapshotHistorySource(scope.step_store), scope="conversation"))
    agent_catalog = AgentCatalogSnapshot(await scope.agent_catalog.list_agents())
    if agent_catalog.items:
        child_agents = _build_child_agents(scope, agent_catalog.items, model_factory=model_factory, parent_model=parent_model)
        capabilities.append(
            SubAgents(
                agents=[SubAgent(child) for child in child_agents],
                agent_folders=None,
                inherit_tools=False,
                forward_usage=True,
            )
        )
        capabilities.append(DynamicWorkflow(agents=child_agents))
    capabilities.append(_build_compaction(scope.context_target_tokens))
    _logger.info(
        "Harness platform capabilities composed: agent=%s conversation=%s step=%s capability_count=%s inherited_count=%s agent_count=%s namespace_digest=%s",
        scope.agent_name,
        bool(scope.conversation_id),
        scope.step_run_id,
        len(capabilities),
        len(scope.inherited_capabilities),
        len(agent_catalog.items),
        None if scope.memory_namespace is None else _namespace_digest(scope.memory_namespace),
    )
    return tuple(capabilities)


def _build_child_agents(
    scope: AgentRunScope,
    items: tuple[AgentCatalogItem, ...],
    *,
    model_factory: Callable[["str | Model | None"], "str | Model"],
    parent_model: "str | Model",
) -> "list[Agent[None, str]]":
    children: "list[Agent[None, str]]" = []
    for item in items:
        model = model_factory(item.model if item.model is not None else parent_model)
        child_scope = AgentRunScope(
            root=scope.root,
            agent_name=item.name,
            conversation_id=None,
            step_run_id="",
            segment_sequence=None,
            memory_namespace=scope.memory_namespace,
            step_store=scope.step_store,
            inherited_capabilities=scope.inherited_capabilities,
            agent_catalog=AgentCatalogSnapshot(()),
            memory_store=scope.memory_store,
            context_target_tokens=scope.context_target_tokens,
        )
        child_capabilities = _compose_child_capabilities(child_scope, item.name)
        children.append(
            Agent(
                model,
                name=item.name,
                description=item.description,
                instructions=item.instructions,
                capabilities=child_capabilities,
            )
        )
        _logger.info("Harness child agent composed: agent=%s capability_count=%s", item.name, len(child_capabilities))
    return children


def _compose_child_capabilities(scope: AgentRunScope, agent_name: str) -> "tuple[PydanticAgentCapability[None], ...]":
    _validate_compaction_target(scope.context_target_tokens)
    capabilities: "list[PydanticAgentCapability[None]]" = list(scope.inherited_capabilities)
    capabilities.append(
        StepPersistence(
            store=scope.step_store,
            agent_name=agent_name,
            run_id=None,
            metadata={"capability_scope": "child", "agent_name": agent_name},
        )
    )
    if scope.memory_namespace is not None:
        if scope.memory_store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(Memory(store=scope.memory_store, namespace="", agent_name="memory", inject_memory=False))
    capabilities.append(Planning())
    capabilities.append(_build_compaction(scope.context_target_tokens))
    return tuple(capabilities)


def _build_compaction(context_target_tokens: "int | None") -> PydanticAgentCapability[None]:
    deduplicate = DeduplicateFileReads(file_key=_workspace_file_key)
    if context_target_tokens is None:
        return deduplicate
    return TieredCompaction(
        tiers=[deduplicate, ClearToolResults(max_tokens=1, keep_pairs=3), SummarizingCompaction(max_messages=1, keep_messages=20)],
        target_tokens=context_target_tokens,
    )


def _validate_compaction_target(context_target_tokens: "int | None") -> None:
    if context_target_tokens is not None and (not isinstance(context_target_tokens, int) or isinstance(context_target_tokens, bool) or context_target_tokens <= 0):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _workspace_file_key(part: ToolCallPart) -> "str | None":
    if part.tool_name != "read_file":
        return None
    try:
        arguments = part.args_as_dict()
    except (TypeError, ValueError):
        return None
    path = arguments.get("path")
    return path if isinstance(path, str) else None


def _namespace_digest(namespace: str) -> str:
    return canonical_sha256(namespace)


@dataclass(frozen=True, slots=True)
class EmptySkillCatalog(SkillCatalogView):
    async def list_skills(self) -> "tuple[SkillDescriptor, ...]":
        return ()

    async def load_skill(self, skill_id: str) -> "SkillSpec | None":
        del skill_id
        return None


@dataclass(frozen=True, slots=True)
class EmptyAgentCatalog(AgentCatalogView):
    async def list_agents(self) -> "tuple[AgentCatalogItem, ...]":
        return ()


__all__ = ["AgentRunScope", "EmptyAgentCatalog", "EmptySkillCatalog", "compose_platform_capabilities"]
