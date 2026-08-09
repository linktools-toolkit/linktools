#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime Asset adapters for authorized Agent and Skill catalog views."""

import hashlib
import keyword
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.models import Model

from ..asset import AssetInfo, AssetKey, AssetStore
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec
from ._skill import SkillCatalogView, SkillDescriptor, SkillSpec


class AgentCatalogView(Protocol):
    async def list_agents(self) -> "tuple[AgentCatalogItem, ...]": ...


@dataclass(frozen=True, slots=True)
class AgentCatalogItem:
    id: str
    name: str
    description: str
    instructions: str
    model: "str | Model | None"

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip() or not self.instructions.strip():
            raise ValueError("agent catalog item is incomplete")
        if not self.name.isidentifier() or keyword.iskeyword(self.name):
            raise ValueError("agent catalog item name is not a valid workflow name")


@dataclass(frozen=True, slots=True)
class AgentCatalogSnapshot(AgentCatalogView):
    items: "tuple[AgentCatalogItem, ...]"

    def __post_init__(self) -> None:
        items = tuple(sorted(self.items, key=lambda item: item.id))
        object.__setattr__(self, "items", items)
        ids = tuple(item.id for item in items)
        names = tuple(item.name for item in items)
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("agent catalog contains duplicate ids or names")

    async def list_agents(self) -> "tuple[AgentCatalogItem, ...]":
        return self.items


@dataclass(frozen=True, slots=True)
class AssetAgentCatalog(AgentCatalogView):
    """Expose the already-authorized Runtime Agent assets as one run view."""

    store: AssetStore

    async def list_agents(self) -> "tuple[AgentCatalogItem, ...]":
        items: list[AgentCatalogItem] = []
        for info in await self._list_infos():
            specification = await self.store.get(info.key, expected=AgentSpec)
            if specification is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            instructions = "\n".join(specification.instructions).strip() or "Execute the assigned task."
            items.append(
                AgentCatalogItem(
                    id=specification.id,
                    name=_workflow_agent_name(specification.id),
                    description=f"Authorized agent {specification.id}",
                    instructions=instructions,
                    model=specification.model,
                )
            )
        return tuple(sorted(items, key=lambda item: item.id))
    async def _list_infos(self) -> "list[AssetInfo]":
        infos: "list[AssetInfo]" = []
        cursor = None
        while True:
            page = await self.store.list_info(kind="agent", cursor=cursor, limit=200)
            infos.extend(page.items)
            if page.next_cursor is None:
                return infos
            if page.next_cursor == cursor or not page.items:
                raise AIError(ErrorCode.CURSOR_INVALID)
            cursor = page.next_cursor


def _workflow_agent_name(agent_id: str) -> str:
    if agent_id.isidentifier() and not keyword.iskeyword(agent_id):
        return agent_id
    return "agent_" + hashlib.sha256(agent_id.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AssetSkillCatalog(SkillCatalogView):
    """Expose the already-authorized Runtime Skill assets as one run view."""

    store: AssetStore

    async def list_skills(self) -> "tuple[SkillDescriptor, ...]":
        descriptors: list[SkillDescriptor] = []
        for info in await self._list_infos():
            specification = await self.store.get(info.key, expected=SkillSpec)
            if specification is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            descriptors.append(SkillDescriptor(info.key.id, specification.revision, f"Authorized skill {info.key.id}"))
        return tuple(sorted(descriptors, key=lambda item: item.id))

    async def load_skill(self, skill_id: str) -> "SkillSpec | None":
        return await self.store.get(AssetKey("skill", skill_id), expected=SkillSpec)

    async def _list_infos(self) -> "list[AssetInfo]":
        infos: "list[AssetInfo]" = []
        cursor = None
        while True:
            page = await self.store.list_info(kind="skill", cursor=cursor, limit=200)
            infos.extend(page.items)
            if page.next_cursor is None:
                return infos
            if page.next_cursor == cursor or not page.items:
                raise AIError(ErrorCode.CURSOR_INVALID)
            cursor = page.next_cursor


__all__ = [
    "AgentCatalogItem", "AgentCatalogSnapshot", "AgentCatalogView", "AssetAgentCatalog", "AssetSkillCatalog",
]
