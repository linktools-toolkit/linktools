#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composition adapters from raw Asset files to capability catalogs."""

import hashlib
import keyword
from dataclasses import dataclass

from ..agent import AgentCatalogItem, AgentCatalogView
from ..asset import AssetInfo, AssetKey, AssetStore
from ..capability import (
    SkillCatalogView,
    SkillDescriptor,
)
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec, AgentSpecCodec, SkillSpec, SkillSpecCodec

_AGENT_CODEC = AgentSpecCodec()
_SKILL_CODEC = SkillSpecCodec()


@dataclass(frozen=True, slots=True)
class AssetAgentCatalog(AgentCatalogView):
    """Expose authorized agent files through the capability catalog contract."""

    store: AssetStore

    async def list_agents(self) -> "tuple[AgentCatalogItem, ...]":
        items: list[AgentCatalogItem] = []
        for info in await self._list_infos():
            content = await self.store.get(info.key)
            if content is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            specification = _decode_agent(content)
            if specification.id != info.key.id:
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
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


@dataclass(frozen=True, slots=True)
class AssetSkillCatalog(SkillCatalogView):
    """Expose authorized skill files through the capability catalog contract."""

    store: AssetStore

    async def list_skills(self) -> "tuple[SkillDescriptor, ...]":
        descriptors: list[SkillDescriptor] = []
        for info in await self._list_infos():
            content = await self.store.get(info.key)
            if content is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            specification = _decode_skill(content)
            if specification.id != info.key.id:
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
            descriptors.append(SkillDescriptor(info.key.id, specification.revision, f"Authorized skill {info.key.id}"))
        return tuple(sorted(descriptors, key=lambda item: item.id))

    async def load_skill(self, skill_id: str) -> "SkillSpec | None":
        content = await self.store.get(AssetKey("skill", skill_id))
        if content is None:
            return None
        specification = _decode_skill(content)
        if specification.id != skill_id:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        return specification

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


def _workflow_agent_name(agent_id: str) -> str:
    if agent_id.isidentifier() and not keyword.iskeyword(agent_id):
        return agent_id
    return "agent_" + hashlib.sha256(agent_id.encode("utf-8")).hexdigest()


def _decode_agent(content: bytes) -> AgentSpec:
    try:
        return _AGENT_CODEC.decode(content)
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error


def _decode_skill(content: bytes) -> SkillSpec:
    try:
        return _SKILL_CODEC.decode(content)
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error


__all__ = ["AssetAgentCatalog", "AssetSkillCatalog"]
