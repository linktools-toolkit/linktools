#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent catalog contracts and immutable snapshots."""

import keyword
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.models import Model


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


__all__ = ["AgentCatalogItem", "AgentCatalogSnapshot", "AgentCatalogView"]
