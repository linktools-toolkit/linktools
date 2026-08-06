#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent release lookup protocol."""

from typing import Protocol


class AgentReleaseRepository(Protocol):
    async def get(self, agent_id: str, revision: int) -> "object | None": ...
    async def get_enabled(self, agent_id: str) -> "object | None": ...
    async def save(self, release: object) -> object: ...


__all__ = ["AgentReleaseRepository"]
