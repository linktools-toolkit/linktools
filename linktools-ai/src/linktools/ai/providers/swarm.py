#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmSpecProvider: source-agnostic surface for SwarmSpec objects."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..swarm.spec import SwarmSpec


@runtime_checkable
class SwarmSpecProvider(Protocol):
    """Provides SwarmSpec objects from any configuration source."""

    async def list_ids(self) -> "tuple[str, ...]":
        ...

    async def get(self, swarm_id: str) -> "SwarmSpec":
        ...
