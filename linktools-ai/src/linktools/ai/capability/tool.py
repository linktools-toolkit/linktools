#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool state and policy boundaries."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ToolState:
    operation_id: str
    state: str
    result_digest: "str | None" = None

    def __post_init__(self) -> None:
        if not self.operation_id.strip() or not self.state.strip():
            raise ValueError("tool state is incomplete")


class ToolStateStore(Protocol):
    async def get(self, operation_id: str) -> 'ToolState | None': ...
    async def put(self, state: ToolState) -> ToolState: ...


class ToolPolicy(Protocol):
    def allowed(self, tool_id: str, profile: str) -> bool: ...


__all__ = ["ToolPolicy", "ToolState", "ToolStateStore"]
