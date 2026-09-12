#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application-owned pure TaskGraph expansion contracts."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel
from pydantic_ai.messages import UserContent

from ..core import JsonValue, Principal, ThinkingValue
from ..task import TaskExpanderRef, TaskNode


class TaskExpansionContext(Protocol):
    @property
    def principal(self) -> Principal: ...

    @property
    def graph_id(self) -> str: ...

    @property
    def source_node(self) -> TaskNode: ...

    @property
    def output(self) -> JsonValue: ...

    def agent_task(
        self,
        agent_id: str,
        node_id: str,
        user_prompt: str | Sequence[UserContent],
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
        output: type[BaseModel] | None = None,
        planning: bool | None = None,
        thinking: ThinkingValue | None = None,
        expander: TaskExpanderRef | None = None,
    ) -> TaskNode: ...


@runtime_checkable
class TaskExpander(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def version(self) -> int: ...

    def expand(self, context: TaskExpansionContext) -> Sequence[TaskNode]: ...


__all__ = ["TaskExpander", "TaskExpansionContext"]
