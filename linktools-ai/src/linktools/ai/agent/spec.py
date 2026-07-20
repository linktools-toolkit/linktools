#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentSpec: an immutable, serializable Agent declaration.
Holds no runtime state -- no Session, no Run, no Store, no working directory."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel

from ..model.policy import ModelPolicy
from ..tool.models import ToolRef  # re-exported below; ToolRef is a tool-domain type


@dataclass(frozen=True, slots=True)
class PromptSpec:
    instructions: str
    sections: "Mapping[str, str]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str):
            raise TypeError("PromptSpec.instructions must be a string")
        from ..utils.freeze import freeze_value

        object.__setattr__(self, "sections", freeze_value(dict(self.sections)))


# ToolRef now lives in ..tool.models (its proper home -- a tool reference, not
# an agent-specific type). The import above re-exports it, so existing
# `from linktools.ai.agent.spec import ToolRef` call sites keep working; new
# code should import it from linktools.ai.tool.models.


@dataclass(frozen=True, slots=True)
class MiddlewareRef:
    name: str
    config: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("MiddlewareRef.name must be a non-empty string")
        from ..utils.freeze import freeze_value

        object.__setattr__(self, "config", freeze_value(dict(self.config)))


@dataclass(frozen=True, slots=True)
class AgentSpec:
    id: str
    name: str
    model: ModelPolicy
    instructions: PromptSpec
    # tools three-state: None = unset (runtime applies its default
    # builtin toolset when an execution backend is present); () = explicitly no
    # tools; a non-empty tuple = only the declared capabilities are assembled.
    tools: "tuple[ToolRef, ...] | None" = None
    middleware: "tuple[MiddlewareRef, ...]" = ()
    output_schema: "type[BaseModel] | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("AgentSpec.id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("AgentSpec.name must be a non-empty string")
        if not isinstance(self.model, ModelPolicy):
            raise TypeError("AgentSpec.model must be a ModelPolicy")
        if not isinstance(self.instructions, PromptSpec):
            raise TypeError("AgentSpec.instructions must be a PromptSpec")
        if self.tools is not None:
            if not isinstance(self.tools, tuple) or not all(
                isinstance(t, ToolRef) for t in self.tools
            ):
                raise TypeError("AgentSpec.tools must be None or tuple[ToolRef]")
        if not isinstance(self.middleware, tuple) or not all(
            isinstance(m, MiddlewareRef) for m in self.middleware
        ):
            raise TypeError("AgentSpec.middleware must be tuple[MiddlewareRef]")
        from ..utils.freeze import freeze_value

        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


@runtime_checkable
class AgentSpecProvider(Protocol):
    """Provides AgentSpec objects from any configuration source. Any backend
    -- file registry, DB, config center, HTTP API -- can implement it; the
    Runtime never imports a concrete registry."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, agent_id: str) -> "AgentSpec": ...
