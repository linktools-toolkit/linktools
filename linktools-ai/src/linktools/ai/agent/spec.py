#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentSpec: an immutable, serializable Agent declaration.
Holds no runtime state -- no Session, no Run, no Store, no working directory."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel

from ..model.policy import ModelPolicy
from .assembly.models import AgentFeatureRef


@dataclass(frozen=True, slots=True)
class PromptSpec:
    instructions: str
    sections: "Mapping[str, str]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str):
            raise TypeError("PromptSpec.instructions must be a string")
        from ..json import freeze_value

        object.__setattr__(self, "sections", freeze_value(dict(self.sections)))


@dataclass(frozen=True, slots=True)
class MiddlewareRef:
    name: str
    config: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("MiddlewareRef.name must be a non-empty string")
        from ..json import freeze_value

        object.__setattr__(self, "config", freeze_value(dict(self.config)))


@dataclass(frozen=True, slots=True)
class AgentSpec:
    id: str
    name: str
    model: ModelPolicy
    instructions: PromptSpec
    features: "tuple[AgentFeatureRef, ...]" = ()
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
        if not isinstance(self.features, tuple) or not all(
            isinstance(feature, AgentFeatureRef) for feature in self.features
        ):
            raise TypeError("AgentSpec.features must be tuple[AgentFeatureRef]")
        if not isinstance(self.middleware, tuple) or not all(
            isinstance(m, MiddlewareRef) for m in self.middleware
        ):
            raise TypeError("AgentSpec.middleware must be tuple[MiddlewareRef]")
        from ..json import freeze_value

        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


@runtime_checkable
class AgentSpecProvider(Protocol):
    """Provides AgentSpec objects from any configuration source. Any backend
    -- file registry, DB, config center, HTTP API -- can implement it; the
    Runtime never imports a concrete registry."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, agent_id: str) -> "AgentSpec": ...
