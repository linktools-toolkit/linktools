#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentSpec: an immutable, serializable Agent declaration, per spec section 7.1.
Holds no runtime state -- no Session, no Run, no Store, no working directory."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import BaseModel

from ..model.policy import ModelPolicy


@dataclass(frozen=True, slots=True)
class PromptSpec:
    instructions: str
    sections: "Mapping[str, str]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolRef:
    name: str


@dataclass(frozen=True, slots=True)
class MiddlewareRef:
    name: str
    config: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    id: str
    name: str
    model: ModelPolicy
    instructions: PromptSpec
    tools: "tuple[ToolRef, ...]" = ()
    middleware: "tuple[MiddlewareRef, ...]" = ()
    output_schema: "type[BaseModel] | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
