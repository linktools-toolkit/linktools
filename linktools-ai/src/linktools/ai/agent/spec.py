#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentSpec: an immutable, serializable Agent declaration.
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
    # kind identifies the capability provider ("builtin", "skill", "mcp",
    # "subagent", "package", "package-resource", "package-entrypoint"). When
    # None the resolver treats a bare name as a builtin tool (backward compat
    # with ``tools: [file, terminal]``). A "kind:name" string (e.g. "skill:sql")
    # is split into kind + name by parse_tool_refs.
    kind: "str | None" = None
    config: "Mapping[str, Any]" = field(default_factory=dict)


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
    # tools three-state: None = unset (runtime applies its default
    # builtin toolset when an execution backend is present); () = explicitly no
    # tools; a non-empty tuple = only the declared capabilities are assembled.
    tools: "tuple[ToolRef, ...] | None" = None
    middleware: "tuple[MiddlewareRef, ...]" = ()
    output_schema: "type[BaseModel] | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
