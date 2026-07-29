#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extension entrypoint types. An entrypoint is an addressable
object inside an extension -- an agent, skill, tool, mcp, workflow, or script.
Scoped entrypoints are namespaced by their ExtensionScope so two extensions can
both expose ``agent:grader`` without colliding (internal key
``extension:<id>:<kind>:<name>``)."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .scope import ExtensionScope

_ENTRYPOINT_KINDS = ("agent", "skill", "tool", "mcp", "workflow", "script")


@dataclass(frozen=True, slots=True)
class EntrypointRef:
    kind: str
    name: str
    scope: "ExtensionScope | None" = None

    def internal_key(self) -> str:
        scope_id = self.scope.extension_id if self.scope is not None else "_global"
        return f"extension:{scope_id}:{self.kind}:{self.name}"


class EntrypointInfo(BaseModel):
    kind: str
    name: str
    extension_id: "str | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)


class EntrypointListResult(BaseModel):
    items: "list[EntrypointInfo]"
    next_cursor: "str | None" = None
