#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package entrypoint types. An entrypoint is an addressable
object inside a package -- an agent, skill, tool, mcp, workflow, or script.
Scoped entrypoints are namespaced by their PackageScope so two packages can
both expose ``agent:grader`` without colliding (internal key
``package:<id>:<kind>:<name>``)."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .scope import PackageScope

_ENTRYPOINT_KINDS = ("agent", "skill", "tool", "mcp", "workflow", "script")


@dataclass(frozen=True, slots=True)
class EntrypointRef:
    kind: str
    name: str
    scope: "PackageScope | None" = None

    def internal_key(self) -> str:
        scope_id = self.scope.package_id if self.scope is not None else "_global"
        return f"package:{scope_id}:{self.kind}:{self.name}"


class EntrypointInfo(BaseModel):
    kind: str
    name: str
    package_id: "str | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)


class EntrypointListResult(BaseModel):
    items: "list[EntrypointInfo]"
    next_cursor: "str | None" = None
