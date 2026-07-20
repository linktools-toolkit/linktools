#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExtensionSpec: an extension declaration. An extension bundles
resources + entrypoints under one id/kind; ``skill`` is one built-in kind, but
the model is general (agentpack / toolpack / mcp-pack / workflow / custom)."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .resource import ResourceContent, ResourceListResult, ResourceRef
from .scope import ExtensionScope

_BUILTIN_KINDS = ("skill", "agentpack", "toolpack", "mcp-pack", "workflow", "custom")


@dataclass(frozen=True, slots=True)
class ExtensionSpec:
    id: str
    name: str
    kind: str
    version: "str | None" = None
    root: "ResourceRef | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    @property
    def scope(self) -> ExtensionScope:
        return ExtensionScope(extension_id=self.id, extension_kind=self.kind)


@runtime_checkable
class ExtensionSpecProvider(Protocol):
    """Provides ExtensionSpec objects from any configuration source."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, extension_id: str) -> "ExtensionSpec": ...


@runtime_checkable
class ExtensionResourceProvider(Protocol):
    """Reads extension resources. ``list_resources`` MUST paginate -- never
    return an entire extension tree in one call."""

    async def list_resources(
        self,
        scope: ExtensionScope,
        path: str = "",
        *,
        limit: int = 50,
        cursor: "str | None" = None,
    ) -> ResourceListResult: ...

    async def read_resource(
        self,
        ref: ResourceRef,
        *,
        max_bytes: "int | None" = None,
    ) -> ResourceContent: ...
