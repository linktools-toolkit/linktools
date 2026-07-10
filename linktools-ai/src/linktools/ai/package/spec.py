#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityPackageSpec: a package declaration. A package bundles
resources + entrypoints under one id/kind; ``skill`` is one built-in kind, but
the model is general (agentpack / toolpack / mcp-pack / workflow / custom)."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from .resource import ResourceRef
from .scope import PackageScope

_BUILTIN_KINDS = ("skill", "agentpack", "toolpack", "mcp-pack", "workflow", "custom")


@dataclass(frozen=True, slots=True)
class CapabilityPackageSpec:
    id: str
    name: str
    kind: str
    version: "str | None" = None
    root: "ResourceRef | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    @property
    def scope(self) -> PackageScope:
        return PackageScope(package_id=self.id, package_kind=self.kind)
