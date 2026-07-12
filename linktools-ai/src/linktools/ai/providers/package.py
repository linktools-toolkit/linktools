#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PackageSpecProvider + PackageResourceProvider: source-agnostic surfaces for
Capability Packages (skill-creator-style bundles of resources + entrypoints).

PackageSpecProvider lists package declarations; PackageResourceProvider reads
their resources with pagination + size limits. Both are implemented by the
default DirectoryPackageRegistry and may be replaced by any business backend
(object storage, DB, git repo). The package-domain types live in
``linktools.ai.package``; only TYPE_CHECKING references are used here so this
module imports cleanly before the package package exists at runtime."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..package.resource import ResourceContent, ResourceListResult, ResourceRef
    from ..package.scope import PackageScope
    from ..package.spec import CapabilityPackageSpec


@runtime_checkable
class PackageSpecProvider(Protocol):
    """Provides CapabilityPackageSpec objects from any configuration source."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, package_id: str) -> "CapabilityPackageSpec": ...


@runtime_checkable
class PackageResourceProvider(Protocol):
    """Reads package resources. ``list_resources`` MUST paginate -- never return
    an entire package tree in one call."""

    async def list_resources(
        self,
        scope: "PackageScope",
        path: str = "",
        *,
        limit: int = 50,
        cursor: "str | None" = None,
    ) -> "ResourceListResult": ...

    async def read_resource(
        self,
        ref: "ResourceRef",
        *,
        max_bytes: "int | None" = None,
    ) -> "ResourceContent": ...
