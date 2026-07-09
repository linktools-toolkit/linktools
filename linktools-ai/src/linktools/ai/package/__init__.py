#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.package: package-scoped capability resolution (spec §13). A
package bundles resources + entrypoints under a PackageScope; Skill is one
built-in package kind, not a special case."""

from .capability_provider import PackageProvider
from .entrypoint import EntrypointInfo, EntrypointListResult, EntrypointRef
from .provider import DirectoryPackageResourceProvider, PackageResourceProvider
from .resolver import (
    DEFAULT_ENTRYPOINT_LIMIT,
    DirectoryEntrypointResolver,
    DirectoryPackageRegistry,
    EntrypointResolver,
    PackageRegistry,
)
from .resource import (
    ResourceContent,
    ResourceInfo,
    ResourceListResult,
    ResourceRef,
    sanitize_package_path,
)
from .scope import PackageScope
from .spec import CapabilityPackageSpec

__all__ = [
    "CapabilityPackageSpec",
    "PackageScope",
    "ResourceRef", "ResourceInfo", "ResourceContent", "ResourceListResult",
    "sanitize_package_path",
    "EntrypointRef", "EntrypointInfo", "EntrypointListResult",
    "PackageResourceProvider",  # re-exported Protocol
    "DirectoryPackageResourceProvider",
    "EntrypointResolver", "DirectoryEntrypointResolver",
    "PackageRegistry", "DirectoryPackageRegistry",
    "PackageProvider",
    "DEFAULT_ENTRYPOINT_LIMIT",
]
