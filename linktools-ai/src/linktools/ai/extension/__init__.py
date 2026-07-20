#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.extension: extension-scoped capability resolution. An
extension bundles resources + entrypoints under a ExtensionScope; Skill is one
built-in extension kind, not a special case."""

from .capability_provider import ExtensionProvider
from .entrypoint import EntrypointInfo, EntrypointListResult, EntrypointRef
from .provider import DirectoryExtensionResourceProvider, ExtensionResourceProvider
from .resolver import (
    DEFAULT_ENTRYPOINT_LIMIT,
    DirectoryEntrypointResolver,
    DirectoryExtensionRegistry,
    EntrypointResolver,
    ExtensionRegistry,
)
from .resource import (
    ResourceContent,
    ResourceInfo,
    ResourceListResult,
    ResourceRef,
    sanitize_extension_path,
)
from .scope import ExtensionScope
from .spec import ExtensionSpec

__all__ = [
    "ExtensionSpec",
    "ExtensionScope",
    "ResourceRef",
    "ResourceInfo",
    "ResourceContent",
    "ResourceListResult",
    "sanitize_extension_path",
    "EntrypointRef",
    "EntrypointInfo",
    "EntrypointListResult",
    "ExtensionResourceProvider",  # re-exported Protocol
    "DirectoryExtensionResourceProvider",
    "EntrypointResolver",
    "DirectoryEntrypointResolver",
    "ExtensionRegistry",
    "DirectoryExtensionRegistry",
    "ExtensionProvider",
    "DEFAULT_ENTRYPOINT_LIMIT",
]
