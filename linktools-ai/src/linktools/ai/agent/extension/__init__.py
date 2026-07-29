#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.agent.extension: extension-scoped feature resolution. An
extension bundles assets + entrypoints under a ExtensionScope; Skill is one
built-in extension kind, not a special case."""

from .provider import ExtensionProvider
from .entrypoint import EntrypointInfo, EntrypointListResult, EntrypointRef
from .content_source import DirectoryExtensionContentSource, ExtensionContentSource
from .resolver import (
    DEFAULT_ENTRYPOINT_LIMIT,
    DirectoryEntrypointResolver,
    DirectoryExtensionRegistry,
    EntrypointResolver,
    ExtensionRegistry,
)
from .content import (
    ExtensionContent,
    ExtensionContentInfo,
    ExtensionContentPage,
    ExtensionContentRef,
    sanitize_extension_path,
)
from .scope import ExtensionScope
from .spec import ExtensionSpec

__all__ = [
    "ExtensionSpec",
    "ExtensionScope",
    "ExtensionContentRef",
    "ExtensionContentInfo",
    "ExtensionContent",
    "ExtensionContentPage",
    "sanitize_extension_path",
    "EntrypointRef",
    "EntrypointInfo",
    "EntrypointListResult",
    "ExtensionContentSource",  # re-exported Protocol
    "DirectoryExtensionContentSource",
    "EntrypointResolver",
    "DirectoryEntrypointResolver",
    "ExtensionRegistry",
    "DirectoryExtensionRegistry",
    "ExtensionProvider",
    "DEFAULT_ENTRYPOINT_LIMIT",
]
