#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime: the top-level integration surface and build kernel.

Public API: ``from linktools.ai.runtime import Runtime, build_runtime``.
The package splits the facade (:class:`Runtime`), the build kernel
(:func:`build_runtime_components`), and the dependency graph
(:class:`RuntimeDependencies`) into sibling modules; callers use the
re-exports below or :meth:`Runtime.build`. ``build_runtime`` is the module-level
alias for :meth:`Runtime.build` -- the public build entry that returns a
fully-wired :class:`Runtime`."""

from .builder import (
    RuntimeBuildConfig,
    RuntimeComponents,
    RuntimeSettings,
    build_runtime_components,
)
from .dependencies import MappingProvider, ProviderPrefixes, RuntimeDependencies
from .facade import Runtime

# The public build entry: a module-level handle equivalent to Runtime.build,
# so ``from linktools.ai.runtime import build_runtime`` works as a function call.
build_runtime = Runtime.build

__all__: "list[str]" = [
    "Runtime",
    "build_runtime",
    "RuntimeDependencies",
    "ProviderPrefixes",
    "MappingProvider",
    "RuntimeBuildConfig",
    "RuntimeComponents",
    "RuntimeSettings",
    "build_runtime_components",
]
