#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime: the top-level integration surface and build kernel.

Public API: ``from linktools.ai.runtime import Runtime, build_runtime``.
The package splits the facade (:class:`Runtime`), the build kernel
(:func:`build_runtime_components`), and the dependency graph
(:class:`RuntimeDependencies`) into sibling modules. ``build_runtime`` is the
ONLY public construction entry point -- a module-level function, not a
``Runtime`` classmethod; there is no ``build_runtime``/``Runtime.create``/
``RuntimeFactory``. ``Runtime`` itself accepts only already-assembled
dependencies (``components=``)."""

from .builder import (
    RuntimeBuildConfig,
    RuntimeComponents,
    RuntimeSettings,
    build_runtime_components,
)
from .dependencies import MappingProvider, ProviderPrefixes, RuntimeDependencies
from .facade import Runtime, build_runtime

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
