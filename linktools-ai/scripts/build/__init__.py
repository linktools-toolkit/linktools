#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Static build-time API."""

from .agent_bundle import AgentBundle, build_bundle
from .bundle import AgentBundleCompiler, AssemblyInput, BundleCompilation, CapabilityAssemblyEntry, CapabilityAssemblyPlan
from .architecture import build_report
from .architecture import ArchitectureCheckResult, ArchitecturePolicyChecker
from .compatibility import build_manifest, validate_manifest
from .entry import build_agent_bundle
from .inventory import build_inventory
from .inventory import SourceInventoryBuilder
from .traceability import load_matrix, validate_matrix

__all__ = [
    "AgentBundle", "AgentBundleCompiler", "ArchitectureCheckResult", "ArchitecturePolicyChecker", "AssemblyInput",
    "BundleCompilation", "CapabilityAssemblyEntry", "CapabilityAssemblyPlan", "SourceInventoryBuilder", "build_inventory", "build_manifest", "build_report", "load_matrix",
    "build_agent_bundle", "build_bundle", "validate_manifest", "validate_matrix",
]
