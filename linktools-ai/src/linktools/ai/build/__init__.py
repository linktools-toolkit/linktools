#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Static build-time API."""

from .assembly import AgentBundleCompiler, AssemblyInput, BundleCompilation, CapabilityAssemblyEntry, CapabilityAssemblyPlan
from .architecture import build_report
from .architecture import ArchitectureCheckResult, ArchitecturePolicyChecker
from .compatibility import build_manifest, validate_manifest
from .inventory import build_inventory
from .inventory import SourceInventoryBuilder
from .docs import DocsSnapshotBuilder
from .probe import UpstreamReleaseManifestBuilder
from .manifest import build_source_manifest
from .signing import BundleSigner
from .traceability import load_matrix, validate_matrix

__all__ = [
    "AgentBundleCompiler", "ArchitectureCheckResult", "ArchitecturePolicyChecker", "AssemblyInput",
    "BundleCompilation", "BundleSigner", "CapabilityAssemblyEntry", "CapabilityAssemblyPlan", "DocsSnapshotBuilder", "SourceInventoryBuilder", "UpstreamReleaseManifestBuilder", "build_inventory", "build_manifest", "build_report", "build_source_manifest", "load_matrix",
    "validate_manifest", "validate_matrix",
]
