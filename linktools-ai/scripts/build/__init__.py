
"""Static build-time API."""

from .architecture import (
    ArchitectureCheckResult,
    ArchitecturePolicyChecker,
    build_report,
)
from .compatibility import build_manifest, validate_manifest
from .inventory import SourceInventoryBuilder, build_inventory
from .traceability import load_matrix, validate_matrix

__all__ = [
    "ArchitectureCheckResult",
    "ArchitecturePolicyChecker",
    "SourceInventoryBuilder",
    "build_inventory",
    "build_manifest",
    "build_report",
    "load_matrix",
    "validate_manifest",
    "validate_matrix",
]
