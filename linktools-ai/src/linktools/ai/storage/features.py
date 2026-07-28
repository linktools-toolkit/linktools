#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage capability vocabulary: the enums + value-types a downstream Storage
declares as its capability surface. Pure standard-library (no domain imports),
so this module stays in the storage kernel.

The runtime-composition surface that BUILDS ON these enums (``StorageFeatures``
dataclass + composition-derived features
presets + the ``StorageFeatures.from_components`` derivation) lives at
``linktools.ai.runtime.persistence.features`` -- it is runtime-shaped (a frozen
declaration of what THIS Storage provides), not a storage-kernel primitive.
``ComponentCapabilities`` lives HERE so each storage-layer store adapter can
return one without importing the runtime layer."""

from dataclasses import dataclass
from enum import Enum


class TransactionScope(str, Enum):
    NONE = "none"
    PROCESS_LOCAL = "process_local"
    DATABASE = "database"


class CoordinationScope(str, Enum):
    NONE = "none"
    PROCESS_LOCAL = "process_local"
    DISTRIBUTED = "distributed"


@dataclass(frozen=True, slots=True)
class ComponentCapabilities:
    """Per-component capability declaration. Each domain store adapter exposes
    a ``capabilities`` property returning one of these; ``StorageFeatures.
    from_components`` (in the runtime layer) aggregates them so a Storage's
    declared features are a FUNCTION of the wired stores rather than an
    optimistic all-components default. Defaults are conservative (every
    capability False) so a store that does not declare ``capabilities`` is
    treated as offering nothing, never silently offering everything."""

    transaction_participation: bool = False
    optimistic_concurrency: bool = False
    idempotency: bool = False
    append_only: bool = False


__all__: "list[str]" = [
    "ComponentCapabilities",
    "CoordinationScope",
    "TransactionScope",
]
