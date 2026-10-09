#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit database schema provisioning for deployment tooling."""

from typing import TYPE_CHECKING

from ._metrics import (
    provision_metrics_database,
    provision_metrics_sqlite,
    validate_metrics_database,
    validate_metrics_sqlite,
)

if TYPE_CHECKING:
    from ._database import (
        build_sql_schema_metadata,
        provision_asset_database,
        provision_database,
        provision_runtime_database,
    )

__all__ = [
    "build_sql_schema_metadata",
    "provision_asset_database",
    "provision_database",
    "provision_metrics_database",
    "provision_metrics_sqlite",
    "provision_runtime_database",
    "validate_metrics_database",
    "validate_metrics_sqlite",
]


def __getattr__(name: str) -> object:
    if name not in {
        "build_sql_schema_metadata",
        "provision_asset_database",
        "provision_database",
        "provision_runtime_database",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from ._database import (
        build_sql_schema_metadata,
        provision_asset_database,
        provision_database,
        provision_runtime_database,
    )

    globals().update(
        build_sql_schema_metadata=build_sql_schema_metadata,
        provision_asset_database=provision_asset_database,
        provision_database=provision_database,
        provision_runtime_database=provision_runtime_database,
    )
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
