#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolSpec: an immutable tool declaration (moved here from registry/tool.py so
the tool domain owns its spec type). Mirrors the policy layer's metadata
shape; the registry/codec fills it from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..governance.policy.rule import (
    ApprovalMode,
    Permission,
    RiskLevel,
    SideEffectKind,
)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str = ""
    enabled: bool = True
    permissions: "frozenset[Permission]" = field(
        default_factory=lambda: frozenset({Permission.READ})
    )
    risk: RiskLevel = RiskLevel.LOW
    side_effect: SideEffectKind = SideEffectKind.READ_ONLY
    approval: ApprovalMode = ApprovalMode.NEVER
    idempotent: "bool | None" = None
    timeout_seconds: "float | None" = None
    max_retries: "int | None" = None
    idempotency_strategy: "str | None" = None
    idempotency_key_field: "str | None" = None
    # bump when a tool's input contract changes shape so an idempotency
    # hash computed under the old schema is never mistaken for a match
    # against the new one (see tool/idempotency.py compute_request_hash).
    schema_version: str = "1"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


__all__: "list[str]" = ["ToolSpec"]
