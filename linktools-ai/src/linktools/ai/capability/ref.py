#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityRef: a resolved (kind, name, config) triple targeting one
CapabilityProvider. ``kind`` selects the provider; ``name`` is interpreted by
that provider (e.g. builtin:file, skill:sql, mcp:risk-data)."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CapabilityRef:
    kind: str
    name: str
    config: "Mapping[str, Any]" = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"
