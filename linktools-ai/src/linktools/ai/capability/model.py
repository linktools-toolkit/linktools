#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability DTOs and manifest values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapabilityRef:
    kind: str
    id: str
    revision: "int | None" = None
    required: bool = True


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    entries: "tuple[CapabilityRef, ...]"
    digest: str


__all__ = ["CapabilityManifest", "CapabilityRef"]
