#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace declarations that connect Asset kinds to capability providers."""

from dataclasses import dataclass

from ..asset import AssetTypeBinding
from ..capability import CapabilityProvider


@dataclass(frozen=True, slots=True)
class CapabilitySource:
    asset_binding: "AssetTypeBinding[object]"
    provider: CapabilityProvider


__all__ = ["CapabilitySource"]
