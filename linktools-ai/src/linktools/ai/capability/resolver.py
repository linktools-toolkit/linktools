#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityResolver is the spec §5.2 name for the capability orchestrator; it
is an alias of CapabilityAssembler (same class, same assemble() entry point)."""

from .assembler import CapabilityAssembler

CapabilityResolver = CapabilityAssembler

__all__ = ["CapabilityResolver"]
