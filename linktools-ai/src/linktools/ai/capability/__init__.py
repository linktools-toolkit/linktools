#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.capability: the capability domain's public model (spec §18.2).
The minimal surface: CapabilityRef / CapabilityRuntimeOptions /
CapabilityInspection / CapabilityProvider. The bundle, context, assembler,
builtin provider, and exposure policy live in their submodules
(``capability.models``, ``capability.provider``, ``capability.assembler``,
``capability.builtin``, ``capability.exposure``)."""

from .models import CapabilityInspection, CapabilityRef, CapabilityRuntimeOptions
from .provider import CapabilityProvider

__all__ = [
    "CapabilityRef",
    "CapabilityRuntimeOptions",
    "CapabilityInspection",
    "CapabilityProvider",
]
