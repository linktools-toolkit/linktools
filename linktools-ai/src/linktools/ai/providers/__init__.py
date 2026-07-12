#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.providers: the providers domain's public surface (spec §18.2).
ProviderBundle is the single declaration bundle handed to Runtime.build. The
individual spec-provider Protocols (AgentSpecProvider, SkillSpecProvider, ...)
live in their submodules (``providers.agent``, ``providers.skill``, ...)."""

from .bundle import ProviderBundle

__all__ = ["ProviderBundle"]
