#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build entry point for deterministic Agent bundles."""

from linktools.ai.spec import AgentSpec

from .agent_bundle import AgentBundle, build_bundle


def build_agent_bundle(spec: AgentSpec, capability_manifest_digest: str) -> AgentBundle:
    return build_bundle(spec, capability_manifest_digest)


__all__ = ["build_agent_bundle"]
