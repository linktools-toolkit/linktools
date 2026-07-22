#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunnableSpec: the union of the two runnable kinds an Runtime accepts.

Runtime.run takes this union and dispatches explicitly (``isinstance(spec,
SwarmSpec)`` in RunCoordinator) -- there is no shared duck-typed Protocol, so a
partial or cross-kind spec fails the dispatch check instead of slipping through
on attribute proximity. Callers that type a generic runnable use this alias."""

from .agent.spec import AgentSpec
from .swarm.spec import SwarmSpec

RunnableSpec = AgentSpec | SwarmSpec

__all__: "list[str]" = ["RunnableSpec"]
