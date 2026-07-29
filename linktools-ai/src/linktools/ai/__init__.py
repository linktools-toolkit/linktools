#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai public API. The package root exports exactly one symbol:
``Runtime``. Every other type lives behind its domain submodule
(``linktools.ai.agent``, ``linktools.ai.spec``, ``linktools.ai.agent.tool``,
...) -- import it from there.

Importing this package has no heavy side effects: no file scans, no DB/MCP
connections, no Runtime construction.

The ``linktools.ai`` namespace is provided by this distribution's core wheel.
Importing it does not load test support or other development-only modules."""

from .runtime import Runtime

__all__ = ["Runtime"]
