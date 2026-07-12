#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai public API. The package root exports exactly one symbol:
``Runtime``. Every other type lives behind its domain submodule
(``linktools.ai.agent``, ``linktools.ai.capability``, ``linktools.ai.tool``,
...) -- import it from there.

Importing this package has no heavy side effects: no file scans, no DB/MCP
connections, no Runtime construction."""

from .runtime import Runtime

__all__ = ["Runtime"]
