#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai public API. The package root exports exactly one symbol:
``Runtime``. Every other type lives behind its domain submodule
(``linktools.ai.agent``, ``linktools.ai.capability``, ``linktools.ai.tool``,
...) -- import it from there.

Importing this package has no heavy side effects: no file scans, no DB/MCP
connections, no Runtime construction.

The ``linktools.ai`` namespace spans two distributions: this one (the core
wheel) and the separate ``linktools-ai-testing`` wheel that ships the storage
conformance testkit at ``linktools.ai.testing``. ``pkgutil.extend_path``
stitches the namespace at import time so ``linktools.ai.testing`` resolves
into the other wheel without this one carrying test code."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .runtime import Runtime

__all__ = ["Runtime"]
