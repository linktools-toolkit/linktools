#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ToolRunner: execute a resolved managed tool via the runtime Process (§10.11).

Standalone build (Phase 5 PR 11). Pairs with ToolDefinition/Registry
(core/_tools_registry.py): a ResolvedTool carries the executable + environment a
runner needs; the runner spawns it through linktools.runtime.popen using the
environment's subprocess_env (so managed tools resolve without mutating the
global PATH, §10.11). The legacy Tool (core/_tools.py) stays until consumers
migrate.
"""

import os
from typing import Any, Dict, List, Optional, Sequence

from ..runtime import popen
from ..types import TimeoutType

__all__ = ["ResolvedTool", "ToolRunner"]


class ResolvedTool(object):
    """A tool resolved to a concrete executable + environment (§10.2/§10.4)."""

    def __init__(self, executable, env=None, source="managed", version=None):
        # type: (str, Optional[Dict[str, str]], str, Optional[str]) -> None
        self.executable = executable
        self.env = dict(env or {})
        self.source = source  # configured / system / managed
        self.version = version


class ToolRunner(object):
    """Runs resolved tools through the runtime Process (§10.11)."""

    def __init__(self, environ):
        # type: (Any) -> None
        self._environ = environ

    def popen(self, resolved, args=(), *, include_tools=True, env_overrides=None,
              **kwargs):
        # type: (ResolvedTool, Sequence[str], bool, Optional[Dict[str, str]], Any) -> Any
        """Spawn the resolved tool; return the runtime Process (do not wait)."""
        env = self._environ.subprocess_env(
            include_tools=include_tools, overrides=env_overrides)
        env.update(resolved.env)
        command = [resolved.executable] + [str(a) for a in args]
        return popen(*command, env=env, **kwargs)

    def run(self, resolved, args=(), *, check=True, timeout=None, **kwargs):
        # type: (ResolvedTool, Sequence[str], bool, TimeoutType, Any) -> int
        """Run the resolved tool to completion; return its exit code.

        With ``check=True`` (default) a non-zero exit raises.
        """
        process = self.popen(resolved, args, **kwargs)
        if check:
            return process.check_call(timeout=timeout)
        return process.call(timeout=timeout)
