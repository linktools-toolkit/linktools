#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tool definitions + dependency-graph validation (spec §10.3, §10.5).

Standalone build (Phase 5 PR 10). The legacy Tools/Tool (core/_tools.py) stays
until common/mobile migrate; this formalises the §10.5 dependency contract:

* every ``depends_on`` target exists;
* no self-dependency;
* no cyclic dependency (error names the full chain);
* platform/architecture availability;
* a stable topological install order.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from ..errors import ToolDependencyError
from .. import system as _system

__all__ = ["ToolDefinition", "ToolRegistry"]


class ToolDefinition(object):
    """Static description of a managed tool (§10.3)."""

    def __init__(self, name, version=None, depends_on=(), description="",
                 platforms=None, architectures=None, sha256=None, size=None,
                 entrypoint=None):
        # type: (str, Optional[str], Sequence[str], str, Optional[Sequence[str]], Optional[Sequence[str]], Optional[str], Optional[int], Optional[str]) -> None
        self.name = name
        self.version = version
        self.depends_on = tuple(depends_on)
        self.description = description
        self.platforms = tuple(platforms) if platforms is not None else None
        self.architectures = tuple(architectures) if architectures is not None else None
        self.sha256 = sha256
        self.size = size
        self.entrypoint = entrypoint


class ToolRegistry(object):
    """A set of ToolDefinitions with dependency-graph validation (§10.5)."""

    def __init__(self, current_system=None, current_arch=None):
        # type: (Optional[str], Optional[str]) -> None
        self._tools = {}  # type: Dict[str, ToolDefinition]
        self._system = current_system  # lazily resolved if None
        self._arch_raw = current_arch

    # -- registration ------------------------------------------------------

    def add(self, tool):
        # type: (ToolDefinition) -> "ToolRegistry"
        if tool.name in self._tools:
            raise ToolDependencyError("duplicate tool name: %s" % tool.name)
        self._tools[tool.name] = tool
        return self

    def __contains__(self, name):
        # type: (str) -> bool
        return name in self._tools

    def get(self, name):
        # type: (str) -> Optional[ToolDefinition]
        return self._tools.get(name)

    @property
    def names(self):
        # type: () -> List[str]
        return list(self._tools.keys())

    # -- §10.5 validation --------------------------------------------------

    def validate(self):
        # type: () -> None
        """Existence + self-dep + cycle checks; raises with the full chain."""
        for name, tool in self._tools.items():
            for dep in tool.depends_on:
                if dep == name:
                    raise ToolDependencyError("tool %s depends on itself" % name)
                if dep not in self._tools:
                    raise ToolDependencyError(
                        "tool %s depends on missing tool %s" % (name, dep))
        # Cycle detection over the depends_on graph.
        for start in self._tools:
            self._detect_cycle(start, [])

    def _detect_cycle(self, name, stack):
        # type: (str, List[str]) -> None
        if name in stack:
            chain = stack[stack.index(name):] + [name]
            raise ToolDependencyError("cyclic tool dependency: " + " -> ".join(chain))
        tool = self._tools.get(name)
        if tool is None:
            return
        stack.append(name)
        for dep in tool.depends_on:
            self._detect_cycle(dep, stack)
        stack.pop()

    # -- platform/arch availability ---------------------------------------

    def _current_system(self):
        if self._system is None:
            self._system = _system.get_system()
        return self._system

    def _current_arch(self):
        # Canonical arch (amd64 -> x86_64, aarch64 -> arm64).
        if self._arch_raw is None:
            self._arch_raw = _system.normalize_arch(_system.get_machine())
        return self._arch_raw

    def available(self):
        # type: () -> List[str]
        sysname = self._current_system()
        arch = self._current_arch()
        out = []
        for name, tool in self._tools.items():
            if tool.platforms is not None and sysname not in tool.platforms:
                continue
            if tool.architectures is not None and arch not in tool.architectures:
                continue
            out.append(name)
        return out

    # -- ordering ----------------------------------------------------------

    def topological_sort(self):
        # type: () -> List[str]
        """Return tool names in dependency order (deps first), stable on ties."""
        self.validate()
        order = []  # type: List[str]
        seen = set()  # type: set
        visiting = set()  # type: set

        def visit(name):
            if name in seen:
                return
            if name in visiting:  # cycle (already validated, defensive)
                return
            visiting.add(name)
            tool = self._tools.get(name)
            if tool is not None:
                for dep in tool.depends_on:
                    visit(dep)
            visiting.discard(name)
            seen.add(name)
            order.append(name)

        # Iterate in insertion order for stability.
        for name in self._tools:
            visit(name)
        return order
