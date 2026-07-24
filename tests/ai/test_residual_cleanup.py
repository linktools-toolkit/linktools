#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Residual-cleanup checks: the core package carries no old hook
wrapper, no requirement-source comment markers, no default security policy in
build_runtime, and no unimplemented public Protocol methods. Also confirms
``import linktools.ai`` stays SQLAlchemy-free."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

_AI_SRC = (
    Path(__file__).resolve().parents[2] / "linktools-ai" / "src" / "linktools" / "ai"
)

# Requirement-origin markers that must not appear in core comments/docstrings.
# Match single-character packaging tags without embedding historical labels.
# not legitimate prose like "Extension skills".
_MARKER_RE = re.compile(
    r"review3|" + r"review" + r"[- ]?doc|" + r"Package [A-Z0-9](?![a-z])|Task [0-9]+|"
    r"P[01]-[0-9]+|\bG[0-9]\b|GAP-[0-9]+|Decision D|Decision #[0-9]+|"
    r"actionable-fix|"
    + r"spec "
    + "§"
    + r"|"
    + "§"
    + r"[0-9]|spec section|spec docs|docs/linktools-ai\.md|"
    r"Phase [0-9]",
    re.IGNORECASE,
)


def _py_files():
    return [p for p in _AI_SRC.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_hooked_builtin_toolset_in_core():
    hits = []
    for p in _py_files():
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "HookedBuiltinToolset" in line:
                hits.append(f"{p}:{n}")
    assert not hits, f"HookedBuiltinToolset still in core: {hits}"


def test_no_requirement_source_comment_markers():
    hits = []
    for p in _py_files():
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#") and _MARKER_RE.search(line):
                hits.append(f"{p}:{n}: {line.strip()}")
    assert not hits, "requirement-source markers in comments:\n" + "\n".join(hits[:30])


def test_runtime_build_has_no_default_command_rule():
    rt = (_AI_SRC / "runtime" / "facade.py").read_text(encoding="utf-8")
    assert "DEFAULT_DENIED_COMMAND_PATTERNS" not in rt, (
        "build_runtime must not inject a default command denylist"
    )


def test_entrypoint_resolver_public_surface_has_no_unimplemented_methods():
    resolver = (_AI_SRC / "extension" / "resolver.py").read_text(encoding="utf-8")
    assert "def resolve_toolset" not in resolver
    assert "def resolve_workflow" not in resolver
    assert "reserved for a later phase" not in resolver
    assert "NotImplementedError" not in resolver


@pytest.mark.asyncio
async def test_import_linktools_ai_without_sqlalchemy():
    # A fresh interpreter with sqlalchemy/aiosqlite blocked must still import the
    # Core package must remain free of historical requirement labels.
    blocker = (
        "import importlib.abc, sys\n"
        "_B={'sqlalchemy','aiosqlite','asyncpg','asyncmy'}\n"
        "class _F(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self,n,p=None,t=None):\n"
        "        if n.split('.')[0] in _B: raise ModuleNotFoundError(n,name=n)\n"
        "sys.meta_path.insert(0,_F())\n"
        "import linktools.ai, linktools.ai.storage\n"
        "from linktools.ai.storage import Storage, FilesystemStorage\n"
    )
    env = {
        "PYTHONPATH": str(_AI_SRC.parents[1])
        + ":"
        + str(_AI_SRC.parents[2] / "linktools" / "src"),
        "PATH": "/usr/bin:/bin",
    }
    r = subprocess.run(
        [sys.executable, "-c", blocker], capture_output=True, text=True, env=env
    )
    assert r.returncode == 0, f"STDERR:\n{r.stderr}"


def test_resolve_methods_removed_from_public_api():
    rt = (_AI_SRC / "runtime" / "facade.py").read_text(encoding="utf-8")
    # No-compat simplification: resolve_agent / resolve_swarm / assemble are
    # removed. Runtime.inspect is the single assembly-inspection entry point;
    # by-id resolution is the caller's job via the RuntimeDependencies directly.
    assert "def resolve_agent" not in rt
    assert "def resolve_swarm" not in rt
    assert "def assemble" not in rt
    assert "DeprecationWarning" not in rt


def test_tool_exposure_counting_uses_descriptors_not_introspection():
    # The runner must count exposed tools from ToolContribution descriptors
    # (the single source of truth), NOT via toolset introspection
    # (toolset_names / getattr(ts, "tools")) -- opaque toolsets would otherwise
    # be miscounted.
    runner = (_AI_SRC / "agent" / "engine.py").read_text(encoding="utf-8")
    assert "toolset_names" not in runner, "runner must not count via toolset_names"
    assert 'getattr(ts, "tools"' not in runner
