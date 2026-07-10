#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Residual-cleanup checks (spec §16.4): the core package carries no old hook
wrapper, no requirement-source comment markers, no default security policy in
Runtime.build, and no unimplemented public Protocol methods. Also confirms
``import linktools.ai`` stays SQLAlchemy-free."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

_AI_SRC = Path(__file__).resolve().parents[2] / "linktools-ai" / "src" / "linktools" / "ai"

# Requirement-origin markers that must not appear in core comments/docstrings.
# "Package X" matches the single-char review-packaging tags (Package A / Package 8),
# not legitimate prose like "Package skills".
_MARKER_RE = re.compile(
    r"review3|review-doc|review doc|Package [A-Z0-9](?![a-z])|Task [0-9]+|"
    r"P[01]-[0-9]+|\bG[0-9]\b|GAP-[0-9]+|Decision D|Decision #[0-9]+|"
    r"actionable-fix|spec §|§[0-9]|spec section|spec docs|docs/linktools-ai\.md|"
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
    assert not hits, f"requirement-source markers in comments:\n" + "\n".join(hits[:30])


def test_runtime_build_has_no_default_command_rule():
    rt = (_AI_SRC / "runtime.py").read_text(encoding="utf-8")
    assert "DEFAULT_DENIED_COMMAND_PATTERNS" not in rt, (
        "Runtime.build must not inject a default command denylist"
    )


def test_entrypoint_resolver_public_surface_has_no_unimplemented_methods():
    resolver = (_AI_SRC / "package" / "resolver.py").read_text(encoding="utf-8")
    assert "def resolve_toolset" not in resolver
    assert "def resolve_workflow" not in resolver
    assert "reserved for a later phase" not in resolver
    assert "NotImplementedError" not in resolver


@pytest.mark.asyncio
async def test_import_linktools_ai_without_sqlalchemy():
    # A fresh interpreter with sqlalchemy/aiosqlite blocked must still import the
    # core package (spec §21.10 / §16.4 #3).
    blocker = (
        "import importlib.abc, sys\n"
        "_B={'sqlalchemy','aiosqlite','asyncpg','asyncmy'}\n"
        "class _F(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self,n,p=None,t=None):\n"
        "        if n.split('.')[0] in _B: raise ModuleNotFoundError(n,name=n)\n"
        "sys.meta_path.insert(0,_F())\n"
        "import linktools.ai, linktools.ai.storage\n"
        "from linktools.ai.storage import Storage, FileStorage\n"
    )
    env = {
        "PYTHONPATH": str(_AI_SRC.parents[1]) + ":" + str(_AI_SRC.parents[2] / "linktools" / "src"),
        "PATH": "/usr/bin:/bin",
    }
    r = subprocess.run([sys.executable, "-c", blocker], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"STDERR:\n{r.stderr}"


def test_deprecated_runtime_api_has_removal_plan():
    rt = (_AI_SRC / "runtime.py").read_text(encoding="utf-8")
    # Each deprecated method documents its replacement + removal target.
    assert rt.count("Removal target: next major version") >= 3
    assert "runtime.providers.agents.get" in rt
    assert "runtime.providers.swarms.get" in rt
    assert "runtime.capability_assembler.assemble" in rt


def test_tool_exposure_counting_uses_shared_helper():
    # Runner must count tools via capability.toolset_names (same helper the
    # assembler uses for conflict detection), not a private getattr.
    runner = (_AI_SRC / "agent" / "runner.py").read_text(encoding="utf-8")
    assert "toolset_names" in runner
    assert 'getattr(ts, "tools"' not in runner


def test_spec_loader_from_resources_uses_resourcestore_api():
    parser = (_AI_SRC / "registry" / "parser.py").read_text(encoding="utf-8")
    body = parser.split("def from_resources")[1].split("return cls")[0]
    # Must not call the non-existent ResourceStore.list / global .revision.
    assert "resource_store.list" not in body
    assert "resource_store.revision" not in body
