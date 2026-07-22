#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SECONDARY import guard, NOT the proof (: '仅扫描内部 E2E 是否
import 私有模块' is explicitly listed as non-evidence).

The strong form -- a from-scratch EXTERNAL adapter driving the FULL
connected run -> approval -> resume -> artifact -> job chain through the public
Protocol surface alone -- lives in
``tests/ai/storage/test_external_adapter_full_chain.py::
test_external_adapter_full_connected_chain_run_approval_resume_artifact_job``.
That test is the evidence; it runs the connected chain through the
adapter's public stores with real persistence assertions.

This test is retained only as a defense-in-depth IMPORT GUARD: it AST-scans the
in-repo chain-driving modules (run/complete, pause/approve/resume, MCP) and
fails if any reaches into ``linktools.ai.runtime.builder`` or an underscore-private
module -- catching a regression where an in-repo test quietly uses a private
surface. It is deliberately NOT counted as evidence; the connected
external-adapter chain is."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]

# The modules that drive the chain end-to-end. If any of these reaches into
# ``linktools.ai.runtime.builder`` or any underscore-private submodule, the public
# surface is no longer sufficient to drive the chain.
_CHAIN_TEST_MODULES = [
    "tests/ai/e2e/test_file_runtime_complete.py",  # run -> complete
    "tests/ai/test_runtime_resume.py",  # pause -> approve -> resume -> SUCCEEDED
    "tests/ai/e2e/test_runtime_mcp.py",  # MCP-tool run end-to-end
]


def _public_linktools_imports(path: Path) -> "set[str]":
    """Every ``linktools.ai.*`` module imported by ``path``, resolved against
    relative imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg = path.relative_to(_REPO).with_suffix("")
    parts = ("linktools", "ai") + tuple(pkg.parts[:-1])
    base = ".".join(parts)
    out: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("linktools.ai"):
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith("linktools.ai"):
                    out.add(node.module)
            else:
                resolved = base
                for _ in range(node.level - 1):
                    resolved = resolved.rsplit(".", 1)[0] if "." in resolved else ""
                mod = f"{resolved}.{node.module}" if node.module else resolved
                if mod.startswith("linktools.ai"):
                    out.add(mod)
    return out


def _is_private(mod: str) -> bool:
    """A linktools.ai module is private if any segment after ``linktools.ai``
    starts with ``_`` (e.g. ``linktools.ai.runtime.builder``,
    ``linktools.ai.governance._internal``)."""
    tail = mod[len("linktools.ai") :].lstrip(".")
    if not tail:
        return False
    return any(seg.startswith("_") for seg in tail.split("."))


@pytest.mark.parametrize("rel", _CHAIN_TEST_MODULES)
def test_chain_driver_uses_only_public_surface(rel: str) -> None:
    path = _REPO / rel
    assert path.is_file(), f"chain test module missing: {rel}"
    imports = _public_linktools_imports(path)
    private = {m for m in imports if _is_private(m)}
    assert not private, (
        f"{rel} imports private linktools.ai modules -- the run/approval/resume "
        f"chain must be drivable through the public surface only: {sorted(private)}"
    )
