#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AC-15 evidence (plan evidence type: "import graph"): the run -> approval ->
resume -> artifact -> job chain is drivable through the PUBLIC linktools.ai
surface alone -- the chain-driving tests reach no private ``_runtime`` or
underscore-private module. (The chain's correctness is covered by the
component tests -- run/complete here, pause/approve/resume in
test_runtime_resume.py, artifact and job in their suites; this test pins the
public-surface-only property so a future change cannot quietly reach into
runtime internals to drive the chain.)

The strong AC-15 form -- a from-scratch EXTERNAL adapter running the full chain
-- needs the adapter to implement the full Storage surface (RunStore /
SessionStore / EventStore / CheckpointStore / ApprovalStore / IdempotencyStore
+ a transaction manager), which is a separate, larger effort. Protocol
sufficiency for the artifact-domain Protocols the external adapter does cover
(blob / record / lease) is proven by tests/ai/storage/test_external_adapter_
conformance.py (Phase 9 op 4)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]

# The modules that drive the chain end-to-end. If any of these reaches into
# ``linktools.ai._runtime`` or any underscore-private submodule, the public
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
    starts with ``_`` (e.g. ``linktools.ai._runtime.build``,
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
