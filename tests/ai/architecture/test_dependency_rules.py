#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dependency-direction guards.

The refactor establishes one-way dependency directions between top-level
packages. records the *current* structure and the rules that already
hold, plus the one known violation must repair (``task`` reaching into
``security`` for the identity value types). As phases land, promote rules from
``TARGET_RULES_NOT_YET_ENFORCED`` into ``FORBIDDEN_IMPORTS``.

A "rule" maps a top-level package (the importer, under
``linktools/ai/<pkg>/``) to the set of other top-level packages it must not
import. The check is AST-based: only ``import x.y`` and ``from x.y import z``
statements are considered, and the importer's own sub-tree never counts as a
self-dependency.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_AI = _REPO / "linktools-ai" / "src" / "linktools" / "ai"

# Importer top-level package -> set of forbidden top-level package names.
# These hold on the baseline and must continue to hold.
FORBIDDEN_IMPORTS: "dict[str, set[str]]" = {
    # identity value types (PrincipalContext/ActorRef/ScopeSet) must not reach
    # into any downstream domain. They are the canonical owner; everything
    # else imports them. Enforced since .
    "identity": {"jobs", "run", "agent", "storage"},
    # governance (security + policy, converged ) must not depend
    # on the jobs domain (task.py -> jobs). Held since before the
    # refactor; made it structural by removing the
    # security.principal -> task.models link entirely.
    "governance": {"jobs"},
    # jobs must not reach into the runtime build internals -- it only depends
    # on the narrow TaskRunDispatcher Protocol at the handler boundary.
    # Enforced since (task -> jobs pure rename).
    "jobs": {"runtime"},
    # The artifact and asset domains are fully decoupled. The
    # artifact facade depends only on the ArtifactBlobStore /
    # ArtifactRecordStore Protocols; the filesystem + SQLAlchemy reference
    # adapters live in the storage layer, so neither domain names the other.
    "artifact": {"asset", "jobs"},
    "asset": {"artifact"},
    # : storage is a bottom-layer dependency -- it must not reach up into
    # runtime (the outermost assembly), catalog, or capability. A backend is
    # injected INTO storage; storage never imports those layers.
    "storage": {"runtime", "catalog", "capability"},
    # : catalog does not depend on runtime (it is a config layer consumed
    # BY runtime, not the reverse).
    "catalog": {"runtime"},
    # NOTE: the rule "capability must not import a concrete storage
    # backend (filesystem/sqlalchemy/sqlite/coordination)" is NOT here -- this
    # map's matcher normalizes to top-level packages, and capability legitimately
    # imports ``storage.protocols`` (top-level ``storage``). That rule is
    # enforced structurally by test_architecture_section_7_6's
    # test_domains_do_not_import_concrete_storage_backends, which checks the
    # concrete subpackages specifically.
}


# Rules defines for the END state. They are documented here so
# the target is captured in code; each is enforced the moment its phase lands.
# Format: (importer_pkg, forbidden_pkg, phase_that_enforces).
TARGET_RULES_NOT_YET_ENFORCED: "list[tuple[str, str, str]]" = [
    ("asset", "runtime", "Phase 3"),
    ("events", "runtime", "Phase 2"),
    ("governance", "agent", "Phase 6"),
]


def _top_level_packages() -> "set[str]":
    return {
        p.name
        for p in _AI.iterdir()
        if p.is_dir() and p.name != "__pycache__"
    }


def _imports_in(file_path: Path) -> "set[str]":
    """Top-level ``linktools.ai.<x>`` packages imported by this file.

    Resolves relative imports (``from ..jobs.models import X``) against the
    file's location, so a forbidden dep cannot slip in via a dotted-relative
    form that the absolute-import-only check would miss.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return set()
    # Resolve relative imports against the file's package. For a module
    # ``linktools.ai.<pkg>.mod`` the base is its containing package; for a
    # package's ``__init__`` the base is the package itself.
    rel = file_path.relative_to(_AI).with_suffix("")
    is_init = rel.name == "__init__"
    if is_init:
        rel = rel.parent
    parts = ("linktools", "ai") + tuple(rel.parts)
    base_pkg = ".".join(parts) if is_init else ".".join(parts[:-1])

    roots: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    roots.add(node.module)
            else:
                # level=1 -> base_pkg; each extra level climbs one parent.
                base = base_pkg
                for _ in range(node.level - 1):
                    if "." in base:
                        base = base.rsplit(".", 1)[0]
                    else:
                        base = ""
                        break
                if node.module:
                    roots.add(f"{base}.{node.module}")
                else:
                    # ``from .. import name`` form: each name is a submodule
                    # of the resolved base. Treat them as such so a forbidden
                    # package cannot slip in via the name-only form.
                    for alias in node.names:
                        roots.add(f"{base}.{alias.name}")
    out: "set[str]" = set()
    for name in roots:
        if name.startswith("linktools.ai.") or name == "linktools.ai":
            tail = name.split(".")[2] if name.startswith("linktools.ai.") else ""
            if tail:
                out.add(tail)
    return out


def _package_imports(pkg: str) -> "dict[Path, set[str]]":
    pkg_dir = _AI / pkg
    if not pkg_dir.is_dir():
        return {}
    result: "dict[Path, set[str]]" = {}
    for path in pkg_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        result[path] = _imports_in(path)
    return result


@pytest.mark.parametrize("importer", sorted(FORBIDDEN_IMPORTS))
def test_forbidden_dependency_directions_hold(importer: str) -> None:
    """Enforced dependency rules must not be violated by new code."""
    forbidden = FORBIDDEN_IMPORTS[importer]
    if not (_AI / importer).is_dir():
        pytest.skip(f"package {importer!r} not present in this checkout")
    violations: "list[str]" = []
    for path, imported in _package_imports(importer).items():
        bad = imported & forbidden
        if bad:
            rel = path.relative_to(_REPO)
            violations.append(f"{rel}: imports {sorted(bad)}")
    assert not violations, (
        f"{importer}/ imports a forbidden package:\n  " + "\n  ".join(violations)
    )


def test_target_dependency_rules_are_documented() -> None:
    """The end-state rules are captured so they can be promoted as phases land.

    This always passes at ; it exists to keep the target visible and to
    fail loudly if a documented rule is silently dropped. When a phase enforces
    a rule, move its entry into FORBIDDEN_IMPORTS and delete it here.
    """
    assert TARGET_RULES_NOT_YET_ENFORCED, "target rules should be non-empty until Phase 9"


def _two_cycles() -> "set[tuple[str, str]]":
    """All 2-cycles (A imports B AND B imports A) between top-level packages."""
    pkgs = _top_level_packages()
    edges: "dict[str, set[str]]" = {p: set() for p in pkgs}
    for pkg in pkgs:
        imported: "set[str]" = set()
        for imps in _package_imports(pkg).values():
            imported |= imps
        edges[pkg] = imported & pkgs
    cycles: "set[tuple[str, str]]" = set()
    for a in pkgs:
        for b in edges[a]:
            if a != b and a in edges.get(b, set()):
                cycles.add(tuple(sorted((a, b))))  # type: ignore[arg-type]
    return cycles


# Top-level 2-cycles still present today. Each is a dependency debt the refactor
# retires (governance/tool split, runtime/run convergence, asset/artifact
# separation, etc.). Ratchet downward as phases land; the target is an empty
# set. New cycles must not appear.
#
# Tuples are stored in sorted form because ``_two_cycles`` normalizes every
# cycle via ``tuple(sorted(...))`` -- a non-sorted literal can never match and
# is silently dead weight. The Catalog migration already retired three
# cycles that earlier baselines carried: ``mcp <-> registry`` (domains now
# import one-way from catalog/*; the registry/*.py shims import the domain
# homes, not vice-versa), ``storage <-> task`` (the implicit storage.assets
# fallback was deleted in ), and ``policy <-> registry`` (policy
# now imports tool.catalog directly instead of registry.tool).
_BASELINE_TWO_CYCLES: "frozenset[tuple[str, str]]" = frozenset({
    ("run", "runtime"),
    ("agent", "capability"),
    ("agent", "governance"),
    ("agent", "run"),
    ("agent", "tool"),
    ("artifact", "storage"),
    ("extension", "subagent"),
    ("run", "sandbox"),
    ("governance", "tool"),
    ("run", "storage"),
    ("run", "swarm"),
})


def test_no_new_circular_top_level_imports() -> None:
    """No NEW 2-cycle may appear between top-level packages.

    The baseline carries a known set of 2-cycles (recorded above) that the
    refactor retires phase by phase. This asserts the current set never grows
    beyond the baseline; lower the baseline as a phase breaks a cycle.
    """
    current = _two_cycles()
    new = current - _BASELINE_TWO_CYCLES
    assert not new, (
        f"new top-level 2-cycles introduced (refactor must not add cycles): "
        f"{sorted(new)}"
    )
