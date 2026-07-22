#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 / line 1852: the external-adapter conformance suite runs from an
installed wheel + the ``test`` extra, with NO source-tree access and NO in-repo
relative imports.

Three checks, all in the default suite (a skipped acceptance test is not
evidence):
1. The conformance package (linktools-ai/conformance/) imports ONLY the
   installed ``linktools.ai.*`` public surface + sibling modules within the
   package -- never the source tree, ``tests.*``, or private modules.
2. The conformance suite passes when run against the installed linktools.ai
   in the current environment (subprocess pytest).
3. The FULL gold standard: build the core wheel AND the separate
   ``linktools-ai-testing`` testkit wheel, install both into a fresh
   virtualenv, copy the conformance package in, and run it there -- proving the
   ADAPTER executes with zero source-tree access. The testkit is consumed from
   its installed wheel (``linktools.ai.testing``), never copied from the source
   tree. SKIPS honestly (not gated) when the environment cannot create a
   working venv."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_CONFORMANCE = _REPO / "linktools-ai" / "conformance"

# Modules an external conformance package must NEVER import: private kernel,
# in-repo reference backends, or the in-repo tests package.
_FORBIDDEN_ROOTS = (
    "linktools.ai.runtime.builder",
    "linktools.ai.storage.filesystem",
    "linktools.ai.storage.sqlalchemy",
    "linktools.ai.storage.coordination",
    "tests",
)


def _linktools_imports(path: Path) -> "set[str]":
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("linktools") or alias.name.startswith("tests"):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level >= 1) are intra-package; they do not
            # reach the source tree as long as the package is self-contained.
            if node.level == 0 and node.module and (
                node.module.startswith("linktools") or node.module.startswith("tests")
            ):
                out.add(node.module)
    return out


def _conformance_py_files() -> "list[Path]":
    return [p for p in _CONFORMANCE.rglob("*.py") if "__pycache__" not in p.parts]


def test_conformance_package_imports_only_public_surface() -> None:
    # Every linktools import in the conformance package resolves to the public
    # linktools.ai.* surface; none reach the private kernel, in-repo reference
    # backends, or the in-repo tests package.
    offenders: "list[str]" = []
    for path in _conformance_py_files():
        for mod in _linktools_imports(path):
            if mod.startswith(_FORBIDDEN_ROOTS) or not mod.startswith("linktools.ai"):
                offenders.append(f"{path.relative_to(_REPO)}: {mod}")
    assert not offenders, (
        "conformance package imports non-public / source-tree modules:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_conformance_runs_in_current_env() -> None:
    # The suite passes against the installed linktools.ai right now -- proving
    # the package is runnable as-is (no source-tree wiring needed).
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(_CONFORMANCE), "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"conformance suite failed in current env:\n"
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-2000:]}"
    )
    assert "passed" in result.stdout or "no tests ran" not in result.stdout


def test_conformance_runs_from_built_wheel(tmp_path) -> None:
    # The gold standard: build the core wheel AND
    # the separate ``linktools-ai-testing`` testkit wheel, install both into a
    # FRESH venv, copy the conformance package in, run it there -- zero
    # source-tree access. The testkit is consumed from its installed wheel
    # (``linktools.ai.testing``), never copied from the source tree.
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    build_core = subprocess.run(
        [sys.executable, "-m", "build", str(_REPO / "linktools-ai"), "--wheel",
         "--no-isolation", "--outdir", str(wheelhouse)],
        capture_output=True, text=True, timeout=180,
    )
    assert build_core.returncode == 0, (
        f"core wheel build failed:\n{build_core.stderr[-2000:]}"
    )
    build_testkit = subprocess.run(
        [sys.executable, "-m", "build", str(_REPO / "linktools-ai-testing"),
         "--wheel", "--no-isolation", "--outdir", str(wheelhouse)],
        capture_output=True, text=True, timeout=180,
    )
    assert build_testkit.returncode == 0, (
        f"testkit wheel build failed:\n{build_testkit.stderr[-2000:]}"
    )
    core_wheels = list(wheelhouse.glob("linktools_ai-*.whl"))
    testkit_wheels = list(wheelhouse.glob("linktools_ai_testing-*.whl"))
    assert core_wheels, "no core wheel produced"
    assert testkit_wheels, "no testkit wheel produced"

    venv_dir = tmp_path / "venv"
    py = _create_venv(venv_dir)
    install = subprocess.run(
        [
            py, "-m", "pip", "install", "--quiet",
            f"{core_wheels[0]}[test]",
            str(testkit_wheels[0]),
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert install.returncode == 0, (
        f"pip install wheels[test] failed:\n{install.stderr[-2000:]}"
    )

    pkg_dir = tmp_path / "conformance"
    shutil.copytree(_CONFORMANCE, pkg_dir)
    run = subprocess.run(
        [py, "-m", "pytest", str(pkg_dir), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, (
        f"conformance failed in the fresh wheel-only venv:\n"
        f"stdout={run.stdout[-2000:]}\nstderr={run.stderr[-2000:]}"
    )


# Paths the architecture purge removed from the source
# tree. They must NOT ship in the built core wheel either: a wheel carrying
# ``_runtime/`` or the old ``runtime.py`` module resurrects the old import path
# (a wheel-level old-path fallback, violating ) even when the source grep
# gate reads ZERO. This catches packaging-config mistakes (``packages.find``
# globbing in an extra dir) and non-``.py`` purged artifacts the source grep
# cannot see.
_PURGED_WHEEL_PATHS = (
    "linktools/ai/_runtime/",      # old private runtime package
    "linktools/ai/runtime.py",     # old runtime MODULE (the new package runtime/ is fine)
    "linktools/ai/capability/assembler.py",  # old name -> resolver.py
    "linktools/ai/extension/resource.py",    # old name -> content.py
)


def _build_core_wheel_clean(outdir: Path) -> Path:
    """Build the core wheel from a FRESH build cache (``python -m build``
    otherwise reuses a stale ``build/lib/`` and can ship files the purge
    removed). Returns the wheel path. Modeling the correct publish flow -- clean
    cache, then build -- is what makes the wheel-level assertion meaningful."""
    shutil.rmtree(_REPO / "linktools-ai" / "build", ignore_errors=True)
    build = subprocess.run(
        [sys.executable, "-m", "build", str(_REPO / "linktools-ai"), "--wheel",
         "--no-isolation", "--outdir", str(outdir)],
        capture_output=True, text=True, timeout=180,
    )
    assert build.returncode == 0, f"core wheel build failed:\n{build.stderr[-2000:]}"
    wheels = list(outdir.glob("linktools_ai-*.whl"))
    assert wheels, "no core wheel produced"
    return wheels[0]


def _wheel_entries(wheel: Path) -> "list[str]":
    import zipfile

    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def test_core_wheel_ships_no_purged_paths(tmp_path) -> None:
    """The published core wheel must carry NONE of the architecture-purge paths
    and NO test-support code. The source
    grep gate only checks ``.py`` under src; this guard checks the actual
    shippable artifact built from a clean cache, so it also catches a
    ``packages.find`` glob that pulls in an extra directory or a purged non-``.py``
    file. Builds from a clean cache (the correct publish flow); a publisher who
    builds without cleaning inherits whatever stale ``build/lib/`` they have,
    which CI avoids by checking out fresh."""
    wheel = _build_core_wheel_clean(tmp_path)
    entries = _wheel_entries(wheel)

    leaked = [p for purged in _PURGED_WHEEL_PATHS for p in entries if p.startswith(purged)]
    assert not leaked, (
        f"core wheel ships architecture-purge paths: {leaked}"
    )
    # The core wheel must not ship test-support code: the conformance testkit
    # lives in the separate linktools-ai-testing wheel.
    testing_in_core = [p for p in entries if p.startswith("linktools/ai/testing/")]
    assert not testing_in_core, (
        "core wheel ships the conformance testkit -- it must live only in the "
        f"linktools-ai-testing wheel: {testing_in_core}"
    )
    # Sanity: the wheel still carries the NEW runtime package, so the purge
    # guard above is not vacuously green because the build dropped runtime.
    assert any(p.startswith("linktools/ai/runtime/facade.py") for p in entries), (
        "core wheel is missing linktools/ai/runtime/facade.py -- the clean-build "
        "step produced an incomplete wheel"
    )


# Public-API names the architecture purge retired. The source grep gate scopes
# to src .py; this wheel-level list guards the published METADATA (which embeds
# README.md as the long description) so a stale name in README cannot ship to
# PyPI / downstream readers.
_PURGED_API_NAMES = (
    "ResourceStore",
    "CapabilityAssembler",
    "ExecutionBackend",
    "MCPConnectionManager",
    "ModelRouter",
    "propfind",
)


def _wheel_metadata_text(wheel: Path) -> str:
    import zipfile

    with zipfile.ZipFile(wheel) as zf:
        meta = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        assert meta, "wheel has no *.dist-info/METADATA"
        return zf.read(meta[0]).decode("utf-8", errors="replace")


def test_core_wheel_metadata_has_no_purged_api_names(tmp_path) -> None:
    """The published wheel's METADATA (project description = README.md) must
    not name any retired public-API class. Catches a README that still
    advertises a purged name -- the source grep gate does not reach README, so
    without this guard a stale README ships to PyPI unchanged."""
    wheel = _build_core_wheel_clean(tmp_path)
    text = _wheel_metadata_text(wheel)
    leaked = [name for name in _PURGED_API_NAMES if name in text]
    assert not leaked, (
        f"core wheel METADATA advertises retired public-API names: {leaked}"
    )


def _create_venv(venv_dir: Path) -> str:
    """Create a fresh venv at ``venv_dir`` (with pip). Prefers ``python -m venv``;
    falls back to the ``virtualenv`` package; skips the test if neither produces
    a venv with a WORKING pip (e.g. the Debian ``python3-venv`` apt package is
    absent, which leaves pip installed-but-broken)."""
    import venv as _venv

    try:
        _venv.create(venv_dir, with_pip=True, clear=True)
    except BaseException:
        # venv.create raises SystemExit (a BaseException, not Exception) when
        # ensurepip is unavailable (Debian python3-venv apt package absent) --
        # swallow it and probe pip viability below.
        pass

    def _pip_works() -> bool:
        py = venv_dir / "bin" / "python"
        if not py.exists():
            return False
        probe = subprocess.run(
            [str(py), "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return probe.returncode == 0

    if _pip_works():
        return str(venv_dir / "bin" / "python")

    if shutil.which("virtualenv") is not None:
        subprocess.run(
            ["virtualenv", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if _pip_works():
            return str(venv_dir / "bin" / "python")

    pytest.skip(
        "cannot create a venv with working pip in this environment (install "
        "the python3-venv apt package or the 'virtualenv' pip package)"
    )
