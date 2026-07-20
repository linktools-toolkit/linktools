#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 / §7.2 line 1852: the external-adapter conformance suite runs from an
installed wheel + the ``test`` extra, with NO source-tree access and NO in-repo
relative imports.

Three checks:
1. The conformance package (linktools-ai/conformance/) imports ONLY the
   installed ``linktools.ai.*`` public surface + sibling modules within the
   package -- never the source tree, ``tests.*``, or private modules.
2. The conformance suite passes when run against the installed linktools.ai
   in the current environment (subprocess pytest).
3. (opt-in, slow) The FULL gold standard: build the wheel, install it + the
   ``test`` extra into a fresh virtualenv, copy the conformance package in,
   and run it there -- proving it executes with zero source-tree access. Set
   RUN_WHEEL_CONFORMANCE=1 to run this (it builds + creates a venv)."""

from __future__ import annotations

import ast
import os
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
    "linktools.ai._runtime",
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


@pytest.mark.skipif(
    not os.environ.get("RUN_WHEEL_CONFORMANCE"),
    reason="slow gold-standard wheel-build test; set RUN_WHEEL_CONFORMANCE=1 to run",
)
def test_conformance_runs_from_built_wheel(tmp_path) -> None:
    # The gold standard (§7.2 line 1852): build the wheel, install it + the
    # [test] extra into a FRESH venv, copy the conformance package in, run it
    # there -- zero source-tree access.
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    build = subprocess.run(
        [sys.executable, "-m", "build", str(_REPO / "linktools-ai"), "--wheel",
         "--no-isolation", "--outdir", str(wheelhouse)],
        capture_output=True, text=True, timeout=180,
    )
    assert build.returncode == 0, f"wheel build failed:\n{build.stderr[-2000:]}"
    wheels = list(wheelhouse.glob("linktools_ai-*.whl"))
    assert wheels, "no wheel produced"
    wheel = wheels[0]

    venv_dir = tmp_path / "venv"
    py = _create_venv(venv_dir)
    install = subprocess.run(
        [py, "-m", "pip", "install", "--quiet", f"{wheel}[test]"],
        capture_output=True, text=True, timeout=300,
    )
    assert install.returncode == 0, (
        f"pip install wheel[test] failed:\n{install.stderr[-2000:]}"
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
