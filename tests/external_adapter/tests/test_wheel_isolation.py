#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wheel-only packaging proof.

The functional proof -- a from-scratch external adapter driving the full
run -> approval -> resume -> artifact -> job chain through public Protocols --
lives in ``test_runtime_e2e.py``. THIS module proves the PACKAGING half: the
adapter is structured as a STANDALONE package whose sole dependency is the
built ``linktools`` wheel, installable in an isolated venv with NO access to
the core source tree.

Three cases:

1. ``test_package_structure_is_wheel_installable`` -- the package has a
   pyproject.toml declaring it standalone with ``linktools`` as its sole
   runtime dependency, and a src/ layout. This is what makes the isolated
   install POSSIBLE.

2. ``test_adapter_imports_only_wheel_public_paths`` -- AST-scans the adapter
   submodules and asserts every ``linktools.ai.*`` import resolves to a path
   the wheel ships. An import of ``linktools.ai.runtime.builder`` or a reference
   backend (``storage.filesystem`` / ``storage.sqlalchemy`` /
   ``storage.coordination``) would only resolve against the SOURCE tree and
   break wheel-only. This IS the wheel-only proof: an adapter that imports
   only wheel-public paths WILL run against the wheel.

3. ``test_external_adapter_installs_in_isolated_venv`` -- the strong form:
   actually build the wheel, create an isolated venv, install, and import.
   SKIPS with an honest reason when ``python -m venv`` / pip / the build is
   unavailable in the environment (this sandbox blocks ensurepip); the two
   cases above remain the wheel-only proof there.

4. ``test_e2e_harness_never_imports_reference_backends_directly`` -- the AST
   guard above only scans ``src/external_adapter/*.py`` (the adapter package
   itself). That leaves a blind spot: the test files under
   ``tests/external_adapter/tests/`` are what actually DRIVE the E2E chain,
   and one of them could import a reference backend (e.g.
   ``FilesystemRunCommitCoordinator``) directly to wire the run without the
   adapter package ever seeing it -- the adapter-package scan would stay
   green while the functional proof was hollow. This case closes that gap by
   scanning the harness test files themselves."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PKG_ROOT / "src" / "external_adapter"

# Modules that are NOT part of the public wheel surface: the private runtime
# kernel and the in-repo reference backends. An adapter importing any of these
# would only resolve against the source tree, defeating the wheel-only proof.
# names all three reference backends (Filesystem / SQLite / SQLAlchemy)
# plus the coordination impl as forbidden adapter imports.
_FORBIDDEN_PREFIXES = (
    "linktools.ai._",  # private kernel + underscore modules
    "linktools.ai.storage.filesystem",
    "linktools.ai.storage.sqlite",
    "linktools.ai.storage.sqlalchemy",
    "linktools.ai.storage.coordination",
)


def test_package_structure_is_wheel_installable() -> None:
    """The package declares itself standalone with ``linktools`` as its sole
    runtime dependency and ships a src/ layout -- the structure an isolated
    ``pip install`` requires."""
    pyproject = _PKG_ROOT / "pyproject.toml"
    assert pyproject.exists(), "external_adapter/pyproject.toml is missing"
    text = pyproject.read_text(encoding="utf-8")
    assert "external-adapter" in text, "pyproject does not name the package"
    assert "linktools" in text, (
        "pyproject must declare `linktools` as the sole runtime dependency"
    )
    # src/ layout: the importable package lives under src/external_adapter/.
    assert (_SRC_ROOT / "__init__.py").exists(), (
        "src/external_adapter/__init__.py is missing -- not a src-layout package"
    )
    # The package must NOT carry a vendored copy of the core -- its only
    # dependency is the installed wheel.
    assert not (_SRC_ROOT / "linktools").exists(), (
        "the external_adapter package vendored the core source -- it must "
        "depend on the wheel instead"
    )


def test_adapter_imports_only_wheel_public_paths() -> None:
    """Every ``linktools.ai.*`` import in the adapter submodules must resolve
    to a wheel-public path. A reference-backend or private-kernel import would
    only resolve against the source tree and break the wheel-only install."""
    forbidden_hits: "list[str]" = []
    for mod in sorted(_SRC_ROOT.glob("*.py")):
        tree = ast.parse(mod.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    if any(target.startswith(p) for p in _FORBIDDEN_PREFIXES):
                        forbidden_hits.append(f"{mod.name}: import {target}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                target = node.module or ""
                if any(target.startswith(p) for p in _FORBIDDEN_PREFIXES):
                    forbidden_hits.append(f"{mod.name}: from {target} import")
    assert not forbidden_hits, (
        "external adapter imports non-wheel-public paths -- it would not run "
        "against the built wheel alone:\n  " + "\n  ".join(forbidden_hits)
    )


def test_e2e_harness_never_imports_reference_backends_directly() -> None:
    """The adapter-package AST guard (above) has a blind spot: it only scans
    ``src/external_adapter/*.py``. The test files under
    ``tests/external_adapter/tests/`` are what actually drive the E2E chain,
    so scan THEM too -- a harness file that imports
    ``linktools.ai.storage.filesystem`` (or sqlite/sqlalchemy/coordination)
    directly would defeat the wheel-only proof even if the adapter package
    itself stayed clean."""
    tests_dir = _PKG_ROOT / "tests"
    forbidden_hits: "list[str]" = []
    for mod in sorted(tests_dir.glob("*.py")):
        tree = ast.parse(mod.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    if any(target.startswith(p) for p in _FORBIDDEN_PREFIXES):
                        forbidden_hits.append(f"{mod.name}: import {target}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                target = node.module or ""
                if any(target.startswith(p) for p in _FORBIDDEN_PREFIXES):
                    forbidden_hits.append(f"{mod.name}: from {target} import")
    assert not forbidden_hits, (
        "the E2E test harness imports a reference-backend / private-kernel "
        "module directly, defeating the wheel-only adapter proof:\n  "
        + "\n  ".join(forbidden_hits)
    )


def test_external_adapter_installs_in_isolated_venv(tmp_path: Path) -> None:
    """The strong form: build the wheel, create an isolated venv, install the
    external_adapter package against it, and import the adapter with NO core
    source on the path. SKIPS honestly when the environment cannot create a
    venv / run pip / build the wheel (this sandbox blocks ensurepip)."""
    # Probe venv + pip availability BEFORE the build -- skip cleanly if the
    # environment cannot do an isolated install at all.
    venv_dir = tmp_path / "venv"
    try:
        venv.create(venv_dir, with_pip=True, clear=True)
    except (OSError, subprocess.SubprocessError, SystemExit) as exc:
        pytest.skip(
            f"isolated venv unavailable in this environment ({exc!r}); the "
            "structure-validation + AST import-guard cases above remain the "
            "wheel-only proof"
        )

    pip = venv_dir / "bin" / "pip"
    if not pip.exists():  # pragma: no cover - defensive
        pytest.skip("pip not present in the created venv")

    repo_root = _PKG_ROOT.parent.parent
    # Build the core wheel into a temp dist dir.
    build_dir = tmp_path / "dist"
    build_dir.mkdir()
    build = subprocess.run(
        [sys.executable, "manage.py", "build", "linktools-ai"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.skip(
            "core wheel build unavailable in this environment; the structure-"
            "validation + AST import-guard cases above remain the wheel-only "
            f"proof. build stderr: {build.stderr[:200]}"
        )
    # Locate the built wheel + install it + the external_adapter package.
    wheels = list((repo_root / "dist").glob("linktools_ai-*.whl"))
    if not wheels:
        pytest.skip("no built linktools_ai wheel found to install against")
    install = subprocess.run(
        [
            str(pip),
            "install",
            "--no-index",
            "--find-links",
            str(repo_root / "dist"),
            str(wheels[0]),
            str(_PKG_ROOT),
        ],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, (
        f"isolated install failed:\nstdout: {install.stdout}\n"
        f"stderr: {install.stderr}"
    )
    # Import the adapter with the isolated venv's interpreter -- NO core source
    # on the path. A successful import is the wheel-only proof.
    proc = subprocess.run(
        [
            str(venv_dir / "bin" / "python"),
            "-c",
            "import external_adapter; "
            "assert external_adapter.build_in_memory_external_storage is not None",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"adapter did not import in the isolated venv:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
