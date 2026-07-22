#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""the core wheel carries NO environment-specific DB driver (postgres/mysql/
etc.) or external coordination client (redis/etcd). The core stays
backend-neutral; a deployment using those brings its own driver + injects a
distributed coordinator.

Three gates, all in the default suite (a skipped acceptance test is not
evidence):
1. requirements.yml -- the source the wheel's METADATA is generated from --
   declares no env driver. Catches the common regression at zero build cost.
2. build the real wheel + parse its METADATA Requires-Dist + assert no env
   driver. This is 's mandated artifact scan (the source scan alone
   could miss a build-pipeline surprise).
3. install the wheel in a fresh venv WITHOUT extras + import linktools.ai --
   the root surface needs no optional dep. SKIPS honestly (not gated) when the
   environment cannot create a working venv (no ensurepip/virtualenv)."""

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_PKG = _REPO / "linktools-ai"
_REQS = _PKG / "requirements.yml"

# Drivers/clients the core wheel must NOT carry. A deployment using one of
# these brings it itself; the core never pins it. The forbidden-token set
# is asyncpg|asyncmy|aiomysql|redis|boto|s3|mysql|postgres; pymysql/etcd are
# kept as reasonable additions in the same class.
_FORBIDDEN_DEPS = (
    "asyncpg",
    "asyncmy",
    "aiomysql",
    "pymysql",
    "redis",
    "etcd",
    "boto",
    "s3",
    "mysql",
    "postgres",
)


def _parse_requires_yml() -> "dict[str, list[str]]":
    """Minimal requirements.yml reader: returns {section: [dep lines]} for the
    dependencies + optional-dependencies.* sections (the sources the wheel
    METADATA is generated from)."""
    import re

    text = _REQS.read_text(encoding="utf-8")
    sections: "dict[str, list[str]]" = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            current = stripped[:-1]
            sections.setdefault(current, [])
        elif line.startswith("  - ") and current is not None:
            # normalize: drop the 'name[extra]>=...' to the bare name
            dep = stripped.lstrip("- ").split("|")[0]
            name = re.split(r"[\[<>=!~ ]", dep)[0].lower()
            sections[current].append(name)
    return sections


def test_requirements_yml_carries_no_env_drivers() -> None:
    # The default gate: every deps section of requirements.yml is free of the
    # env drivers/coordination clients. The wheel METADATA is generated from
    # this file, so a clean source means a clean wheel.
    sections = _parse_requires_yml()
    offenders: "list[str]" = []
    for section, deps in sections.items():
        for dep in deps:
            if any(dep == f or dep.startswith(f) for f in _FORBIDDEN_DEPS):
                offenders.append(f"{section}: {dep}")
    assert not offenders, (
        "requirements.yml declares an environment-specific driver/client the "
        "core wheel must not carry:\n  " + "\n  ".join(sorted(offenders))
    )


def test_wheel_metadata_carries_no_env_drivers(tmp_path) -> None:
    # The artifact scan: build the real wheel + parse its
    # METADATA Requires-Dist. No env driver may appear in any extra.
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            str(_PKG),
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheelhouse),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert build.returncode == 0, f"wheel build failed:\n{build.stderr[-2000:]}"
    wheels = list(wheelhouse.glob("linktools_ai-*.whl"))
    assert wheels, "no wheel produced"
    metadata = _read_wheel_metadata(wheels[0])
    # Requires-Dist lines look like: 'asyncpg>=0.29,<1; extra == "postgres"'
    offenders = [
        line
        for line in (metadata.get_all("Requires-Dist") or [])
        if any(f in line.lower() for f in _FORBIDDEN_DEPS)
    ]
    assert not offenders, (
        "wheel METADATA Requires-Dist carries an environment-specific "
        "driver/client:\n  " + "\n  ".join(sorted(offenders))
    )


def test_root_imports_in_minimal_venv(tmp_path) -> None:
    # The root surface (``import linktools.ai``) needs no optional dep: install
    # the built wheel in a fresh venv WITHOUT extras + import it.
    import zipfile

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            str(_PKG),
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheelhouse),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert build.returncode == 0, f"wheel build failed:\n{build.stderr[-2000:]}"
    wheels = list(wheelhouse.glob("linktools_ai-*.whl"))
    assert wheels, "no wheel produced"
    wheel = wheels[0]

    py = _create_venv(tmp_path / "venv")
    install = subprocess.run(
        [py, "-m", "pip", "install", "--quiet", str(wheel)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert install.returncode == 0, (
        f"pip install wheel (no extras) failed:\n{install.stderr[-2000:]}"
    )
    probe = subprocess.run(
        [py, "-c", "import linktools.ai; print('ok')"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, (
        f"import linktools.ai failed in a minimal venv (no extras):\n"
        f"stdout={probe.stdout}\nstderr={probe.stderr[-2000:]}"
    )


def _read_wheel_metadata(wheel: Path):
    import email

    with zipfile.ZipFile(wheel) as zf:
        name = [n for n in zf.namelist() if n.endswith("METADATA")][0]
        return email.message_from_string(zf.read(name).decode("utf-8"))


def _create_venv(venv_dir: Path) -> str:
    """Create a fresh venv with working pip (skip if the environment can't)."""
    import venv as _venv

    try:
        _venv.create(venv_dir, with_pip=True, clear=True)
    except BaseException:
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

    pytest.skip("cannot create a venv with working pip in this environment")
