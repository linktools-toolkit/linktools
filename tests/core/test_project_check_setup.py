#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path

import pytest
import yaml

import manage
from scripts.check.matrix import package_checks


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))
    monkeypatch.delenv("RELEASE", raising=False)
    monkeypatch.delenv("SETUP_EDITABLE_MODE", raising=False)
    return tmp_path


def _package(repository: Path, name: str, setup: object = None, dependencies: tuple = ()) -> Path:
    path = repository / name
    path.mkdir()
    data = {"checks": {}, "dev-dependencies": list(dependencies)}
    if setup is not None:
        data["checks"]["setup"] = setup
    (path / "linktools.yml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _install(*names: str, check_setup: bool = True) -> None:
    manage.handle_install(argparse.Namespace(
        module=list(names), editable=True, quiet=True, no_isolation=False,
        check_setup=check_setup,
    ))


@pytest.mark.parametrize("setup", ("echo ready", [], {}, [1], [""], [" \n "]))
def test_invalid_setup_fails_before_install(
    repository: Path, monkeypatch: pytest.MonkeyPatch, setup: object,
) -> None:
    _package(repository, "linktools", setup)
    calls = []
    monkeypatch.setattr(manage.subprocess, "check_call", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(SystemExit, match="1"):
        _install("linktools")
    assert not calls


def test_null_setup_is_invalid(repository: Path) -> None:
    path = _package(repository, "linktools")
    (path / "linktools.yml").write_text("checks:\n  setup: null\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="1"):
        manage.load_project_checks("linktools", str(path))


def test_setup_uses_dependency_closure_once_after_pip(
    repository: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(repository, "linktools", ["echo core"])
    _package(repository, "linktools-ai", ["echo ai", "echo smoke"], ("linktools",))
    _package(repository, "linktools-unused", ["echo unused"])
    calls = []
    monkeypatch.setattr(manage.subprocess, "check_call", lambda *args, **kwargs: calls.append((args, kwargs)))
    _install("linktools-ai", "linktools", "linktools-ai")
    assert calls[0][0][0][1:4] == ["-m", "pip", "install"]
    assert [call[0][0] for call in calls[1:]] == [
        ["bash", "-euo", "pipefail", "-c", command]
        for command in ("echo ai", "echo smoke", "echo core")
    ]
    assert all(call[1] == {"cwd": str(repository)} for call in calls[1:])


def test_core_matrix_empty_install_runs_all_package_setup(
    repository: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(repository, "linktools")
    _package(repository, "linktools-ai", ["echo ai"])
    row, = package_checks(("linktools",))
    assert row["install"] == ""
    calls = []
    monkeypatch.setattr(manage.subprocess, "check_call", lambda *args, **kwargs: calls.append(args[0]))
    _install(*row["install"].split())
    assert calls[1:] == [["bash", "-euo", "pipefail", "-c", "echo ai"]]


@pytest.mark.parametrize("enabled,setup", ((False, ["sudo forbidden"]), (True, None)))
def test_ordinary_install_and_packages_without_setup_only_run_pip(
    repository: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, setup: object,
) -> None:
    _package(repository, "linktools", setup)
    calls = []
    monkeypatch.setattr(manage.subprocess, "check_call", lambda *args, **kwargs: calls.append(args[0]))
    _install("linktools", check_setup=enabled)
    assert len(calls) == 1
    assert calls[0][1:4] == ["-m", "pip", "install"]


@pytest.mark.parametrize("failure_index", (0, 1))
def test_pip_and_setup_failures_propagate_and_stop_remaining_scripts(
    repository: Path, monkeypatch: pytest.MonkeyPatch, failure_index: int,
) -> None:
    _package(repository, "linktools", ["exit 7", "echo unreachable"])
    calls = []

    def run(command: list, **kwargs: object) -> None:
        calls.append(command)
        if len(calls) == failure_index + 1:
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(manage.subprocess, "check_call", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        _install("linktools")
    assert error.value.returncode == 7
    assert len(calls) == failure_index + 1


def test_check_setup_requires_explicit_install_flag() -> None:
    parser = manage.build_parser()
    assert not parser.parse_args(["install"]).check_setup
    assert parser.parse_args(["install", "--check-setup"]).check_setup
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "--check-setup"])


def test_ai_setup_and_workflow_have_one_configuration_owner() -> None:
    root = Path(manage.PROJECT_PATH)
    setup = manage.load_project_checks("linktools-ai", str(root / "linktools-ai"))["setup"]
    assert len(setup) == 1
    assert "sudo apt-get update" in setup[0]
    assert "sudo apt-get install -y file" in setup[0]
    assert 'command -v file' in setup[0]
    assert '${GITHUB_ENV:-}' in setup[0]
    assert 'LINKTOOLS_REQUIRE_LIBMAGIC=1' in setup[0]
    assert 'magic.from_buffer' in setup[0]
    workflow = (root / ".github/workflows/python-check.yml").read_text(encoding="utf-8")
    assert '--editable --quiet --check-setup' in workflow
    assert 'sudo apt-get' not in workflow
    assert 'LINKTOOLS_REQUIRE_LIBMAGIC' not in workflow
    assert 'magic.from_buffer' not in workflow


def test_read_only_check_never_runs_setup(
    repository: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _package(repository, "linktools", ["sudo forbidden"])
    (path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.6"\n', encoding="utf-8")
    calls = []
    monkeypatch.setattr(manage.subprocess, "check_call", lambda *args, **kwargs: calls.append(args))
    manage.handle_check(argparse.Namespace(
        module=["linktools"], skip_compatibility=True, compatibility=False, test_tier="merge",
    ))
    assert not calls
