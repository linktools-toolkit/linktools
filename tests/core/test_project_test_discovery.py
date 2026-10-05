#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path

import pytest

import manage


def _new_project(root: Path, name: str = "linktools-example") -> Path:
    project = root / name
    project.mkdir()
    (project / "src").mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.6"\n', encoding="utf-8",
    )
    template = Path(manage.TEMPLATE_PATH) / "linktools.yml"
    (project / "linktools.yml").write_text(
        template.read_text(encoding="utf-8").format(name="example"), encoding="utf-8",
    )
    return project


def _write_test(directory: Path, name: str = "test_example.py") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("def test_example():\n    pass\n", encoding="utf-8")


@pytest.mark.parametrize("test_paths", (
    ("linktools-example/tests",),
    ("tests/example",),
    ("linktools-example/tests", "tests/example"),
))
def test_new_package_discovers_conventional_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_paths: "tuple",
) -> None:
    project = _new_project(tmp_path)
    for path in test_paths:
        _write_test(tmp_path / path)
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    checks = manage._load_project_checks("linktools-example", str(project))

    assert checks["pytest"]["paths"] == tuple(str(tmp_path / path) for path in test_paths)


def test_new_package_without_pytest_files_needs_no_pytest_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _new_project(tmp_path)
    (project / "tests").mkdir()
    _write_test(tmp_path / "tests" / "example", "helper.py")
    _write_test(tmp_path / "tests" / "unrelated")
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    checks = manage._load_project_checks("linktools-example", str(project))

    assert "pytest" not in checks


def test_check_reports_no_pytest_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    _new_project(tmp_path)
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))
    monkeypatch.setattr(manage, "_run_ruff", lambda *args: None)

    manage.handle_check(argparse.Namespace(
        module=[], compatibility=False, skip_compatibility=True, test_tier="merge",
    ))

    assert "linktools-example: pytest skipped (no declared paths or conventional test files)" in capsys.readouterr().out


def test_explicit_pytest_paths_cannot_hide_conventional_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _new_project(tmp_path)
    for path in (project / "tests", tmp_path / "tests" / "example", project / "acceptance"):
        _write_test(path)
    with (project / "linktools.yml").open("a", encoding="utf-8") as file:
        file.write("  pytest:\n    paths:\n      - acceptance\n")
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    checks = manage._load_project_checks("linktools-example", str(project))

    assert checks["pytest"]["paths"] == (
        str(project / "acceptance"), str(project / "tests"), str(tmp_path / "tests" / "example"),
    )


def test_explicit_pytest_paths_do_not_duplicate_conventional_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _new_project(tmp_path)
    for path in (project / "tests", tmp_path / "tests" / "example"):
        _write_test(path)
    with (project / "linktools.yml").open("a", encoding="utf-8") as file:
        file.write("  pytest:\n    paths:\n      - tests\n      - ../tests\n")
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    checks = manage._load_project_checks("linktools-example", str(project))

    assert checks["pytest"]["paths"] == (str(project / "tests"), str(tmp_path / "tests"))


def test_conventional_directory_replaces_narrow_explicit_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _new_project(tmp_path)
    _write_test(project / "tests")
    _write_test(project / "tests", "test_new.py")
    with (project / "linktools.yml").open("a", encoding="utf-8") as file:
        file.write("  pytest:\n    paths:\n      - tests/test_example.py\n")
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    checks = manage._load_project_checks("linktools-example", str(project))

    assert checks["pytest"]["paths"] == (str(project / "tests"),)


@pytest.mark.parametrize("name", ("test_nested.py", "nested_test.py"))
def test_conventional_discovery_recurses_for_pytest_default_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    project = _new_project(tmp_path)
    _write_test(project / "tests" / "nested", name)
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    checks = manage._load_project_checks("linktools-example", str(project))

    assert checks["pytest"]["paths"] == (str(project / "tests"),)


def test_core_keeps_explicit_core_and_cli_test_paths() -> None:
    root = Path(manage.PROJECT_PATH)

    checks = manage._load_project_checks("linktools", str(root / "linktools"))

    assert checks["pytest"]["paths"] == (str(root / "tests" / "core"), str(root / "tests" / "cli"))


def test_default_core_discovery_does_not_collect_other_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _new_project(tmp_path, "linktools")
    _write_test(tmp_path / "tests" / "unrelated")
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    checks = manage._load_project_checks("linktools", str(project))

    assert "pytest" not in checks


def test_discovered_tests_cannot_escape_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    project = _new_project(root)
    outside = tmp_path / "outside-tests"
    outside.mkdir()
    (project / "tests").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(manage, "PROJECT_PATH", str(root))

    with pytest.raises(SystemExit, match="1"):
        manage._load_project_checks("linktools-example", str(project))

    assert "pytest path escapes repository" in capsys.readouterr().err


def test_default_check_runs_discovered_package_tests_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _new_project(tmp_path)
    tests = tmp_path / "tests" / "example"
    _write_test(tests)
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))
    monkeypatch.setattr(manage.shutil, "which", lambda command: "/usr/bin/ruff")
    commands = []
    monkeypatch.setattr(manage.subprocess, "check_call", lambda command, **kwargs: commands.append((command, kwargs)))
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    manage.handle_check(argparse.Namespace(
        module=[], compatibility=False, skip_compatibility=True, test_tier="daily",
    ))

    assert len(commands) == 2
    assert commands[0][0][1:5] == ["check", "--quiet", "--no-cache", "--select"]
    assert commands[1][0] == [
        manage.sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--test-tier", "daily", str(tests),
    ]
    for _, kwargs in commands:
        assert kwargs["cwd"] == str(tmp_path)
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_failing_discovered_test_fails_check_without_writing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture,
) -> None:
    source = Path(manage.PROJECT_PATH)
    for name in ("conftest.py", "pytest.ini"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    project = _new_project(tmp_path)
    tests = project / "tests"
    tests.mkdir()
    (tests / "test_failure.py").write_text("def test_failure():\n    assert False\n", encoding="utf-8")
    _write_test(tests)
    with (project / "linktools.yml").open("a", encoding="utf-8") as file:
        file.write("  pytest:\n    paths:\n      - tests/test_example.py\n")
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))
    monkeypatch.setattr(manage, "_run_ruff", lambda *args: None)
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONPATH", str(source))
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(subprocess.CalledProcessError) as error:
        manage.handle_check(argparse.Namespace(
            module=[], compatibility=False, skip_compatibility=True, test_tier="merge",
        ))

    assert error.value.returncode == 1
    assert "1 failed, 1 passed" in capfd.readouterr().out
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
