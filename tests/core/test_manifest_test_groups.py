#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package manifests own both the CI matrix and pytest file partitions."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

import manage
from scripts.check.matrix import package_checks


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = Path(__file__).resolve().parents[2]
    for name in ("conftest.py", "pytest.ini"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("LINKTOOLS_PYTEST", raising=False)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    return tmp_path


def _write_test(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def test_example():\n    pass\n", encoding="utf-8")


def _write_manifest(project: Path, groups: object = None, paths: "tuple" = ("tests",)) -> None:
    check = {"paths": list(paths)}
    if groups is not None:
        check["groups"] = groups
    (project / "linktools.yml").write_text(
        yaml.safe_dump({"checks": {"pytest": check}}, sort_keys=False), encoding="utf-8",
    )


def _new_package(root: Path, name: str = "linktools-example") -> Path:
    project = root / name
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.6"\n', encoding="utf-8")
    _write_test(project / "tests" / "test_alpha.py")
    _write_manifest(project)
    return project


def _collect_package(project: Path, group: str, capfd: pytest.CaptureFixture, *extra: str) -> "set":
    check = manage.load_project_checks(project.name, str(project))["pytest"]
    environment = manage._check_environment()
    environment["PYTEST_ADDOPTS"] = " ".join(
        shlex.quote(arg) for arg in ("--collect-only", "--test-group", group) + extra
    )
    manage._run_pytest(project.name, check, environment, "merge")
    return {line for line in capfd.readouterr().out.splitlines() if "::test_example" in line}


@pytest.mark.parametrize("groups,message", (
    ([], "must be a mapping"),
    ({}, "exactly one"),
    ({"specific": ["test_*.py"]}, "exactly one"),
    ({"first": [], "second": []}, "exactly one"),
    ({"all": []}, "reserved"),
    ({1: []}, "invalid"),
    ({"bad name": []}, "invalid"),
    ({"fallback": [], "specific": "test_*.py"}, "must be a list"),
    ({"fallback": [], "specific": [1]}, "list of strings"),
    ({"fallback": [], "specific": [""]}, "list of strings"),
    ({"fallback": [], "specific": ["tests/test_*.py"]}, "filenames"),
    ({"fallback": [], "specific": [r"tests\test_*.py"]}, "filenames"),
))
def test_invalid_group_declarations_fail_before_matrix_generation(
    repository: Path, capsys: pytest.CaptureFixture, groups: object, message: str,
) -> None:
    project = _new_package(repository)
    _write_manifest(project, groups)
    for load in (
        lambda: manage.load_project_checks(project.name, str(project)),
        lambda: package_checks((project.name,)),
    ):
        with pytest.raises(SystemExit, match="1"):
            load()
        assert message in capsys.readouterr().err


@pytest.mark.parametrize("manifest", (
    "checks: {}\nchecks: {}\n",
    "checks:\n  pytest:\n    paths: [tests]\n    groups:\n      fallback: []\n      fallback: [test_*.py]\n",
))
def test_duplicate_yaml_keys_are_not_silently_overwritten(
    repository: Path, capsys: pytest.CaptureFixture, manifest: str,
) -> None:
    project = _new_package(repository)
    (project / "linktools.yml").write_text(manifest, encoding="utf-8")
    with pytest.raises(SystemExit, match="1"):
        package_checks((project.name,))
    assert "duplicate YAML key" in capsys.readouterr().err


def test_yaml_only_changes_update_matrix_and_subprocess_selection(
    repository: Path, capfd: pytest.CaptureFixture,
) -> None:
    project = _new_package(repository)
    _write_test(project / "tests" / "test_beta.py")
    for name, pattern in (("feature", "test_alpha.py"), ("renamed", "test_beta.py")):
        _write_manifest(project, {name: [pattern], "remaining": []})
        rows = package_checks((project.name,))
        assert [row["group"] for row in rows] == [name, "remaining"]
        assert all("--test-group=" + row["group"] in shlex.split(row["pytest-args"]) for row in rows)
        assert _collect_package(project, name, capfd) == {
            "%s/tests/%s::test_example" % (project.name, pattern),
        }


def test_new_package_and_new_files_remain_in_fallback(
    repository: Path, capfd: pytest.CaptureFixture,
) -> None:
    project = _new_package(repository, "linktools-future")
    _write_manifest(project, {"specific": ["test_alpha.py"], "remaining": []})
    _write_test(project / "tests" / "nested" / "test_new_feature.py")
    _write_test(repository / "tests" / "future" / "test_shared_feature.py")
    assert [row["group"] for row in package_checks((project.name,))] == ["specific", "remaining"]
    assert _collect_package(project, "remaining", capfd) == {
        "linktools-future/tests/nested/test_new_feature.py::test_example",
        "tests/future/test_shared_feature.py::test_example",
    }


def test_package_without_groups_uses_all_and_discovers_new_files(
    repository: Path, capfd: pytest.CaptureFixture,
) -> None:
    project = _new_package(repository, "linktools-future")
    _write_test(project / "tests" / "test_new_feature.py")
    rows = package_checks((project.name,))
    assert len(rows) == 1
    assert rows[0]["group"] == "all"
    assert "--test-group=all" in shlex.split(rows[0]["pytest-args"])
    assert _collect_package(project, "all", capfd) == {
        "linktools-future/tests/test_alpha.py::test_example",
        "linktools-future/tests/test_new_feature.py::test_example",
    }


def test_configured_roots_and_default_discovery_share_the_manifest_groups(
    repository: Path, capfd: pytest.CaptureFixture,
) -> None:
    project = _new_package(repository)
    roots = (project / "acceptance", project / "tests", repository / "tests" / "example")
    for index, root in enumerate(roots):
        _write_test(root / ("test_feature_%d.py" % index))
    _write_manifest(project, {"specific": ["test_feature_*.py"], "remaining": []}, ("acceptance",))
    check = manage.load_project_checks(project.name, str(project))["pytest"]
    assert check["paths"] == tuple(str(root) for root in roots)
    assert check["groups"] == {"specific": ("test_feature_*.py",), "remaining": ()}
    assert _collect_package(project, "specific", capfd) == {
        "%s/test_feature_%d.py::test_example" % (root.relative_to(repository), index)
        for index, root in enumerate(roots)
    }


def test_group_globs_match_basenames_only_within_owned_roots(
    repository: Path, capfd: pytest.CaptureFixture,
) -> None:
    project = _new_package(repository)
    _write_manifest(project, {"specific": ["test_feature*.py"], "remaining": []})
    _write_test(project / "tests" / "nested" / "test_feature_match.py")
    _write_test(project / "tests" / "test_feature_directory.py" / "test_plain.py")
    unrelated = repository / "unrelated" / "test_other.py"
    _write_test(unrelated)
    assert _collect_package(project, "specific", capfd, str(unrelated)) == {
        "linktools-example/tests/nested/test_feature_match.py::test_example",
        "unrelated/test_other.py::test_example",
    }


@pytest.mark.parametrize("with_payload", (False, True))
def test_unknown_named_groups_fail_clearly_even_without_manifest_payload(
    repository: Path, capfd: pytest.CaptureFixture, with_payload: bool,
) -> None:
    project = _new_package(repository)
    _write_manifest(project, {"specific": ["test_alpha.py"], "remaining": []})
    if with_payload:
        with pytest.raises(subprocess.CalledProcessError) as error:
            _collect_package(project, "unknown", capfd)
        assert error.value.returncode == pytest.ExitCode.USAGE_ERROR
        output = capfd.readouterr()
        assert "Unknown test group 'unknown'" in output.err
    else:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--test-group=unknown", str(project / "tests")],
            cwd=repository, env=manage._check_environment(), capture_output=True, text=True,
        )
        assert result.returncode == pytest.ExitCode.USAGE_ERROR
        assert "Unknown test group 'unknown'" in result.stderr


@pytest.mark.parametrize("group", ("all", "specific"))
def test_overlapping_nonfallback_groups_fail_instead_of_duplicating_coverage(
    repository: Path, capfd: pytest.CaptureFixture, group: str,
) -> None:
    project = _new_package(repository)
    _write_manifest(project, {"specific": ["test_alpha.py"], "broad": ["test_*.py"], "remaining": []})
    with pytest.raises(subprocess.CalledProcessError) as error:
        _collect_package(project, group, capfd)
    assert error.value.returncode == pytest.ExitCode.USAGE_ERROR
    assert "Ambiguous test groups" in capfd.readouterr().err


def test_pytest_payload_does_not_leak_across_package_invocations(
    repository: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    grouped = _new_package(repository, "linktools-grouped")
    ungrouped = _new_package(repository, "linktools-ungrouped")
    _write_manifest(grouped, {"specific": ["test_alpha.py"], "remaining": []})
    inherited = '{"groups": {"stale": []}}'
    monkeypatch.setenv("LINKTOOLS_PYTEST", inherited)
    calls = []
    monkeypatch.setattr(manage, "_run_check", lambda command, environment: calls.append((command, environment)))

    manage.handle_check(argparse.Namespace(
        module=[grouped.name, ungrouped.name], compatibility=False, skip_compatibility=True, test_tier="merge",
    ))

    assert len(calls) == 2
    assert calls[0][1] is not calls[1][1]
    payloads = [json.loads(environment["LINKTOOLS_PYTEST"]) for _, environment in calls]
    assert payloads == [
        {"paths": [str(grouped / "tests")], "groups": {"specific": ["test_alpha.py"], "remaining": []}},
        {"paths": [str(ungrouped / "tests")]},
    ]
    assert os.environ["LINKTOOLS_PYTEST"] == inherited


def test_groups_only_manifest_uses_conventional_test_directories(
    repository: Path, capfd: pytest.CaptureFixture,
) -> None:
    project = _new_package(repository)
    _write_test(repository / "tests" / "example" / "test_new_feature.py")
    (project / "linktools.yml").write_text(
        "checks:\n  pytest:\n    groups:\n      specific: [test_alpha.py]\n      remaining: []\n",
        encoding="utf-8",
    )
    check = manage.load_project_checks(project.name, str(project))["pytest"]
    assert check["paths"] == (str(project / "tests"), str(repository / "tests" / "example"))
    assert [row["group"] for row in package_checks((project.name,))] == ["specific", "remaining"]
    assert _collect_package(project, "remaining", capfd) == {
        "tests/example/test_new_feature.py::test_example",
    }


@pytest.mark.parametrize("check", ({"groups": {"remaining": []}}, {}))
def test_declared_pytest_without_any_test_paths_fails_closed(
    repository: Path, capsys: pytest.CaptureFixture, check: dict,
) -> None:
    project = repository / "linktools-empty"
    project.mkdir()
    (project / "linktools.yml").write_text(
        yaml.safe_dump({"checks": {"pytest": check}}), encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="1"):
        package_checks((project.name,))
    assert "declares pytest checks but has no test paths" in capsys.readouterr().err


@pytest.mark.parametrize("paths", ([], None))
def test_explicit_invalid_paths_are_not_replaced_by_conventional_discovery(
    repository: Path, capsys: pytest.CaptureFixture, paths: object,
) -> None:
    project = _new_package(repository)
    (project / "linktools.yml").write_text(
        yaml.safe_dump({"checks": {"pytest": {"paths": paths, "groups": {"remaining": []}}}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="1"):
        manage.load_project_checks(project.name, str(project))
    assert "checks.pytest.paths must be a non-empty list of strings" in capsys.readouterr().err


def _set_ci_pool(project: Path, pool: object) -> None:
    path = project / "linktools.yml"
    data = yaml.safe_load(path.read_text())
    data["checks"]["ci-pool"] = pool
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("pool", (None, "", True, [], {}, "bad name", "-bad"))
def test_invalid_ci_pool_names_fail_closed(
    repository: Path, capsys: pytest.CaptureFixture, pool: object,
) -> None:
    project = _new_package(repository)
    _set_ci_pool(project, pool)
    with pytest.raises(SystemExit, match="1"):
        package_checks((project.name,))
    assert "checks.ci-pool must be a non-empty pool name" in capsys.readouterr().err


def test_ci_pools_are_manifest_owned_and_new_packages_stay_independent(repository: Path) -> None:
    first = _new_package(repository, "linktools-first")
    second = _new_package(repository, "linktools-second")
    new = _new_package(repository, "linktools-new")
    for project in (first, second):
        _set_ci_pool(project, "quick")
    rows = package_checks((second.name, first.name, new.name))
    assert [(row["name"], row["packages"]) for row in rows] == [
        ("quick checks", "%s %s" % (second.name, first.name)),
        ("linktools-new checks", new.name),
    ]
    assert [row["install"] for row in rows] == [row["packages"] for row in rows]
    assert all(row["group"] == "all" for row in rows)
    _set_ci_pool(second, "other")
    assert len(package_checks((second.name, first.name, new.name))) == 3


def test_ci_pool_including_core_keeps_full_install_regardless_of_order(repository: Path) -> None:
    core = _new_package(repository, "linktools")
    other = _new_package(repository, "linktools-other")
    for project in (core, other):
        _set_ci_pool(project, "quick")
    rows = package_checks((other.name, core.name))
    assert len(rows) == 1
    assert rows[0]["packages"] == "%s %s" % (other.name, core.name)
    assert rows[0]["install"] == ""


def test_ci_pool_cannot_hide_an_independent_package(
    repository: Path, capsys: pytest.CaptureFixture,
) -> None:
    first = _new_package(repository, "linktools-first")
    second = _new_package(repository, "linktools-second")
    _set_ci_pool(second, first.name)
    with pytest.raises(SystemExit, match="1"):
        package_checks((first.name, second.name))
    assert "conflicts with an independent package" in capsys.readouterr().err


def test_ci_pool_cannot_mix_with_pytest_groups(
    repository: Path, capsys: pytest.CaptureFixture,
) -> None:
    project = _new_package(repository)
    _write_manifest(project, {"specific": ["test_alpha.py"], "remaining": []})
    _set_ci_pool(project, "quick")
    with pytest.raises(SystemExit, match="1"):
        package_checks((project.name,))
    assert "cannot combine checks.ci-pool and pytest.groups" in capsys.readouterr().err


def test_repository_pool_preserves_all_discovered_package_coverage() -> None:
    modules = manage.get_modules()
    rows = package_checks(modules)
    core = next(row for row in rows if row["name"] == "linktools checks")
    assert {"linktools", "linktools-common", "linktools-mobile"} <= set(core["packages"].split())
    assert core["install"] == ""
    ai = [row for row in rows if row["packages"] == "linktools-ai"]
    assert {"evaluation", "runtime"} <= {row["group"] for row in ai}
    assert all(row["install"] == "linktools-ai" for row in ai)
    cntr = next(row for row in rows if row["name"] == "linktools-cntr checks")
    assert cntr["packages"] == cntr["install"] == "linktools-cntr"
    expected = {
        package: len(manage.load_project_checks(package, module["path"]).get("pytest", {}).get("groups", {"all": ()}))
        for package, module in modules.items()
    }
    assert Counter(package for row in rows for package in row["packages"].split()) == expected
