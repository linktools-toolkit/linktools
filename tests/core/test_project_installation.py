#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path

import pytest
import yaml

import manage


@pytest.fixture(autouse=True)
def clear_build_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RELEASE", raising=False)
    monkeypatch.delenv("SETUP_EDITABLE_MODE", raising=False)


def _modules(tmp_path: Path, configs: dict) -> dict:
    modules = {}
    for name, config in configs.items():
        path = tmp_path / name
        path.mkdir()
        (path / "linktools.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
        modules[name] = {"path": str(path)}
    return modules


def _requirements(modules: dict, *names: str, editable: bool = True) -> list:
    return manage._install_requirements(argparse.Namespace(module=names, editable=editable), modules)


@pytest.mark.parametrize("package,expected", (
    ("linktools", [("linktools", "")]),
    ("linktools-ai", [("linktools-ai", "[web]"), ("linktools", "")]),
    ("linktools-cntr", [("linktools-cntr", ""), ("linktools", "[cli,git]")]),
    ("linktools-common", [("linktools-common", ""), ("linktools", "[cli]")]),
    ("linktools-mobile", [("linktools-mobile", ""), ("linktools", "[cli]")]),
    ("linktools-mobile[ssh]", [("linktools-mobile", "[ssh]"), ("linktools", "[cli,ssh]")]),
))
def test_named_install_uses_only_local_dependency_closure(package: str, expected: list) -> None:
    modules = manage.get_modules()
    assert _requirements(modules, package) == [
        (name, modules[name]["path"] + extras) for name, extras in expected
    ]


def test_default_install_preserves_all_discovered_projects(tmp_path: Path) -> None:
    modules = {"linktools-new": {"path": str(tmp_path / "linktools-new")}}
    assert _requirements(modules) == [("linktools-new", modules["linktools-new"]["path"])]


@pytest.mark.parametrize("editable,release,develop,expected", (
    (False, "false", "false", ["linktools", "linktools-base"]),
    (True, "false", "false", ["linktools", "linktools-base", "linktools-dev"]),
    (False, "true", "false", ["linktools", "linktools-base", "linktools-release"]),
    (False, "1", "yes", ["linktools", "linktools-base", "linktools-dev", "linktools-release"]),
    (True, "YES", "false", ["linktools", "linktools-base", "linktools-dev", "linktools-release"]),
))
def test_install_matches_backend_dependency_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    editable: bool, release: str, develop: str, expected: list,
) -> None:
    modules = _modules(tmp_path, {
        "linktools": {
            "dependencies": ["linktools-base>=1"],
            "dev-dependencies": ["linktools-dev"],
            "release-dependencies": ["linktools-release"],
        },
        "linktools-base": {}, "linktools-dev": {}, "linktools-release": {},
    })
    monkeypatch.setenv("RELEASE", release)
    monkeypatch.setenv("SETUP_EDITABLE_MODE", develop)
    assert [name for name, _ in _requirements(modules, "linktools", editable=editable)] == expected


def test_transitive_extras_merge_and_cycles_terminate(tmp_path: Path) -> None:
    modules = _modules(tmp_path, {
        "linktools": {"optional-dependencies": {
            "first": ["linktools-one[all]>=1"], "later": ["linktools-two"],
        }},
        "linktools-one": {
            "dependencies": ["linktools[later]"],
            "optional-dependencies": {"feature": ["linktools-three"]},
        },
        "linktools-two": {}, "linktools-three": {}, "linktools-unused": {},
    })
    result = dict(_requirements(modules, "linktools[first]", "linktools-one", "linktools[first]"))
    assert result == {
        "linktools": modules["linktools"]["path"] + "[first,later]",
        "linktools-one": modules["linktools-one"]["path"] + "[all]",
        "linktools-two": modules["linktools-two"]["path"],
        "linktools-three": modules["linktools-three"]["path"],
    }


def test_markers_and_optional_groups_are_evaluated_in_their_own_context(tmp_path: Path) -> None:
    modules = _modules(tmp_path, {
        "linktools": {
            "dependencies": [
                "linktools-base; python_version >= '3.6'",
                "linktools-never; python_version < '2'",
                "linktools-extra; extra == 'feature'",
            ],
            "optional-dependencies": {
                "feature": ["linktools-option; extra == 'feature'"],
                "unused": ["linktools-never"],
            },
        },
        "linktools-base": {}, "linktools-never": {}, "linktools-extra": {}, "linktools-option": {},
    })
    assert [name for name, _ in _requirements(modules, "linktools[feature]")] == [
        "linktools", "linktools-base", "linktools-extra", "linktools-option",
    ]
    assert [name for name, _ in _requirements(modules, "linktools[all]")] == [
        "linktools", "linktools-base", "linktools-never",
    ]


def test_dependency_names_and_extras_use_standard_normalization(tmp_path: Path) -> None:
    modules = _modules(tmp_path, {
        "linktools": {"dependencies": ["Linktools_One[My_Feature]>=1"]},
        "linktools-one": {"optional-dependencies": {"my.feature": ["linktools-two"]}},
        "linktools-two": {},
    })
    assert _requirements(modules, "linktools", "linktools-one[MY-feature]") == [
        ("linktools", modules["linktools"]["path"]),
        ("linktools-one", modules["linktools-one"]["path"] + "[my-feature]"),
        ("linktools-two", modules["linktools-two"]["path"]),
    ]


def test_normalized_extra_aliases_preserve_all_dependencies(tmp_path: Path) -> None:
    modules = _modules(tmp_path, {
        "linktools": {"optional-dependencies": {
            "my-feature": ["linktools-one"], "my_feature": ["linktools-two"],
        }},
        "linktools-one": {}, "linktools-two": {},
    })
    assert [name for name, _ in _requirements(modules, "linktools[my.feature]")] == [
        "linktools", "linktools-one", "linktools-two",
    ]


def test_all_extra_keeps_explicit_and_synthesized_dependencies(tmp_path: Path) -> None:
    modules = _modules(tmp_path, {
        "linktools": {"optional-dependencies": {
            "all": ["linktools-one"], "feature": ["linktools-two"],
        }},
        "linktools-one": {}, "linktools-two": {},
    })
    assert [name for name, _ in _requirements(modules, "linktools[all]")] == [
        "linktools", "linktools-one", "linktools-two",
    ]


def test_external_and_explicit_url_dependencies_are_left_to_pip(tmp_path: Path) -> None:
    modules = _modules(tmp_path, {
        "linktools": {"dependencies": [
            "external[feature]>=1", "linktools-one @ https://example.org/package.whl",
        ]},
        "linktools-one": {},
    })
    assert _requirements(modules, "linktools") == [("linktools", modules["linktools"]["path"])]


@pytest.mark.parametrize("selection", ("unknown", "linktools[]", "linktools[feature,]", "linktools>=1"))
def test_invalid_selection_fails_before_running_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, selection: str,
) -> None:
    modules = _modules(tmp_path, {"linktools": {}})
    monkeypatch.setattr(manage, "get_modules", lambda: modules)
    calls = []
    monkeypatch.setattr(manage.subprocess, "check_call", calls.append)
    with pytest.raises(SystemExit) as error:
        manage.handle_install(argparse.Namespace(module=[selection], editable=True))
    assert error.value.code == 2
    assert "Unknown project module" in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("editable", (False, True))
def test_install_uses_one_pip_resolution_with_original_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, editable: bool,
) -> None:
    modules = _modules(tmp_path, {
        "linktools": {}, "linktools-one": {"dependencies": ["linktools[cli]>=1"]},
    })
    monkeypatch.setattr(manage, "get_modules", lambda: modules)
    calls = []
    monkeypatch.setattr(manage.subprocess, "check_call", calls.append)
    manage.handle_install(argparse.Namespace(
        module=["linktools-one"], editable=editable, quiet=True, no_isolation=True,
    ))
    expected = [sys.executable, "-m", "pip", "install", "--quiet"]
    for name, extras in (("linktools-one", ""), ("linktools", "[cli]")):
        if editable:
            expected.append("-e")
        expected.append(os.path.join(str(tmp_path), name) + extras)
    assert calls == [expected + ["--no-build-isolation"]]
