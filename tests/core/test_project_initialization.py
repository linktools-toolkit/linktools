#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

import pytest
import tomlkit
import yaml

import manage


@pytest.mark.parametrize("package", ["linktools-ai", "linktools-cntr", "linktools-common", "linktools-mobile"])
def test_init_preserves_existing_package_policy(tmp_path, monkeypatch, package):
    source = Path(manage.PROJECT_PATH) / package
    target = tmp_path / package
    target.mkdir()
    for name in ("pyproject.toml", "linktools.yml"):
        shutil.copy2(str(source / name), str(target / name))
    before = tomlkit.parse((target / "pyproject.toml").read_text(encoding="utf-8"))
    yaml_before = (target / "linktools.yml").read_bytes()
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    manage.handle_init(argparse.Namespace(module=[package]))

    after = tomlkit.parse((target / "pyproject.toml").read_text(encoding="utf-8"))
    assert after == before
    assert (target / "linktools.yml").read_bytes() == yaml_before
    first = (target / "pyproject.toml").read_bytes()
    manage.handle_init(argparse.Namespace(module=[package]))
    assert (target / "pyproject.toml").read_bytes() == first


def test_init_completes_missing_metadata_without_replacing_explicit_values(tmp_path, monkeypatch):
    target = tmp_path / "linktools-example"
    target.mkdir()
    (target / "pyproject.toml").write_text('''[project]
name = "custom-package"
requires-python = ">=3.10"
authors = []

[project.urls]
Homepage = "https://example.org/project"

[tool.setuptools]
include-package-data = false

[tool.setuptools.package-data]
linktools = []
''', encoding="utf-8")
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    manage.handle_init(argparse.Namespace(module=["linktools-example"]))

    result = tomlkit.parse((target / "pyproject.toml").read_text(encoding="utf-8"))
    assert result["project"]["name"] == "custom-package"
    assert result["project"]["requires-python"] == ">=3.10"
    assert result["project"]["authors"] == []
    assert result["project"]["urls"]["Homepage"] == "https://example.org/project"
    assert result["project"]["urls"]["Repository"] == "https://github.com/linktools-toolkit/linktools.git"
    assert result["tool"]["setuptools"]["include-package-data"] is False
    assert result["tool"]["setuptools"]["package-data"]["linktools"] == []
    assert result["build-system"]["build-backend"] == "linktools_setup.build_meta"


def test_init_creates_complete_new_package_metadata(tmp_path, monkeypatch):
    target = tmp_path / "linktools-example"
    target.mkdir()
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))

    manage.handle_init(argparse.Namespace(module=["linktools-example"]))

    result = tomlkit.parse((target / "pyproject.toml").read_text(encoding="utf-8"))
    assert result["project"]["name"] == "linktools-example"
    assert result["project"]["requires-python"] == ">=3.6"
    assert result["project"]["urls"]["Repository"] == "https://github.com/linktools-toolkit/linktools.git"
    assert result["build-system"]["requires"] == ["linktools-setup==0.0.3"]
    assert result["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert result["tool"]["setuptools"]["package-data"]["linktools"] == ["**/assets/**"]
    config = yaml.safe_load((target / "linktools.yml").read_text(encoding="utf-8"))
    assert config["name"] == "example"
    assert config["optional-dependencies"] == {}
    assert config["scripts"]["capability"] == "linktools.capabilities.example:__cap_example__"
    assert (target / "capability.jinja2").is_file()
