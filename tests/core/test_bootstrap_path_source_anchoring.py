#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A relative STORAGE_PATH/DATA_PATH/TEMP_PATH must resolve against the
config *file* that set it, never the process's current working directory.

Regression: the old ``_resolve_bootstrap_paths`` worked off
``LinktoolsFileConfigLoader().load().environment`` -- the merged, source-
blind dict -- so every relative bootstrap value (however it was set) was
normalized with a blanket ``os.path.abspath()`` relative to the process
CWD. A relative STORAGE_PATH in the GLOBAL ``~/.linktools/linktools.json``
happened to look correct only because every existing test's CWD was set to
the local root at construction time; it silently drifted the moment the
same global file was read from a different CWD (a real scenario: any two
commands invoked from different directories).
"""
import json

import pytest

from linktools.core._environ import BaseEnviron, Environ
from linktools.errors import ConfigValidationError
from linktools.types import MISSING


def _reset_global_config():
    descriptor = BaseEnviron.__dict__.get("global_config")
    if descriptor is not None and hasattr(descriptor, "val"):
        descriptor.val = MISSING


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated_bootstrap(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    for name in ("LINKTOOLS_PATH", "LINKTOOLS_STORAGE_PATH", "LINKTOOLS_DATA_PATH", "LINKTOOLS_TEMP_PATH"):
        monkeypatch.delenv(name, raising=False)
    _reset_global_config()
    yield home, cwd
    _reset_global_config()


def test_global_relative_storage_path_anchors_to_global_file_dir_not_cwd(monkeypatch, tmp_path):
    home = tmp_path / "home"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": "./storage"}})

    # CWD is a directory that is NOT the global file's own directory.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _reset_global_config()

    env = Environ()
    assert str(env.paths.storage) == str(home / ".linktools" / "storage")


def test_global_relative_storage_path_survives_cwd_change_after_resolution(monkeypatch, tmp_path):
    home = tmp_path / "home"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": "./storage"}})
    _reset_global_config()

    cwd_a = tmp_path / "cwd-a"
    cwd_a.mkdir()
    monkeypatch.chdir(cwd_a)
    env_a = Environ()
    result_a = str(env_a.paths.storage)

    _reset_global_config()
    cwd_b = tmp_path / "cwd-b"
    cwd_b.mkdir()
    monkeypatch.chdir(cwd_b)
    env_b = Environ()
    result_b = str(env_b.paths.storage)

    assert result_a == result_b == str(home / ".linktools" / "storage")


def test_local_relative_storage_path_still_anchors_to_local_file_dir(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": "./storage"}})
    _reset_global_config()
    env = Environ()
    assert str(env.paths.storage) == str(cwd / "storage")


def test_local_file_relative_path_wins_over_global_file_relative_path(monkeypatch, tmp_path):
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": "./global-storage"}})
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": "./local-storage"}})
    _reset_global_config()
    env = Environ()
    assert str(env.paths.storage) == str(cwd / "local-storage")


def test_env_var_relative_path_still_resolves_against_cwd(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    monkeypatch.setenv("LINKTOOLS_PATH", "./env-storage")
    _reset_global_config()
    env = Environ()
    assert str(env.paths.storage) == str(cwd / "env-storage")


def test_empty_string_storage_path_in_global_file_raises(monkeypatch, tmp_path):
    home = tmp_path / "home"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": ""}})
    _reset_global_config()
    with pytest.raises(ConfigValidationError):
        Environ().paths


def test_data_and_temp_path_relative_values_anchor_to_their_own_file(monkeypatch, tmp_path):
    home = tmp_path / "home"
    _write(home / ".linktools" / "linktools.json", {
        "environment": {"STORAGE_PATH": str(tmp_path / "storage"), "DATA_PATH": "./gdata"},
    })
    _reset_global_config()
    env = Environ()
    assert str(env.data_path) == str(home / ".linktools" / "gdata")
    # TEMP_PATH unset anywhere -- falls back to storage_path/temp.
    assert str(env.temp_path) == str(tmp_path / "storage" / "temp")
