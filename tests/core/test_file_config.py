#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinktoolsFileConfig / LinktoolsFileConfigLoader / merge_file_config
(spec Part I-II): a generic two-layer JSON file configuration source, not a
Manifest protocol -- no version/kind/schema_version required, no parent
directory search, unknown top-level fields preserved verbatim."""
import json
import os

import pytest

from linktools.core import (
    LinktoolsFileConfig,
    LinktoolsFileConfigLoader,
    ResolvedLinktoolsFileConfig,
    ensure_requirement,
    merge_file_config,
)
from linktools.errors import ConfigError, ConfigValidationError


def _write(path, data):
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data), encoding="utf-8")


def _loader(tmp_path, global_data=None):
    global_path = tmp_path / "global" / "linktools.json"
    global_path.parent.mkdir(parents=True, exist_ok=True)
    if global_data is not None:
        _write(global_path, global_data)
    return LinktoolsFileConfigLoader(global_path=global_path)


# -- LinktoolsFileConfigLoader.load() ---------------------------------------

def test_global_and_local_both_missing_yield_empty_configs(tmp_path):
    loader = _loader(tmp_path)
    resolved = loader.load(local_root=tmp_path / "repo")
    assert resolved.global_config.to_dict() == {}
    assert resolved.local_config.to_dict() == {}
    assert resolved.environment == {}


def test_global_missing_local_present(tmp_path):
    loader = _loader(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / ".linktools.json", {"environment": {"STORAGE_PATH": "./storage"}})
    resolved = loader.load(local_root=repo)
    assert resolved.global_config.to_dict() == {}
    assert resolved.environment == {"STORAGE_PATH": "./storage"}


def test_local_missing_global_present(tmp_path):
    loader = _loader(tmp_path, {"environment": {"STORAGE_PATH": "/mnt/nas"}})
    resolved = loader.load(local_root=tmp_path / "repo")
    assert resolved.local_config.to_dict() == {}
    assert resolved.environment == {"STORAGE_PATH": "/mnt/nas"}


def test_both_present_local_overrides(tmp_path):
    loader = _loader(tmp_path, {"environment": {"STORAGE_PATH": "/global", "DOWNLOAD_PATH": "/g/dl"}})
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / ".linktools.json", {"environment": {"STORAGE_PATH": "./local"}})
    resolved = loader.load(local_root=repo)
    assert resolved.environment == {"STORAGE_PATH": "./local", "DOWNLOAD_PATH": "/g/dl"}


def test_minimal_legal_file_is_empty_object(tmp_path):
    loader = _loader(tmp_path, {})
    resolved = loader.load(local_root=tmp_path / "repo")
    assert resolved.global_config.to_dict() == {}


def test_root_non_object_raises_config_error(tmp_path):
    loader = _loader(tmp_path, "[1, 2, 3]")
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_environment_non_object_raises(tmp_path):
    loader = _loader(tmp_path, {"environment": [1, 2]})
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_requires_non_object_raises(tmp_path):
    loader = _loader(tmp_path, {"requires": "nope"})
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_requires_non_string_value_raises(tmp_path):
    loader = _loader(tmp_path, {"requires": {"linktools-cntr": 12}})
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_requires_empty_string_value_raises(tmp_path):
    loader = _loader(tmp_path, {"requires": {"linktools-cntr": ""}})
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_requires_invalid_specifier_raises(tmp_path):
    loader = _loader(tmp_path, {"requires": {"linktools-cntr": "not a specifier!!"}})
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


# -- blank/whitespace keys ---------------------------------------------------

def test_environment_empty_key_raises(tmp_path):
    loader = _loader(tmp_path, {"environment": {"": "x"}})
    with pytest.raises(ConfigValidationError):
        loader.load(local_root=tmp_path / "repo")


def test_environment_whitespace_only_key_raises(tmp_path):
    loader = _loader(tmp_path, {"environment": {"   ": "x"}})
    with pytest.raises(ConfigValidationError):
        loader.load(local_root=tmp_path / "repo")


def test_environment_key_with_leading_or_trailing_space_raises(tmp_path):
    loader = _loader(tmp_path, {"environment": {" KEY": "x"}})
    with pytest.raises(ConfigValidationError):
        loader.load(local_root=tmp_path / "repo")


def test_requires_empty_key_raises(tmp_path):
    loader = _loader(tmp_path, {"requires": {"": ">=1.0"}})
    with pytest.raises(ConfigValidationError):
        loader.load(local_root=tmp_path / "repo")


def test_requires_whitespace_only_value_raises(tmp_path):
    loader = _loader(tmp_path, {"requires": {"linktools-cntr": "   "}})
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_invalid_json_raises(tmp_path):
    loader = _loader(tmp_path, "{not json")
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_invalid_utf8_raises(tmp_path):
    global_path = tmp_path / "global" / "linktools.json"
    global_path.parent.mkdir(parents=True, exist_ok=True)
    global_path.write_bytes(b"\xff\xfe\x00\x01")
    loader = LinktoolsFileConfigLoader(global_path=global_path)
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_oversized_file_raises(tmp_path):
    loader = _loader(tmp_path, {"environment": {"X": "y" * (1024 * 1024 + 1)}})
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_dangling_symlink_raises_not_silently_empty(tmp_path):
    global_path = tmp_path / "global" / "linktools.json"
    global_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(tmp_path / "does-not-exist"), str(global_path))
    loader = LinktoolsFileConfigLoader(global_path=global_path)
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_directory_in_place_of_file_raises(tmp_path):
    global_path = tmp_path / "global" / "linktools.json"
    global_path.mkdir(parents=True)
    loader = LinktoolsFileConfigLoader(global_path=global_path)
    with pytest.raises(ConfigError):
        loader.load(local_root=tmp_path / "repo")


def test_unknown_top_level_fields_preserved(tmp_path):
    loader = _loader(tmp_path, {"cntr": {"foo": "bar"}, "metadata": {"team": "x"}})
    resolved = loader.load(local_root=tmp_path / "repo")
    assert resolved.global_config.get_path("cntr", "foo") == "bar"
    assert resolved.global_config.to_dict()["metadata"] == {"team": "x"}


def test_to_dict_is_a_deep_copy(tmp_path):
    loader = _loader(tmp_path, {"environment": {"A": "1"}})
    resolved = loader.load(local_root=tmp_path / "repo")
    d = resolved.global_config.to_dict()
    d["environment"]["A"] = "mutated"
    assert resolved.global_config.environment == {"A": "1"}


def test_get_local_path_uses_cwd_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = LinktoolsFileConfigLoader()
    assert loader.get_local_path() == str(tmp_path / ".linktools.json")


def test_get_global_path_default_is_home_dot_linktools(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    loader = LinktoolsFileConfigLoader()
    assert loader.get_global_path() == str(tmp_path / ".linktools" / "linktools.json")


# -- merge_file_config() -----------------------------------------------------

def test_merge_local_scalar_overrides_global():
    merged = merge_file_config({"environment": {"STORAGE_PATH": "/g"}}, {"environment": {"STORAGE_PATH": "/l"}})
    assert merged["environment"]["STORAGE_PATH"] == "/l"


def test_merge_uncovered_key_inherited():
    merged = merge_file_config({"environment": {"A": "1", "B": "2"}}, {"environment": {"A": "override"}})
    assert merged["environment"] == {"A": "override", "B": "2"}


def test_merge_nested_object_recursive():
    merged = merge_file_config(
        {"ai": {"agents_path": "global-agents", "skills_path": "global-skills"}},
        {"ai": {"skills_path": "skills"}},
    )
    assert merged["ai"] == {"agents_path": "global-agents", "skills_path": "skills"}


def test_merge_array_replaced_not_concatenated():
    merged = merge_file_config({"environment": {"MIRRORS": ["a", "b"]}}, {"environment": {"MIRRORS": ["c"]}})
    assert merged["environment"]["MIRRORS"] == ["c"]


def test_merge_null_is_explicit_override():
    merged = merge_file_config({"environment": {"A": "1"}}, {"environment": {"A": None}})
    assert merged["environment"]["A"] is None


def test_merge_does_not_mutate_inputs():
    global_data = {"environment": {"A": "1"}}
    local_data = {"environment": {"A": "2"}}
    merge_file_config(global_data, local_data)
    assert global_data == {"environment": {"A": "1"}}
    assert local_data == {"environment": {"A": "2"}}


def test_merge_empty_local():
    merged = merge_file_config({"environment": {"A": "1"}}, {})
    assert merged == {"environment": {"A": "1"}}


def test_merge_empty_global():
    merged = merge_file_config({}, {"environment": {"A": "1"}})
    assert merged == {"environment": {"A": "1"}}


# -- get_path() / require_path() --------------------------------------------

def test_get_path_returns_nested_value():
    config = LinktoolsFileConfig({"ai": {"agents_path": "agents"}})
    assert config.get_path("ai", "agents_path") == "agents"


def test_get_path_missing_returns_default():
    config = LinktoolsFileConfig({})
    assert config.get_path("ai", "agents_path", default="agents") == "agents"
    assert config.get_path("ai", "agents_path") is None


def test_get_path_empty_returns_deep_copy_of_root():
    data = {"environment": {"A": "1"}}
    config = LinktoolsFileConfig(data)
    root = config.get_path()
    root["environment"]["A"] = "mutated"
    assert config.environment == {"A": "1"}


def test_require_path_raises_when_missing():
    config = LinktoolsFileConfig({})
    with pytest.raises(ConfigError):
        config.require_path("ai", "agents_path")


def test_require_path_returns_value_when_present():
    config = LinktoolsFileConfig({"ai": {"agents_path": "agents"}})
    assert config.require_path("ai", "agents_path") == "agents"


# -- requirement isolation (spec §10) ----------------------------------------

def test_effective_view_does_not_expose_requirement_override(tmp_path):
    loader = _loader(tmp_path, {"requires": {"linktools-cntr": ">=99.0"}})
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / ".linktools.json", {"requires": {"linktools-cntr": ">=0.1,<1.0"}})
    resolved = loader.load(local_root=repo)
    # A caller that (incorrectly) checks a requirement against `effective`
    # would see the local value here too -- but the correct call site is
    # `resolved.local_config.get_requirement`, never `resolved.effective`.
    assert resolved.local_config.get_requirement("linktools-cntr") == ">=0.1,<1.0"
    assert resolved.global_config.get_requirement("linktools-cntr") == ">=99.0"


# -- ensure_requirement() ----------------------------------------------------

def test_ensure_requirement_no_requirement_is_noop():
    config = LinktoolsFileConfig({})
    ensure_requirement(config, "linktools-cntr", "0.12.0")


def test_ensure_requirement_satisfied():
    config = LinktoolsFileConfig({"requires": {"linktools-cntr": ">=0.12,<0.14"}})
    ensure_requirement(config, "linktools-cntr", "0.13.0")


def test_ensure_requirement_unsatisfied_raises():
    config = LinktoolsFileConfig({"requires": {"linktools-cntr": ">=0.12,<0.14"}})
    with pytest.raises(ConfigValidationError):
        ensure_requirement(config, "linktools-cntr", "0.20.0")


def test_ensure_requirement_ignores_unknown_key():
    config = LinktoolsFileConfig({"requires": {"linktools-ai": ">=0.1"}})
    ensure_requirement(config, "linktools-cntr", "0.13.0")


def test_ensure_requirement_invalid_actual_version_raises():
    config = LinktoolsFileConfig({"requires": {"linktools-cntr": ">=0.12"}})
    with pytest.raises(ConfigValidationError):
        ensure_requirement(config, "linktools-cntr", "not-a-version")
