#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProjectProfile: reads a single ``linktools.json``/``.linktools.json``
file, or merges an ordered list of them (earlier paths take precedence).
Not a Manifest protocol: no version/kind/schema_version required, no parent
directory search, and no shape validation of ``environment``/``requires`` at
load time -- a malformed section is read verbatim and it's up to the
specific consumer (e.g. cntr's ``ensure_requirement``) to reject it when it
actually needs to interpret that section.
"""
import json
import os
from pathlib import Path

import pytest

from linktools.core import ProjectProfile
from linktools.errors import ConfigError


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data), encoding="utf-8")


# -- construction / merging ---------------------------------------------------

def test_zero_paths_returns_empty_profile():
    profile = ProjectProfile()
    assert profile.to_dict() == {}


def test_dict_input_is_an_internal_construction_form():
    profile = ProjectProfile({"environment": {"A": "1"}})
    assert profile.to_dict() == {"environment": {"A": "1"}}


def test_none_is_treated_as_empty():
    profile = ProjectProfile(None)
    assert profile.to_dict() == {}


def test_single_missing_path_yields_empty_profile(tmp_path):
    profile = ProjectProfile(str(tmp_path / "does-not-exist.json"))
    assert profile.to_dict() == {}


def test_single_present_path_is_read(tmp_path):
    path = tmp_path / "linktools.json"
    _write(path, {"environment": {"A": "1"}})
    profile = ProjectProfile(str(path))
    assert profile.to_dict() == {"environment": {"A": "1"}}


def test_two_paths_earlier_wins_on_conflicting_key(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _write(first, {"environment": {"KEY": "first"}})
    _write(second, {"environment": {"KEY": "second"}})
    profile = ProjectProfile(str(first), str(second))
    assert profile.get("environment")["KEY"] == "first"


def test_two_paths_uncovered_key_inherited_from_later_path(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _write(first, {"environment": {"A": "1"}})
    _write(second, {"environment": {"A": "override", "B": "2"}})
    profile = ProjectProfile(str(first), str(second))
    assert profile.get("environment") == {"A": "1", "B": "2"}


def test_three_paths_merge_in_order(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    third = tmp_path / "c.json"
    _write(first, {"environment": {"A": "1", "B": "1"}})
    _write(second, {"environment": {"B": "2", "C": "2"}})
    _write(third, {"environment": {"C": "3"}})
    profile = ProjectProfile(str(first), str(second), str(third))
    # Earlier paths win on conflicting keys; a key absent from every earlier
    # path is inherited from whichever later path sets it.
    assert profile.get("environment") == {"A": "1", "B": "1", "C": "2"}


def test_missing_path_in_the_middle_contributes_nothing(tmp_path):
    first = tmp_path / "a.json"
    missing = tmp_path / "does-not-exist.json"
    third = tmp_path / "c.json"
    _write(first, {"environment": {"A": "1"}})
    _write(third, {"environment": {"C": "3"}})
    profile = ProjectProfile(str(first), str(missing), str(third))
    assert profile.get("environment") == {"A": "1", "C": "3"}


def test_non_path_non_dict_argument_raises_type_error():
    with pytest.raises(TypeError):
        ProjectProfile(123)


# -- load errors (still enforced) ---------------------------------------------

def test_root_non_object_raises_config_error(tmp_path):
    path = tmp_path / "linktools.json"
    _write(path, "[1, 2, 3]")
    with pytest.raises(ConfigError):
        ProjectProfile(str(path))


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "linktools.json"
    _write(path, "{not json")
    with pytest.raises(ConfigError):
        ProjectProfile(str(path))


def test_invalid_utf8_raises(tmp_path):
    path = tmp_path / "linktools.json"
    path.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ConfigError):
        ProjectProfile(str(path))


def test_oversized_file_raises(tmp_path):
    path = tmp_path / "linktools.json"
    _write(path, {"environment": {"X": "y" * (1024 * 1024 + 1)}})
    with pytest.raises(ConfigError):
        ProjectProfile(str(path))


def test_dangling_symlink_raises_not_silently_empty(tmp_path):
    path = tmp_path / "linktools.json"
    os.symlink(str(tmp_path / "does-not-exist"), str(path))
    with pytest.raises(ConfigError):
        ProjectProfile(str(path))


def test_directory_in_place_of_file_raises(tmp_path):
    path = tmp_path / "linktools.json"
    path.mkdir(parents=True)
    with pytest.raises(ConfigError):
        ProjectProfile(str(path))


# -- no shape validation at load time ------------------------------------------

def test_environment_non_object_does_not_raise(tmp_path):
    path = tmp_path / "linktools.json"
    _write(path, {"environment": [1, 2]})
    profile = ProjectProfile(str(path))
    assert profile.get("environment") == [1, 2]


def test_requires_non_object_does_not_raise(tmp_path):
    path = tmp_path / "linktools.json"
    _write(path, {"requires": "nope"})
    profile = ProjectProfile(str(path))
    assert profile.get("requires") == "nope"


def test_unknown_top_level_fields_preserved(tmp_path):
    path = tmp_path / "linktools.json"
    _write(path, {"cntr": {"foo": "bar"}, "metadata": {"team": "x"}})
    profile = ProjectProfile(str(path))
    assert profile.get_path("cntr", "foo") == "bar"
    assert profile.to_dict()["metadata"] == {"team": "x"}


# -- get() / get_path() / require_path() / to_dict() ---------------------------

def test_get_returns_top_level_value_and_default():
    profile = ProjectProfile({"metadata": {"team": "core"}})
    assert profile.get("metadata") == {"team": "core"}
    assert profile.get("missing", "fallback") == "fallback"


def test_get_path_returns_nested_value():
    profile = ProjectProfile({"ai": {"agents_path": "agents"}})
    assert profile.get_path("ai", "agents_path") == "agents"


def test_get_path_missing_returns_default():
    profile = ProjectProfile({})
    assert profile.get_path("ai", "agents_path", default="agents") == "agents"
    assert profile.get_path("ai", "agents_path") is None


def test_get_path_empty_returns_deep_copy_of_root():
    profile = ProjectProfile({"environment": {"A": "1"}})
    root = profile.get_path()
    root["environment"]["A"] = "mutated"
    assert profile.get("environment") == {"A": "1"}


def test_require_path_raises_when_missing():
    profile = ProjectProfile({})
    with pytest.raises(ConfigError):
        profile.require_path("ai", "agents_path")


def test_require_path_returns_value_when_present():
    profile = ProjectProfile({"ai": {"agents_path": "agents"}})
    assert profile.require_path("ai", "agents_path") == "agents"


def test_to_dict_is_a_deep_copy():
    profile = ProjectProfile({"environment": {"A": "1"}})
    d = profile.to_dict()
    d["environment"]["A"] = "mutated"
    assert profile.get("environment") == {"A": "1"}


# -- global_path() / local_path() ----------------------------------------------

def test_local_path_uses_cwd_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ProjectProfile.local_path() == str(tmp_path / ".linktools.json")


def test_local_path_uses_given_root(tmp_path):
    assert ProjectProfile.local_path(tmp_path) == str(tmp_path / ".linktools.json")


def test_global_path_is_home_dot_linktools(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert ProjectProfile.global_path() == str(tmp_path / ".linktools" / "linktools.json")


# -- ProjectProfile._merge() ---------------------------------------------------

def test_merge_overlay_scalar_overrides_base():
    merged = ProjectProfile._merge({"environment": {"KEY": "base"}}, {"environment": {"KEY": "overlay"}})
    assert merged["environment"]["KEY"] == "overlay"


def test_merge_uncovered_key_inherited():
    merged = ProjectProfile._merge({"environment": {"A": "1", "B": "2"}}, {"environment": {"A": "override"}})
    assert merged["environment"] == {"A": "override", "B": "2"}


def test_merge_nested_object_recursive():
    merged = ProjectProfile._merge(
        {"ai": {"agents_path": "base-agents", "skills_path": "base-skills"}},
        {"ai": {"skills_path": "overlay-skills"}},
    )
    assert merged["ai"] == {"agents_path": "base-agents", "skills_path": "overlay-skills"}


def test_merge_array_replaced_not_concatenated():
    merged = ProjectProfile._merge({"environment": {"MIRRORS": ["a", "b"]}}, {"environment": {"MIRRORS": ["c"]}})
    assert merged["environment"]["MIRRORS"] == ["c"]


def test_merge_null_is_explicit_override():
    merged = ProjectProfile._merge({"environment": {"A": "1"}}, {"environment": {"A": None}})
    assert merged["environment"]["A"] is None


def test_merge_does_not_mutate_inputs():
    base = {"environment": {"A": "1"}}
    overlay = {"environment": {"A": "2"}}
    ProjectProfile._merge(base, overlay)
    assert base == {"environment": {"A": "1"}}
    assert overlay == {"environment": {"A": "2"}}


def test_merge_empty_overlay():
    merged = ProjectProfile._merge({"environment": {"A": "1"}}, {})
    assert merged == {"environment": {"A": "1"}}


def test_merge_empty_base():
    merged = ProjectProfile._merge({}, {"environment": {"A": "1"}})
    assert merged == {"environment": {"A": "1"}}


# -- requirement isolation ------------------------------------------------------

def test_merged_view_does_not_expose_the_other_paths_requirement(tmp_path):
    """A ``requires`` check that must never be satisfied by a different
    path's declaration (e.g. cntr's per-repo gate must never be satisfied by
    the user's global ``linktools.json``) reads a single path's own
    ``ProjectProfile`` in isolation, never a multi-path merged one."""
    global_path = tmp_path / "global.json"
    local_path = tmp_path / "local.json"
    _write(global_path, {"requires": {"linktools-cntr": ">=99.0"}})
    _write(local_path, {"requires": {"linktools-cntr": ">=0.1,<1.0"}})

    local_only = ProjectProfile(str(local_path))
    global_only = ProjectProfile(str(global_path))

    assert local_only.get("requires")["linktools-cntr"] == ">=0.1,<1.0"
    assert global_only.get("requires")["linktools-cntr"] == ">=99.0"
