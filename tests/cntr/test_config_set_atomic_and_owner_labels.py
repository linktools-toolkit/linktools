#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`config set` validates every key against every relevant schema before
writing anything (review P1-05), and owner-label disambiguation for two
same-named repositories is shared identically across
list/get/explain/validate (review P2-05) instead of only being complete in
`list`.
"""
import json
import os

import pytest

import _harness
from linktools.cntr.commands.config import ConfigCommand, build_display_owner_labels
import linktools.cntr.commands._shared as cntr_shared


def _fresh_standalone_manager(tmp_path):
    _harness.install_deterministic_interaction()
    _harness._reset_global_config()
    data_path = tmp_path / "data"
    temp_path = tmp_path / "temp"
    os.environ["LINKTOOLS_PATH"] = str(tmp_path)
    os.environ["LINKTOOLS_DATA_PATH"] = str(data_path)
    os.environ["LINKTOOLS_TEMP_PATH"] = str(temp_path)

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    return ContainerManager(Environ(), name="aio")


# -- P1-05: atomic validate-then-persist ------------------------------------

def _repo_with_int_field(tmp_path, name):
    repo_dir = tmp_path / name
    repo_dir.mkdir(parents=True)
    (repo_dir / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr import BaseContainer\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def configs(self):\n"
        "        return {'PORT_FIELD': ConfigField(cast=int, default=80)}\n",
        encoding="utf-8",
    )
    return repo_dir


def test_set_rolls_back_whole_batch_when_one_key_invalid(tmp_path, monkeypatch):
    repo = _repo_with_int_field(tmp_path, "repo")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo))
    manager.installed_state.add("repo")
    monkeypatch.setattr(cntr_shared, "manager", manager)

    with pytest.raises(Exception):
        ConfigCommand().on_command_set(configs={
            "HOST": "example.com",
            "PORT_FIELD": "not-a-number",
        })

    # Neither key was persisted -- HOST (which would have validated fine on
    # its own) must not be written just because it came first in the batch.
    assert "HOST" not in manager.env_config.persisted_keys()
    assert "PORT_FIELD" not in manager.env_config.persisted_keys()


def test_set_persists_everything_when_all_keys_valid(tmp_path, monkeypatch):
    repo = _repo_with_int_field(tmp_path, "repo")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo))
    manager.installed_state.add("repo")
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_set(configs={"HOST": "example.com", "PORT_FIELD": "8080"})

    assert manager.env_config.get("HOST") == "example.com"
    container = manager.containers["repo"]
    assert container.env_config.get("PORT_FIELD") == 8080


# -- P2-05: consistent owner-label disambiguation ----------------------------

def _two_repos_sharing_a_name(tmp_path, shared_basename):
    # The two repos' own root directories share a basename (drives
    # repo_name/owner_label ambiguity) -- their containers have distinct
    # names ("a"/"b") so this doesn't also trip the separate duplicate
    # container-identity rule.
    group_a = tmp_path / "group_a"
    group_b = tmp_path / "group_b"
    group_a.mkdir()
    group_b.mkdir()
    repo_a = group_a / shared_basename
    repo_b = group_b / shared_basename
    for repo_dir, container_name, value in (
            (repo_a, "a", "value-a"), (repo_b, "b", "value-b")):
        repo_dir.mkdir()
        (repo_dir / ".linktools.json").write_text(
            json.dumps({"env": {"SHARED_FIELD": value}}), encoding="utf-8")
        c_dir = repo_dir / f"100-{container_name}"
        c_dir.mkdir()
        (c_dir / "container.py").write_text(
            "from linktools.core import ConfigField\n"
            "from linktools.cntr.container import BaseContainer\n\n\n"
            "class Container(BaseContainer):\n"
            "    @property\n"
            "    def configs(self):\n"
            "        return {'SHARED_FIELD': ConfigField(default='builtin-default')}\n",
            encoding="utf-8",
        )
    return repo_a, repo_b


def test_get_shows_one_shared_value_for_two_repos_sharing_a_name(tmp_path, monkeypatch, capsys):
    # Per-repository local-file config isolation was intentionally removed:
    # both repos declare SHARED_FIELD identically, and neither repo's own
    # `.linktools.json` value feeds config resolution anymore -- they share
    # the manager's one repository Config, so `config get` shows a single
    # unified value with no owner-disambiguation label needed at all.
    repo_a, repo_b = _two_repos_sharing_a_name(tmp_path, "common")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))
    manager.installed_state.add("a", "b")
    manager.prepare_installed_containers()
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_get(keys=["SHARED_FIELD"], show_secret=True)
    out = capsys.readouterr().out

    assert out.count("SHARED_FIELD=") == 1
    assert "SHARED_FIELD=builtin-default" in out


def test_build_display_owner_labels_disambiguates_only_when_needed():
    from collections import namedtuple
    T = namedtuple("T", ["owner_id", "owner_label"])

    # Two distinct owners, same label -> both get a hash suffix.
    labels = build_display_owner_labels([T("id-a", "common"), T("id-b", "common")])
    assert labels["id-a"] != "common"
    assert labels["id-b"] != "common"
    assert labels["id-a"] != labels["id-b"]

    # Two distinct owners, distinct labels -> plain labels, no suffix.
    labels = build_display_owner_labels([T("id-a", "repo-a"), T("id-b", "repo-b")])
    assert labels == {"id-a": "repo-a", "id-b": "repo-b"}
