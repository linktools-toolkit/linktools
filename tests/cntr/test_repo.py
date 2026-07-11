#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repo trust prompt + repo status.

Interactive `repo add` asks for confirmation (a repo may carry executable Python
container definitions); --force and non-interactive runs skip the prompt.
`repo status` is read-only.
"""
import pytest

import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
import linktools.cntr.commands.repo as cntr_repo
from linktools.cntr.container import ContainerError


@pytest.fixture
def local_repo(tmp_path):
    repo = tmp_path / "myrepo"
    repo.mkdir()
    (repo / "container.py").write_text("# placeholder\n")
    return str(repo)


def test_repo_add_interactive_confirmed_adds(monkeypatch, fresh_manager, local_repo):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(cntr_repo, "is_no_input", lambda: False)
    monkeypatch.setattr(cntr_repo, "confirm", lambda *a, **k: True)
    cntr_main.RepoCommand().on_command_add(url=local_repo, force=False)
    assert local_repo in fresh_manager.repo_store.get_all()


def test_repo_add_interactive_canceled_raises(monkeypatch, fresh_manager, local_repo):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(cntr_repo, "is_no_input", lambda: False)
    monkeypatch.setattr(cntr_repo, "confirm", lambda *a, **k: False)
    with pytest.raises(ContainerError):
        cntr_main.RepoCommand().on_command_add(url=local_repo, force=False)
    assert fresh_manager.repo_store.get_all() == {}


def test_repo_add_force_skips_prompt(monkeypatch, fresh_manager, local_repo):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    asked = []
    monkeypatch.setattr(cntr_repo, "confirm", lambda *a, **k: asked.append(1) or False)
    cntr_main.RepoCommand().on_command_add(url=local_repo, force=True)
    assert asked == []  # prompt skipped
    assert local_repo in fresh_manager.repo_store.get_all()


def test_repo_add_noninteractive_skips_prompt(monkeypatch, fresh_manager, local_repo):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    asked = []
    monkeypatch.setattr(cntr_repo, "is_no_input", lambda: True)
    monkeypatch.setattr(cntr_repo, "confirm", lambda *a, **k: asked.append(1) or False)
    cntr_main.RepoCommand().on_command_add(url=local_repo, force=False)
    assert asked == []  # prompt skipped in non-interactive mode
    assert local_repo in fresh_manager.repo_store.get_all()


def test_repo_status_is_readonly(monkeypatch, fresh_manager, local_repo):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    fresh_manager.repo_store.add(local_repo, force=True)
    before = dict(fresh_manager.repo_store.get_all())
    cntr_main.RepoCommand().on_command_status()  # must not raise or mutate
    assert fresh_manager.repo_store.get_all() == before


def test_repo_status_no_repos(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    cntr_main.RepoCommand().on_command_status()  # no repos -> no crash
    assert fresh_manager.repo_store.get_all() == {}
