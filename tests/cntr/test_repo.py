#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repo trust prompt + repo status.

Interactive `repo add` asks for confirmation (a repo may carry executable Python
container definitions); the global --yes flag (is_no_input()) and non-interactive
runs skip the prompt. There is no repo-add-specific flag for this -- only
--replace, which is independent of it. `repo status` is read-only.
"""
import pytest

from linktools.cntr.commands.repo import RepoCommand
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
    RepoCommand().on_command_add(url=local_repo)
    assert local_repo in fresh_manager.repos.get_all()


def test_repo_add_interactive_canceled_raises(monkeypatch, fresh_manager, local_repo):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(cntr_repo, "is_no_input", lambda: False)
    monkeypatch.setattr(cntr_repo, "confirm", lambda *a, **k: False)
    with pytest.raises(ContainerError):
        RepoCommand().on_command_add(url=local_repo)
    assert fresh_manager.repos.get_all() == {}


def test_repo_add_yes_skips_prompt(monkeypatch, fresh_manager, local_repo):
    """The global --yes flag sets is_no_input() -- `add` must skip the
    prompt through that alone, with no repo-add-specific flag needed."""
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(cntr_repo, "is_no_input", lambda: True)
    asked = []
    monkeypatch.setattr(cntr_repo, "confirm", lambda *a, **k: asked.append(1) or False)
    RepoCommand().on_command_add(url=local_repo)
    assert asked == []  # prompt skipped
    assert local_repo in fresh_manager.repos.get_all()


def test_repo_add_yes_does_not_imply_replace(monkeypatch, fresh_manager, local_repo):
    """Skipping the trust prompt (is_no_input()) must never also allow
    replacing an already-added repository."""
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(cntr_repo, "is_no_input", lambda: True)
    RepoCommand().on_command_add(url=local_repo)
    with pytest.raises(ContainerError, match="already exists"):
        RepoCommand().on_command_add(url=local_repo)


def test_repo_add_replace_does_not_skip_trust_prompt(monkeypatch, fresh_manager, local_repo):
    """--replace only allows replacing an existing repository -- it must
    never also skip the trust confirmation prompt on its own."""
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(cntr_repo, "is_no_input", lambda: False)
    monkeypatch.setattr(cntr_repo, "confirm", lambda *a, **k: False)
    with pytest.raises(ContainerError, match="Canceled"):
        RepoCommand().on_command_add(url=local_repo, replace=True)


def test_repo_add_noninteractive_skips_prompt(monkeypatch, fresh_manager, local_repo):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    asked = []
    monkeypatch.setattr(cntr_repo, "is_no_input", lambda: True)
    monkeypatch.setattr(cntr_repo, "confirm", lambda *a, **k: asked.append(1) or False)
    RepoCommand().on_command_add(url=local_repo)
    assert asked == []  # prompt skipped in non-interactive mode
    assert local_repo in fresh_manager.repos.get_all()


def test_repo_status_is_readonly(monkeypatch, fresh_manager, local_repo):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    fresh_manager.repos.add(local_repo, replace=True)
    before = dict(fresh_manager.repos.get_all())
    RepoCommand().on_command_status()  # must not raise or mutate
    assert fresh_manager.repos.get_all() == before


def test_repo_status_no_repos(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    RepoCommand().on_command_status()  # no repos -> no crash
    assert fresh_manager.repos.get_all() == {}
