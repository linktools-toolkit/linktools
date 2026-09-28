#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository URL credentials must never be persisted or displayed (review
P1-07): `repo add` rejects an HTTP/HTTPS URL carrying userinfo outright;
SSH usernames are routing information, not secrets, and remain allowed.
"""
import json

import pytest

from linktools.cntr.container import ContainerError
from linktools.cntr.repo.service import safe_display_url, _reject_credential_url


@pytest.mark.parametrize("url", [
    "https://user:pass@example.com/repo.git",
    "https://token@example.com/repo.git",
    "http://user:pass@example.com/repo.git",
])
def test_add_rejects_credential_bearing_https_url(fresh_manager, url):
    with pytest.raises(ContainerError, match="credentials"):
        fresh_manager.repos.add(url)
    assert fresh_manager.repos.get_all() == {}


def test_add_allows_ssh_username(fresh_manager, monkeypatch):
    """An SSH username (git@host) is routing information, not a secret --
    real auth happens via SSH keys outside the URL entirely."""
    import os

    def fake_clone(url, repo_path, branch=None):
        os.makedirs(repo_path)

    monkeypatch.setattr(fresh_manager.repos.git, "clone", fake_clone)
    monkeypatch.setattr(fresh_manager.repos, "_validate_new_repo_requirement", lambda repo_path: None)

    fresh_manager.repos.add("ssh://git@example.com/repo.git")
    assert "ssh://git@example.com/repo.git" in fresh_manager.repos.get_all()


def test_reject_error_message_does_not_contain_the_token():
    with pytest.raises(ContainerError) as exc_info:
        _reject_credential_url("https://supersecrettoken@example.com/repo.git")
    assert "supersecrettoken" not in str(exc_info.value)


class TestSafeDisplayUrl:
    def test_strips_username_and_password(self):
        assert safe_display_url("https://user:pass@example.com/repo.git") \
            == "https://example.com/repo.git"

    def test_strips_token_as_username(self):
        assert safe_display_url("https://token@example.com/repo.git") \
            == "https://example.com/repo.git"

    def test_preserves_port(self):
        assert safe_display_url("https://user:pass@example.com:8443/repo.git") \
            == "https://example.com:8443/repo.git"

    def test_noop_for_credential_free_url(self):
        assert safe_display_url("https://example.com/repo.git") == "https://example.com/repo.git"

    def test_noop_for_local_path(self):
        assert safe_display_url("/some/local/path") == "/some/local/path"

    def test_preserves_ssh_username(self):
        # Not a secret -- left as-is.
        assert safe_display_url("ssh://git@example.com/repo.git") == "ssh://git@example.com/repo.git"


# -- legacy (pre-fix) credential URLs already persisted ----------------------

def _inject_legacy_credential_repo(manager, url="https://legacytoken@example.com/repo.git"):
    manager.repos._dump({url: dict(type="git", repo_path=str(manager.data_path), repo_name="repo")})
    return url


def test_describe_fails_closed_for_legacy_credential_url(fresh_manager):
    url = _inject_legacy_credential_repo(fresh_manager)
    meta = fresh_manager.repos.get_all()[url]

    info = fresh_manager.repos.describe(url, meta)

    assert info["available"] is False
    assert "repository_error" in info
    assert "legacytoken" not in json.dumps(info)
    assert info["url"] == "https://example.com/repo.git"


def test_update_fails_closed_for_legacy_credential_url_without_syncing(fresh_manager, monkeypatch):
    url = _inject_legacy_credential_repo(fresh_manager)

    def fail_if_called(*a, **k):
        raise AssertionError("must not sync a repo with an embedded credential")

    monkeypatch.setattr(fresh_manager.repos.git, "update", fail_if_called)

    results = fresh_manager.repos.update()

    assert len(results) == 1
    assert results[0].updated is False
    assert "legacytoken" not in json.dumps([results[0].url, results[0].error])


def test_repo_list_output_does_not_contain_legacy_token(fresh_manager, monkeypatch, capsys):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.repo import RepoCommand

    _inject_legacy_credential_repo(fresh_manager)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    RepoCommand().on_command_status()
    out = capsys.readouterr().out
    assert "legacytoken" not in out


def test_doctor_check_repos_does_not_contain_legacy_token(fresh_manager):
    """review P1-07: Doctor.check_repos() findings (text and --json) are
    user-visible output too -- must never echo a legacy credential URL,
    same as repo list/status/validate."""
    import json as json_module
    from linktools.cntr.doctor import Doctor

    url = _inject_legacy_credential_repo(fresh_manager)
    doctor = Doctor(fresh_manager)

    findings = doctor.check_repos()

    for finding in findings:
        assert "legacytoken" not in finding.message
        assert "legacytoken" not in (finding.component or "")
    assert "legacytoken" not in json_module.dumps([f.__dict__ for f in findings], default=str)


def test_loader_warning_logs_do_not_contain_legacy_token(tmp_path, monkeypatch):
    """review P1-07: ContainerLoader's debug/warning logs about a
    repository must never echo a legacy credential URL."""
    import linktools.cntr.registry.loader as loader_module
    from _harness import install_deterministic_interaction, _reset_global_config

    install_deterministic_interaction()
    _reset_global_config()
    monkeypatch.setenv("LINKTOOLS_PATH", str(tmp_path))
    monkeypatch.setenv("LINKTOOLS_DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("LINKTOOLS_TEMP_PATH", str(tmp_path / "temp"))
    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    manager = ContainerManager(Environ(), name="aio")
    url = "https://legacytoken@example.com/repo.git"
    # Points at a nonexistent repo_path so the loader hits its
    # "not found, skip" warning branch.
    manager.repos._dump({url: dict(type="git", repo_path=str(tmp_path / "missing"), repo_name="repo")})

    messages = []
    monkeypatch.setattr(manager.logger, "warning", lambda msg: messages.append(msg))
    monkeypatch.setattr(manager.logger, "debug", lambda msg: messages.append(msg))

    loader_module.ContainerLoader(manager).load_all()

    assert messages  # sanity: the warning branch actually fired
    for message in messages:
        assert "legacytoken" not in message


def test_repo_validate_json_output_does_not_contain_legacy_token(fresh_manager, monkeypatch, capsys):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.repo import RepoCommand

    _inject_legacy_credential_repo(fresh_manager)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    with pytest.raises(ContainerError):
        RepoCommand().on_command_validate(as_json=True)
    out = capsys.readouterr().out
    assert "legacytoken" not in out
