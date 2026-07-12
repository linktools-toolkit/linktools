#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A ConfigField a third-party repository declares for itself -- secret=True,
a custom cast/validator -- lives ONLY in that repository's own Config
schema, never in Manager Config's. `config set/get/explain/validate/list`
must discover it there, not assume "absent from Manager's schema" means
"not a secret" / "no repository-specific cast or validator applies".

Regression: `on_command_set`/`on_command_get`/`on_command_explain`/
`on_command_validate` only ever consulted `_shared.manager.env_config`
(Manager Config) -- a repository-only secret field with an innocuous
persisted value sailed straight through as plain text, and `validate` never
applied a repository's own cast/validator at all.
"""
import logging

import pytest

import _harness

_SECRET_VALUE = "credential-value-739102"


def _fresh_standalone_manager(tmp_path):
    import os
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


def _repo_with_secret_field(tmp_path, name="repo_secret"):
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    (repo_dir / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def configs(self):\n"
        "        return {'CREDENTIAL': ConfigField(secret=True)}\n",
        encoding="utf-8",
    )
    return repo_dir


def _install_repo_with_secret(tmp_path, name="repo_secret"):
    repo_dir = _repo_with_secret_field(tmp_path, name)
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_dir))
    manager.installed_state.add(name)
    manager.prepare_installed_containers()
    return manager


def _repo_with_port_field(tmp_path, name, field_expr):
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    (repo_dir / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def configs(self):\n"
        "        return {'PORT': %s}\n" % field_expr,
        encoding="utf-8",
    )
    return repo_dir


def test_manager_schema_does_not_know_about_repo_field(tmp_path):
    manager = _install_repo_with_secret(tmp_path)
    assert manager.env_config.schema.get("CREDENTIAL") is None
    container = manager.containers["repo_secret"]
    assert container.env_config.schema.get("CREDENTIAL").secret is True


def test_set_redacts_repo_only_secret(monkeypatch, tmp_path, caplog):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_repo_with_secret(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    with caplog.at_level(logging.INFO):
        ConfigCommand().on_command_set(configs={"CREDENTIAL": _SECRET_VALUE})

    messages = "\n".join(caplog.messages)
    assert _SECRET_VALUE not in messages
    assert "CREDENTIAL: ***" in messages


def test_get_default_redacts_repo_only_secret(monkeypatch, tmp_path, caplog):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_repo_with_secret(tmp_path)
    manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    with caplog.at_level(logging.INFO):
        ConfigCommand().on_command_get(keys=["CREDENTIAL"], show_secret=False)

    messages = "\n".join(caplog.messages)
    assert _SECRET_VALUE not in messages
    assert "CREDENTIAL=***" in messages


def test_get_show_secret_reveals_repo_only_value(monkeypatch, tmp_path, capsys):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_repo_with_secret(tmp_path)
    manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_get(keys=["CREDENTIAL"], show_secret=True)
    out = capsys.readouterr().out
    assert f"CREDENTIAL={_SECRET_VALUE}" in out


def test_explain_json_does_not_leak_repo_only_secret(monkeypatch, tmp_path, capsys):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_repo_with_secret(tmp_path)
    manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_explain(key="CREDENTIAL", as_json=True)
    out = capsys.readouterr().out
    assert _SECRET_VALUE not in out


def test_explain_text_does_not_leak_repo_only_secret(monkeypatch, tmp_path, caplog):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_repo_with_secret(tmp_path)
    manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    with caplog.at_level(logging.INFO):
        ConfigCommand().on_command_explain(key="CREDENTIAL", as_json=False)

    messages = "\n".join(caplog.messages)
    assert _SECRET_VALUE not in messages


def test_list_redacts_repo_only_secret(monkeypatch, tmp_path, caplog):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_repo_with_secret(tmp_path)
    manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    with caplog.at_level(logging.INFO):
        ConfigCommand().on_command_list(names=[], show_secret=False)

    messages = "\n".join(caplog.messages)
    assert _SECRET_VALUE not in messages
    assert "CREDENTIAL=***" in messages


def test_validate_uses_repo_field_secret_and_never_prints_it(monkeypatch, tmp_path, capsys):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_repo_with_secret(tmp_path)
    manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_validate(as_json=True)
    out = capsys.readouterr().out
    assert _SECRET_VALUE not in out


def test_validate_reports_each_repo_independently(monkeypatch, tmp_path, caplog):
    """Two repositories declaring the SAME key with DIFFERENT cast/validator
    rules: validate() must apply each repository's own rule independently
    and name the failing owner, never stop at the first repo checked."""
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.container import ContainerError
    from linktools.cntr.commands.config import ConfigCommand

    repo_a = _repo_with_port_field(tmp_path, "repo_a", "ConfigField(cast=int)")
    repo_b = _repo_with_port_field(
        tmp_path, "repo_b",
        "ConfigField(validator=lambda value: value.startswith('tcp-'))",
    )

    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))
    manager.installed_state.add("repo_a", "repo_b")
    manager.prepare_installed_containers()
    manager.env_config.persist("PORT", "8080")
    monkeypatch.setattr(cntr_shared, "manager", manager)

    with caplog.at_level(logging.INFO):
        with pytest.raises(ContainerError):
            ConfigCommand().on_command_validate(as_json=False)

    invalid_lines = [m for m in caplog.messages if "[INVALID]" in m]
    assert len(invalid_lines) == 1
    assert "repo_b" in invalid_lines[0]


def _install_two_repos_disagreeing_on_secret(tmp_path):
    """Two repos share the SAME persisted key; only one flags it secret --
    the shared value must be redacted everywhere regardless of which
    repo's row/target is being rendered."""
    repo_a = _repo_with_port_field(tmp_path, "repo_a", "ConfigField(secret=True)")
    repo_b = _repo_with_port_field(tmp_path, "repo_b", "ConfigField(secret=False)")

    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))
    manager.installed_state.add("repo_a", "repo_b")
    manager.prepare_installed_containers()
    manager.env_config.persist("PORT", _SECRET_VALUE)
    return manager


def test_explain_redacts_across_targets_even_when_this_targets_own_field_is_not_secret(monkeypatch, tmp_path, capsys):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_two_repos_disagreeing_on_secret(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_explain(key="PORT", as_json=True)
    out = capsys.readouterr().out
    assert _SECRET_VALUE not in out


def test_list_redacts_across_targets_even_when_this_entrys_own_field_is_not_secret(monkeypatch, tmp_path, caplog):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_two_repos_disagreeing_on_secret(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    with caplog.at_level(logging.INFO):
        ConfigCommand().on_command_list(names=[], show_secret=False)

    messages = "\n".join(caplog.messages)
    assert _SECRET_VALUE not in messages
    # Both repos' rows for the shared key must be masked -- not just repo_a's.
    assert messages.count("***") >= 2


def _plain_repo(tmp_path, name):
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    (repo_dir / "container.py").write_text(
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n"
        "    pass\n",
        encoding="utf-8",
    )
    return repo_dir


def _install_two_plain_repos(tmp_path):
    repo_a = _plain_repo(tmp_path, "repo_a")
    repo_b = _plain_repo(tmp_path, "repo_b")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))
    manager.installed_state.add("repo_a", "repo_b")
    manager.prepare_installed_containers()
    return manager


def test_get_stays_single_target_for_plain_manager_key_with_multiple_repos(monkeypatch, tmp_path, capsys):
    """A repository's Config schema is seeded with a COPY of every Manager
    field (by reference) so its containers can still resolve manager-owned
    keys like HOST -- that must never be mistaken for the repository
    declaring HOST itself. With two plain (no custom config) repos
    installed, an ordinary manager key must still resolve as a single
    target, not fan out into one row per installed repo."""
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_two_plain_repos(tmp_path)
    manager.env_config.persist("HOST", "10.0.0.5")
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_get(keys=["HOST"], show_secret=True)
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["HOST=10.0.0.5"]


def test_explain_stays_single_target_for_plain_manager_key_with_multiple_repos(monkeypatch, tmp_path, capsys):
    import json

    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_two_plain_repos(tmp_path)
    manager.env_config.persist("HOST", "10.0.0.5")
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_explain(key="HOST", as_json=True)
    info = json.loads(capsys.readouterr().out)
    assert "targets" not in info
    assert info["resolved_value"] == "10.0.0.5"


def test_set_get_explain_validate_work_before_any_container_is_installed(monkeypatch, tmp_path, capsys):
    """`prepare_installed_containers()` raises when nothing is installed
    yet -- set/get/explain/validate must still work against Manager Config
    alone (e.g. setting HOST as one of the very first commands run on a
    fresh install, before any repository/container has ever been added)."""
    import json

    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _fresh_standalone_manager(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_set(configs={"HOST": "10.0.0.5"})
    ConfigCommand().on_command_get(keys=["HOST"], show_secret=True)
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["HOST=10.0.0.5"]

    ConfigCommand().on_command_explain(key="HOST", as_json=True)
    info = json.loads(capsys.readouterr().out)
    assert info["resolved_value"] == "10.0.0.5"

    ConfigCommand().on_command_validate(as_json=True)
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
