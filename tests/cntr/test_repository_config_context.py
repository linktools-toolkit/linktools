#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manager-owned vs repository-owned config keys.

Regression: `build_repository_config()` used to copy the manager's own
ConfigFields onto a fresh per-repository schema, but shared the manager's
Environment/RuntimeOverride/Persistent sources verbatim and then let the
repository's own local `.linktools.json` sit ahead of the field's provider
in the source chain -- so a third-party repo could locally override a
manager-owned key (e.g. DOCKER_APP_PATH), and `container.get_app_path()`
(which always reads `manager.app_path`) would disagree with whatever
`container.get_config("DOCKER_APP_PATH")` returned for that repo.
`ManagerConfigSource` closes that: a manager-owned key always resolves
through the manager's own Config, no matter which repository asks.
"""
import os

import _harness


def _fresh_standalone_manager(tmp_path):
    # Not the `fresh_manager` fixture -- it already memoizes `.containers`
    # over just the builtins before a test gets a chance to add repos.
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


def _repo_with_container(tmp_path, name, environment=None, extra_config_field=None):
    import json

    repo_dir = tmp_path / name
    repo_dir.mkdir()
    if environment:
        (repo_dir / ".linktools.json").write_text(
            json.dumps({"environment": environment}), encoding="utf-8")

    configs_body = ""
    if extra_config_field:
        configs_body = (
            "    @property\n"
            "    def configs(self):\n"
            f"        return {{'{extra_config_field}': ConfigField(cast='path', default='./x')}}\n"
        )
    (repo_dir / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n"
        + (configs_body or "    pass\n"),
        encoding="utf-8",
    )
    return repo_dir


def test_repo_local_cannot_override_manager_key(tmp_path):
    manager = _fresh_standalone_manager(tmp_path)
    manager.env_config.persist("DOCKER_APP_PATH", str(tmp_path / "manager-app"))

    repo_dir = _repo_with_container(tmp_path, "repo_a", environment={"DOCKER_APP_PATH": "./repo-app"})
    manager.repos.add(str(repo_dir))
    manager.installed_state.add("repo_a")
    manager.prepare_installed_containers()

    container = manager.containers["repo_a"]
    assert container.get_config("DOCKER_APP_PATH") == str(tmp_path / "manager-app")
    assert str(container.get_app_path()).startswith(str(tmp_path / "manager-app"))


def test_repo_local_custom_key_is_scoped_per_repo(tmp_path):
    manager = _fresh_standalone_manager(tmp_path)

    repo_a = _repo_with_container(tmp_path, "repo_a", environment={"CUSTOM_PATH": "./a"},
                                   extra_config_field="CUSTOM_PATH")
    repo_b = _repo_with_container(tmp_path, "repo_b", environment={"CUSTOM_PATH": "./b"},
                                   extra_config_field="CUSTOM_PATH")
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))
    manager.installed_state.add("repo_a", "repo_b")
    manager.prepare_installed_containers()

    container_a = manager.containers["repo_a"]
    container_b = manager.containers["repo_b"]
    assert container_a.get_config("CUSTOM_PATH") != container_b.get_config("CUSTOM_PATH")
    assert str(container_a.get_config("CUSTOM_PATH")).endswith(os.sep + "a")
    assert str(container_b.get_config("CUSTOM_PATH")).endswith(os.sep + "b")


def test_runtime_override_still_wins_over_manager_and_repo(tmp_path):
    manager = _fresh_standalone_manager(tmp_path)
    repo_dir = _repo_with_container(tmp_path, "repo_a", environment={"DOCKER_APP_PATH": "./repo-app"})
    manager.repos.add(str(repo_dir))
    manager.installed_state.add("repo_a")
    manager.prepare_installed_containers()

    container = manager.containers["repo_a"]
    manager.env_config.set("DOCKER_APP_PATH", str(tmp_path / "runtime-app"))
    assert container.get_config("DOCKER_APP_PATH") == str(tmp_path / "runtime-app")


def test_persisted_value_still_wins_over_manager_default_and_repo(tmp_path):
    manager = _fresh_standalone_manager(tmp_path)
    repo_dir = _repo_with_container(tmp_path, "repo_a", environment={"DOCKER_APP_PATH": "./repo-app"})
    manager.repos.add(str(repo_dir))
    manager.installed_state.add("repo_a")
    manager.prepare_installed_containers()

    container = manager.containers["repo_a"]
    manager.env_config.persist("DOCKER_APP_PATH", str(tmp_path / "persisted-app"))
    assert container.get_config("DOCKER_APP_PATH") == str(tmp_path / "persisted-app")


def test_repo_declaring_manager_key_is_reported_as_ignored(tmp_path):
    manager = _fresh_standalone_manager(tmp_path)
    repo_dir = _repo_with_container(
        tmp_path, "repo_a",
        environment={"DOCKER_APP_PATH": "./repo-app", "COMPOSE_PROJECT_NAME": "repo-project"},
    )
    manager.repos.add(str(repo_dir))

    url = str(repo_dir)
    meta = manager.repos.get_all()[url]
    info = manager.repos.describe(url, meta)

    assert info["ignored_environment_keys"] == ["COMPOSE_PROJECT_NAME", "DOCKER_APP_PATH"]


def test_repo_without_reserved_keys_reports_empty_list(tmp_path):
    manager = _fresh_standalone_manager(tmp_path)
    repo_dir = _repo_with_container(tmp_path, "repo_a", environment={"CUSTOM_ONLY": "value"})
    manager.repos.add(str(repo_dir))

    url = str(repo_dir)
    meta = manager.repos.get_all()[url]
    info = manager.repos.describe(url, meta)

    assert info["ignored_environment_keys"] == []


def test_manager_config_source_reflects_already_cached_manager_write(tmp_path):
    """ManagerConfigSource.revision delegates to the manager's own Config
    revision, so a repository's Resolver notices a manager-side write even
    if it already cached the manager-owned key's old value."""
    manager = _fresh_standalone_manager(tmp_path)
    repo_dir = _repo_with_container(tmp_path, "repo_a")
    manager.repos.add(str(repo_dir))
    manager.installed_state.add("repo_a")
    manager.prepare_installed_containers()

    container = manager.containers["repo_a"]
    manager.env_config.persist("DOCKER_APP_PATH", str(tmp_path / "first-app"))
    assert container.get_config("DOCKER_APP_PATH") == str(tmp_path / "first-app")  # cache it

    manager.env_config.persist("DOCKER_APP_PATH", str(tmp_path / "second-app"))
    assert container.get_config("DOCKER_APP_PATH") == str(tmp_path / "second-app")

    manager.env_config.set("DOCKER_APP_PATH", str(tmp_path / "runtime-app"))
    assert container.get_config("DOCKER_APP_PATH") == str(tmp_path / "runtime-app")
