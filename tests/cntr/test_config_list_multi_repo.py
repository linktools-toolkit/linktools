#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repositories share config values while container selection limits listed keys."""
import json

import _harness


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


def _repo_with_port(tmp_path, name, port):
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    (repo_dir / ".linktools.json").write_text(
        json.dumps({"env": {"PORT": port}}), encoding="utf-8")
    (repo_dir / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def configs(self):\n"
        "        return {'PORT': ConfigField(cast=int, default=0)}\n",
        encoding="utf-8",
    )
    return repo_dir


def _install_two_repos_with_ports(tmp_path):
    repo_a = _repo_with_port(tmp_path, "repo_a", "8001")
    repo_b = _repo_with_port(tmp_path, "repo_b", "8002")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))
    manager.installed_state.add("repo_a", "repo_b")
    manager.prepare_installed_containers()
    return manager


def test_different_repos_declaring_same_field_share_one_value(monkeypatch, tmp_path, capsys):
    # repo_a/repo_b each declare PORT (cast=int, default=0) identically, and
    # each repo's own `.linktools.json` PORT value is no longer consulted
    # for config resolution -- both containers share the manager's one
    # repository Config, so exactly one PORT entry is listed.
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_two_repos_with_ports(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_list(names=[], show_secret=True)
    out = capsys.readouterr().out

    assert out.count("PORT=") == 1
    assert "PORT=0" in out


def test_same_repo_shared_config_key_shown_once(monkeypatch, tmp_path, capsys):
    """Two containers in the SAME repository share one Config -- the same
    key from both must collapse to a single listed entry, not one per
    container."""
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    repo_dir = tmp_path / "repo_shared"
    repo_dir.mkdir()
    (repo_dir / "1-alpha").mkdir()
    (repo_dir / "1-alpha" / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def configs(self):\n"
        "        return {'SHARED_KEY': ConfigField(default='v')}\n",
        encoding="utf-8",
    )
    (repo_dir / "2-beta").mkdir()
    (repo_dir / "2-beta" / "container.py").write_text(
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n"
        "    pass\n",
        encoding="utf-8",
    )

    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_dir))
    manager.installed_state.add("alpha", "beta")
    manager.prepare_installed_containers()
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_list(names=[], show_secret=True)
    out = capsys.readouterr().out

    assert out.count("SHARED_KEY=v") == 1


def test_manager_persistent_extra_shown_once_and_unambiguous(monkeypatch, tmp_path, capsys):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_two_repos_with_ports(tmp_path)
    manager.env_config.persist("DOCKER_DOWNLOAD_PATH", "/srv/downloads")
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_list(names=[], show_secret=True)
    out = capsys.readouterr().out

    assert out.count("DOCKER_DOWNLOAD_PATH=/srv/downloads") == 1
    # Manager-owned, not ambiguous with anything else -- no owner prefix.
    assert "manager:DOCKER_DOWNLOAD_PATH" not in out


def test_same_repo_name_different_repos_share_one_value(monkeypatch, tmp_path, capsys):
    """Two DIFFERENT repositories that happen to share a repo_name (both
    named "common", e.g. cloned from team-a/common.git and
    team-b/common.git) declare the identical PORT field -- since repos no
    longer have their own isolated Config, this is just one shared entry,
    not two values needing an owner-disambiguation label."""
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    team_a = tmp_path / "team-a"
    team_a.mkdir()
    team_b = tmp_path / "team-b"
    team_b.mkdir()
    repo_a = _repo_with_port(team_a, "common", "8001")
    repo_b = _repo_with_port(team_b, "common", "8002")

    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))
    manager.installed_state.add("common", "common_0")
    manager.prepare_installed_containers()
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_list(names=[], show_secret=True)
    out = capsys.readouterr().out

    lines = [line for line in out.splitlines() if "PORT=" in line]
    assert len(lines) == 1
    assert str(tmp_path) not in lines[0]  # never leak an absolute path


def test_filtered_list_includes_only_selected_declarations_and_optional_dependencies(monkeypatch, tmp_path, capsys):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    repo = tmp_path / "repo"
    for name, dependencies, key in (
            ("app", ["dependency"], "APP_VALUE"),
            ("dependency", [], "DEPENDENCY_VALUE"),
            ("other", [], "OTHER_VALUE")):
        directory = repo / name
        directory.mkdir(parents=True)
        (directory / "container.py").write_text(
            "from linktools.core import ConfigField\n"
            "from linktools.cntr import BaseContainer\n"
            "class Container(BaseContainer):\n"
            "    @property\n"
            "    def dependencies(self):\n"
            "        return %r\n"
            "    @property\n"
            "    def configs(self):\n"
            "        return {%r: ConfigField(default='value')}\n"
            "    @property\n"
            "    def extend_configs(self):\n"
            "        return {'EXTENDED': ConfigField(default='extended'),\n"
            "                'OPTIONAL': ConfigField(secret=True)}\n"
            % (dependencies, key), encoding="utf-8")

    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo))
    manager.installed_state.add("app", "dependency", "other")
    manager.env_config.persist_many({
        "APP_VALUE": "persisted", "EXTENDED": "extended", "UNDECLARED": "extra"})
    monkeypatch.setattr(cntr_shared, "manager", manager)

    command = ConfigCommand()
    command.on_command_list(names=["app"], show_secret=True)
    assert capsys.readouterr().out.splitlines() == ["APP_VALUE=persisted", "EXTENDED=extended"]

    command.on_command_list(names=["app"], with_dependencies=True, show_secret=True)
    assert capsys.readouterr().out.splitlines() == [
        "APP_VALUE=persisted", "DEPENDENCY_VALUE=value", "EXTENDED=extended"]
