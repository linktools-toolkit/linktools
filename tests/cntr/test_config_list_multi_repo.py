#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`ct-cntr config list` across multiple repositories with a colliding key
name.

Regression: `on_command_list` used to dedup purely by key name (a plain
``set()``), so when two different repositories each declared their own
``PORT`` field with a different value, only the first-seen repo's PORT ever
made it into the listing -- the second repo's real, different value for the
same key was silently dropped. Dedup must be per (Config identity, key), and
a genuinely ambiguous key (same name, different owners) must show which
owner each value belongs to.
"""
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
        json.dumps({"environment": {"PORT": port}}), encoding="utf-8")
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


def test_different_repos_same_key_both_shown(monkeypatch, tmp_path, capsys):
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.commands.config import ConfigCommand

    manager = _install_two_repos_with_ports(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    ConfigCommand().on_command_list(names=[], show_secret=True)
    out = capsys.readouterr().out

    assert "repo_a:PORT=8001" in out
    assert "repo_b:PORT=8002" in out


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
