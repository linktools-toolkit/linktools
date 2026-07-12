# -*- coding: utf-8 -*-
"""`ct-cntr config list` must never print the same config key twice.

Regression: a repo-backed (non-builtin) container's own `env_config` is a
different Config *object* from the shared `manager.env_config` (only the
Environment/RuntimeOverride/Persistent sources are actually shared across
them -- the local-file layer differs per repo). `on_command_list` used to
dedup by `(id(config), key)`, so a key declared by that container's own
`configs` (added via its own env_config) was never recognized as the same
key later re-discovered through `manager.env_config.persisted_keys()`
(which enumerates the whole shared Persistent namespace, not just
manager-declared fields) -- printing it twice.
"""
import os

import _harness

from linktools.cntr.commands.config import ConfigCommand
import linktools.cntr.commands._shared as cntr_shared


def _install_repo_with_secret_field(tmp_path):
    _harness.install_deterministic_interaction()
    _harness._reset_global_config()
    data_path = tmp_path / "data"
    temp_path = tmp_path / "temp"
    os.environ["LINKTOOLS_PATH"] = str(tmp_path)
    os.environ["LINKTOOLS_DATA_PATH"] = str(data_path)
    os.environ["LINKTOOLS_TEMP_PATH"] = str(temp_path)

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    manager = ContainerManager(Environ(), name="aio")

    repo = tmp_path / "repo"
    c_dir = repo / "100-aionui"
    c_dir.mkdir(parents=True)
    (c_dir / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr import BaseContainer\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def configs(self):\n"
        "        return dict(AIONUI_JWT_SECRET=ConfigField(secret=True, default='generated-secret'))\n"
    )
    manager.repos.add(str(repo))
    manager.installed_state.add("aionui")
    return manager


def test_repo_field_persisted_key_is_not_listed_twice(monkeypatch, tmp_path, capsys):
    manager = _install_repo_with_secret_field(tmp_path)
    container = manager.containers["aionui"]
    # Register the field, then persist it (as a cached secret provider would
    # on first resolution) -- this is what makes it show up via the shared
    # manager.env_config.persisted_keys() sweep as well as the container's
    # own configs.
    container.env_config.update_defaults(**container.configs)
    container.env_config.persist("AIONUI_JWT_SECRET", "s3cr3t")

    monkeypatch.setattr(cntr_shared, "manager", manager)
    ConfigCommand().on_command_list(names=[], show_secret=True)

    out = capsys.readouterr().out
    assert out.count("AIONUI_JWT_SECRET=") == 1
