#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root command diagnosability (review P2-08):

- `ct-cntr list` must work on a fresh install with nothing installed yet
  (previously routed through `prepare_installed_containers()`, which raises
  "No container installed").
- `add`/`remove` raise a real ContainerError instead of a bare `assert`,
  which Python's `-O` flag strips entirely, silently turning a "nothing
  happened" outcome into no error at all.
"""
import os

import pytest

import _harness
from linktools.cntr.container import ContainerError
from linktools.cntr.commands.root import Command as RootCommand
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


def test_root_list_works_on_fresh_install_with_nothing_installed(tmp_path, monkeypatch):
    manager = _fresh_standalone_manager(tmp_path)
    assert manager.installed_state.get(resolve=False) == []
    monkeypatch.setattr(cntr_shared, "manager", manager)

    # Must not raise "No container installed".
    RootCommand().on_command_list(names=[])


def test_add_raises_container_error_not_bare_assert(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    with pytest.raises(ContainerError, match="No container added"):
        RootCommand().on_command_add(names=["does-not-exist"])


def test_remove_raises_container_error_not_bare_assert(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    with pytest.raises(ContainerError, match="No container removed"):
        RootCommand().on_command_remove(names=["does-not-exist"])
