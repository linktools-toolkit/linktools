#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ContainerManager stability fixes (spec §6.1, §6.2).

- __init__ must assign self.environ before using it for the default name,
  so ContainerManager(environ, name=None) does not AttributeError.
- change_file_mode() must probe `chmod`, not `chown`.
"""
import os
import shutil
import tempfile
from pathlib import Path

from linktools.cntr.manager import ContainerManager


class _FakeLogger:
    def __getattr__(self, _name):
        return lambda *a, **kw: None


class _FakeEnvConfig:
    def update_defaults(self, **kwargs):
        pass


class _FakeEnviron:
    name = "test-env"

    def get_logger(self, _name):
        return _FakeLogger()

    def wrap_config(self, namespace=None, env_prefix=None):
        return _FakeEnvConfig()

    def get_data_path(self, *parts):
        # configs property builds str(self.data_path.joinpath(...)) defaults.
        return Path(tempfile.gettempdir())


def test_default_name_uses_environ_name_when_none():
    mgr = ContainerManager(environ=_FakeEnviron(), name=None)
    assert mgr.name == "test-env"


def test_default_name_uses_environ_name_when_empty():
    mgr = ContainerManager(environ=_FakeEnviron(), name="")
    assert mgr.name == "test-env"


def test_explicit_name_wins():
    mgr = ContainerManager(environ=_FakeEnviron(), name="custom")
    assert mgr.name == "custom"


def test_change_file_mode_probes_chmod_not_chown(monkeypatch):
    calls = []

    def fake_which(cmd):
        calls.append(cmd)
        return None  # force early return so only the probe is observed

    monkeypatch.setattr(shutil, "which", fake_which)

    mgr = ContainerManager(environ=_FakeEnviron(), name="x")
    mgr.system = "linux"  # _is_chown_supported is linux-only

    with tempfile.NamedTemporaryFile() as fp:
        mgr.change_file_mode(fp.name, mode=0o755)

    assert "chmod" in calls
    assert "chown" not in calls
