# -*- coding: utf-8 -*-
"""Tests for :class:`linktools.core._environ.Environ` isolation.

Spec (first-batch checklist) and (ENV-001): ``BaseEnviron`` factories
must resolve paths through ``self`` rather than the module-level ``environ``
singleton, otherwise a custom :class:`Environment` instance silently shares
state with the default global.
"""
import os

import linktools.core._environ as env_mod
from linktools.core._environ import Environ


def test_create_tools_resolves_paths_via_self_not_global_environ(monkeypatch):
    env = Environ()

    calls = []
    orig_path, orig_data = env.get_path, env.get_data_path

    def spy_path(*args, **kwargs):
        calls.append(("get_path", args))
        return orig_path(*args, **kwargs)

    def spy_data(*args, **kwargs):
        calls.append(("get_data_path", args))
        return orig_data(*args, **kwargs)

    monkeypatch.setattr(env, "get_path", spy_path)
    monkeypatch.setattr(env, "get_data_path", spy_data)

    def boom(*args, **kwargs):  # pragma: no cover - only hit by the bug
        raise AssertionError("_create_tools must use self, not the global environ singleton")

    monkeypatch.setattr(env_mod.environ, "get_path", boom)
    monkeypatch.setattr(env_mod.environ, "get_data_path", boom)

    # _create_tools appends the tool stub dir to PATH; snapshot so the test
    # does not leak that mutation.
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    # Must not raise (the global spies would otherwise fire).
    env._create_tools()

    assert any(name == "get_path" for name, _ in calls), "self.get_path was not used"
    assert any(name == "get_data_path" for name, _ in calls), "self.get_data_path was not used"
