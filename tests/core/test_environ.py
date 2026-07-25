# -*- coding: utf-8 -*-
"""Tests for :class:`linktools.core._environ.Environ` isolation."""

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

    # Must not raise (the global spies would otherwise fire).
    env._create_tools()

    assert not any(name == "get_path" for name, _ in calls)
    assert any(name == "get_data_path" for name, _ in calls)
