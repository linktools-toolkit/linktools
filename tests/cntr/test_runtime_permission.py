#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chown/chmod must resolve their `path` argument identically: expanduser +
abspath via env_config.cast(type="path"), regardless of whether the caller
passed a Path object, a plain string, a relative path, or a `~/...` path.

Regression: chmod() checked os.path.exists(path) on the raw argument while
chown() cast it first, so `manager.change_file_mode("~/mytarget", ...)` raised
FileNotFoundError on the literal "~/mytarget" instead of resolving it like
change_file_owner() does.
"""
import os

import pytest


class _FakeProcess:
    def check_call(self):
        return 0


def _record(monkeypatch, manager):
    calls = []
    monkeypatch.setattr(manager, "create_process", lambda *a, **k: (calls.append(a), _FakeProcess())[1])
    return calls


@pytest.fixture(autouse=True)
def _linux(monkeypatch, fresh_manager):
    # chown/chmod are no-ops on non-Linux (bind-mount ownership isn't
    # reflected in the container there); force the Linux path so every
    # assertion here actually reaches create_process.
    monkeypatch.setattr(fresh_manager, "system", "linux")


@pytest.mark.parametrize("make_path", [
    lambda p: p,
    lambda p: str(p),
    lambda p: os.path.relpath(str(p)),
])
def test_chmod_resolves_path_object_string_and_relative_forms(fresh_manager, tmp_path, monkeypatch, make_path):
    target = tmp_path / "mytarget"
    target.mkdir()
    monkeypatch.chdir(tmp_path)
    calls = _record(monkeypatch, fresh_manager)

    fresh_manager.runtime.chmod(make_path(target), 0o700)

    assert calls
    assert str(target) in calls[0]


def test_chmod_resolves_tilde_path(fresh_manager, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "mytarget"
    target.mkdir()
    calls = _record(monkeypatch, fresh_manager)

    fresh_manager.runtime.chmod("~/mytarget", 0o700)

    assert calls
    assert str(target) in calls[0]


def test_chown_and_chmod_resolve_the_same_tilde_path(fresh_manager, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "mytarget"
    target.mkdir()
    calls = _record(monkeypatch, fresh_manager)

    fresh_manager.runtime.chown("~/mytarget", fresh_manager.user)
    fresh_manager.runtime.chmod("~/mytarget", 0o700)

    assert len(calls) == 2
    assert calls[0][-1] == calls[1][-1] == str(target)
