#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifies consistent path normalization for chown and chmod: both resolve
their `path` argument identically via env_config.cast(type="path"), for a
Path object, a plain string, a relative path, or a `~/...` path.
"""
import os
import shutil

import pytest

from linktools.cntr.runtime import process as process_module


class _FakeProcess:
    def check_call(self):
        return 0


def _record(monkeypatch, manager):
    calls = []
    monkeypatch.setattr(manager.runtime, "create_process", lambda *a, **k: (calls.append(a), _FakeProcess())[1])
    return calls


@pytest.fixture(autouse=True)
def _linux(monkeypatch, fresh_manager):
    # chown/chmod are no-ops on non-Linux (bind-mount ownership isn't
    # reflected in the container there); force the Linux path so every
    # assertion here actually reaches create_process. shutil.which/get_uid/
    # get_gid are mocked too so this test only exercises path resolution,
    # not whatever chown/chmod binaries or user database happen to be on
    # the host running the suite.
    monkeypatch.setattr(fresh_manager, "system", "linux")
    monkeypatch.setattr(shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(process_module, "get_uid", lambda user: 1000)
    monkeypatch.setattr(process_module, "get_gid", lambda user: 1000)


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
