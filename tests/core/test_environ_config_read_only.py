#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Environment config reads must not require writable storage."""

import errno
import json
import os
from pathlib import Path

import pytest

from linktools.core import Environ
from linktools.errors import ConfigError


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.chdir(home)
    for key in tuple(os.environ):
        if key.startswith("LINKTOOLS_"):
            monkeypatch.delenv(key)
    return home


def _deny_mkdir(monkeypatch):
    def denied(self, *args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system", str(self))
    monkeypatch.setattr(Path, "mkdir", denied)


@pytest.mark.parametrize("debug", [None, "true", "false"])
def test_debug_and_logging_do_not_create_storage(monkeypatch, isolated_home, debug):
    if debug is not None:
        monkeypatch.setenv("LINKTOOLS_DEBUG", debug)
    _deny_mkdir(monkeypatch)
    env = Environ()
    assert env.debug is (debug == "true")
    try:
        raise ValueError("original business failure")
    except ValueError:
        env.get_logger("read-only").warning("operation failed", exc_info=env.debug)
    assert env.get_config("MISSING_FIELD", default="fallback") == "fallback"
    assert env.build_config_store("other.json").keys() == []
    assert not (isolated_home / ".linktools").exists()


@pytest.mark.parametrize("override, expected", [(None, True), ("false", False)])
def test_existing_persistent_debug_remains_readable(monkeypatch, isolated_home, override, expected):
    settings = isolated_home / ".linktools" / "config" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"main.DEBUG": True}))
    if override is not None:
        monkeypatch.setenv("LINKTOOLS_DEBUG", override)
    _deny_mkdir(monkeypatch)
    assert Environ().debug is expected
    assert not (isolated_home / ".linktools" / "cache").exists()


def test_persist_creates_storage_only_when_writing(isolated_home):
    env = Environ()
    assert env.debug is False
    assert not env.paths.storage.exists()
    env.config.persist("DEBUG", True)
    assert env.paths.config.is_dir()
    assert (env.paths.cache / "locks").is_dir()
    assert Environ().debug is True


def test_persist_still_reports_read_only_storage(monkeypatch, isolated_home):
    _deny_mkdir(monkeypatch)
    env = Environ()
    assert env.debug is False
    with pytest.raises(OSError) as error:
        env.config.persist("DEBUG", True)
    assert error.value.errno == errno.EROFS
    assert env.debug is False


def test_invalid_persistent_config_still_fails_closed(isolated_home):
    settings = isolated_home / ".linktools" / "config" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("{invalid")
    with pytest.raises(ConfigError, match="not valid JSON"):
        Environ().debug
