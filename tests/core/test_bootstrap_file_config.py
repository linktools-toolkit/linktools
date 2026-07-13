# -*- coding: utf-8 -*-
"""Bootstrap path resolution from linktools.json.

STORAGE_PATH/DATA_PATH/TEMP_PATH must be resolvable from the profile
environment (not just OS environment variables). They are ordinary values
computed once by `global_config` and exposed via `environ.paths`/
`data_path`/`temp_path`; `get_config`/`set_config` have no special handling
for them.
"""
import os

import pytest

from linktools.core._environ import BaseEnviron, Environ
from linktools.types import MISSING


def _reset_global_config():
    descriptor = BaseEnviron.__dict__.get("global_config")
    if descriptor is not None and hasattr(descriptor, "val"):
        descriptor.val = MISSING


def _write(path, data):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated_bootstrap(monkeypatch, tmp_path):
    """Every test in this file gets a private HOME and CWD so the real
    developer machine's ~/.linktools/linktools.json / cwd .linktools.json
    (if any) can never leak in, and a reset global_config classproperty."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    _reset_global_config()
    yield home, cwd
    _reset_global_config()


def _make_environ(monkeypatch, storage):
    monkeypatch.delenv("LINKTOOLS_PATH", raising=False)
    monkeypatch.delenv("LINKTOOLS_STORAGE_PATH", raising=False)
    monkeypatch.delenv("LINKTOOLS_DATA_PATH", raising=False)
    monkeypatch.delenv("LINKTOOLS_TEMP_PATH", raising=False)
    if storage is not None:
        monkeypatch.setenv("LINKTOOLS_PATH", str(storage))
    _reset_global_config()
    return Environ()


# -- STORAGE_PATH resolution priority ---------------------------------------

def test_storage_path_default_when_nothing_configured(monkeypatch, tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    env = _make_environ(monkeypatch, storage=None)
    assert str(env.paths.storage) == str(home / ".linktools")


def test_profile_config_is_read_by_environment_source(monkeypatch, tmp_path):
    _write(tmp_path / "cwd" / ".linktools.json", {"config": {"PROFILE_KEY": "profile"}})
    env = _make_environ(monkeypatch, storage=tmp_path / "storage")

    assert env.get_config("PROFILE_KEY") == "profile"
    assert env.global_config == {"DEBUG": False}


def test_storage_path_from_global_file(monkeypatch, tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "global-storage")}})
    env = _make_environ(monkeypatch, storage=None)
    assert str(env.paths.storage) == str(tmp_path / "global-storage")


def test_storage_path_global_file_overrides_local_file(monkeypatch, tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "global-storage")}})
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "local-storage")}})
    env = _make_environ(monkeypatch, storage=None)
    assert str(env.paths.storage) == str(tmp_path / "global-storage")


def test_profile_outranks_os_env(monkeypatch, tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "global-storage")}})
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "local-storage")}})
    env = _make_environ(monkeypatch, storage=str(tmp_path / "env-storage"))
    assert str(env.paths.storage) == str(tmp_path / "global-storage")


def test_corrupt_local_file_raises_not_silently_ignored(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    (cwd / ".linktools.json").write_text("{not json", encoding="utf-8")
    _reset_global_config()
    monkeypatch.delenv("LINKTOOLS_PATH", raising=False)
    with pytest.raises(Exception):  # ConfigError, surfaced through global_config
        Environ().paths


def test_set_config_storage_path_is_a_plain_settable_field(monkeypatch, tmp_path):
    # STORAGE_PATH/DATA_PATH/TEMP_PATH are no longer special-cased: set_config
    # on them behaves like any other key and does not raise.
    env = _make_environ(monkeypatch, storage=None)
    env.paths
    env.set_config("STORAGE_PATH", str(tmp_path / "elsewhere"))
    assert env.get_config("STORAGE_PATH") == str(tmp_path / "elsewhere")


def test_set_config_unrelated_field_still_works(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, storage=None)
    env.paths
    env.set_config("DEBUG", True)
    assert env.get_config("DEBUG", bool) is True


def test_reload_does_not_raise(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "local-storage")}})
    env = _make_environ(monkeypatch, storage=None)
    env.paths
    env.config.reload()  # re-reads the same file -- must not raise


# -- relative STORAGE_PATH ----------------------------------------------------

def test_relative_storage_path_in_local_file_is_normalized(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": "./storage"}})
    env = _make_environ(monkeypatch, storage=None)
    assert str(env.paths.storage) == str(cwd / "storage")
    assert os.path.isabs(str(env.data_path))


def test_relative_storage_path_does_not_crash_config_store(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": "./storage"}})
    env = _make_environ(monkeypatch, storage=None)
    env.config_store  # must not raise "Unsafe path" from mixing relative/absolute paths
