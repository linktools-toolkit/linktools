# -*- coding: utf-8 -*-
"""Bootstrap path resolution from linktools.json (spec Part IV).

STORAGE_PATH/DATA_PATH/TEMP_PATH must be resolvable from the local/global
file config (not just OS environment variables), and must stay consistent
between `environ.paths.storage` and `environ.get_config("STORAGE_PATH")`
(spec §27). They are never stored in `Config` at all -- `get_config()`
reads straight from `self.paths`/`data_path`/`temp_path`, and `set_config()`
always rejects them (spec §29) -- so no Config-level state (a runtime
override, `config.persist()`, `config.reload()`, or a stale/rogue
PersistentSource value) can ever make them disagree with `environ.paths`.
"""
import json
import os

import pytest

from linktools.core._environ import BaseEnviron, Environ
from linktools.errors import ConfigValidationError
from linktools.types import MISSING


def _reset_global_config():
    descriptor = BaseEnviron.__dict__.get("global_config")
    if descriptor is not None and hasattr(descriptor, "val"):
        descriptor.val = MISSING


def _write(path, data):
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


def test_storage_path_from_global_file(monkeypatch, tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "global-storage")}})
    env = _make_environ(monkeypatch, storage=None)
    assert str(env.paths.storage) == str(tmp_path / "global-storage")


def test_storage_path_local_file_overrides_global_file(monkeypatch, tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "global-storage")}})
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "local-storage")}})
    env = _make_environ(monkeypatch, storage=None)
    assert str(env.paths.storage) == str(tmp_path / "local-storage")


def test_os_env_outranks_both_files(monkeypatch, tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "global-storage")}})
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "local-storage")}})
    env = _make_environ(monkeypatch, storage=str(tmp_path / "env-storage"))
    assert str(env.paths.storage) == str(tmp_path / "env-storage")


def test_corrupt_local_file_raises_not_silently_ignored(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    (cwd / ".linktools.json").write_text("{not json", encoding="utf-8")
    _reset_global_config()
    monkeypatch.delenv("LINKTOOLS_PATH", raising=False)
    with pytest.raises(Exception):  # ConfigError, surfaced through global_config
        Environ().paths


# -- consistency: get_config("STORAGE_PATH") == paths.storage (spec §27) ----

def test_get_config_storage_path_matches_paths_storage_default(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, storage=None)
    assert env.get_config("STORAGE_PATH") == str(env.paths.storage)


def test_get_config_storage_path_matches_paths_storage_from_global_file(monkeypatch, tmp_path):
    home = tmp_path / "home"
    _write(home / ".linktools" / "linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "global-storage")}})
    env = _make_environ(monkeypatch, storage=None)
    assert env.get_config("STORAGE_PATH") == str(env.paths.storage) == str(tmp_path / "global-storage")


def test_get_config_storage_path_matches_paths_storage_from_local_file(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "local-storage")}})
    env = _make_environ(monkeypatch, storage=None)
    assert env.get_config("STORAGE_PATH") == str(env.paths.storage) == str(tmp_path / "local-storage")


def test_get_config_storage_path_matches_paths_storage_from_env(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, storage=str(tmp_path / "env-storage"))
    assert env.get_config("STORAGE_PATH") == str(env.paths.storage) == str(tmp_path / "env-storage")


def test_get_config_data_and_temp_path_match(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, storage=None)
    assert env.get_config("DATA_PATH") == str(env.data_path)
    assert env.get_config("TEMP_PATH") == str(env.temp_path)


# -- runtime mutation of bootstrap keys is always rejected -------------------

def test_set_config_storage_path_rejected_once_paths_initialized(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, storage=None)
    env.paths  # force initialization
    with pytest.raises(ConfigValidationError):
        env.set_config("STORAGE_PATH", str(tmp_path / "elsewhere"))


def test_set_config_storage_path_rejected_even_before_paths_touched(monkeypatch, tmp_path):
    # Unlike the old lock-once-initialized design, set_config() rejects
    # bootstrap keys unconditionally -- there is no "before touch" window in
    # which it would silently succeed and then be forgotten (nothing reads a
    # RuntimeOverrideSource entry for these keys in the first place).
    env = _make_environ(monkeypatch, storage=None)
    with pytest.raises(ConfigValidationError):
        env.set_config("STORAGE_PATH", str(tmp_path / "elsewhere"))


def test_persist_storage_path_has_no_effect(monkeypatch, tmp_path):
    # The low-level Config.persist() API has no knowledge of STORAGE_PATH at
    # all (it is not a schema field), so this "succeeds" -- but the value it
    # wrote is inert: get_config()/paths.storage never look at PersistentSource
    # for this key, so they cannot be made to disagree by it.
    env = _make_environ(monkeypatch, storage=None)
    before = str(env.paths.storage)
    env.config.persist("STORAGE_PATH", str(tmp_path / "elsewhere"))
    assert env.get_config("STORAGE_PATH") == before == str(env.paths.storage)


def test_set_config_unrelated_field_still_works(monkeypatch, tmp_path):
    # Sanity: the guard is specific to the three bootstrap keys -- an
    # *unrelated* field must remain freely settable at any time.
    env = _make_environ(monkeypatch, storage=None)
    env.paths
    env.set_config("DEBUG", True)  # must not raise -- only bootstrap keys are protected
    assert env.get_config("DEBUG", bool) is True


# -- reload() never touches bootstrap keys -----------------------------------

def test_reload_does_not_raise(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": str(tmp_path / "local-storage")}})
    env = _make_environ(monkeypatch, storage=None)
    env.paths
    env.config.reload()  # re-reads the same file -- must not raise


def test_reload_after_file_changes_bootstrap_value_has_no_effect(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    local_file = cwd / ".linktools.json"
    _write(local_file, {"environment": {"STORAGE_PATH": str(tmp_path / "local-storage")}})
    env = _make_environ(monkeypatch, storage=None)
    env.paths  # lock in local-storage
    before = str(env.paths.storage)
    _write(local_file, {"environment": {"STORAGE_PATH": str(tmp_path / "changed-storage")}})
    env.config.reload()  # must not raise -- STORAGE_PATH isn't a Config field
    assert env.get_config("STORAGE_PATH") == before == str(env.paths.storage)


# -- relative STORAGE_PATH (the spec's own canonical example) ---------------

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


def test_relative_storage_path_get_config_matches_paths_storage(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    _write(cwd / ".linktools.json", {"environment": {"STORAGE_PATH": "./storage"}})
    env = _make_environ(monkeypatch, storage=None)
    assert env.get_config("STORAGE_PATH") == str(env.paths.storage)
    assert env.get_config("DATA_PATH") == str(env.data_path)
    assert env.get_config("TEMP_PATH") == str(env.temp_path)


# -- a pre-existing PersistentSource value can never diverge ----------------

def test_preexisting_persisted_bootstrap_value_is_inert(monkeypatch, tmp_path):
    """A `main.STORAGE_PATH` key already sitting in ConfigStore (e.g. a
    hand-edited settings.json, or leftover from unrelated experimentation)
    must not be able to make get_config("STORAGE_PATH") disagree with
    environ.paths.storage -- not because it's detected and rejected, but
    because STORAGE_PATH is never a Config field in the first place, so
    nothing ever asks PersistentSource for it.
    """
    env = _make_environ(monkeypatch, storage=None)
    env.paths  # fix the real storage path first
    env.config_store.set("main.STORAGE_PATH", str(tmp_path / "rogue-persisted-storage"))
    _reset_global_config()
    other_env = _make_environ(monkeypatch, storage=None)
    other_env.paths  # same fixed path as before (file/env unchanged)
    other_env.config  # building the full Config must not raise
    assert other_env.get_config("STORAGE_PATH") == str(other_env.paths.storage)
