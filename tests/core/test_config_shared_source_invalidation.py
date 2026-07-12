#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sibling Config objects sharing a RuntimeOverrideSource/PersistentSource
must observe a write made through EITHER sibling immediately -- not just
"eventually" or "if never read before the write".

Regression: Manager Config and per-repository Configs share their
Environment/RuntimeOverride/Persistent triple, but each has its own
ConfigResolver._memo. A write through one Config only ever cleared ITS OWN
memo (Config.set/persist/unset/remove -> self._resolver.clear_memo()) --
a sibling that had already cached the old value kept returning it until
something unrelated (define()/reload()) happened to clear that sibling's
whole memo too. This only reproduces when the sibling is read (and thus
memoized) BEFORE the write -- reading it only after the write trivially
"works" even with the bug, which is why the existing
test_shared_config_sources.py suite (order: write, then read) didn't catch
it.

Fixed via per-source `revision`: a ConfigResolver's memo entry also stores
the token (schema revision, every source's revision) captured when it was
computed; a stale token forces recomputation, so a sibling Config's write
is visible on the next read through ANY Config sharing that source, cache
or no cache.
"""
import json

from linktools.core._config import ConfigSchema, AliasProvider, LazyProvider, ConfigField
from linktools.core._environ import BaseEnviron, Environ
from linktools.types import MISSING


def _reset_global_config():
    descriptor = BaseEnviron.__dict__.get("global_config")
    if descriptor is not None and hasattr(descriptor, "val"):
        descriptor.val = MISSING


def _make_environ(monkeypatch, tmp_path):
    monkeypatch.delenv("LINKTOOLS_PATH", raising=False)
    monkeypatch.setenv("LINKTOOLS_PATH", str(tmp_path / "storage"))
    _reset_global_config()
    return Environ()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _two_sibling_configs(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    shared = env.shared_config_sources("container", "")
    config_a = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_a)
    config_b = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_b)
    return config_a, config_b


def test_runtime_write_through_a_invalidates_already_cached_b(monkeypatch, tmp_path):
    config_a, config_b = _two_sibling_configs(monkeypatch, tmp_path)
    config_a.set("PORT", "8001")

    assert config_b.get("PORT") == "8001"  # cache B's resolution first
    config_a.set("PORT", "9000")
    assert config_b.get("PORT") == "9000"  # must not return the stale cache


def test_persist_write_through_a_invalidates_already_cached_b(monkeypatch, tmp_path):
    config_a, config_b = _two_sibling_configs(monkeypatch, tmp_path)
    config_a.persist("PORT", "8001")

    assert config_b.get("PORT") == "8001"  # cache B's resolution first
    config_a.persist("PORT", "9000")
    assert config_b.get("PORT") == "9000"


def test_write_through_b_invalidates_already_cached_a(monkeypatch, tmp_path):
    """Symmetric: it isn't just A -> B, any sibling can invalidate any
    other."""
    config_a, config_b = _two_sibling_configs(monkeypatch, tmp_path)
    config_b.persist("PORT", "8002")

    assert config_a.get("PORT") == "8002"  # cache A's resolution first
    config_b.persist("PORT", "9001")
    assert config_a.get("PORT") == "9001"


def test_unrelated_already_cached_key_is_unaffected(monkeypatch, tmp_path):
    """A revision bump forces recomputation of every memoized key on the
    next read, but an unrelated key must still resolve to the SAME
    (correct) value afterward -- the invalidation must not corrupt it."""
    config_a, config_b = _two_sibling_configs(monkeypatch, tmp_path)
    config_b.persist("UNRELATED", "still-here")
    assert config_b.get("UNRELATED") == "still-here"  # cache it

    config_a.persist("PORT", "8001")
    assert config_b.get("PORT") == "8001"
    assert config_b.get("UNRELATED") == "still-here"


def test_alias_across_sibling_configs_reflects_shared_write(monkeypatch, tmp_path):
    config_a, config_b = _two_sibling_configs(monkeypatch, tmp_path)
    config_a.define(ConfigField(name="SOURCE", default="old"))
    config_a.define(ConfigField(name="ALIAS", provider=AliasProvider("SOURCE")))
    config_b.define(ConfigField(name="SOURCE", default="old"))
    config_b.define(ConfigField(name="ALIAS", provider=AliasProvider("SOURCE")))

    assert config_b.get("ALIAS") == "old"  # cache B's Alias resolution
    config_a.persist("SOURCE", "new")
    assert config_b.get("ALIAS") == "new"


def test_lazy_across_sibling_configs_reflects_shared_write(monkeypatch, tmp_path):
    config_a, config_b = _two_sibling_configs(monkeypatch, tmp_path)
    for cfg in (config_a, config_b):
        cfg.define(ConfigField(name="SOURCE", default="old"))
        cfg.define(ConfigField(name="DERIVED", provider=LazyProvider(
            lambda resolver: "value:%s" % resolver.get("SOURCE"))))

    assert config_b.get("DERIVED") == "value:old"  # cache B's Lazy resolution
    config_a.persist("SOURCE", "new")
    assert config_b.get("DERIVED") == "value:new"


def test_file_source_reload_invalidates_cache(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / ".linktools.json", {"environment": {"KEY": "before"}})

    shared = env.shared_config_sources("container", "")
    config = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo)

    assert config.get("KEY") == "before"  # cache it
    _write(repo / ".linktools.json", {"environment": {"KEY": "after"}})
    config.reload()
    assert config.get("KEY") == "after"
