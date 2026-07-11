# -*- coding: utf-8 -*-
"""shared_config_sources()/build_config() (spec §33, §71): multiple sibling
Config objects (e.g. cntr's per-repository configs) sharing one
Environment/RuntimeOverride/Persistent triple while each keeps its own
local-file layer."""
import json

from linktools.core._environ import BaseEnviron, Environ
from linktools.core import ConfigField
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


def test_two_repo_configs_have_independent_local_files(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    _write(repo_a / ".linktools.json", {"environment": {"STORAGE_PATH": "./a"}})
    _write(repo_b / ".linktools.json", {"environment": {"STORAGE_PATH": "./b"}})

    shared = env.shared_config_sources("container", "")
    from linktools.core._config import ConfigSchema
    config_a = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_a)
    config_b = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_b)

    assert config_a.get("STORAGE_PATH") == "./a"
    assert config_b.get("STORAGE_PATH") == "./b"


def test_load_order_does_not_matter(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    _write(repo_a / ".linktools.json", {"environment": {"KEY": "a"}})
    _write(repo_b / ".linktools.json", {"environment": {"KEY": "b"}})

    shared = env.shared_config_sources("container", "")
    from linktools.core._config import ConfigSchema
    config_b = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_b)
    config_a = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_a)

    assert config_a.get("KEY") == "a"
    assert config_b.get("KEY") == "b"


def test_global_file_inherited_by_both(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    _write(home / ".linktools" / "linktools.json", {"environment": {"SHARED": "global-value"}})
    env = _make_environ(monkeypatch, tmp_path)
    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()

    shared = env.shared_config_sources("container", "")
    from linktools.core._config import ConfigSchema
    config_a = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_a)
    config_b = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_b)

    assert config_a.get("SHARED") == "global-value"
    assert config_b.get("SHARED") == "global-value"


def test_persistent_value_overrides_both_repos(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    _write(repo_a / ".linktools.json", {"environment": {"KEY": "a"}})
    _write(repo_b / ".linktools.json", {"environment": {"KEY": "b"}})

    shared = env.shared_config_sources("container", "")
    from linktools.core._config import ConfigSchema
    config_a = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_a)
    config_b = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_b)

    config_a.persist("KEY", "persisted-everywhere")
    assert config_a.get("KEY") == "persisted-everywhere"
    assert config_b.get("KEY") == "persisted-everywhere"


def test_runtime_override_overrides_both_repos(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    _write(repo_a / ".linktools.json", {"environment": {"KEY": "a"}})
    _write(repo_b / ".linktools.json", {"environment": {"KEY": "b"}})

    shared = env.shared_config_sources("container", "")
    from linktools.core._config import ConfigSchema
    config_a = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_a)
    config_b = env.build_config(ConfigSchema(allow_unknown=True), shared, local_root=repo_b)

    config_a.set("KEY", "runtime-everywhere")
    assert config_a.get("KEY") == "runtime-everywhere"
    assert config_b.get("KEY") == "runtime-everywhere"


def test_schema_fields_copy_to_sibling_schema(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    from linktools.core._config import ConfigSchema

    base_schema = ConfigSchema(allow_unknown=True)
    base_schema.define(ConfigField(name="HOST", default="localhost"))
    copied = ConfigSchema(allow_unknown=True)
    for field in base_schema.fields():
        copied.define(field)
    assert "HOST" in copied
    assert copied.get("HOST").default == "localhost"
