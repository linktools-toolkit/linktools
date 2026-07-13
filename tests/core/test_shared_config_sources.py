# -*- coding: utf-8 -*-
"""shared_config_sources()/build_config(): multiple sibling Config objects
(e.g. cntr's manager Config and its per-repository Config) sharing one
Environment/RuntimeOverride/Persistent/global-profile state -- a runtime
override, a persisted value, or the profile's own values apply uniformly to
every sibling, since none of them keep a local-file layer of their own
anymore (every repository shares this process's own merged profile)."""
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


def test_persistent_value_overrides_both_siblings(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    from linktools.core._config import ConfigSchema

    shared = env.shared_config_sources("container", "")
    config_a = env.build_config(ConfigSchema(), shared)
    config_b = env.build_config(ConfigSchema(), shared)

    config_a.persist("KEY", "persisted-everywhere")
    assert config_a.get("KEY") == "persisted-everywhere"
    assert config_b.get("KEY") == "persisted-everywhere"


def test_runtime_override_overrides_both_siblings(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    from linktools.core._config import ConfigSchema

    shared = env.shared_config_sources("container", "")
    config_a = env.build_config(ConfigSchema(), shared)
    config_b = env.build_config(ConfigSchema(), shared)

    config_a.set("KEY", "runtime-everywhere")
    assert config_a.get("KEY") == "runtime-everywhere"
    assert config_b.get("KEY") == "runtime-everywhere"


def test_global_profile_value_inherited_by_every_sibling(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    import json
    home.joinpath(".linktools").mkdir(parents=True, exist_ok=True)
    home.joinpath(".linktools", "linktools.json").write_text(
        json.dumps({"environment": {"SHARED": "global-value"}}), encoding="utf-8")

    env = _make_environ(monkeypatch, tmp_path)
    from linktools.core._config import ConfigSchema

    shared = env.shared_config_sources("container", "")
    config_a = env.build_config(ConfigSchema(), shared)
    config_b = env.build_config(ConfigSchema(), shared)

    assert config_a.get("SHARED") == "global-value"
    assert config_b.get("SHARED") == "global-value"


def test_schema_fields_copy_to_sibling_schema(monkeypatch, tmp_path):
    env = _make_environ(monkeypatch, tmp_path)
    from linktools.core._config import ConfigSchema

    base_schema = ConfigSchema()
    base_schema.define(ConfigField(name="HOST", default="localhost"))
    copied = ConfigSchema()
    for field in base_schema.fields():
        copied.define(field)
    assert "HOST" in copied
    assert copied.get("HOST").default == "localhost"
