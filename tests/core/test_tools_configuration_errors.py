import json
from pathlib import Path

import pytest

from linktools import metadata
from linktools.core import Environ
from linktools.core._tools import Tools
from linktools.errors import ToolDefinitionError


class Capability:
    develop = False

    def __init__(self, name, path):
        self.name = name
        self.path = Path(path)

    def get_asset_path(self, *parts):
        return self.path


class EntryPoint:
    def __init__(self, capability, name=None, value=None):
        self.capability = capability
        self.name = name or "%s-capability" % capability.name
        self.value = value or "pkg.%s:capability" % capability.name

    def load(self):
        return self.capability


class FailingEntryPoint:
    name = "broken-capability"
    value = "pkg.broken:capability"

    def load(self):
        raise ImportError("missing optional dependency")


def _write_tools(path, tools, **extra):
    payload = {"schema": 1, "tools": tools}
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _environment(tmp_path, monkeypatch, capabilities):
    env = Environ()
    monkeypatch.setattr(env, "get_data_path", lambda *parts, **kwargs: tmp_path.joinpath(*parts))
    monkeypatch.setattr("linktools.core._entrypoint.select_entry_points",
                        lambda group: tuple(EntryPoint(capability) for capability in capabilities))
    return env


def test_duplicate_capability_is_a_definition_error(tmp_path, monkeypatch):
    first = Capability("mobile", tmp_path / "first.json")
    second = Capability("mobile", tmp_path / "second.json")
    monkeypatch.setattr("linktools.core._entrypoint.select_entry_points", lambda group: (
        EntryPoint(first, "mobile-first", "pkg.first:capability"),
        EntryPoint(second, "mobile-second", "pkg.second:capability"),
    ))
    with pytest.raises(ToolDefinitionError) as raised:
        Environ()._create_tools()
    message = str(raised.value)
    assert "duplicate capability mobile" in message
    assert "mobile-first" in message and "pkg.first:capability" in message
    assert "mobile-second" in message and "pkg.second:capability" in message


def test_capability_entry_point_load_error_includes_source_and_cause(monkeypatch):
    entry_point = FailingEntryPoint()
    monkeypatch.setattr("linktools.core._entrypoint.select_entry_points", lambda group: (entry_point,))
    with pytest.raises(ToolDefinitionError) as raised:
        Environ()._create_tools()
    message = str(raised.value)
    assert metadata.__capability_group__ in message
    assert entry_point.name in message
    assert entry_point.value in message
    assert "ImportError" in message
    assert "missing optional dependency" in message
    assert isinstance(raised.value.__cause__, ImportError)


def test_duplicate_tool_reports_both_capability_sources(tmp_path, monkeypatch):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_tools(first_path, {"adb": {"run": {"lookup": "adb"}}})
    _write_tools(second_path, {"adb": {"run": {"lookup": "adb"}}})
    capabilities = [Capability("mobile", first_path), Capability("custom-mobile", second_path)]
    with pytest.raises(ToolDefinitionError) as raised:
        _environment(tmp_path, monkeypatch, capabilities)._create_tools()
    message = str(raised.value)
    assert "adb" in message
    assert "mobile" in message and str(first_path) in message
    assert "custom-mobile" in message and str(second_path) in message


def test_invalid_schema_reports_capability_and_path(tmp_path, monkeypatch):
    path = tmp_path / "mobile.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ToolDefinitionError, match=r"mobile.*mobile\.json"):
        _environment(tmp_path, monkeypatch, [Capability("mobile", path)])._create_tools()


def test_new_user_tool_must_be_complete(tmp_path, monkeypatch):
    user_path = tmp_path / "tools" / "tools.json"
    user_path.parent.mkdir()
    _write_tools(user_path, {"custom": {"version": "1"}})
    with pytest.raises(ToolDefinitionError, match=r"custom.*tools\.json"):
        _environment(tmp_path, monkeypatch, [])._create_tools()


@pytest.mark.parametrize("config, missing", [
    ({"dependencies": ["missing"]}, "missing"),
    ({"run": {"runner": "missing"}}, "missing"),
])
def test_missing_dependency_and_runner_include_source(tmp_path, config, missing):
    source = "mobile: %s" % (tmp_path / "mobile.json")
    with pytest.raises(ToolDefinitionError, match="demo.*mobile.*missing"):
        Tools(type("Env", (), {
            "system": "linux", "machine": "x86_64",
            "get_logger": lambda self, name: None,
            "get_data_path": lambda self, *parts, **kwargs: tmp_path.joinpath(*parts),
            "build_config": lambda self, *args: type("Config", (), {"get": lambda self, *a, **k: k.get("default")})(),
        })(), {"demo": config}, sources={"demo": source})


def test_cyclic_dependency_includes_source(tmp_path):
    source = "mobile: %s" % (tmp_path / "mobile.json")
    with pytest.raises(ToolDefinitionError, match="demo.*mobile.*cyclic"):
        Tools(type("Env", (), {
            "system": "linux", "machine": "x86_64",
            "get_logger": lambda self, name: None,
            "get_data_path": lambda self, *parts, **kwargs: tmp_path.joinpath(*parts),
            "build_config": lambda self, *args: type("Config", (), {"get": lambda self, *a, **k: k.get("default")})(),
        })(), {"demo": {"dependencies": ["demo"]}}, sources={"demo": source})
