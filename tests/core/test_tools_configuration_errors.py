import json
from pathlib import Path

import pytest

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
    def __init__(self, capability):
        self.capability = capability

    def load(self):
        return self.capability


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
    with pytest.raises(ToolDefinitionError, match="duplicate capability mobile"):
        _environment(tmp_path, monkeypatch, [first, second])._create_tools()


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
