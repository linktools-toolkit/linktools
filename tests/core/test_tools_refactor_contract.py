import zipfile
from pathlib import Path

import pytest

from linktools.core._tools import InstallSpec, RunSpec, ToolDefinition, Tools
from linktools.errors import ToolDefinitionError, ToolNotFound


class Env:
    system = "linux"
    machine = "x86_64"
    version = "1"

    def __init__(self, root):
        self.root = Path(root)
        self._cache = None
        self._downloads = None
        from linktools.core._locks import LockManager
        self.locks = LockManager(self.root / "locks")

    @property
    def cache(self):
        if self._cache is None:
            from linktools.core import CacheStore
            self._cache = CacheStore(self.root / "cache.db")
        return self._cache

    @property
    def downloads(self):
        if self._downloads is None:
            from linktools.core import DownloadManager
            self._downloads = DownloadManager(self)
        return self._downloads

    def get_data_path(self, *parts, **kwargs):
        return self.root.joinpath("data", *parts)

    def get_temp_path(self, *parts, **kwargs):
        return self.root.joinpath("temp", *parts)

    def get_logger(self, name):
        import logging
        return logging.getLogger(name)

    def build_config(self, *args, **kwargs):
        class Config:
            def get(self, key, type=None, default=None):
                return default
        return Config()


def test_explicit_specs_reject_unknown_fields_and_string_lists():
    assert InstallSpec(url="x").url == "x"
    assert RunSpec(args=["--ok"]).args == ("--ok",)
    with pytest.raises(ToolDefinitionError):
        InstallSpec.from_mapping({"url": "x", "download_url": "y"})
    with pytest.raises(ToolDefinitionError):
        RunSpec.from_mapping({"args": "--bad"})


def test_tool_definition_uses_mapping_key_as_name():
    definition = ToolDefinition.from_mapping("demo", {"version": "1"})
    assert definition.name == "demo"
    with pytest.raises(ToolDefinitionError):
        ToolDefinition.from_mapping("demo", {"name": "other"})


def test_extract_dir_is_part_of_runtime_install_layout(tmp_path):
    archive = tmp_path / "demo.zip"
    with zipfile.ZipFile(str(archive), "w") as zf:
        zf.writestr("bin/demo", "#!/bin/sh\n")
    env = Env(tmp_path)
    tools = Tools(env, {"demo": {
        "version": "1",
        "install": {"url": str(archive), "extract_dir": "nested/path", "entrypoint": "bin/demo"},
    }})
    tool = tools["demo"]
    tool.prepare()
    assert tool.artifact_path == tool.content_dir / "bin" / "demo"
    assert (tool.install_dir / "nested" / "path" / "bin" / "demo").exists()


def test_runtime_copy_only_accepts_version(tmp_path):
    tool = Tools(Env(tmp_path), {"demo": {"version": "1"}})["demo"]
    with pytest.raises(TypeError):
        tool.copy(run={})


def test_runner_is_an_implicit_dependency_and_path_is_not_used_as_lookup(tmp_path):
    tools = Tools(Env(tmp_path), {
        "runner": {"run": {"path": "/bin/echo"}},
        "child": {"run": {"runner": "runner", "args": ["fixed"]}},
    })
    child = tools["child"]
    assert child.argv == ["/bin/echo", "fixed"]
    assert child.environment == {}


def test_multiple_matching_variants_fail(tmp_path):
    with pytest.raises(ToolDefinitionError, match="multiple variants"):
        Tools(Env(tmp_path), {"demo": {"variants": [
            {"match": {"platform": "linux"}, "run": {"lookup": "one"}},
            {"match": {"platform": ["linux", "darwin"]}, "run": {"lookup": "two"}},
        ]}})


def test_template_can_reference_another_tool_artifact(tmp_path):
    tools = Tools(Env(tmp_path), {
        "child": {"run": {"runner": "runner", "args": ["{tools[runner].artifact_path}"]}},
        "runner": {"run": {"path": "/bin/echo"}},
    })
    assert str(tools["runner"].artifact_path) in tools["child"].argv


def test_explicit_path_never_falls_back_to_install(tmp_path):
    missing = tmp_path / "missing"
    tools = Tools(Env(tmp_path), {"demo": {
        "install": {"url": "https://example.invalid/demo"},
        "run": {"path": str(missing)},
    }})
    with pytest.raises(ToolNotFound) as raised:
        tools["demo"].prepare()
    assert "explicit path" in str(raised.value)


def test_entry_point_metadata_is_cached(monkeypatch):
    from linktools.core import _entrypoint

    calls = []
    entries = [type("EP", (), {"group": "linktools.capability", "name": "demo"})()]
    monkeypatch.setattr(_entrypoint, "get_entry_points", lambda: entries)
    assert _entrypoint.select_entry_points("linktools.capability") == (entries[0],)
    assert _entrypoint.select_entry_points("linktools.command") == ()
