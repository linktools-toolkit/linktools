import zipfile
from pathlib import Path

import pytest

from linktools.core._tools import InstallSpec, RunSpec, ToolDefinition, Tools
from linktools.errors import ToolDefinitionError, ToolInstallError, ToolNotFound


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
    assert RunSpec(args=("--ok",)).args == ("--ok",)
    with pytest.raises(ToolDefinitionError):
        InstallSpec.from_mapping({"url": "x", "download_url": "y"})
    with pytest.raises(ToolDefinitionError):
        RunSpec.from_mapping({"args": "--bad"})
    for invalid in ([1], [True], [None]):
        with pytest.raises(ToolDefinitionError, match=r"demo.*mobile.*run\.args"):
            RunSpec(name="demo", source="mobile: /tmp/mobile.json", args=invalid)


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


def test_managed_directory_entrypoint_fails_before_manifest_and_active(tmp_path):
    archive = tmp_path / "demo.zip"
    with zipfile.ZipFile(str(archive), "w") as zf:
        zf.writestr("bin/demo", "#!/bin/sh\n")
    env = Env(tmp_path)
    tools = Tools(env, {"demo": {
        "version": "1",
        "install": {"url": str(archive), "extract_dir": "payload", "entrypoint": "bin"},
    }})
    tool = tools["demo"]
    with pytest.raises(ToolInstallError, match="entrypoint missing"):
        tool.prepare()
    assert not (tool.install_dir / "manifest.json").exists()
    assert not (tmp_path / "data" / "tools" / "demo" / "active.json").exists()
    assert not list((tmp_path / "data" / "tools" / "demo").glob(".staging-*"))


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


def test_explicit_path_accepts_regular_file_and_file_symlink(tmp_path):
    target = tmp_path / "program"
    target.write_text("program", encoding="utf-8")

    tools = Tools(Env(tmp_path), {"demo": {"run": {"path": str(target)}}})
    assert tools["demo"].exists
    tools["demo"].prepare()

    link = tmp_path / "program-link"
    link.symlink_to(target)
    linked_tools = Tools(Env(tmp_path), {"demo": {"run": {"path": str(link)}}})
    assert linked_tools["demo"].exists
    linked_tools["demo"].prepare()


@pytest.mark.parametrize("path", ["directory", "missing"])
def test_explicit_path_rejects_non_regular_file(tmp_path, path):
    explicit = tmp_path / path
    if path == "directory":
        explicit.mkdir()

    tools = Tools(Env(tmp_path), {"demo": {"run": {"path": str(explicit)}}})
    assert not tools["demo"].exists
    with pytest.raises(ToolNotFound, match="explicit path is not a regular file"):
        tools["demo"].prepare()


def test_entry_point_metadata_is_cached(monkeypatch):
    from linktools.core import _entrypoint

    class EntryPoints(list):
        def select(self, **kwargs):
            return EntryPoints(item for item in self if item.group == kwargs["group"])

    entries = EntryPoints([
        type("EP", (), {"group": "linktools.capability", "name": "demo"})(),
        type("EP", (), {"group": "linktools.command", "name": "command"})(),
    ])
    calls = []
    _entrypoint.get_entry_points.cache_clear()
    monkeypatch.setattr(_entrypoint.metadata, "entry_points", lambda: calls.append(1) or entries)
    assert _entrypoint.select_entry_points("linktools.capability")[0].name == "demo"
    assert _entrypoint.select_entry_points("linktools.command")[0].name == "command"
    assert calls == [1]
    _entrypoint.get_entry_points.cache_clear()


def test_entry_point_cache_supports_legacy_mapping(monkeypatch):
    from linktools.core import _entrypoint

    _entrypoint.get_entry_points.cache_clear()
    monkeypatch.setattr(_entrypoint.metadata, "entry_points", lambda: {
        "linktools.capability": ("demo",),
    })
    assert _entrypoint.select_entry_points("linktools.capability") == ("demo",)
    assert _entrypoint.select_entry_points("linktools.command") == ()
    _entrypoint.get_entry_points.cache_clear()
