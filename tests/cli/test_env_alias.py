from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.cli import _command as command_module
from linktools.cli.env import get_commands


class FakeEnviron:
    name = "test"
    version = "1"
    system = "linux"
    debug = False

    def __init__(self, root):
        self.root = Path(root)
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: None,
                                      warning=lambda *args, **kwargs: None)

    def get_data_path(self, *parts, **kwargs):
        return self.root.joinpath(*parts)


def _alias(environ):
    return next(item for item in get_commands(environ) if item.name == "alias")


@pytest.fixture(autouse=True)
def _quiet_file_helpers(monkeypatch):
    monkeypatch.setattr("linktools.utils._files.get_environ",
                        lambda: SimpleNamespace(debug=False))


def _command_info(name="hello"):
    return SimpleNamespace(id=name, command=True, command_name=name, parent_id="",
                           module="demo.commands.%s" % name)


def test_alias_ignores_broken_command_entry_point_without_warning(tmp_path, monkeypatch, capsys):
    calls = []

    def discover(group, *, onerror):
        calls.append(onerror)
        yield _command_info()

    monkeypatch.setattr(command_module, "iter_entry_point_commands", discover)
    result = _alias(FakeEnviron(tmp_path)).run(SimpleNamespace(shell="bash", reload=True))
    captured = capsys.readouterr()

    assert result in (None, 0)
    assert calls == ["ignore"]
    assert "warning" not in captured.err.lower()
    assert (tmp_path / "scripts").exists()


def test_alias_does_not_initialize_tools_or_load_capabilities(tmp_path, monkeypatch):
    from linktools.core import _environ
    from linktools import metadata

    def fail_tools(self):
        raise AssertionError("alias must not initialize Tools")

    monkeypatch.setattr(_environ.Environ, "_create_tools", fail_tools)
    capability = SimpleNamespace(load=lambda: (_ for _ in ()).throw(AssertionError("capability loaded")))
    monkeypatch.setattr("linktools.core.select_entry_points",
                        lambda group: (capability,) if group == metadata.__capability_group__ else ())
    monkeypatch.setattr(command_module, "iter_entry_point_commands",
                        lambda group, *, onerror: iter((_command_info(),)))

    assert _alias(FakeEnviron(tmp_path)).run(SimpleNamespace(shell="bash", reload=True)) in (None, 0)


def test_non_alias_entry_point_discovery_still_warns(monkeypatch):
    from linktools.cli._command import _iter_entry_points

    class Broken:
        name = "broken"

        def load(self):
            raise RuntimeError("broken entry point")

    warnings = []
    monkeypatch.setattr("linktools.cli._command.environ",
                        SimpleNamespace(logger=SimpleNamespace(warning=lambda message, **kwargs: warnings.append(message)),
                                         debug=False))
    # The discovery backend is patched locally so this test checks the warning
    # policy without depending on installed distributions.
    monkeypatch.setattr("linktools.core.select_entry_points", lambda group: (Broken(),))
    assert list(_iter_entry_points("commands", onerror="warn")) == []
    assert "broken" in warnings[-1]


def test_broken_entry_point_is_silent_when_alias_requests_ignore(monkeypatch):
    from linktools.cli._command import _iter_entry_points

    class Broken:
        name = "broken"

        def load(self):
            raise RuntimeError("broken entry point")

    warnings = []
    monkeypatch.setattr("linktools.cli._command.environ",
                        SimpleNamespace(logger=SimpleNamespace(warning=lambda message, **kwargs: warnings.append(message)),
                                         debug=False))
    monkeypatch.setattr("linktools.core.select_entry_points", lambda group: (Broken(),))
    assert list(_iter_entry_points("commands", onerror="ignore")) == []
    assert warnings == []
