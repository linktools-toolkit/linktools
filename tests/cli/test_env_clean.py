#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Tuple

from linktools.cli.env import get_commands
from linktools.system import get_interpreter_ident


class FakeEnviron:
    name = "test"
    version = "1"
    system = "linux"
    debug = False

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.clean_calls = []
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    def get_data_path(self, *parts: str, **kwargs: Any) -> Path:
        return self.root.joinpath(*parts)

    def clean_temp_files(self, *paths: str, expire_days: int = 7) -> None:
        self.clean_calls.append((paths, expire_days))


def _write_shell_artifacts(root: Path, version: str = "1") -> "Tuple[Path, Path]":
    scripts = root / "scripts" / get_interpreter_ident()
    stub = scripts / ("env_v%s" % version) / "ct-demo"
    alias = scripts / ("alias_v%s" % version) / "alias.bash"
    stub.parent.mkdir(parents=True)
    alias.parent.mkdir(parents=True)
    stub.write_text("stub", encoding="utf-8")
    alias.write_text("alias", encoding="utf-8")
    return stub, alias


def test_clean_preserves_generated_alias_and_command_stubs(tmp_path: Path) -> None:
    environ = FakeEnviron(tmp_path)
    stub, alias = _write_shell_artifacts(tmp_path)

    clean = next(command for command in get_commands(environ) if command.name == "clean")
    assert clean.run(SimpleNamespace(days=3, all=False)) is None

    assert stub.read_text(encoding="utf-8") == "stub"
    assert alias.read_text(encoding="utf-8") == "alias"
    assert environ.clean_calls == [((), 3)]


def test_clean_all_removes_all_generated_alias_and_command_stubs(tmp_path: Path) -> None:
    environ = FakeEnviron(tmp_path)
    current_stub, current_alias = _write_shell_artifacts(tmp_path)
    old_stub, old_alias = _write_shell_artifacts(tmp_path, version="0")

    clean = next(command for command in get_commands(environ) if command.name == "clean")
    assert clean.run(SimpleNamespace(days=3, all=True)) is None

    assert not current_stub.exists()
    assert not current_alias.exists()
    assert not old_stub.exists()
    assert not old_alias.exists()
    assert environ.clean_calls == [((), 3)]
