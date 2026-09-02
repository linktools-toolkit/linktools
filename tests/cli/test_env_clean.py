#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from types import SimpleNamespace

from linktools.cli.env import get_commands
from linktools.system import get_interpreter_ident


class FakeEnviron:
    name = "test"
    version = "1"
    system = "linux"
    debug = False

    def __init__(self, root):
        self.root = Path(root)
        self.clean_calls = []
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    def get_data_path(self, *parts, **kwargs):
        return self.root.joinpath(*parts)

    def clean_temp_files(self, *paths, expire_days=7):
        self.clean_calls.append((paths, expire_days))


def test_clean_preserves_generated_alias_and_command_stubs(tmp_path):
    environ = FakeEnviron(tmp_path)
    scripts = tmp_path / "scripts" / get_interpreter_ident()
    stub = scripts / "env_v1" / "ct-demo"
    alias = scripts / "alias_v1" / "alias.bash"
    stub.parent.mkdir(parents=True)
    alias.parent.mkdir(parents=True)
    stub.write_text("stub", encoding="utf-8")
    alias.write_text("alias", encoding="utf-8")

    clean = next(command for command in get_commands(environ) if command.name == "clean")
    assert clean.run(SimpleNamespace(days=3)) is None

    assert stub.read_text(encoding="utf-8") == "stub"
    assert alias.read_text(encoding="utf-8") == "alias"
    assert environ.clean_calls == [((), 3)]
