#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""grep must not import its optional lief/magic deps at module load (spec §6.3).

Command discovery imports every command module; if grep imported lief/magic at
the top level, `ct grep` (and the whole `common` group) would vanish whenever
those optional deps are absent. They must load only when grep actually runs.
"""
import sys


def test_grep_import_does_not_load_optional_deps():
    # Drop any pre-existing optional deps so we observe grep's own effect.
    for name in list(sys.modules):
        if name.split(".")[0] in ("lief", "magic"):
            del sys.modules[name]

    import linktools.commands.common.grep  # noqa: F401

    assert "lief" not in sys.modules
    assert "magic" not in sys.modules
    import linktools.commands.common.grep as grep
    assert not hasattr(grep, "lief")
    assert not hasattr(grep, "magic")


def test_grep_loader_raises_commanderror_when_absent(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name.split(".")[0] in ("lief", "magic"):
            raise ImportError("No module named '%s'" % name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    import importlib
    from linktools.cli import CommandError
    import linktools.commands.common.grep as grep

    import pytest
    with pytest.raises(CommandError):
        grep._load_grep_optional_deps()
