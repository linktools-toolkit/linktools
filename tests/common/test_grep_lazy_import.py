#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The grep command loads optional binary-analysis dependencies only when run."""

import builtins
import sys
from types import SimpleNamespace

import pytest

from linktools.cli import CommandError


def test_grep_import_does_not_load_optional_dependencies() -> None:
    for name in list(sys.modules):
        if name.split(".")[0] in ("lief", "magic"):
            del sys.modules[name]

    import linktools.commands.common.grep  # noqa: F401

    assert "lief" not in sys.modules
    assert "magic" not in sys.modules


def test_grep_reports_missing_optional_dependencies_when_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocking_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in ("lief", "magic"):
            raise ImportError("No module named '%s'" % name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    from linktools.commands.common.grep import Command

    with pytest.raises(CommandError):
        Command().run(
            SimpleNamespace(ignore_case=False, pattern="needle", files=[])
        )
