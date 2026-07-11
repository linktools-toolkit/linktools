#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.git.support: Python-version + Dulwich-presence gating (spec
Part I). Importing linktools.git must never require dulwich itself --
only actually using GitRepository/GitHead does."""
import sys

import pytest

from linktools.errors import GitUnavailableError
from linktools.git import support


def test_git_support_py_lt_310(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 9, 0))
    assert support.is_git_supported_python() is False
    assert support.is_git_available() is False
    reason = support.get_git_unavailable_reason()
    assert "Python 3.10" in reason


def test_git_support_py_ge_310_missing_dulwich(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 10, 0))
    monkeypatch.setitem(sys.modules, "dulwich", None)  # force ImportError
    assert support.is_git_available() is False
    reason = support.get_git_unavailable_reason()
    assert "linktools[git]" in reason


def test_git_support_available(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 10, 0))
    monkeypatch.delitem(sys.modules, "dulwich", raising=False)
    assert support.get_git_unavailable_reason() is None
    assert support.is_git_available() is True


def test_require_git_available_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 6, 0))
    with pytest.raises(GitUnavailableError):
        support.require_git_available("Cloning a Git repository")


def test_require_git_available_is_noop_when_available(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 10, 0))
    monkeypatch.delitem(sys.modules, "dulwich", raising=False)
    support.require_git_available()  # must not raise


def test_linktools_git_import_without_dulwich(monkeypatch):
    """import linktools.git must succeed even when Python itself doesn't
    support Git (repository.py must never be eagerly imported)."""
    import importlib
    import linktools.git as git_module

    monkeypatch.setattr(sys, "version_info", (3, 6, 0))
    reloaded = importlib.reload(git_module)
    try:
        assert reloaded.is_git_available() is False
        with pytest.raises(Exception):
            reloaded.GitRepository(None, ".")
    finally:
        monkeypatch.undo()
        importlib.reload(git_module)
