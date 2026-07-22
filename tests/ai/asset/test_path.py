#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/asset/test_path.py"""

import pytest

from linktools.ai.errors import InvalidAssetPathError
from linktools.ai.asset.path import AssetPath


def test_normalizes_repeated_and_trailing_slashes():
    assert (
        AssetPath("//agents//security//agent.md/").value
        == "/agents/security/agent.md"
    )


def test_equality_and_hash_based_on_normalized_value():
    a = AssetPath("/a//b")
    b = AssetPath("/a/b/")
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "relative/path", "/a/../b", "/a/./b", "/a/..", "/..", "/a\x00b"],
)
def test_rejects_invalid_paths(raw):
    with pytest.raises(InvalidAssetPathError):
        AssetPath(raw)


def test_parts_and_namespace():
    p = AssetPath("/agents/security/agent.md")
    assert p.parts == ("agents", "security", "agent.md")
    assert p.namespace == "agents"


def test_child_and_truediv():
    p = AssetPath("/agents/security")
    assert p.child("agent.md") == AssetPath("/agents/security/agent.md")
    assert (p / "agent.md") == AssetPath("/agents/security/agent.md")


def test_str():
    assert str(AssetPath("/a/b")) == "/a/b"
