#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/resource/test_path.py"""

import pytest

from linktools.ai.errors import InvalidResourcePathError
from linktools.ai.storage.resource.path import ResourcePath


def test_normalizes_repeated_and_trailing_slashes():
    assert (
        ResourcePath("//agents//security//agent.md/").value
        == "/agents/security/agent.md"
    )


def test_equality_and_hash_based_on_normalized_value():
    a = ResourcePath("/a//b")
    b = ResourcePath("/a/b/")
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "relative/path", "/a/../b", "/a/./b", "/a/..", "/..", "/a\x00b"],
)
def test_rejects_invalid_paths(raw):
    with pytest.raises(InvalidResourcePathError):
        ResourcePath(raw)


def test_parts_and_namespace():
    p = ResourcePath("/agents/security/agent.md")
    assert p.parts == ("agents", "security", "agent.md")
    assert p.namespace == "agents"


def test_child_and_truediv():
    p = ResourcePath("/agents/security")
    assert p.child("agent.md") == ResourcePath("/agents/security/agent.md")
    assert (p / "agent.md") == ResourcePath("/agents/security/agent.md")


def test_str():
    assert str(ResourcePath("/a/b")) == "/a/b"
