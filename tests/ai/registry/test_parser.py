#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for registry/parser.py: shared spec-loading primitives."""

import asyncio
from decimal import Decimal

import pytest

from linktools.ai.agent.spec import ToolRef
from linktools.ai.errors import InvalidSpecError, RegistryNotFoundError, RegistryParseError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.registry.parser import (
    SpecLoader,
    parse_json_text,
    parse_markdown_text,
    parse_model_policy,
    parse_tool_refs,
    parse_yaml_text,
)
from fnmatch import fnmatch


class _StubResourceFile:
    """Minimal duck-typed stand-in satisfying SpecLoader.from_resources."""

    __slots__ = ("path", "content")

    def __init__(self, path: str, content: str) -> None:
        self.path = path
        self.content = content


class _StubResourceStore:
    """Minimal in-memory store exercising SpecLoader.from_resources without
    pulling in any concrete ResourceStore implementation."""

    def __init__(self) -> None:
        self._entries: "dict[str, str]" = {}
        self._revision = 0

    async def get(self, path: str) -> "_StubResourceFile | None":
        if path not in self._entries:
            return None
        return _StubResourceFile(path, self._entries[path])

    async def list(self, *, pattern: "str | None" = None) -> "list[_StubResourceFile]":
        return [
            _StubResourceFile(p, c)
            for p, c in self._entries.items()
            if pattern is None or fnmatch(p, pattern)
        ]

    async def put(self, path: str, content: str) -> _StubResourceFile:
        self._entries[path] = content
        self._revision += 1
        return _StubResourceFile(path, content)

    async def revision(self) -> int:
        return self._revision


# 1. parse_yaml_text
def test_parse_yaml_text_parses_object():
    assert parse_yaml_text("a: 1\nb: [2,3]") == {"a": 1, "b": [2, 3]}


def test_parse_yaml_text_rejects_malformed():
    with pytest.raises(RegistryParseError):
        parse_yaml_text("a: [unterminated")


# 2. parse_markdown_text
def test_parse_markdown_text_parses_frontmatter_and_body():
    # load_markdown_text returns the body verbatim after the closing "---";
    # the newline immediately following the delimiter is preserved.
    front, body = parse_markdown_text("---\nname: x\n---\nbody")
    assert front == {"name": "x"}
    assert body == "\nbody"


def test_parse_markdown_text_returns_empty_frontmatter_when_none():
    front, body = parse_markdown_text("just body")
    assert front == {}
    assert body == "just body"


def test_parse_markdown_text_rejects_malformed_frontmatter():
    with pytest.raises(RegistryParseError):
        parse_markdown_text("---\nname: [unterminated\n---\nbody")


# 3. parse_json_text
def test_parse_json_text_parses_object():
    assert parse_json_text('{"k":"v"}') == {"k": "v"}


def test_parse_json_text_rejects_malformed_json():
    with pytest.raises(RegistryParseError):
        parse_json_text('{"k":')


def test_parse_json_text_rejects_non_object_top_level():
    with pytest.raises(RegistryParseError):
        parse_json_text("[1,2]")


# 4. SpecLoader.from_filesystem
def test_spec_loader_from_filesystem_read_returns_content(tmp_path):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "a1.md").write_text("hello agent", encoding="utf-8")
    (tmp_path / "swarms").mkdir()
    (tmp_path / "swarms" / "s1.yaml").write_text("id: s1", encoding="utf-8")

    loader = SpecLoader.from_filesystem(tmp_path)

    async def run():
        return await loader.read("agents/a1.md")

    assert asyncio.run(run()) == "hello agent"


def test_spec_loader_from_filesystem_read_missing_raises(tmp_path):
    loader = SpecLoader.from_filesystem(tmp_path)

    async def run():
        await loader.read("missing.md")

    with pytest.raises(RegistryNotFoundError):
        asyncio.run(run())


def test_spec_loader_from_filesystem_list_ids_lists_files_with_suffix(tmp_path):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "a1.md").write_text("a1", encoding="utf-8")
    (tmp_path / "agents" / "a2.md").write_text("a2", encoding="utf-8")
    (tmp_path / "agents" / "ignore.txt").write_text("nope", encoding="utf-8")

    loader = SpecLoader.from_filesystem(tmp_path / "agents")

    async def run():
        return await loader.list_ids(".md")

    assert asyncio.run(run()) == ("a1", "a2")


def test_spec_loader_from_filesystem_revision_is_deterministic(tmp_path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    loader = SpecLoader.from_filesystem(tmp_path)

    async def run():
        return await loader.revision()

    first = asyncio.run(run())
    assert isinstance(first, int)
    assert first > 0


# 4b. SpecLoader.from_resources
def test_spec_loader_from_resources_read_and_list():
    async def run():
        store = _StubResourceStore()
        await store.put("agents/a1.md", "agent body")
        await store.put("agents/a2.md", "agent body 2")
        await store.put("agents/skip.txt", "ignored")

        loader = SpecLoader.from_resources(store, prefix="agents")
        body = await loader.read("a1.md")
        ids = await loader.list_ids(".md")
        rev = await loader.revision()
        return body, ids, rev

    body, ids, rev = asyncio.run(run())
    assert body == "agent body"
    assert set(ids) == {"a1", "a2"}
    assert isinstance(rev, int)


def test_spec_loader_from_resources_read_missing_raises():
    async def run():
        store = _StubResourceStore()
        loader = SpecLoader.from_resources(store, prefix="agents")
        await loader.read("missing.md")

    with pytest.raises(RegistryNotFoundError):
        asyncio.run(run())


# 5. parse_model_policy
def test_parse_model_policy_builds_policy_with_decimal_budget():
    policy = parse_model_policy({"primary": "gpt-4o", "budget": "1.50"})
    assert isinstance(policy, ModelPolicy)
    assert policy.primary == "gpt-4o"
    assert policy.budget == Decimal("1.50")


def test_parse_model_policy_accepts_model_alias_for_primary():
    policy = parse_model_policy({"model": "claude-haiku"})
    assert policy.primary == "claude-haiku"


def test_parse_model_policy_rejects_missing_primary():
    with pytest.raises(InvalidSpecError):
        parse_model_policy({})


# 6. parse_tool_refs
def test_parse_tool_refs_builds_tuple_from_strings_and_dicts():
    refs = parse_tool_refs(["tool_a", {"name": "tool_b"}])
    assert isinstance(refs, tuple)
    assert len(refs) == 2
    assert all(isinstance(r, ToolRef) for r in refs)
    assert refs[0].name == "tool_a"
    assert refs[1].name == "tool_b"


def test_parse_tool_refs_rejects_invalid_item():
    with pytest.raises(InvalidSpecError):
        parse_tool_refs(["ok", 42])


def test_parse_tool_refs_none_is_unset():
    # No tools key -> None (unset), distinct from tools: [] -> () (spec §10.7).
    assert parse_tool_refs(None) is None
    assert parse_tool_refs([]) == ()


def test_parse_tool_refs_rejects_non_list():
    with pytest.raises(InvalidSpecError):
        parse_tool_refs("not-a-list")
