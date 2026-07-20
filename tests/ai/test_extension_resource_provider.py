#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DirectoryExtensionResourceProvider (contract/contract): path sandbox,
pagination, and max_bytes read clamp."""

import pytest

from linktools.ai.errors import (
    ExtensionResourceAccessDeniedError,
    ExtensionResourceNotFoundError,
)
from linktools.ai.extension.provider import DirectoryExtensionResourceProvider
from linktools.ai.extension.resource import ResourceRef
from linktools.ai.extension.scope import ExtensionScope


@pytest.fixture
def provider(tmp_path):
    root = tmp_path / "skill-creator"
    (root / "references").mkdir(parents=True)
    (root / "agents").mkdir()
    (root / "SKILL.md").write_text("# skill\nbody", encoding="utf-8")
    (root / "references" / "a.md").write_text("aaa", encoding="utf-8")
    (root / "references" / "b.md").write_text("bbb", encoding="utf-8")
    (root / "agents" / "grader.md").write_text("# grader", encoding="utf-8")
    return DirectoryExtensionResourceProvider({"skill-creator": root}), root


SCOPE = ExtensionScope("skill-creator", "skill")


@pytest.mark.asyncio
async def test_list_resources_paginates(provider):
    p, _ = provider
    page1 = await p.list_resources(SCOPE, "references", limit=1)
    assert len(page1.items) == 1
    assert page1.next_cursor is not None
    page2 = await p.list_resources(
        SCOPE, "references", limit=1, cursor=page1.next_cursor
    )
    assert len(page2.items) == 1
    names = {i.path for i in page1.items + page2.items}
    assert names == {"references/a.md", "references/b.md"}


@pytest.mark.asyncio
async def test_list_resources_unknown_extension_raises(provider):
    p, _ = provider
    from linktools.ai.errors import ExtensionNotFoundError

    with pytest.raises(ExtensionNotFoundError):
        await p.list_resources(ExtensionScope("nope"), "")


@pytest.mark.asyncio
async def test_read_resource_returns_content(provider):
    p, _ = provider
    content = await p.read_resource(ResourceRef(scope=SCOPE, path="SKILL.md"))
    assert b"skill" in (
        content.content
        if isinstance(content.content, bytes)
        else content.content.encode()
    )
    assert content.size_bytes > 0


@pytest.mark.asyncio
async def test_read_resource_clamps_to_max_bytes(provider):
    p, root = provider
    (root / "big.txt").write_text("x" * 1000, encoding="utf-8")
    content = await p.read_resource(
        ResourceRef(scope=SCOPE, path="big.txt"), max_bytes=10
    )
    assert len(content.content) == 10
    assert content.size_bytes == 1000
    assert content.metadata.get("truncated") is True


@pytest.mark.asyncio
async def test_read_resource_bounds_io_not_just_payload(tmp_path):
    # A resource larger than max_bytes must not be fully read into memory just to
    # be truncated -- size_bytes reflects cap+1 (the read bound), not the file.
    from linktools.ai.extension.provider import DirectoryExtensionResourceProvider

    root = tmp_path / "pkg"
    root.mkdir()
    (root / "huge.txt").write_text("y" * 100_000, encoding="utf-8")
    prov = DirectoryExtensionResourceProvider({"pkg": root})
    content = await prov.read_resource(
        ResourceRef(scope=ExtensionScope("pkg"), path="huge.txt"), max_bytes=16
    )
    assert len(content.content) == 16  # payload bounded to max_bytes
    assert content.size_bytes == 100_000  # true file size (stat), not bytes read
    assert content.metadata.get("truncated") is True


@pytest.mark.asyncio
async def test_read_resource_rejects_parent_traversal(provider):
    p, _ = provider
    with pytest.raises(ValueError):
        await p.read_resource(ResourceRef(scope=SCOPE, path="../etc/passwd"))


@pytest.mark.asyncio
async def test_read_resource_rejects_absolute_path(provider):
    p, _ = provider
    with pytest.raises(ValueError):
        await p.read_resource(ResourceRef(scope=SCOPE, path="/etc/passwd"))


@pytest.mark.asyncio
async def test_read_resource_missing_raises(provider):
    p, _ = provider
    with pytest.raises(ExtensionResourceNotFoundError):
        await p.read_resource(ResourceRef(scope=SCOPE, path="nope.md"))


@pytest.mark.asyncio
async def test_read_resource_requires_scope(provider):
    p, _ = provider
    with pytest.raises(ExtensionResourceAccessDeniedError):
        await p.read_resource(ResourceRef(scope=None, path="SKILL.md"))


def test_sanitize_rejects_null_byte_and_drive():
    from linktools.ai.extension.resource import sanitize_extension_path

    with pytest.raises(ValueError):
        sanitize_extension_path("a\x00b")
    with pytest.raises(ValueError):
        sanitize_extension_path("C:/x")


def test_sanitize_collapses_dot_and_rejects_parent():
    from linktools.ai.extension.resource import sanitize_extension_path

    assert sanitize_extension_path("a/./b/") == "a/b"
    with pytest.raises(ValueError):
        sanitize_extension_path("a/../b")


@pytest.mark.asyncio
async def test_read_resource_extension_allow_deny(tmp_path):
    from linktools.ai.errors import ExtensionResourceAccessDeniedError
    from linktools.ai.extension.provider import DirectoryExtensionResourceProvider

    root = tmp_path / "pkg"
    root.mkdir()
    (root / "a.md").write_text("md", encoding="utf-8")
    (root / "b.bin").write_text("bin", encoding="utf-8")
    # allow only .md
    prov = DirectoryExtensionResourceProvider({"pkg": root}, allow_extensions=(".md",))
    await prov.read_resource(ResourceRef(scope=ExtensionScope("pkg"), path="a.md"))
    with pytest.raises(ExtensionResourceAccessDeniedError):
        await prov.read_resource(ResourceRef(scope=ExtensionScope("pkg"), path="b.bin"))
    # deny .bin
    prov2 = DirectoryExtensionResourceProvider({"pkg": root}, deny_extensions=(".bin",))
    with pytest.raises(ExtensionResourceAccessDeniedError):
        await prov2.read_resource(ResourceRef(scope=ExtensionScope("pkg"), path="b.bin"))
