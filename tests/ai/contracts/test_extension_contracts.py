#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scenario provider + entrypoint resolver contracts (contract),
exercised against the default DirectoryExtensionContentSource /
DirectoryEntrypointResolver and reusable for business backends."""

import pytest

from linktools.ai.extension.provider import DirectoryExtensionContentSource
from linktools.ai.extension.resolver import DirectoryEntrypointResolver

from ._assertions import (
    assert_entrypoint_resolver_contract,
    assert_extension_content_source_contract,
)


@pytest.fixture
def env(tmp_path):
    for pkg in ("skill-creator", "another-skill"):
        root = tmp_path / pkg
        (root / "agents").mkdir(parents=True)
        (root / "references").mkdir()
        (root / "SKILL.md").write_text("# s", encoding="utf-8")
        (root / "references" / "r.md").write_text("ref", encoding="utf-8")
        (root / "agents" / "grader.md").write_text(
            "---\nname: grader\nmodel:\n  primary: gpt-4o\n---\nGrade.\n",
            encoding="utf-8",
        )
    roots = {p: tmp_path / p for p in ("skill-creator", "another-skill")}
    return (
        DirectoryExtensionContentSource(roots),
        DirectoryEntrypointResolver(roots),
    )


@pytest.mark.asyncio
async def test_extension_content_source_contract(env):
    provider, _ = env
    await assert_extension_content_source_contract(
        provider, extension_id="skill-creator", sample_path="SKILL.md"
    )


@pytest.mark.asyncio
async def test_entrypoint_resolver_contract(env):
    _, resolver = env
    await assert_entrypoint_resolver_contract(
        resolver, extension_id="skill-creator", agent_name="grader"
    )


@pytest.mark.asyncio
async def test_same_entrypoint_name_in_two_extensions_stays_distinct(env):
    _, resolver = env
    from linktools.ai.extension.entrypoint import EntrypointRef
    from linktools.ai.extension.scope import ExtensionScope

    a = await resolver.resolve_agent(
        EntrypointRef("agent", "grader", ExtensionScope("skill-creator"))
    )
    b = await resolver.resolve_agent(
        EntrypointRef("agent", "grader", ExtensionScope("another-skill"))
    )
    assert a.id != b.id
