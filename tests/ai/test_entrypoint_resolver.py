#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DirectoryEntrypointResolver (contract/contract/contract): list entrypoints,
resolve scoped agents, and namespace isolation across extensions."""

import pytest

from linktools.ai.errors import ExtensionEntrypointNotFoundError
from linktools.ai.extension.entrypoint import EntrypointRef
from linktools.ai.extension.resolver import DirectoryEntrypointResolver
from linktools.ai.extension.scope import ExtensionScope


def _make_pkg(base, pkg_id, agents=("grader",)):
    root = base / pkg_id
    (root / "agents").mkdir(parents=True)
    for a in agents:
        (root / "agents" / f"{a}.md").write_text(
            "---\nname: {n}\nmodel:\n  primary: gpt-4o\n---\nYou are {n}.\n".format(
                n=a
            ),
            encoding="utf-8",
        )
    return root


@pytest.fixture
def resolver(tmp_path):
    _make_pkg(tmp_path, "skill-creator", agents=("grader", "comparator"))
    _make_pkg(tmp_path, "another-skill", agents=("grader",))
    return DirectoryEntrypointResolver(
        {
            "skill-creator": tmp_path / "skill-creator",
            "another-skill": tmp_path / "another-skill",
        }
    )


@pytest.mark.asyncio
async def test_list_entrypoints_kind_filter(resolver):
    scope = ExtensionScope("skill-creator", "skill")
    result = await resolver.list_entrypoints(scope, kind="agent")
    names = sorted(i.name for i in result.items)
    assert names == ["comparator", "grader"]
    assert all(i.kind == "agent" for i in result.items)
    assert all(i.extension_id == "skill-creator" for i in result.items)


@pytest.mark.asyncio
async def test_list_entrypoints_pagination(resolver):
    scope = ExtensionScope("skill-creator", "skill")
    page1 = await resolver.list_entrypoints(scope, kind="agent", limit=1)
    assert len(page1.items) == 1
    assert page1.next_cursor is not None
    page2 = await resolver.list_entrypoints(
        scope, kind="agent", limit=1, cursor=page1.next_cursor
    )
    assert len(page2.items) == 1
    assert {i.name for i in page1.items + page2.items} == {"grader", "comparator"}


@pytest.mark.asyncio
async def test_resolve_scoped_agent_has_namespaced_id(resolver):
    scope = ExtensionScope("skill-creator", "skill")
    agent = await resolver.resolve_agent(
        EntrypointRef(kind="agent", name="grader", scope=scope)
    )
    assert agent.id == "extension:skill-creator:agent:grader"
    assert agent.model.primary == "gpt-4o"


@pytest.mark.asyncio
async def test_namespace_isolation_same_name_different_extensions(resolver):
    s1 = ExtensionScope("skill-creator", "skill")
    s2 = ExtensionScope("another-skill", "skill")
    a1 = await resolver.resolve_agent(
        EntrypointRef(kind="agent", name="grader", scope=s1)
    )
    a2 = await resolver.resolve_agent(
        EntrypointRef(kind="agent", name="grader", scope=s2)
    )
    # Same entrypoint name, distinct scoped ids -> no global namespace clash.
    assert a1.id != a2.id
    assert a1.id == "extension:skill-creator:agent:grader"
    assert a2.id == "extension:another-skill:agent:grader"


@pytest.mark.asyncio
async def test_resolve_missing_entrypoint_raises(resolver):
    scope = ExtensionScope("skill-creator", "skill")
    with pytest.raises(ExtensionEntrypointNotFoundError):
        await resolver.resolve_agent(
            EntrypointRef(kind="agent", name="ghost", scope=scope)
        )


@pytest.mark.asyncio
async def test_resolver_only_exposes_implemented_methods(resolver):
    # First version exposes only list_entrypoints + resolve_agent; workflow /
    # toolset resolution are not part of the public Protocol surface.
    assert hasattr(resolver, "list_entrypoints")
    assert hasattr(resolver, "resolve_agent")
    assert not hasattr(resolver, "resolve_toolset")
    assert not hasattr(resolver, "resolve_workflow")
