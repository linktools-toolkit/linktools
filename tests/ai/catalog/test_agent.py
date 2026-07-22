#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for agent/catalog.py: AgentCatalog resolves AgentSpec from
{name}.md (markdown + YAML frontmatter), revision-cached."""

import asyncio
import os
import time

import pytest

from linktools.ai.agent.spec import (
    AgentSpec,
    MiddlewareRef,
    PromptSpec,
    ToolRef,
)
from linktools.ai.errors import (
    InvalidSpecError,
    RegistryNotFoundError,
    RegistryParseError,
)
from linktools.ai.agent.catalog import AgentCatalog
from linktools.ai.agent.codec import parse_agent_spec
from linktools.ai.catalog.parsing import SpecLoader


def _write_agents(tmp_path) -> None:
    """Write writer.md + default.md under tmp_path/agents."""
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "writer.md").write_text(
        "---\n"
        "name: writer\n"
        "model:\n"
        "  primary: gpt-4o\n"
        "  fallbacks: [gpt-4o-mini]\n"
        "tools:\n"
        "  - {kind: builtin, name: file}\n"
        "  - {kind: builtin, name: terminal}\n"
        "middleware:\n"
        "  - budget\n"
        "---\n"
        "You are a careful writer. Be concise.\n",
        encoding="utf-8",
    )
    (agents / "minimal.md").write_text(
        "---\nname: minimal\nmodel:\n  primary: gpt-4o-mini\n---\nJust do the thing.\n",
        encoding="utf-8",
    )


# 1. get() parses markdown frontmatter into an AgentSpec with full fields.
def test_get_returns_agent_spec_from_markdown(tmp_path):
    _write_agents(tmp_path)
    registry = AgentCatalog.from_specloader(SpecLoader.from_filesystem(tmp_path / "agents"))

    async def run():
        return await registry.get("writer")

    spec = asyncio.run(run())
    assert isinstance(spec, AgentSpec)
    assert spec.id == "writer"
    assert spec.name == "writer"
    # model slice
    assert spec.model.primary == "gpt-4o"
    assert spec.model.fallbacks == ("gpt-4o-mini",)
    # instructions come from the markdown body (stripped)
    assert isinstance(spec.instructions, PromptSpec)
    assert spec.instructions.instructions == "You are a careful writer. Be concise."
    # tools / middleware refs
    assert len(spec.tools) == 2
    assert all(isinstance(t, ToolRef) for t in spec.tools)
    assert tuple(t.name for t in spec.tools) == ("file", "terminal")
    assert len(spec.middleware) == 1
    assert all(isinstance(m, MiddlewareRef) for m in spec.middleware)
    assert spec.middleware[0].name == "budget"
    assert spec.output_schema is None


# 2. list_ids() returns every agent id under the loader root.
def test_list_ids_returns_all_agent_ids(tmp_path):
    _write_agents(tmp_path)
    registry = AgentCatalog.from_specloader(SpecLoader.from_filesystem(tmp_path / "agents"))

    async def run():
        return await registry.list_ids()

    ids = asyncio.run(run())
    assert ids == ("minimal", "writer")


# 3. get() caches the parsed spec per revision: second read returns the same object.
def test_get_caches_spec_per_revision(tmp_path):
    _write_agents(tmp_path)
    registry = AgentCatalog.from_specloader(SpecLoader.from_filesystem(tmp_path / "agents"))

    async def run():
        a = await registry.get("writer")
        b = await registry.get("writer")
        return a, b

    a, b = asyncio.run(run())
    assert a is b


# 4. Cache invalidation: bumping a file's mtime changes revision -> next get re-reads.
def test_get_re_reads_after_file_change(tmp_path):
    _write_agents(tmp_path)
    agents = tmp_path / "agents"
    registry = AgentCatalog.from_specloader(SpecLoader.from_filesystem(agents))

    async def run():
        return await registry.get("writer")

    spec1 = asyncio.run(run())
    assert spec1.instructions.instructions == "You are a careful writer. Be concise."

    # Rewrite the body and force mtime to advance past the original second.
    (agents / "writer.md").write_text(
        "---\n"
        "name: writer\n"
        "model:\n"
        "  primary: gpt-4o\n"
        "---\n"
        "You are now a verbose explainer.\n",
        encoding="utf-8",
    )
    future = time.time() + 10
    os.utime(agents / "writer.md", (future, future))

    spec2 = asyncio.run(run())
    assert spec2.instructions.instructions == "You are now a verbose explainer."
    assert spec2 is not spec1


# 5. Missing agent -> RegistryNotFoundError (propagated from the loader).
def test_get_missing_agent_raises_not_found(tmp_path):
    _write_agents(tmp_path)
    registry = AgentCatalog.from_specloader(SpecLoader.from_filesystem(tmp_path / "agents"))

    async def run():
        await registry.get("nope")

    with pytest.raises(RegistryNotFoundError):
        asyncio.run(run())


# 6a. Missing 'model' block -> InvalidSpecError.
def test_get_missing_model_raises_invalid_spec(tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "nomodel.md").write_text(
        "---\nname: nomodel\n---\nNo model declared.\n",
        encoding="utf-8",
    )
    registry = AgentCatalog.from_specloader(SpecLoader.from_filesystem(agents))

    async def run():
        await registry.get("nomodel")

    with pytest.raises(InvalidSpecError):
        asyncio.run(run())


# 6b. Malformed YAML in the frontmatter -> RegistryParseError.
def test_get_malformed_frontmatter_raises_parse_error(tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "broken.md").write_text(
        "---\nname: broken\ntools: [unterminated\n---\nbody\n",
        encoding="utf-8",
    )
    registry = AgentCatalog.from_specloader(SpecLoader.from_filesystem(agents))

    async def run():
        await registry.get("broken")

    with pytest.raises(RegistryParseError):
        asyncio.run(run())


# 7. Defaults: an agent with only name + minimal model + body -> tools==(), middleware==().
def test_get_applies_defaults_when_minimal(tmp_path):
    _write_agents(tmp_path)
    registry = AgentCatalog.from_specloader(SpecLoader.from_filesystem(tmp_path / "agents"))

    async def run():
        return await registry.get("minimal")

    spec = asyncio.run(run())
    assert spec.name == "minimal"
    assert spec.model.primary == "gpt-4o-mini"
    assert spec.tools is None  # no tools key -> unset (three-state, contract)
    assert spec.middleware == ()
    assert spec.instructions.instructions == "Just do the thing."


# 8. parse_agent_spec accepts {name, config} middleware dict entries.
def test_parse_agent_spec_supports_dict_middleware():
    payload = {
        "name": "agent",
        "model": {"primary": "gpt-4o"},
        "middleware": [{"name": "budget", "config": {"limit": 5}}],
    }
    spec = parse_agent_spec("agent", payload, "do things")
    assert len(spec.middleware) == 1
    assert spec.middleware[0].name == "budget"
    assert dict(spec.middleware[0].config) == {"limit": 5}


# -- explicit null vs missing for tools/middleware -------------------


def test_parse_agent_spec_tools_null_rejected_missing_ok():
    """tools:null is an explicit null (reject); tools absent is unset (None);
    tools:[] is an explicit empty list (())."""
    from linktools.ai.agent.codec import parse_agent_spec

    model = {"primary": "gpt"}
    # missing -> None
    assert parse_agent_spec("a", {"model": model}, "body").tools is None
    # [] -> ()
    assert parse_agent_spec("a", {"model": model, "tools": []}, "body").tools == ()
    # null -> reject
    with pytest.raises(InvalidSpecError, match="tools"):
        parse_agent_spec("a", {"model": model, "tools": None}, "body")


def test_parse_agent_spec_middleware_null_rejected():
    from linktools.ai.agent.codec import parse_agent_spec

    with pytest.raises(InvalidSpecError, match="middleware"):
        parse_agent_spec("a", {"model": {"primary": "gpt"}, "middleware": None}, "body")


def test_parse_agent_spec_rejects_empty_name(tmp_path):
    """an explicit empty/whitespace 'name' is a config error, not a
    silent fall-back to the agent id. Missing 'name' still falls back."""
    from linktools.ai.errors import InvalidSpecError

    body = "instructions body"
    # Missing name -> falls back to the agent id (no error).
    spec = parse_agent_spec("fallback", {"model": {"primary": "m"}}, body)
    assert spec.name == "fallback"
    # Explicit empty name -> InvalidSpecError.
    with pytest.raises(InvalidSpecError):
        parse_agent_spec("blank", {"name": "", "model": {"primary": "m"}}, body)
    # Whitespace-only name -> InvalidSpecError.
    with pytest.raises(InvalidSpecError):
        parse_agent_spec("space", {"name": "   ", "model": {"primary": "m"}}, body)
