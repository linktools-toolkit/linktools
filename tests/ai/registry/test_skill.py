#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for registry/skill.py: SkillRegistry resolves SkillSpec from {name}.md
(markdown + YAML frontmatter), revision-cached. The body becomes instructions."""

import asyncio

import pytest

from linktools.ai.errors import (
    RegistryNotFoundError,
    RegistryParseError,
)
from linktools.ai.registry.parser import SpecLoader
from linktools.ai.registry.skill import SkillRegistry, SkillSpec, parse_skill_spec


def _write_skills(tmp_path) -> None:
    """Write skills/greeter.md (full frontmatter + body) under tmp_path."""
    skills = tmp_path / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "greeter.md").write_text(
        "---\n"
        "name: greeter\n"
        "description: says hello\n"
        "---\n"
        "Greet the user warmly.\n",
        encoding="utf-8",
    )


def _write_minimal(tmp_path) -> None:
    """Write skills/minimal.md -- only name + body, exercising the description default."""
    skills = tmp_path / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "minimal.md").write_text(
        "---\nname: minimal\n---\nJust do the thing.\n",
        encoding="utf-8",
    )


# 1. get() parses markdown frontmatter + body into a SkillSpec.
def test_get_returns_skill_spec_from_markdown(tmp_path):
    _write_skills(tmp_path)
    registry = SkillRegistry(SpecLoader.from_filesystem(tmp_path / "skills"))

    async def _run():
        return await registry.get("greeter")

    spec = asyncio.run(_run())
    assert isinstance(spec, SkillSpec)
    assert spec.id == "greeter"
    assert spec.name == "greeter"
    assert spec.description == "says hello"
    assert spec.instructions == "Greet the user warmly."


# 2. list_ids() includes greeter.
def test_list_ids_includes_greeter(tmp_path):
    _write_skills(tmp_path)
    _write_minimal(tmp_path)
    registry = SkillRegistry(SpecLoader.from_filesystem(tmp_path / "skills"))

    async def _run():
        return await registry.list_ids()

    ids = asyncio.run(_run())
    assert "greeter" in ids


# 3. Defaults: a skill with only name + body -> description == "".
def test_get_applies_defaults_when_minimal(tmp_path):
    _write_minimal(tmp_path)
    registry = SkillRegistry(SpecLoader.from_filesystem(tmp_path / "skills"))

    async def _run():
        return await registry.get("minimal")

    spec = asyncio.run(_run())
    assert spec.name == "minimal"
    assert spec.description == ""
    assert spec.instructions == "Just do the thing."


# 4. Missing skill -> RegistryNotFoundError.
def test_get_missing_skill_raises_not_found(tmp_path):
    _write_skills(tmp_path)
    registry = SkillRegistry(SpecLoader.from_filesystem(tmp_path / "skills"))

    async def _run():
        await registry.get("nope")

    with pytest.raises(RegistryNotFoundError):
        asyncio.run(_run())


# 5. get() caches the parsed spec per revision: second read does not hit the loader.
def test_get_caches_spec_per_revision():
    files = {"greeter.md": "---\nname: greeter\n---\nGreet warmly.\n"}
    read_count = [0]

    async def read(path):
        read_count[0] += 1
        if path not in files:
            raise RegistryNotFoundError(path)
        return files[path]

    async def list_ids(suffix):
        return tuple(sorted(k[: -len(suffix)] for k in files if k.endswith(suffix)))

    async def revision():
        return 1

    loader = SpecLoader(read=read, list_ids=list_ids, revision=revision)
    registry = SkillRegistry(loader)

    async def _run():
        a = await registry.get("greeter")
        b = await registry.get("greeter")
        return a, b

    a, b = asyncio.run(_run())
    assert a is b
    assert read_count[0] == 1


# 6. name defaults to skill_id when frontmatter omits name.
def test_parse_skill_spec_name_defaults_to_id():
    spec = parse_skill_spec("fallback", {"description": "x"}, "do things")
    assert spec.name == "fallback"
    assert spec.id == "fallback"
    assert spec.description == "x"
    assert spec.instructions == "do things"


# 7. metadata from frontmatter is copied through.
def test_parse_skill_spec_passes_metadata():
    spec = parse_skill_spec(
        "g",
        {"name": "g", "metadata": {"tags": ["a", "b"]}},
        "body",
    )
    assert dict(spec.metadata) == {"tags": ["a", "b"]}


# 8. Malformed YAML frontmatter -> RegistryParseError.
def test_get_malformed_frontmatter_raises_parse_error(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "broken.md").write_text(
        "---\nname: broken\ntags: [unterminated\n---\nbody\n",
        encoding="utf-8",
    )
    registry = SkillRegistry(SpecLoader.from_filesystem(skills))

    async def _run():
        await registry.get("broken")

    with pytest.raises(RegistryParseError):
        asyncio.run(_run())
