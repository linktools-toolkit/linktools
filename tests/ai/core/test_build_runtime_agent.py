#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import shutil
from pathlib import Path

from pydantic_ai.models.function import FunctionModel

from linktools.core import environ
from linktools.ai.core.runtime import build_runtime_agent
from linktools.ai.agent import RuntimeAgent
from linktools.ai.session.local import local_session
from linktools.ai.session.types import FileSession


def _fake_model() -> FunctionModel:
    return FunctionModel(lambda messages, info: None)


def test_build_runtime_agent_with_defaults_returns_runtime_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(environ, "get_temp_path", lambda *paths, **kw: tmp_path / "temp" / Path(*paths))

    async def _run():
        return await build_runtime_agent(model=_fake_model(), model_type="standard")

    agent = asyncio.run(_run())
    assert isinstance(agent, RuntimeAgent)
    assert agent.tools == ["file", "terminal"]
    assert agent.workdir == Path.cwd()


def test_build_runtime_agent_default_registries_are_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(environ, "get_temp_path", lambda *paths, **kw: tmp_path / "temp" / Path(*paths))

    async def _run():
        return await build_runtime_agent(model=_fake_model(), model_type="standard")

    agent = asyncio.run(_run())
    assert agent.skills == []
    assert agent.subagents == []
    assert agent.mcp_servers == []


def test_build_runtime_agent_uses_given_session(tmp_path):
    session = local_session("test-session-1")
    try:
        async def _run():
            return await build_runtime_agent(model=_fake_model(), model_type="standard", session=session)

        agent = asyncio.run(_run())
        assert agent.session is session
    finally:
        shutil.rmtree(session.root, ignore_errors=True)


def test_build_runtime_agent_without_session_creates_ephemeral_file_session(tmp_path, monkeypatch):
    monkeypatch.setattr(environ, "get_temp_path", lambda *paths, **kw: tmp_path / Path(*paths))

    async def _run():
        return await build_runtime_agent(model=_fake_model(), model_type="standard")

    agent = asyncio.run(_run())
    assert isinstance(agent.session, FileSession)
    assert agent.session.session_id


def test_build_runtime_agent_ephemeral_sessions_get_distinct_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(environ, "get_temp_path", lambda *paths, **kw: tmp_path / Path(*paths))

    async def _run():
        first = await build_runtime_agent(model=_fake_model(), model_type="standard")
        second = await build_runtime_agent(model=_fake_model(), model_type="standard")
        return first, second

    first, second = asyncio.run(_run())
    assert first.session.root != second.session.root


def test_build_runtime_agent_respects_workdir_and_allowed_tools(tmp_path):
    session = local_session("test-session-2")
    try:
        async def _run():
            return await build_runtime_agent(
                model=_fake_model(), model_type="standard",
                session=session, workdir=tmp_path, allowed_tools=["file"],
            )

        agent = asyncio.run(_run())
        assert agent.workdir == tmp_path
        assert agent.tools == ["file"]
    finally:
        shutil.rmtree(session.root, ignore_errors=True)


def test_build_runtime_agent_system_prompt_reaches_spec():
    async def _run():
        return await build_runtime_agent(
            model=_fake_model(), model_type="standard", system_prompt="be terse",
        )

    agent = asyncio.run(_run())
    assert agent.spec.system_prompt == "be terse"


def test_build_runtime_agent_loads_skills_from_skill_paths(tmp_path):
    skill_dir = tmp_path / "skills" / "greeting"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.md").write_text(
        "---\ndescription: says hello\n---\nSay hello warmly.\n", encoding="utf-8",
    )

    async def _run():
        return await build_runtime_agent(
            model=_fake_model(), model_type="standard",
            skill_paths=(tmp_path / "skills",),
        )

    agent = asyncio.run(_run())
    assert [s.name for s in agent.skills] == ["greeting"]
