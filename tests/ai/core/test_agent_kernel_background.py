import asyncio
from pathlib import Path

import pytest

from linktools.ai.core.registry import AgentSpec, SpecSource
from linktools.ai.core.runtime import AgentKernel
from linktools.ai.session.local import InMemorySessionStatusStore
from linktools.ai.session.coordination import InMemorySessionCoordinator
from linktools.ai.session.types import FileSession, FileSessionSpec
from linktools.ai.subagent.registry import SubagentSpec


class _Registry:
    def __init__(self, specs):
        self._specs = {spec.name: spec for spec in specs}

    async def preload(self):
        return None

    def all(self):
        return list(self._specs.values())

    def __contains__(self, item):
        return item in self._specs

    def get(self, item):
        return self._specs[item]

    def resolve_by_capability(self, item):
        return None


class _FakeAgentEnv:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.hooks = None
        self._subagent = _Registry([])

    def get_skill_registry(self, cap_store=None):
        del cap_store
        return _Registry([])

    def get_subagent_registry(self, cap_store=None):
        del cap_store
        return self._subagent

    def get_mcp_registry(self, cap_store=None):
        del cap_store
        return _Registry([])

    def get_logger(self, name):
        import logging
        return logging.getLogger(name)


def _make_session(tmp_path: Path, env: _FakeAgentEnv) -> FileSession:
    spec = FileSessionSpec(session_id="s1", coordination=InMemorySessionCoordinator(), status_store=InMemorySessionStatusStore())
    return FileSession.create(env.workspace_root / "s1", spec)


def _make_subagent_spec(tmp_path: Path) -> SubagentSpec:
    agent_dir = tmp_path / "subagents" / "echo"
    agent_dir.mkdir(parents=True)
    return SubagentSpec.from_dict(
        {"description": "echoes input", "system": "echo whatever you're given as JSON"},
        SpecSource(name="echo", path=agent_dir / "agent.md", base_dir=agent_dir),
    )


def test_start_background_returns_run_id_immediately(tmp_path):
    env = _FakeAgentEnv(tmp_path)
    kernel = AgentKernel(
        skill_registry=env.get_skill_registry(),
        subagent_registry=env.get_subagent_registry(),
        mcp_registry=env.get_mcp_registry(),
    )
    session = _make_session(tmp_path, env)
    spec = _make_subagent_spec(tmp_path)

    async def _run():
        run_id = await kernel.start_background(spec, session, {"x": 1}, model_config_resolver=lambda model_type: None)
        assert isinstance(run_id, str) and run_id
        status = await kernel.check_background(run_id)
        assert status.state == "running"
        return run_id

    asyncio.run(_run())


def test_check_background_unknown_run_id_raises_key_error():
    kernel = AgentKernel(skill_registry=object(), subagent_registry=object(), mcp_registry=object())

    async def _run():
        await kernel.check_background("nope")

    with pytest.raises(KeyError):
        asyncio.run(_run())


def test_start_background_eventually_reaches_done_or_failed(tmp_path):
    # The fake subagent has no real model config, so its generate() call will fail
    # (ModelClientUnavailable or similar) -- that's fine, this test only checks that
    # the background task's failure is captured as a terminal status rather than
    # crashing silently or leaving the run stuck at "running" forever.
    env = _FakeAgentEnv(tmp_path)
    kernel = AgentKernel(
        skill_registry=env.get_skill_registry(),
        subagent_registry=env.get_subagent_registry(),
        mcp_registry=env.get_mcp_registry(),
    )
    session = _make_session(tmp_path, env)
    spec = _make_subagent_spec(tmp_path)

    async def _run():
        run_id = await kernel.start_background(spec, session, {"x": 1}, model_config_resolver=lambda model_type: None)
        for _ in range(50):
            status = await kernel.check_background(run_id)
            if status.state != "running":
                return status
            await asyncio.sleep(0.02)
        raise AssertionError("background run never left 'running' state")

    status = asyncio.run(_run())
    assert status.state == "failed"
    assert status.error
