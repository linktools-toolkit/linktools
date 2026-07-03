import asyncio
from pathlib import Path

from linktools.ai.core.registry import SpecSource
from linktools.ai.core.runtime import AgentKernel
from linktools.ai.session.coordination import InMemorySessionCoordinator
from linktools.ai.session.local import InMemorySessionStatusStore
from linktools.ai.session.types import FileSession, FileSessionSpec
from linktools.ai.subagent.registry import SubagentSpec
from linktools.ai.swarm.coordinator import SwarmCoordinator
from linktools.ai.swarm.local import InMemoryTaskQueue
from linktools.ai.swarm.protocols import Task


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

    def trace_root(self, trace_id: str) -> Path:
        return self.workspace_root / "traces" / trace_id


def _make_session(tmp_path: Path, env: _FakeAgentEnv) -> FileSession:
    spec = FileSessionSpec(session_id="s1", trace_id="t1", coordination=InMemorySessionCoordinator(), status_store=InMemorySessionStatusStore())
    return FileSession.create(
        env.workspace_root,
        env.trace_root(spec.trace_id),
        spec,
    )


def _make_subagent_spec(tmp_path: Path) -> SubagentSpec:
    agent_dir = tmp_path / "subagents" / "echo"
    agent_dir.mkdir(parents=True)
    return SubagentSpec.from_dict(
        {"description": "echoes input", "system": "echo whatever you're given as JSON"},
        SpecSource(name="echo", path=agent_dir / "agent.md", base_dir=agent_dir),
    )


def test_swarm_coordinator_runs_all_tasks_to_a_terminal_state(tmp_path):
    # No live model config is available in this test environment, so every
    # agent.generate() call will fail (ModelClientUnavailable or similar) --
    # that's fine and expected, same premise as
    # test_start_background_eventually_reaches_done_or_failed in
    # test_agent_kernel_background.py. This test only checks that the
    # coordinator drives every task to a terminal state (done OR failed),
    # not that generation succeeds.
    env = _FakeAgentEnv(tmp_path)
    kernel = AgentKernel(
        skill_registry=env.get_skill_registry(),
        subagent_registry=env.get_subagent_registry(),
        mcp_registry=env.get_mcp_registry(),
    )
    session = _make_session(tmp_path, env)
    spec = _make_subagent_spec(tmp_path)
    queue = InMemoryTaskQueue()

    async def _run():
        await queue.add([
            Task(task_id="t1", payload={"x": 1}),
            Task(task_id="t2", payload={"x": 2}),
            Task(task_id="t3", payload={"x": 3}),
        ])
        coordinator = SwarmCoordinator(kernel, queue, model_config_resolver=lambda model_type: None)
        return await coordinator.run(spec, session, agent_count=2)

    results = asyncio.run(_run())
    assert len(results) == 3
    assert all(t.status in ("done", "failed") for t in results)


def test_swarm_coordinator_returns_empty_list_for_empty_queue(tmp_path):
    env = _FakeAgentEnv(tmp_path)
    kernel = AgentKernel(
        skill_registry=env.get_skill_registry(),
        subagent_registry=env.get_subagent_registry(),
        mcp_registry=env.get_mcp_registry(),
    )
    session = _make_session(tmp_path, env)
    spec = _make_subagent_spec(tmp_path)
    queue = InMemoryTaskQueue()

    async def _run():
        coordinator = SwarmCoordinator(kernel, queue, model_config_resolver=lambda model_type: None)
        return await coordinator.run(spec, session, agent_count=3)

    results = asyncio.run(_run())
    assert results == []
