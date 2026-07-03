import asyncio
from pathlib import Path

from linktools.ai.core.registry import SpecSource
from linktools.ai.core.runtime import AgentKernel
from linktools.ai.fork.coordinator import ForkCoordinator
from linktools.ai.session.coordination import InMemorySessionCoordinator
from linktools.ai.session.local import InMemorySessionStatusStore
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
    spec = FileSessionSpec(session_id="parent", coordination=InMemorySessionCoordinator(), status_store=InMemorySessionStatusStore())
    return FileSession.create(env.workspace_root / "parent", spec)


def _make_subagent_spec(tmp_path: Path) -> SubagentSpec:
    agent_dir = tmp_path / "subagents" / "echo"
    agent_dir.mkdir(parents=True)
    return SubagentSpec.from_dict(
        {"description": "echoes input", "system": "echo whatever you're given as JSON"},
        SpecSource(name="echo", path=agent_dir / "agent.md", base_dir=agent_dir),
    )


def test_fork_coordinator_runs_isolated_branches(tmp_path):
    # Same no-live-model-config premise as test_swarm_coordinator_runs_all_tasks_to_a_terminal_state:
    # every branch's agent.generate() fails, this test only checks branch
    # isolation and terminal-state bookkeeping, not generation success.
    env = _FakeAgentEnv(tmp_path)
    kernel = AgentKernel(
        skill_registry=env.get_skill_registry(),
        subagent_registry=env.get_subagent_registry(),
        mcp_registry=env.get_mcp_registry(),
    )
    session = _make_session(tmp_path, env)
    workdir = tmp_path / "runtime"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "shared.txt").write_text("original")
    spec = _make_subagent_spec(tmp_path)

    async def _run():
        coordinator = ForkCoordinator(kernel, model_config_resolver=lambda model_type: None)
        return await coordinator.run(spec, session, {"x": 1}, branch_count=3, workdir=workdir)

    results = asyncio.run(_run())
    assert len(results) == 3
    branch_ids = {r["branch_id"] for r in results}
    assert len(branch_ids) == 3
    assert all(r["status"] in ("done", "failed") for r in results)
    # The parent's workdir file must be untouched by branch execution.
    assert (workdir / "shared.txt").read_text() == "original"
    for r in results:
        forked_file = workdir.parent / ".runtime.forks" / "parent" / r["branch_id"] / "shared.txt"
        assert forked_file.read_text() == "original"


def test_fork_coordinator_returns_empty_list_for_zero_branches(tmp_path):
    env = _FakeAgentEnv(tmp_path)
    kernel = AgentKernel(
        skill_registry=env.get_skill_registry(),
        subagent_registry=env.get_subagent_registry(),
        mcp_registry=env.get_mcp_registry(),
    )
    session = _make_session(tmp_path, env)
    workdir = tmp_path / "runtime"
    workdir.mkdir(parents=True, exist_ok=True)
    spec = _make_subagent_spec(tmp_path)

    async def _run():
        coordinator = ForkCoordinator(kernel, model_config_resolver=lambda model_type: None)
        return await coordinator.run(spec, session, {"x": 1}, branch_count=0, workdir=workdir)

    results = asyncio.run(_run())
    assert results == []
