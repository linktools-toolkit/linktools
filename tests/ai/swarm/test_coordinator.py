import asyncio
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from linktools.ai.core.model_runtime import ModelBundle, RuntimeModelConfig
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


def _make_file_writing_subagent_spec(tmp_path: Path) -> SubagentSpec:
    agent_dir = tmp_path / "subagents" / "writer"
    agent_dir.mkdir(parents=True)
    return SubagentSpec.from_dict(
        {
            "description": "writes a marker file",
            "system": "write a marker file",
            "allowed_tools": ["file"],
        },
        SpecSource(name="writer", path=agent_dir / "agent.md", base_dir=agent_dir),
    )


def _drive_single_write_file_call(path: str, content: str):
    """FunctionModel that calls the `write_file` builtin tool once, then satisfies
    SubAgent's structured (dict[str, Any]) output requirement by calling whatever
    output tool pydantic-ai registered for it (info.output_tools), same two-step
    shape as tests/ai/plan/test_capability.py's _drive_single_tool_call."""
    call_state = {"done": False}

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if not call_state["done"]:
            call_state["done"] = True
            return ModelResponse(parts=[ToolCallPart(tool_name="write_file", args={"path": path, "content": content})])
        output_tool_name = info.output_tools[0].name if info.output_tools else None
        if output_tool_name:
            return ModelResponse(parts=[ToolCallPart(tool_name=output_tool_name, args={"response": {"result": "done"}})])
        return ModelResponse(parts=[TextPart('{"result": "done"}')])

    return model_fn


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
        return await coordinator.run(spec, session, agent_count=2, workdir=tmp_path / "runtime")

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
        return await coordinator.run(spec, session, agent_count=3, workdir=tmp_path / "runtime")

    results = asyncio.run(_run())
    assert results == []


def test_swarm_coordinator_workers_write_files_under_the_given_workdir(tmp_path, monkeypatch):
    # Regression test: SwarmCoordinator.run() must thread its `workdir` param into
    # each worker's SubAgent, otherwise RuntimeAgent.__init__ defaults workdir to
    # Path.cwd() and file-tool calls silently land outside the caller-controlled
    # directory. build_model() normally constructs a real OpenAIChatModel, so it's
    # monkeypatched (at the `linktools.ai.agent` import site) to return a
    # FunctionModel-backed bundle that drives one write_file tool call.
    import linktools.ai.agent as agent_module

    def _fake_build_model(config) -> ModelBundle:
        return ModelBundle(
            config=config,
            model=FunctionModel(_drive_single_write_file_call("marker.txt", "hello from swarm")),
            settings=ModelSettings(max_tokens=4096),
            usage_limits=UsageLimits(request_limit=5),
        )

    monkeypatch.setattr(agent_module, "build_model", _fake_build_model)

    env = _FakeAgentEnv(tmp_path)
    kernel = AgentKernel(
        skill_registry=env.get_skill_registry(),
        subagent_registry=env.get_subagent_registry(),
        mcp_registry=env.get_mcp_registry(),
    )
    session = _make_session(tmp_path, env)
    spec = _make_file_writing_subagent_spec(tmp_path)
    queue = InMemoryTaskQueue()
    workdir = tmp_path / "swarm_run"

    fake_config = RuntimeModelConfig(
        model_type="standard", protocol="openai", model="fake", base_url=None,
        api_key=None, auth_token=None, timeout_seconds=300, raw={"max_retries": 1},
    )

    async def _run():
        await queue.add([Task(task_id="t1", payload={"x": 1})])
        coordinator = SwarmCoordinator(kernel, queue, model_config_resolver=lambda model_type: fake_config)
        return await coordinator.run(spec, session, agent_count=1, workdir=workdir)

    results = asyncio.run(_run())
    assert len(results) == 1
    assert results[0].status == "done"
    assert (workdir / "marker.txt").read_text() == "hello from swarm"
    assert not (tmp_path / "marker.txt").exists()
