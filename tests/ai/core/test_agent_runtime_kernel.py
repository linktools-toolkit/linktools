"""Tests for `linktools.ai.core.runtime.AgentKernel` and the `registry`/`session`
building blocks it composes, ported from `sec-smartops-svc`'s
`tests/test_agent_runtime_kernel.py`.

The source test also covered `engine.secops.runtime.AgentRuntime` — a secops-specific
bootstrap (registry preload + hook/observability wiring) that stays in sec-smartops-svc
and has no domain-agnostic equivalent here. Its kernel/session-factory behavior is
exercised instead through `_MinimalAgentRuntime`, a local stand-in built directly from
`AgentKernel` + the same `_Registry` test double, covering the same assertions:
env/kernel wiring, FileSession/RemoteSession defaults, and coordinator sharing across
DB sessions on the same store.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from linktools.ai.core.registry import AgentSpec, SpecSource
from linktools.ai.mcp.registry import MCPServerSpec
from linktools.ai.skill.registry import SkillSpec
from linktools.ai.subagent.registry import SubagentSpec
from linktools.ai.core.runtime import AgentKernel
from linktools.ai.session.types import (
    FileSession,
    FileSessionSpec,
    RemoteSession,
    RemoteSessionSpec,
    Session,
    SessionTranscript,
    SessionTranscriptHead,
)
from linktools.ai.session.local import InMemorySessionStatusStore
from linktools.ai.session.coordination import InMemorySessionCoordinator, coordinator_for_store
from linktools.ai.session.window import NoopSummaryPolicy, RecentWindowPolicy


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
        for spec in self._specs.values():
            if item in getattr(spec, "provides", []):
                return spec
        return None


def test_agent_kernel_takes_registries_directly_not_environ():
    skill_registry = _Registry([])
    subagent_registry = _Registry([])
    mcp_registry = _Registry([])
    kernel = AgentKernel(
        skill_registry=skill_registry,
        subagent_registry=subagent_registry,
        mcp_registry=mcp_registry,
    )
    assert kernel._skill_registry is skill_registry
    assert kernel._subagent_registry is subagent_registry
    assert kernel._mcp_registry is mcp_registry


class _FakeTranscriptStore:
    async def head(self, session_id: str):
        del session_id
        return SessionTranscriptHead()

    async def load(self, session_id: str, *, budget_tokens: int, after_seq: int | None = None, batch_size: int = 64):
        del session_id, budget_tokens, after_seq, batch_size
        return SessionTranscript(head=SessionTranscriptHead())

    async def save(self, transcript):
        del transcript


class _FakeAgentEnv:
    """Minimal environment test double: registries + `hooks`/`trace_root`/`workspace_root`
    (the now-deleted `AgentEnvironment` class this stood in for)."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.hooks = None
        self._stage = _Registry([])
        self._skill = _Registry([])
        self._subagent = _Registry([])
        self._mcp = _Registry([])
        self.trace_store_calls = 0

    def get_stage_registry(self, cap_store=None):
        del cap_store
        return self._stage

    def get_skill_registry(self, cap_store=None):
        del cap_store
        return self._skill

    def get_subagent_registry(self, cap_store=None):
        del cap_store
        return self._subagent

    def get_mcp_registry(self, cap_store=None):
        del cap_store
        return self._mcp

    def with_registries(self, stage, skill, subagent, mcp):
        env = _FakeAgentEnv(self.workspace_root)
        env._stage = stage
        env._skill = skill
        env._subagent = subagent
        env._mcp = mcp
        return env

    def trace_store(self):
        self.trace_store_calls += 1
        return object()


class _MinimalAgentRuntime:
    """Local stand-in for `engine.secops.runtime.AgentRuntime` (secops-specific, stays in
    sec-smartops-svc): preload registries onto a frozen env copy, build an `AgentKernel`
    over it, and expose the same `session`/`db_session` factories used by the ported
    assertions below."""

    def __init__(self, base_env: _FakeAgentEnv, *, capability_store=None) -> None:
        self._base_env = base_env
        self._cs = capability_store
        self._env: _FakeAgentEnv | None = None
        self._kernel: AgentKernel | None = None

    async def prepare(self, *, enable_trace_observer: bool = True):
        if self._env is not None:
            return self._env
        cs = self._cs
        stage = self._base_env.get_stage_registry(cap_store=cs)
        skill = self._base_env.get_skill_registry(cap_store=cs)
        subagent = self._base_env.get_subagent_registry(cap_store=cs)
        mcp = self._base_env.get_mcp_registry(cap_store=cs)
        await asyncio.gather(stage.preload(), skill.preload(), subagent.preload(), mcp.preload())
        env = self._base_env.with_registries(stage, skill, subagent, mcp)
        if enable_trace_observer:
            env.trace_store()
        self._kernel = AgentKernel(
            skill_registry=env.get_skill_registry(),
            subagent_registry=env.get_subagent_registry(),
            mcp_registry=env.get_mcp_registry(),
        )
        self._env = env
        return env

    @property
    def env(self) -> _FakeAgentEnv:
        if self._env is None:
            raise RuntimeError("AgentRuntime.prepare() must be awaited before use")
        return self._env

    @property
    def kernel(self) -> AgentKernel:
        if self._kernel is None:
            raise RuntimeError("AgentRuntime.prepare() must be awaited before use")
        return self._kernel

    def session(self, trace_id: str, session_id: str = "main") -> Session:
        return Session.create(
            self.env.workspace_root / trace_id,
            FileSessionSpec(session_id=session_id),
        )

    def db_session(self, session_id: str, store) -> Session:
        spec = RemoteSessionSpec(
            session_id=session_id,
            store=store,
            coordination=coordinator_for_store(store),
        )
        return Session.create_db(spec)


def _make_session(tmp_path: Path):
    return FileSession(
        session_id="sess",
        root=tmp_path / "TRC" / "session" / "sess",
        status_store=InMemorySessionStatusStore(),
    )


def test_agent_kernel_resolves_capability_bundle(tmp_path):
    env = SimpleNamespace(
        get_skill_registry=lambda: _Registry([
            SkillSpec(
                name="triage",
                path=tmp_path / "skills" / "triage" / "SKILL.md",
                base_dir=None,
                enabled=True,
            )
        ]),
        get_subagent_registry=lambda: _Registry([
            SubagentSpec(
                name="child",
                path=tmp_path / "subagents" / "child" / "agent.md",
                base_dir=None,
                enabled=True,
            )
        ]),
        get_mcp_registry=lambda: _Registry([
            MCPServerSpec(
                name="search",
                path=tmp_path / "adapter" / "search" / "mcp.yaml",
                base_dir=None,
                enabled=True,
                server_name="search",
            ),
            MCPServerSpec(
                name="intel",
                path=tmp_path / "adapter" / "intel" / "mcp.yaml",
                base_dir=None,
                enabled=True,
                server_name="intel",
                provides=["threat-intel"],
            ),
        ]),
        hooks=None,
    )
    spec = AgentSpec(
        name="worker",
        path=tmp_path / "agent" / "worker" / "agent.md",
        base_dir=None,
        enabled=True,
        model="standard",
        allowed_tools=["file", "terminal", "search", "threat-intel", "missing-mcp"],
        allowed_skills=["triage", "missing-skill"],
        allowed_subagents=["child", "missing-subagent"],
    )

    context = AgentKernel(
        skill_registry=env.get_skill_registry(),
        subagent_registry=env.get_subagent_registry(),
        mcp_registry=env.get_mcp_registry(),
    ).build_context(
        spec,
        _make_session(tmp_path),
        builtin_tool_names=frozenset({"file", "terminal"}),
    )

    assert context.capabilities.builtin_tools == ["file", "terminal"]
    assert [skill.name for skill in context.capabilities.skills] == ["triage"]
    assert [subagent.name for subagent in context.capabilities.subagents] == ["child"]
    assert [server.name for server in context.capabilities.mcp_servers] == ["search", "intel"]
    assert context.capabilities.missing_mcp_sources == ["missing-mcp"]


def test_mcp_server_spec_description_falls_back_to_display_name(tmp_path):
    spec = MCPServerSpec.from_dict(
        {
            "display_name": "HR Mock MCP 服务",
            "mcp": {"type": "stdio", "command": "python3", "args": ["script.py"]},
            "provides": ["hr_profile"],
        },
        SpecSource(
            name="hr",
            path=tmp_path / "adapter" / "hr" / "mcp.yaml",
            base_dir=tmp_path / "adapter" / "hr",
        ),
    )

    assert spec.description == "HR Mock MCP 服务"
    assert spec.provides == ["hr_profile"]


def test_agent_runtime_prepare_exposes_kernel_and_session_defaults(tmp_path: Path):
    runtime = _MinimalAgentRuntime(_FakeAgentEnv(tmp_path), capability_store=None)

    prepared_env = asyncio.run(runtime.prepare())
    file_session = runtime.session("trace-1", "main")
    session = runtime.db_session("chat-1", _FakeTranscriptStore())

    assert runtime.env is prepared_env
    assert runtime.kernel._skill_registry is prepared_env.get_skill_registry()
    assert runtime.kernel._subagent_registry is prepared_env.get_subagent_registry()
    assert runtime.kernel._mcp_registry is prepared_env.get_mcp_registry()
    assert isinstance(file_session.coordination, InMemorySessionCoordinator)
    assert isinstance(file_session.window_policy, RecentWindowPolicy)
    assert isinstance(session, RemoteSession)
    assert isinstance(session.window_policy, RecentWindowPolicy)
    assert isinstance(session.summary_policy, NoopSummaryPolicy)


def test_agent_runtime_db_sessions_share_store_coordinator(tmp_path: Path):
    runtime = _MinimalAgentRuntime(_FakeAgentEnv(tmp_path), capability_store=None)
    asyncio.run(runtime.prepare())
    store = _FakeTranscriptStore()

    first = runtime.db_session("chat-1", store)
    second = runtime.db_session("chat-2", store)

    assert first.coordination is second.coordination


def test_agent_runtime_prepare_can_skip_trace_observer(tmp_path: Path):
    env = _FakeAgentEnv(tmp_path)
    runtime = _MinimalAgentRuntime(env, capability_store=None)

    asyncio.run(runtime.prepare(enable_trace_observer=False))

    assert env.trace_store_calls == 0
