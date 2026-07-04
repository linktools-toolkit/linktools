"""Tests for the enable_checkpointing/checkpoint_store wiring, via the extracted
BaseAgent._maybe_save_checkpoint() method (called by _save_call after
self.session.persist(...) -- see Task 5's "Design note" in the plan this was
written from for why testing this method directly, rather than the full
_save_call/generate() path, is deliberate and sufficient).
"""

import asyncio
from pathlib import Path

from linktools.ai.checkpoint.local import FileCheckpointStore
from linktools.ai.agent import SubAgent
from linktools.ai.core.registry import AgentSpec, SpecSource
from linktools.ai.core.runtime import CapabilityBundle
from linktools.ai.session.coordination import InMemorySessionCoordinator
from linktools.ai.session.local import InMemorySessionStatusStore
from linktools.ai.session.types import FileSession, FileSessionSpec


class _FakeExecutionContext:
    # Must be non-None: BaseAgent.__init__'s fallback branch triggers whenever
    # `execution_context.kernel is None` (not merely when `execution_context`
    # itself is falsy), constructing a real `AgentKernel(environ.get_skill_registry(),
    # ...)` that would call methods `_FakeAgentEnv` below doesn't have.
    kernel = object()
    context: "dict" = {}
    capabilities = CapabilityBundle(builtin_tools=[], skills=[], subagents=[], mcp_servers=[], missing_mcp_sources=[])


class _FakeAgentEnv:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.hooks = None

    def get_logger(self, name):
        import logging
        return logging.getLogger(name)


def _make_agent(tmp_path: Path, **toggles) -> SubAgent:
    """Constructs a real FileSession + SubAgent, and pre-populates root/context.json
    the way self.session.persist(...) would have by the time _maybe_save_checkpoint runs
    -- this test intentionally doesn't call persist() itself, since that needs a real
    SessionTurn/RuntimeModelConfig this task has no reason to construct."""
    env = _FakeAgentEnv(tmp_path)
    spec_kwargs = FileSessionSpec(session_id="s1", coordination=InMemorySessionCoordinator(), status_store=InMemorySessionStatusStore())
    session = FileSession.create(env.workspace_root / "s1", spec_kwargs)
    session.root.mkdir(parents=True, exist_ok=True)
    (session.root / "context.json").write_bytes(b'{"messages": []}')
    spec = AgentSpec.from_dict({"description": "test"}, SpecSource(name="a1", path=tmp_path / "agent.md", base_dir=tmp_path))
    return SubAgent(
        spec=spec,
        session=session,
        execution_context=_FakeExecutionContext(),
        **toggles,
    )


def test_no_checkpoint_saved_when_disabled(tmp_path):
    agent = _make_agent(tmp_path, enable_checkpointing=False)
    asyncio.run(agent._maybe_save_checkpoint())
    assert not (tmp_path / "checkpoints").exists()


def test_checkpoint_saved_when_enabled(tmp_path):
    agent = _make_agent(tmp_path, enable_checkpointing=True)
    asyncio.run(agent._maybe_save_checkpoint())
    checkpoints_root = agent.session.root / "checkpoints"
    assert checkpoints_root.exists()
    saved = list(checkpoints_root.rglob("*.bin"))
    assert len(saved) == 1


def test_checkpoint_content_matches_context_json(tmp_path):
    agent = _make_agent(tmp_path, enable_checkpointing=True)
    asyncio.run(agent._maybe_save_checkpoint())
    context_json = (agent.session.root / "context.json").read_bytes()
    checkpoints_root = agent.session.root / "checkpoints"
    saved_file = next(checkpoints_root.rglob("*.bin"))
    assert saved_file.read_bytes() == context_json


def test_custom_checkpoint_store_is_used(tmp_path):
    custom_root = tmp_path / "custom-checkpoints"
    custom_store = FileCheckpointStore(root=custom_root)
    agent = _make_agent(tmp_path, enable_checkpointing=True, checkpoint_store=custom_store)
    asyncio.run(agent._maybe_save_checkpoint())
    assert not (agent.session.root / "checkpoints").exists()
    assert list(custom_root.rglob("*.bin"))


def test_repeated_calls_increment_seq(tmp_path):
    agent = _make_agent(tmp_path, enable_checkpointing=True)
    asyncio.run(agent._maybe_save_checkpoint())
    asyncio.run(agent._maybe_save_checkpoint())
    saved = list((agent.session.root / "checkpoints").rglob("*.bin"))
    assert {p.stem for p in saved} == {"1", "2"}
