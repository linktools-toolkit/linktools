"""Tests for BaseAgent's feature-toggle constructor parameters and the
`_build_feature_capabilities` helper `_build_model_agent` uses to assemble them.

These tests exercise `_build_feature_capabilities` directly rather than the full
`_build_model_agent`/`generate()` path: that path additionally requires a live
model config file, a real Session with a runtime_dir, and an execution_context
built from a preloaded registry -- exactly the heavy fixture set
`_FakeAgentEnv`/`_Registry` in `test_agent_runtime_kernel.py` exist to provide,
but wiring all of that up again here would test far more than this task changed.
`_build_feature_capabilities` is a pure function of `self`'s toggle attributes and
has no such dependencies, which is exactly why it was extracted as its own method.

Construction uses `SubAgent` (not a hand-rolled `BaseAgent` subclass): `BaseAgent`
itself is abstract (`generate`/`stream`/`snapshot` are `@abc.abstractmethod`, verified
against the current code -- `BaseAgent.__abstractmethods__` is
`frozenset({'generate', 'stream', 'snapshot'})`), while `SubAgent` -> `RuntimeAgent` ->
`LlmAgent` already fully implements all three, so it's the cheapest concrete class
that exists specifically for lightweight construction (see its own docstring:
"Lightweight child agent invoked via call_subagent. Recursion stops here.").
"""

from pathlib import Path

from linktools.ai.budget.hook import BudgetCapability
from linktools.ai.agent import SubAgent
from linktools.ai.core.runtime import CapabilityBundle
from linktools.ai.periodic_reminder.capability import PeriodicReminderCapability
from linktools.ai.security.hook import SecurityCapability
from linktools.ai.stuck_loop.capability import StuckLoopCapability
from linktools.ai.tool_search.capability import ToolSearchCapability


class _FakeExecutionContext:
    # `execution_context` is required (no None fallback) on `BaseAgent.__init__`,
    # so any non-None object with a `.kernel` attribute works here.
    kernel = object()
    capabilities = CapabilityBundle(
        builtin_tools=[],
        skills=[],
        subagents=[],
        mcp_servers=[],
        missing_mcp_sources=[],
    )


class _FakeSession:
    session_id = "fake-session-id"
    workspace_root = Path("/tmp/fake-workspace-root")


class _FakeSpec:
    # `agent_id` (used by the task_queue/SwarmCapability toggle) reads
    # `self.spec.name`, so the fake spec needs a `name` attribute -- everything
    # else about `spec` is still untouched by `_build_feature_capabilities`.
    name = "fake-agent-id"


def _construct(**toggles) -> SubAgent:
    # `session` only needs `session_id`/`workspace_root` (both provided by
    # `_FakeSession`), since enable_plan_mode/enable_memory read those off
    # `self.session`. `spec` needs `name` (see `_FakeSpec`).
    return SubAgent(
        spec=_FakeSpec(),
        session=_FakeSession(),
        execution_context=_FakeExecutionContext(),
        model_config_resolver=lambda model_type: None,  # never called by these tests
        **toggles,
    )


def test_construction_no_longer_accepts_environ_kwarg():
    import pytest

    with pytest.raises(TypeError, match="environ"):
        SubAgent(
            environ=object(),
            spec=_FakeSpec(),
            session=_FakeSession(),
            execution_context=_FakeExecutionContext(),
            model_config_resolver=lambda model_type: None,
        )


def test_defaults_produce_only_security_capability():
    agent = _construct()
    capabilities = agent._build_feature_capabilities()
    kinds = {type(c) for c in capabilities}
    assert kinds == {SecurityCapability}, (
        "enable_security_preset defaults to True (the plan's one intentional "
        "default-on toggle); every other toggle defaults off and should add nothing"
    )


def test_security_preset_false_adds_nothing():
    agent = _construct(enable_security_preset=False)
    assert agent._build_feature_capabilities() == []


def test_stuck_loop_toggle():
    agent = _construct(enable_security_preset=False, enable_stuck_loop_detection=True)
    capabilities = agent._build_feature_capabilities()
    assert len(capabilities) == 1
    assert isinstance(capabilities[0], StuckLoopCapability)


def test_periodic_reminders_toggle():
    agent = _construct(enable_security_preset=False, enable_periodic_reminders=True)
    capabilities = agent._build_feature_capabilities()
    assert len(capabilities) == 1
    assert isinstance(capabilities[0], PeriodicReminderCapability)


def test_tool_search_toggle():
    agent = _construct(enable_security_preset=False, enable_tool_search=True)
    capabilities = agent._build_feature_capabilities()
    assert len(capabilities) == 1
    assert isinstance(capabilities[0], ToolSearchCapability)


def test_budget_toggle_only_when_budget_usd_set():
    agent = _construct(enable_security_preset=False, budget_usd=1.5)
    capabilities = agent._build_feature_capabilities()
    assert len(capabilities) == 1
    assert isinstance(capabilities[0], BudgetCapability)
    assert capabilities[0].tracker.budget_usd == 1.5


def test_budget_not_added_when_budget_usd_is_none():
    agent = _construct(enable_security_preset=False, budget_usd=None)
    assert agent._build_feature_capabilities() == []


def test_all_toggles_together_produce_all_capabilities_in_order():
    from linktools.ai.memory.capability import MemoryCapability
    from linktools.ai.plan.capability import PlanCapability
    from linktools.ai.swarm.capability import SwarmCapability
    from linktools.ai.swarm.local import InMemoryTaskQueue

    agent = _construct(
        enable_security_preset=True,
        enable_stuck_loop_detection=True,
        enable_periodic_reminders=True,
        enable_tool_search=True,
        budget_usd=2.0,
        enable_plan_mode=True,
        enable_memory=True,
        task_queue=InMemoryTaskQueue(),
    )
    capabilities = agent._build_feature_capabilities()
    kinds = [type(c) for c in capabilities]
    assert kinds == [
        SecurityCapability,
        StuckLoopCapability,
        PeriodicReminderCapability,
        BudgetCapability,
        PlanCapability,
        MemoryCapability,
        SwarmCapability,
        ToolSearchCapability,
    ]


def test_task_queue_toggle():
    from linktools.ai.swarm.capability import SwarmCapability
    from linktools.ai.swarm.local import InMemoryTaskQueue

    queue = InMemoryTaskQueue()
    agent = _construct(enable_security_preset=False, task_queue=queue)
    capabilities = agent._build_feature_capabilities()
    assert len(capabilities) == 1
    assert isinstance(capabilities[0], SwarmCapability)
    assert capabilities[0].task_queue is queue


def test_inert_toggles_are_accepted_and_stored_without_effect():
    # fallback_models/context_files are accepted per the spec's constructor
    # signature but not yet wired to any capability. enable_checkpointing IS
    # wired, but via _save_call -> _maybe_save_checkpoint, not as a
    # _build_feature_capabilities-appended capability, which is why it still
    # asserts == [] here. enable_plan_mode/enable_memory/task_queue are now
    # wired (see their own dedicated tests) and are deliberately excluded here.
    agent = _construct(
        enable_security_preset=False,
        enable_checkpointing=True,
        checkpoint_store=object(),
        fallback_models=("gpt-4o-mini",),
        context_files=("AGENTS.md",),
    )
    assert agent._build_feature_capabilities() == []
    assert agent.enable_checkpointing is True
    assert agent.fallback_models == ("gpt-4o-mini",)
    assert agent.context_files == ("AGENTS.md",)


def test_plan_mode_toggle():
    from linktools.ai.plan.capability import PlanCapability

    agent = _construct(enable_security_preset=False, enable_plan_mode=True)
    capabilities = agent._build_feature_capabilities()
    assert len(capabilities) == 1
    assert isinstance(capabilities[0], PlanCapability)
    assert capabilities[0].session_id == agent.session.session_id


def test_memory_toggle():
    from linktools.ai.memory.capability import MemoryCapability

    agent = _construct(enable_security_preset=False, enable_memory=True)
    capabilities = agent._build_feature_capabilities()
    assert len(capabilities) == 1
    assert isinstance(capabilities[0], MemoryCapability)
