#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live skill-private subagent routing tests (spec §9/§10/§15).

Exercises the wired chain end-to-end at the tool layer: ``read_skill`` sets the
active-skill contextvar; ``call_subagent(instruction_path=...)`` resolves through
the UnifiedSubagentResolver against the active skill, builds a child AgentSpec
with the permission intersection, and runs it via the executor. No live model."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from linktools.ai.agent.spec import ToolRef
from linktools.ai.errors import SubagentExecutionError, SubagentResolutionError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.skill.private import (
    ActiveSkillContext,
    SkillSubagentSpec,
    get_active_skill,
    reset_active_skill,
    set_active_skill,
    skill_subagent_to_agent_spec,
)
from linktools.ai.skill.toolset import build_skill_toolset
from linktools.ai.subagent.skill_resolver import (
    SkillSubagentProvider,
    UnifiedSubagentResolver,
)
from linktools.ai.subagent.toolset import build_subagent_toolset
from linktools.ai_cli.skill_index import DirectorySkillIndex


def _make_skill(tmp, name="skill-creator") -> Path:
    root = Path(tmp) / name
    (root / "agents").mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: skill-creator\ndescription: x\n---\nbody\n", "utf-8"
    )
    (root / "agents" / "grader.md").write_text("# Grader\nGrade assertions.\n", "utf-8")
    return root


class _FakeExecutor:
    def __init__(self):
        self.specs = []
        self.last_task = None

    async def execute(
        self, *, agent_spec, task, context, parent, scope, timeout_seconds
    ):
        self.specs.append(agent_spec)
        self.last_task = task

        class _Result:
            def model_dump(self):
                return {"status": "succeeded", "output": "ok"}

        return _Result()


class _DummyParent:
    pass


class _NoneAgents:
    async def get(self, name):
        return None


class TestCallSubagentInstructionPath(unittest.IsolatedAsyncioTestCase):
    async def test_routes_through_resolver_and_runs_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(tmp)
            index = DirectorySkillIndex(Path(tmp))
            info = await index.get("skill-creator")
            token = set_active_skill(
                ActiveSkillContext(
                    skill_id="skill-creator",
                    skill_root=info.root,
                    revision=info.revision,
                )
            )
            try:
                resolver = UnifiedSubagentResolver(
                    project_agents=_NoneAgents(),
                    skill_agents=SkillSubagentProvider(
                        skills=index, default_timeout_seconds=120
                    ),
                )
                executor = _FakeExecutor()
                toolset = build_subagent_toolset(
                    allowed_names=set(),
                    subagent_provider=None,
                    entrypoint_resolver=None,
                    executor=executor,
                    depth_provider=lambda: 0,
                    max_depth=3,
                    timeout_seconds=120.0,
                    parent=_DummyParent(),
                    skill_resolver=resolver,
                    active_skill_provider=get_active_skill,
                    child_model_policy=ModelPolicy(primary="standard"),
                )
                call = toolset.tools["call_subagent"].function
                result = await call(
                    instruction_path="agents/grader.md", task="grade it"
                )
                self.assertEqual(result["status"], "succeeded")
                self.assertEqual(len(executor.specs), 1)
                self.assertEqual(executor.specs[0].id, "skill-creator/agents/grader.md")
                self.assertEqual(executor.last_task, "grade it")
            finally:
                reset_active_skill(token)

    async def test_instruction_path_without_resolver_raises(self):
        toolset = build_subagent_toolset(
            allowed_names=set(),
            subagent_provider=None,
            entrypoint_resolver=None,
            executor=_FakeExecutor(),
            depth_provider=lambda: 0,
            max_depth=3,
            timeout_seconds=120.0,
            parent=_DummyParent(),
        )
        call = toolset.tools["call_subagent"].function
        with self.assertRaises(SubagentExecutionError):
            await call(instruction_path="agents/grader.md", task="t")

    async def test_instruction_path_without_active_skill_raises(self):
        token = set_active_skill(None)
        try:
            resolver = UnifiedSubagentResolver(
                project_agents=_NoneAgents(),
                skill_agents=SkillSubagentProvider(
                    skills=DirectorySkillIndex(Path("/nonexistent")),
                    default_timeout_seconds=120,
                ),
            )
            toolset = build_subagent_toolset(
                allowed_names=set(),
                subagent_provider=None,
                entrypoint_resolver=None,
                executor=_FakeExecutor(),
                depth_provider=lambda: 0,
                max_depth=3,
                timeout_seconds=120.0,
                parent=_DummyParent(),
                skill_resolver=resolver,
                active_skill_provider=get_active_skill,
            )
            call = toolset.tools["call_subagent"].function
            with self.assertRaises(SubagentResolutionError):
                await call(instruction_path="agents/grader.md", task="t")
        finally:
            reset_active_skill(token)


class TestPermissionIntersection(unittest.TestCase):
    """A skill-private agent keeps only the tools the parent can delegate (§15)."""

    def _spec(self, tools):
        return SkillSubagentSpec(
            skill_id="skill-creator",
            instruction_path="agents/grader.md",
            name="grader",
            description=None,
            instructions="grade",
            requested_tools=tools,
            timeout_seconds=120,
            max_depth=0,
            fingerprint="abc",
        )

    def test_requested_tools_filtered_to_parent_delegated(self):
        spec = self._spec(
            (
                ToolRef(kind="builtin", name="file-read"),
                ToolRef(kind="builtin", name="terminal"),
                ToolRef(kind="builtin", name="file-write"),
            )
        )
        agent = skill_subagent_to_agent_spec(
            spec,
            model_policy=ModelPolicy(primary="standard"),
            parent_delegated={"file-read"},
        )
        # terminal + file-write dropped: the parent cannot delegate them.
        self.assertEqual([t.name for t in agent.tools], ["file-read"])

    def test_no_constraint_keeps_all_requested(self):
        spec = self._spec((ToolRef(kind="builtin", name="file-read"),))
        agent = skill_subagent_to_agent_spec(
            spec, model_policy=ModelPolicy(primary="standard")
        )
        self.assertEqual([t.name for t in agent.tools], ["file-read"])


class TestReadSkillActivatesSkill(unittest.IsolatedAsyncioTestCase):
    async def test_read_skill_sets_active_skill_contextvar(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(tmp)
            index = DirectorySkillIndex(Path(tmp))
            from linktools.ai.registry.skill import SkillSpec

            class _Prov:
                async def list_ids(self):
                    return ("skill-creator",)

                async def get(self, sid):
                    return SkillSpec(id=sid, name=sid)

            async def lookup(sid):
                i = await index.get(sid)
                return (
                    ActiveSkillContext(
                        skill_id=i.id, skill_root=i.root, revision=i.revision
                    )
                    if i
                    else None
                )

            token = set_active_skill(None)
            try:
                toolset = build_skill_toolset(
                    _Prov(), authorized={"skill-creator"}, active_skill_lookup=lookup
                )
                read = toolset.tools["read_skill"].function
                await read("skill-creator")
                active = get_active_skill()
                self.assertIsNotNone(active)
                self.assertEqual(active.skill_id, "skill-creator")
            finally:
                reset_active_skill(token)


class TestLivePermissionIntersection(unittest.IsolatedAsyncioTestCase):
    """The live call_subagent applies the parent-delegated intersection: a
    skill-private agent that requests a tool the parent lacks drops it before
    the child run executes."""

    async def test_child_keeps_only_parent_delegated_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A grader that requests BOTH file-read and terminal.
            root = Path(tmp) / "skill-creator"
            (root / "agents").mkdir(parents=True)
            (root / "SKILL.md").write_text(
                "---\nname: skill-creator\ndescription: x\n---\nbody\n", "utf-8"
            )
            (root / "agents" / "grader.md").write_text(
                "---\n"
                "name: grader\n"
                "tools:\n"
                "  - kind: builtin\n"
                "    name: file-read\n"
                "  - kind: builtin\n"
                "    name: terminal\n"
                "---\n"
                "Grade it.\n",
                "utf-8",
            )
            index = DirectorySkillIndex(Path(tmp))
            info = await index.get("skill-creator")
            token = set_active_skill(
                ActiveSkillContext(
                    skill_id="skill-creator",
                    skill_root=info.root,
                    revision=info.revision,
                )
            )
            try:
                resolver = UnifiedSubagentResolver(
                    project_agents=_NoneAgents(),
                    skill_agents=SkillSubagentProvider(
                        skills=index, default_timeout_seconds=120
                    ),
                )
                executor = _FakeExecutor()
                # Parent delegates ONLY file-read -> terminal must be dropped.
                toolset = build_subagent_toolset(
                    allowed_names=set(),
                    subagent_provider=None,
                    entrypoint_resolver=None,
                    executor=executor,
                    depth_provider=lambda: 0,
                    max_depth=3,
                    timeout_seconds=120.0,
                    parent=_DummyParent(),
                    skill_resolver=resolver,
                    active_skill_provider=get_active_skill,
                    child_model_policy=ModelPolicy(primary="standard"),
                    parent_delegated_tools={"file-read"},
                )
                call = toolset.tools["call_subagent"].function
                await call(instruction_path="agents/grader.md", task="grade")
                self.assertEqual(len(executor.specs), 1)
                self.assertEqual(
                    [t.name for t in executor.specs[0].tools], ["file-read"]
                )
            finally:
                reset_active_skill(token)


class TestReadThenCallInSameTask(unittest.IsolatedAsyncioTestCase):
    """The realistic one-turn sequence: read_skill activates the skill, then a
    call_subagent(instruction_path=...) in the SAME task resolves under it and
    runs the child. This is the skill-creator acceptance path at the tool layer
    (a full Runtime.run + fake-model E2E is a heavier follow-up)."""

    async def test_read_then_call_chains_through_contextvar(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(tmp)
            index = DirectorySkillIndex(Path(tmp))
            from linktools.ai.registry.skill import SkillSpec

            class _SkillProv:
                async def list_ids(self):
                    return ("skill-creator",)

                async def get(self, sid):
                    return SkillSpec(id=sid, name=sid)

            async def lookup(sid):
                i = await index.get(sid)
                return (
                    ActiveSkillContext(
                        skill_id=i.id, skill_root=i.root, revision=i.revision
                    )
                    if i
                    else None
                )

            skill_toolset = build_skill_toolset(
                _SkillProv(),
                authorized={"skill-creator"},
                active_skill_lookup=lookup,
            )
            read = skill_toolset.tools["read_skill"].function

            resolver = UnifiedSubagentResolver(
                project_agents=_NoneAgents(),
                skill_agents=SkillSubagentProvider(
                    skills=index, default_timeout_seconds=120
                ),
            )
            executor = _FakeExecutor()
            sub_toolset = build_subagent_toolset(
                allowed_names=set(),
                subagent_provider=None,
                entrypoint_resolver=None,
                executor=executor,
                depth_provider=lambda: 0,
                max_depth=3,
                timeout_seconds=120.0,
                parent=_DummyParent(),
                skill_resolver=resolver,
                active_skill_provider=get_active_skill,
                child_model_policy=ModelPolicy(primary="standard"),
            )
            call = sub_toolset.tools["call_subagent"].function

            token = set_active_skill(None)
            try:
                await read("skill-creator")  # activates the skill in this task
                result = await call(
                    instruction_path="agents/grader.md", task="grade"
                )  # resolves under the active skill
            finally:
                reset_active_skill(token)
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(executor.specs[0].id, "skill-creator/agents/grader.md")


class TestSubagentProviderDerivesParentDelegated(unittest.IsolatedAsyncioTestCase):
    """The production path: SubagentProvider.resolve derives the parent's
    delegatable tools from the parent agent's own spec (via context.agent_id),
    and FAILS CLOSED (empty set, not None) when the parent can't be read."""

    def _fake_build(self, captured):
        from pydantic_ai.toolsets import FunctionToolset

        def _build(**kwargs):
            captured.update(kwargs)
            ts = FunctionToolset()

            async def call_subagent(**kw):
                return {}

            ts.add_function(call_subagent)
            return ts

        return _build

    async def _resolve(self, agents, agent_id, ref_name):
        from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
        from linktools.ai.capability.models import CapabilityRef
        from linktools.ai.capability.provider import CapabilityContext
        from linktools.ai.subagent.provider import SubagentProvider

        captured: dict = {}
        provider = SubagentProvider(subagent_provider=agents, executor=object())
        ref = CapabilityRef(kind="subagent", name=ref_name)
        ctx = CapabilityContext(
            agent_id=agent_id, exposure_policy=CapabilityToolExposurePolicy()
        )
        with mock.patch(
            "linktools.ai.subagent.provider.build_subagent_toolset",
            self._fake_build(captured),
        ):
            await provider.resolve(ref, ctx)
        return captured

    async def test_derives_from_parent_spec_tools(self):
        from linktools.ai.agent.spec import AgentSpec, PromptSpec
        from linktools.ai.model.policy import ModelPolicy

        parent_spec = AgentSpec(
            id="parent",
            name="parent",
            model=ModelPolicy(primary="standard"),
            instructions=PromptSpec(instructions="x"),
            tools=(
                ToolRef(kind="builtin", name="file-read"),
                ToolRef(kind="builtin", name="terminal"),
            ),
        )

        class _Agents:
            async def get(self, name):
                if name == "parent":
                    return parent_spec
                raise KeyError(name)

            async def list_ids(self):
                return ("parent",)

        captured = await self._resolve(_Agents(), "parent", "parent")
        self.assertEqual(captured["parent_delegated_tools"], {"file-read", "terminal"})

    async def test_fails_closed_when_parent_unknown(self):
        class _Agents:
            async def get(self, name):
                raise KeyError(name)

            async def list_ids(self):
                return ()

        captured = await self._resolve(_Agents(), "ghost", "*")
        # Unknown parent -> fail closed: empty set, NOT None (no constraint).
        self.assertEqual(captured["parent_delegated_tools"], set())


if __name__ == "__main__":
    unittest.main()
