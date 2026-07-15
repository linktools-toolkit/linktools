#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified subagent resolution tests (spec §9/§13/§14)."""

import tempfile
import unittest
from pathlib import Path

from linktools.ai.errors import RegistryNotFoundError, SubagentResolutionError
from linktools.ai.skill.private import ActiveSkillContext
from linktools.ai.subagent.skill_resolver import (
    CallSubagentInput,
    SkillSubagentProvider,
    UnifiedSubagentResolver,
)
from linktools.commands.ai.skill_index import SkillInfo


def _skill_info(skill_root: Path, revision: str = "r1") -> SkillInfo:
    from linktools.ai.registry.skill import SkillSpec

    return SkillInfo(
        id="skill-creator",
        root=skill_root,
        revision=revision,
        spec=SkillSpec(id="skill-creator", name="skill-creator"),
    )


class _FakeSkillIndex:
    def __init__(self, info):
        self._info = info

    async def get(self, skill_id):
        return self._info if skill_id == self._info.id else None


class _FakeProjectAgents:
    def __init__(self, specs, *, raise_not_found=False):
        self._specs = specs
        self._raise = raise_not_found

    async def get(self, name):
        if name in self._specs:
            return self._specs[name]
        if self._raise:
            raise RegistryNotFoundError(name)
        return None


class TestCallSubagentInput(unittest.TestCase):
    def test_validate_requires_exactly_one_id(self):
        CallSubagentInput(task="t", name="r").validate()
        CallSubagentInput(task="t", instruction_path="agents/g.md").validate()
        with self.assertRaises(SubagentResolutionError):
            CallSubagentInput(task="t").validate()
        with self.assertRaises(SubagentResolutionError):
            CallSubagentInput(
                task="t", name="r", instruction_path="agents/g.md"
            ).validate()

    def test_blank_task_rejected(self):
        with self.assertRaises(SubagentResolutionError):
            CallSubagentInput(task="  ", name="r").validate()


class TestSkillSubagentProvider(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.skill_root = Path(self._tmp.name) / "skill-creator"
        (self.skill_root / "agents").mkdir(parents=True)
        (self.skill_root / "agents" / "grader.md").write_text(
            "# Grader\nGrade assertions.\n", "utf-8"
        )

    def tearDown(self):
        self._tmp.cleanup()

    async def test_resolve_happy_path(self):
        provider = SkillSubagentProvider(
            skills=_FakeSkillIndex(_skill_info(self.skill_root)),
            default_timeout_seconds=120,
        )
        active = ActiveSkillContext(
            skill_id="skill-creator", skill_root=self.skill_root, revision="r1"
        )
        spec = await provider.resolve(
            active_skill=active, instruction_path="agents/grader.md"
        )
        self.assertEqual(spec.skill_id, "skill-creator")
        self.assertEqual(spec.name, "grader")
        self.assertEqual(spec.instruction_path, "agents/grader.md")

    async def test_resolve_revision_mismatch_rejected(self):
        provider = SkillSubagentProvider(
            skills=_FakeSkillIndex(_skill_info(self.skill_root, revision="r2")),
            default_timeout_seconds=120,
        )
        active = ActiveSkillContext(
            skill_id="skill-creator", skill_root=self.skill_root, revision="r1"
        )
        with self.assertRaises(SubagentResolutionError):
            await provider.resolve(
                active_skill=active, instruction_path="agents/grader.md"
            )

    async def test_resolve_missing_skill_rejected(self):
        # The index knows "skill-creator"; querying a different active skill
        # yields None, which the provider must reject.
        provider = SkillSubagentProvider(
            skills=_FakeSkillIndex(_skill_info(self.skill_root)),
            default_timeout_seconds=120,
        )
        active = ActiveSkillContext(
            skill_id="other-skill", skill_root=self.skill_root, revision="r1"
        )
        with self.assertRaises(SubagentResolutionError):
            await provider.resolve(
                active_skill=active, instruction_path="agents/grader.md"
            )


class TestUnifiedSubagentResolver(unittest.IsolatedAsyncioTestCase):
    async def test_name_branch_uses_project_agents(self):
        resolver = UnifiedSubagentResolver(
            project_agents=_FakeProjectAgents({"reviewer": "REVIEWER_SPEC"}),
            skill_agents=SkillSubagentProvider(
                skills=_FakeSkillIndex(None), default_timeout_seconds=120
            ),
        )
        result = await resolver.resolve(
            request=CallSubagentInput(task="review it", name="reviewer"),
            active_skill=None,
        )
        self.assertEqual(result, "REVIEWER_SPEC")

    async def test_unknown_name_rejected(self):
        resolver = UnifiedSubagentResolver(
            project_agents=_FakeProjectAgents({}, raise_not_found=True),
            skill_agents=SkillSubagentProvider(
                skills=_FakeSkillIndex(None), default_timeout_seconds=120
            ),
        )
        with self.assertRaises(SubagentResolutionError):
            await resolver.resolve(
                request=CallSubagentInput(task="t", name="ghost"),
                active_skill=None,
            )

    async def test_instruction_path_requires_active_skill(self):
        resolver = UnifiedSubagentResolver(
            project_agents=_FakeProjectAgents({}),
            skill_agents=SkillSubagentProvider(
                skills=_FakeSkillIndex(None), default_timeout_seconds=120
            ),
        )
        with self.assertRaises(SubagentResolutionError):
            await resolver.resolve(
                request=CallSubagentInput(task="t", instruction_path="agents/g.md"),
                active_skill=None,
            )


if __name__ == "__main__":
    unittest.main()
