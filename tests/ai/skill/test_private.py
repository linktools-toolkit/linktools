#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill-private subagent core tests.

Covers the security-critical path resolver (escape/symlink rejection), the
parser (frontmatter defaults), and the call-subagent request validation. These
are the primitives the UnifiedSubagentResolver composes."""

import os
import tempfile
import unittest
from pathlib import Path

from linktools.ai.errors import SkillAssetAccessError, SubagentResolutionError
from linktools.ai.skill.private import (
    ActiveSkillContext,
    identity,
    parse_skill_subagent,
    resolve_skill_agent_path,
    validate_call_request,
)


def _make_skill(tmp: str, name: str = "skill-creator") -> Path:
    root = Path(tmp) / name
    (root / "agents").mkdir(parents=True)
    return root


class TestResolveSkillAgentPath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.skill_root = _make_skill(self._tmp.name)
        (self.skill_root / "agents" / "grader.md").write_text("# Grader\n", "utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _resolve(self, instruction_path):
        return resolve_skill_agent_path(
            skill_root=self.skill_root, instruction_path=instruction_path
        )

    def test_valid_path_under_agents(self):
        resolved = self._resolve("agents/grader.md")
        self.assertTrue(resolved.is_file())
        self.assertEqual(resolved.name, "grader.md")

    def test_absolute_path_rejected(self):
        with self.assertRaises(SkillAssetAccessError):
            self._resolve("/etc/passwd")

    def test_non_agents_prefix_rejected(self):
        with self.assertRaises(SkillAssetAccessError):
            self._resolve("references/x.md")

    def test_parent_escape_rejected(self):
        with self.assertRaises(SkillAssetAccessError):
            self._resolve("../agents/x.md")

    def test_non_markdown_rejected(self):
        (self.skill_root / "agents" / "notes.txt").write_text("x", "utf-8")
        with self.assertRaises(SkillAssetAccessError):
            self._resolve("agents/notes.txt")

    def test_missing_file_rejected(self):
        with self.assertRaises(SkillAssetAccessError):
            self._resolve("agents/ghost.md")

    def test_symlink_escape_rejected(self):
        # A symlink inside agents/ that points outside must be rejected after
        # resolve() -- the boundary check catches the resolved target.
        outside = self.skill_root.parent / "outside.md"
        outside.write_text("secret", "utf-8")
        link = self.skill_root / "agents" / "link-to-outside.md"
        try:
            os.symlink(outside, link)
        except OSError:
            self.skipTest("symlink creation not supported on this fs")
        with self.assertRaises(SkillAssetAccessError):
            self._resolve("agents/link-to-outside.md")


class TestParseSkillSubagent(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.skill_root = _make_skill(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_defaults_without_frontmatter(self):
        path = self.skill_root / "agents" / "grader.md"
        path.write_text("# Grader\nEvaluate assertions.\n", "utf-8")
        spec = parse_skill_subagent(
            skill_id="skill-creator",
            instruction_path="agents/grader.md",
            path=path,
            default_timeout_seconds=120,
        )
        self.assertEqual(spec.name, "grader")
        self.assertEqual(spec.max_depth, 0)
        self.assertEqual(spec.timeout_seconds, 120)
        self.assertIn("Evaluate assertions.", spec.instructions)
        # No tools key -> runtime default (file-read).
        self.assertEqual(len(spec.requested_tools), 1)
        self.assertEqual(spec.requested_tools[0].name, "file-read")

    def test_frontmatter_overrides(self):
        path = self.skill_root / "agents" / "grader.md"
        path.write_text(
            "---\n"
            "name: grader\n"
            "description: Grade an evaluation.\n"
            "tools:\n"
            "  - kind: builtin\n"
            "    name: file-read\n"
            "timeout_seconds: 60\n"
            "max_depth: 2\n"
            "---\n"
            "Evaluate each assertion.\n",
            "utf-8",
        )
        spec = parse_skill_subagent(
            skill_id="skill-creator",
            instruction_path="agents/grader.md",
            path=path,
            default_timeout_seconds=120,
        )
        self.assertEqual(spec.description, "Grade an evaluation.")
        self.assertEqual(spec.timeout_seconds, 60)
        self.assertEqual(spec.max_depth, 2)
        self.assertEqual(spec.requested_tools[0].name, "file-read")

    def test_fingerprint_stable_and_distinct(self):
        path = self.skill_root / "agents" / "grader.md"
        path.write_text("body", "utf-8")
        s1 = parse_skill_subagent(
            skill_id="skill-creator",
            instruction_path="agents/grader.md",
            path=path,
            default_timeout_seconds=120,
        )
        s2 = parse_skill_subagent(
            skill_id="skill-creator",
            instruction_path="agents/grader.md",
            path=path,
            default_timeout_seconds=120,
        )
        self.assertEqual(s1.fingerprint, s2.fingerprint)
        # Different instruction path -> different fingerprint.
        path2 = self.skill_root / "agents" / "analyzer.md"
        path2.write_text("body", "utf-8")
        s3 = parse_skill_subagent(
            skill_id="skill-creator",
            instruction_path="agents/analyzer.md",
            path=path2,
            default_timeout_seconds=120,
        )
        self.assertNotEqual(s1.fingerprint, s3.fingerprint)


class TestTwoSkillsSameGraderName(unittest.TestCase):
    """Spec : two skills may both have agents/grader.md without colliding --
    identity is the (skill_id, instruction_path) pair, never the bare name."""

    def test_identity_is_skill_scoped(self):
        self.assertEqual(
            identity("skill-creator", "agents/grader.md"),
            "skill-creator/agents/grader.md",
        )
        self.assertNotEqual(
            identity("skill-creator", "agents/grader.md"),
            identity("code-review", "agents/grader.md"),
        )


class TestValidateCallRequest(unittest.TestCase):
    def test_exactly_one_of_name_or_instruction_path(self):
        validate_call_request(name="reviewer", instruction_path=None, task="t")
        validate_call_request(name=None, instruction_path="agents/g.md", task="t")
        with self.assertRaises(SubagentResolutionError):
            validate_call_request(name=None, instruction_path=None, task="t")
        with self.assertRaises(SubagentResolutionError):
            validate_call_request(name="r", instruction_path="agents/g.md", task="t")

    def test_blank_task_rejected(self):
        with self.assertRaises(SubagentResolutionError):
            validate_call_request(name="r", instruction_path=None, task="   ")


class TestActiveSkillContext(unittest.TestCase):
    def test_is_frozen(self):
        ctx = ActiveSkillContext(skill_id="s", skill_root=Path("/tmp/s"), revision="r1")
        with self.assertRaises(Exception):
            ctx.skill_id = "other"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
