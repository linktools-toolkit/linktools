#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Self-contained skill directory index tests (spec §7)."""

import tempfile
import unittest
from pathlib import Path

from linktools.ai_cli.skill_index import DirectorySkillIndex


def _make_skill(root: Path, skill_id: str, *, body: str = "# x\n", agents=()) -> Path:
    skill = root / skill_id
    (skill).mkdir(parents=True)
    (skill / "SKILL.md").write_text(body, "utf-8")
    if agents:
        ag = skill / "agents"
        ag.mkdir()
        for name in agents:
            (ag / name).write_text(f"# {name}\n", "utf-8")
    return skill


class TestDirectorySkillIndex(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    async def test_list_ids_finds_skill_dirs(self):
        _make_skill(self.root, "skill-creator")
        _make_skill(self.root, "code-review")
        # A directory without SKILL.md is not a skill.
        (self.root / "not-a-skill").mkdir()
        index = DirectorySkillIndex(self.root)
        self.assertEqual(await index.list_ids(), ("code-review", "skill-creator"))

    async def test_get_returns_spec_root_and_revision(self):
        _make_skill(self.root, "skill-creator", agents=("grader.md", "analyzer.md"))
        index = DirectorySkillIndex(self.root)
        info = await index.get("skill-creator")
        self.assertIsNotNone(info)
        self.assertEqual(info.id, "skill-creator")
        self.assertEqual(info.root, self.root / "skill-creator")
        self.assertEqual(len(info.revision), 16)
        names = sorted(p.name for p in info.list_private_agents())
        self.assertEqual(names, ["analyzer.md", "grader.md"])

    async def test_revision_changes_when_agent_added(self):
        _make_skill(self.root, "skill-creator")
        index = DirectorySkillIndex(self.root)
        before = await index.revision("skill-creator")
        # Add an agents/*.md file -> revision changes.
        ag = self.root / "skill-creator" / "agents"
        ag.mkdir()
        (ag / "grader.md").write_text("# g\n", "utf-8")
        after = await index.revision("skill-creator")
        self.assertNotEqual(before, after)

    async def test_missing_skill_returns_none(self):
        index = DirectorySkillIndex(self.root)
        self.assertIsNone(await index.get("ghost"))
        self.assertEqual(await index.list_ids(), ())


if __name__ == "__main__":
    unittest.main()
