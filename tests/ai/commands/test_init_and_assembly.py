#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai init` + `build_cli_runtime` tests."""

import os
import tempfile
import unittest
from pathlib import Path

from linktools.commands.ai import init as init_cmd
from linktools.ai_cli.runtime import build_cli_runtime
from linktools.ai_cli.project import load_project
from linktools.ai.runtime import Runtime


class TestAiInit(unittest.TestCase):
    def setUp(self):
        self._cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def _run(self):
        from argparse import Namespace

        return init_cmd.Command().run(Namespace(path=None))

    def test_ai_init_creates_scaffold(self):
        code = self._run()
        self.assertEqual(code, 0)
        root = Path(self._tmp.name) / ".linktools"
        for relative in (
            "config.yaml",
            "agents/default.md",
            "agents/reviewer.md",
            "skills/code-review/SKILL.md",
            "skills/skill-creator/SKILL.md",
            "skills/skill-creator/agents/grader.md",
            "skills/skill-creator/agents/comparator.md",
            "skills/skill-creator/agents/analyzer.md",
            "mcp/github.yaml.disabled",
        ):
            self.assertTrue((root / relative).is_file(), f"missing {relative}")

    def test_ai_init_does_not_overwrite(self):
        # A pre-existing customized file must be left untouched.
        cfg = Path(self._tmp.name) / ".linktools" / "config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            "version: 1\ndefault_agent: mine\ndefault_session: main\n", "utf-8"
        )
        self._run()
        self.assertIn("default_agent: mine", cfg.read_text("utf-8"))


class TestBuildCliRuntime(unittest.TestCase):
    def _project(self, tmp: str) -> Path:
        # Minimal valid project for assembly: config + a default agent + a
        # self-contained skill with a private agent.
        root = Path(tmp)
        linktools = root / "proj" / ".linktools"
        (linktools / "agents").mkdir(parents=True)
        (linktools / "skills" / "skill-creator" / "agents").mkdir(parents=True)
        (linktools / "config.yaml").write_text(
            "version: 1\ndefault_agent: default\ndefault_session: main\n", "utf-8"
        )
        (linktools / "agents" / "default.md").write_text(
            "---\nname: default\ndescription: x\n---\nbody\n", "utf-8"
        )
        (linktools / "skills" / "skill-creator" / "SKILL.md").write_text(
            "---\nname: skill-creator\ndescription: x\n---\nbody\n", "utf-8"
        )
        (linktools / "skills" / "skill-creator" / "agents" / "grader.md").write_text(
            "# Grader\n", "utf-8"
        )
        return linktools.parent  # the project root

    def test_build_cli_runtime_wires_registries(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj_root = self._project(tmp)
            project = load_project(data_root=Path(tmp) / "data", start=proj_root)
            bundle = build_cli_runtime(project=project, model_resolver=None)
            self.assertIsInstance(bundle.runtime, Runtime)
            self.assertEqual(bundle.project, project)
            # Agents loaded from .linktools/agents.
            self.assertIn(
                "default", __import__("asyncio").run(bundle.agents.list_ids())
            )
            # Skills discovered as self-contained directories.
            ids = __import__("asyncio").run(bundle.skill_index.list_ids())
            self.assertIn("skill-creator", ids)

    def test_build_cli_runtime_state_isolated_per_project(self):
        # state_root comes from the project hash, so each project gets its own.
        with tempfile.TemporaryDirectory() as tmp:
            proj_root = self._project(tmp)
            project = load_project(data_root=Path(tmp) / "data", start=proj_root)
            self.assertEqual(
                project.state_root,
                Path(tmp) / "data" / "projects" / project_hash_root(proj_root),
            )


def project_hash_root(root: Path) -> str:
    from hashlib import sha256

    return sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    unittest.main()
