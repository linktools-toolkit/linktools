#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai list`/`inspect`/`doctor` tests (spec §20/§21/§25).

Runs against a real scaffolded project on disk (created via `lt ai init`) and
exercises the command path end-to-end through the project layer + runtime.inspect.
The commands are synchronous entry points (each calls asyncio.run internally),
so these are plain TestCase methods -- no nested event loop."""

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from linktools.commands.ai import doctor as doctor_cmd
from linktools.commands.ai import init as init_cmd
from linktools.commands.ai import inspect as inspect_cmd
from linktools.commands.ai import list as list_cmd


class _InProject(unittest.TestCase):
    def setUp(self):
        self._cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        init_cmd.Command().run(Namespace(force=False))

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()


class TestList(_InProject):
    def test_list_agents(self):
        code = list_cmd.Command().run(Namespace(kind="agents", verbose=False))
        self.assertEqual(code, 0)

    def test_list_skills_verbose_runs(self):
        # The verbose path iterates each skill's private agents (the per-skill
        # agent listing is unit-tested in test_skill_index); here we assert the
        # command completes cleanly on a scaffold that has private agents.
        code = list_cmd.Command().run(Namespace(kind="skills", verbose=True))
        self.assertEqual(code, 0)

    def test_list_mcp_empty(self):
        # github.yaml.disabled is not a .yaml file, so MCP list is empty.
        code = list_cmd.Command().run(Namespace(kind="mcp", verbose=False))
        self.assertEqual(code, 0)


class TestInspect(_InProject):
    def test_inspect_default_agent(self):
        # Capability resolution succeeds without a live model client.
        code = inspect_cmd.Command().run(Namespace(agent=None, json=False))
        self.assertEqual(code, 0)

    def test_inspect_unknown_agent_fails(self):
        from linktools.cli import CommandError

        with self.assertRaises(CommandError):
            inspect_cmd.Command().run(Namespace(agent="ghost", json=False))


class TestDoctor(_InProject):
    def test_doctor_passes_on_clean_scaffold(self):
        code = doctor_cmd.Command().run(Namespace())
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
