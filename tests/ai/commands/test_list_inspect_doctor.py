#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai doctor` tests.

The `list`/`inspect` commands were removed when the command surface was frozen
to the five `lt ai` commands; their capability coverage now lives
on `RuntimeClient` and is tested in ``tests/ai_cli/test_client.py``. ``doctor``
stays as a top-level command, so its end-to-end check against a scaffolded
project is kept here.

Runs against a real scaffolded project on disk (created via `lt ai init`) and
exercises the command path through the project layer + client.doctor(). The
command is a synchronous entry point (it calls asyncio.run internally), so this
is a plain TestCase method -- no nested event loop."""

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from linktools.commands.ai import doctor as doctor_cmd
from linktools.commands.ai import init as init_cmd


class _InProject(unittest.TestCase):
    def setUp(self):
        self._cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        init_cmd.Command().run(Namespace(path=None))

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()


class TestDoctor(_InProject):
    def test_doctor_passes_on_clean_scaffold(self):
        code = doctor_cmd.Command().run(Namespace(project=None, remote=None, json=False))
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
