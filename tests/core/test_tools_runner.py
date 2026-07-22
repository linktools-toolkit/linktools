# -*- coding: utf-8 -*-
"""Tests for ToolRunner + subprocess_env."""
import sys

import pytest

from linktools.core import environ
from linktools.core._tools_runner import ResolvedTool, ToolRunner
from linktools.runtime.process import _subprocess_env


def test_subprocess_env_returns_fresh_dict():
    # _subprocess_env returns its own mapping, not os.environ itself.
    import os
    env = _subprocess_env()
    assert env is not os.environ


def test_subprocess_env_prepends_tools_stub():
    env = _subprocess_env()
    stub = str(environ.tools.stub_path)
    # stub is first on PATH (managed tools win) without mutating global PATH.
    assert env["PATH"].startswith(stub)


def test_runner_runs_real_command():
    runner = ToolRunner(environ)
    resolved = ResolvedTool(executable=sys.executable)
    code = runner.run(resolved, ["-c", "import sys; sys.exit(0)"], check=True)
    assert code == 0


def test_runner_check_raises_on_nonzero():
    from linktools.errors import ExecError
    runner = ToolRunner(environ)
    resolved = ResolvedTool(executable=sys.executable)
    with pytest.raises(Exception):
        runner.run(resolved, ["-c", "import sys; sys.exit(3)"], check=True)


def test_runner_no_check_returns_exit_code():
    runner = ToolRunner(environ)
    resolved = ResolvedTool(executable=sys.executable)
    code = runner.run(resolved, ["-c", "import sys; sys.exit(3)"], check=False)
    assert code == 3


def test_runner_passes_tool_env():
    runner = ToolRunner(environ)
    resolved = ResolvedTool(executable=sys.executable, env={"MY_TOOL_VAR": "abc"})
    code = runner.run(resolved, ["-c", "import os,sys; sys.exit(0 if os.environ.get('MY_TOOL_VAR')=='abc' else 1)"])
    assert code == 0
