# -*- coding: utf-8 -*-
"""Tests for ToolRunner + subprocess_env (spec §10.11, §5.1)."""
import sys

import pytest

from linktools.core import environ
from linktools.core._tools_runner import ResolvedTool, ToolRunner


def test_subprocess_env_returns_fresh_dict_with_overrides():
    # subprocess_env returns its own mapping (not os.environ) and applies
    # overrides without leaking them into os.environ.
    import os
    env = environ.subprocess_env(overrides={"LT_TEST": "1"})
    assert env is not os.environ
    assert env["LT_TEST"] == "1"
    assert "LT_TEST" not in os.environ  # override did not leak to the process env
    # (The legacy _create_tools still mutates os.environ["PATH"] so subprocesses
    #  that invoke tools by name resolve the stub; removing that is a §10.11
    #  follow-up. subprocess_env itself prepends the stub cleanly here.)


def test_subprocess_env_prepends_tools_stub():
    env = environ.subprocess_env()
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
