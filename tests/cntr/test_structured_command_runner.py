#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StructuredCommandRunner: uniform CommandResult over a captured process,
JSON parsing, timeout/error mapping, and sudo prefixing on
RuntimeProcessFactory.create_process.

Uses plain `python3 -c ...` subprocesses (not Docker) to drive real
Process.fetch()/recursive_kill() behavior end-to-end.
"""
import sys

import pytest

from linktools.runtime import popen
from linktools.cntr.container import ContainerError
from linktools.cntr.runtime.structured import (
    CommandResult, StructuredCommandError, StructuredCommandOutputError, StructuredCommandRunner,
    StructuredCommandTimeout,
)


@pytest.fixture
def runner(fresh_manager):
    return StructuredCommandRunner(fresh_manager)


def _py(code: str):
    return popen(sys.executable, "-c", code, capture_output=True)


def test_stdout_only(runner):
    result = runner.execute(_py("import sys; sys.stdout.write('hello')"))
    assert result.stdout == "hello"
    assert result.stderr == ""
    assert result.succeeded is True
    assert result.returncode == 0
    assert result.timed_out is False


def test_stderr_only(runner):
    result = runner.execute(_py("import sys; sys.stderr.write('oops')"))
    assert result.stdout == ""
    assert result.stderr == "oops"
    assert result.succeeded is True


def test_stdout_and_stderr(runner):
    result = runner.execute(_py(
        "import sys; sys.stdout.write('out'); sys.stderr.write('err')"
    ))
    assert result.stdout == "out"
    assert result.stderr == "err"


def test_nonzero_returncode_raises_by_default(runner):
    with pytest.raises(StructuredCommandError):
        runner.execute(_py("import sys; sys.exit(3)"))


def test_nonzero_returncode_check_false_returns_result(runner):
    result = runner.execute(_py("import sys; sys.exit(3)"), check=False)
    assert result.returncode == 3
    assert result.succeeded is False


def test_custom_error_type_is_raised(runner):
    class CustomError(ContainerError):
        pass

    with pytest.raises(CustomError):
        runner.execute(_py("import sys; sys.exit(1)"), error_type=CustomError)


def test_result_is_frozen_dataclass():
    result = CommandResult(args=("x",), returncode=0, stdout="", stderr="", duration=0.0)
    assert result.succeeded is True
    with pytest.raises(Exception):
        result.returncode = 1


def test_args_are_recorded(runner):
    process = _py("pass")
    result = runner.execute(process)
    assert sys.executable in result.args
    assert "-c" in result.args


# -- command redaction ("敏感参数脱敏"/"命令脱敏") ---------------------------

def _py_with_extra_args(code: str, *extra):
    return popen(sys.executable, "-c", code, *extra, capture_output=True)


def test_build_arg_secret_is_redacted_in_recorded_args(runner):
    process = _py_with_extra_args("pass", "--build-arg", "http_proxy=http://user:secret@host")
    result = runner.execute(process)
    assert "http_proxy=***" in result.args
    assert not any("secret" in arg for arg in result.args)


def test_build_arg_secret_is_redacted_in_failure_message(runner):
    process = _py_with_extra_args(
        "import sys; sys.exit(1)", "--build-arg", "http_proxy=http://user:secret@host",
    )
    with pytest.raises(StructuredCommandError) as exc_info:
        runner.execute(process)
    assert "secret" not in str(exc_info.value)
    assert "http_proxy=***" in str(exc_info.value)


def test_redact_command_helper_is_a_noop_without_value_bearing_flags():
    from linktools.cntr.runtime.structured import redact_command
    assert redact_command(("compose", "up", "--detach")) == ("compose", "up", "--detach")
    assert redact_command(None) is None
    assert redact_command(()) == ()


def test_timeout_raises_and_process_is_reaped(runner):
    calls = []
    process = _py("import time; time.sleep(30)")
    real_kill = process.recursive_kill

    def spy_kill():
        calls.append(1)
        return real_kill()

    process.recursive_kill = spy_kill
    with pytest.raises(StructuredCommandTimeout):
        runner.execute(process, timeout=0.2)
    assert calls == [1]


def test_execute_json_object(runner):
    data = runner.execute_json(_py("import sys; sys.stdout.write('{\"a\": 1}')"))
    assert data == {"a": 1}


def test_execute_json_array(runner):
    data = runner.execute_json(_py("import sys; sys.stdout.write('[1, 2, 3]')"))
    assert data == [1, 2, 3]


def test_execute_json_invalid_raises_output_error_with_context(runner):
    with pytest.raises(StructuredCommandOutputError) as exc_info:
        runner.execute_json(_py("import sys; sys.stdout.write('not json')"))
    message = str(exc_info.value)
    assert "not json" in message


def test_execute_json_only_parses_stdout_not_stderr(runner):
    data = runner.execute_json(_py(
        "import sys; sys.stdout.write('{\"ok\": true}'); sys.stderr.write('noise')"
    ))
    assert data == {"ok": True}


def test_long_output_is_truncated_in_error_message(runner):
    with pytest.raises(StructuredCommandOutputError) as exc_info:
        runner.execute_json(_py("import sys; sys.stdout.write('x' * 10000)"))
    assert "...(truncated)" in str(exc_info.value)
    assert len(str(exc_info.value)) < 10000


# -- sudo prefixing (RuntimeProcessFactory.create_process) -------------------

def test_sudo_is_always_interactive(fresh_manager, monkeypatch):
    """sudo never gets a `-n`: if the sudo policy needs a password, the
    call blocks on the prompt rather than failing fast."""
    fresh_manager.system = "linux"
    fresh_manager.uid = 1000
    recorded = []
    monkeypatch.setattr(
        "linktools.cntr.runtime.process.popen",
        lambda *a, **k: recorded.append(a),
    )
    fresh_manager.runtime.create_process("docker", "ps", privilege=True)
    assert recorded[0][0] == "sudo"
    assert "-n" not in recorded[0]


def test_no_privilege_never_invokes_sudo(fresh_manager, monkeypatch):
    fresh_manager.system = "linux"
    fresh_manager.uid = 1000
    recorded = []
    monkeypatch.setattr(
        "linktools.cntr.runtime.process.popen",
        lambda *a, **k: recorded.append(a),
    )
    fresh_manager.runtime.create_process("echo", "hi", privilege=False)
    assert recorded[0][0] == "echo"
