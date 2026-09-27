#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest

from linktools.system import CommandStub, ShellScript


def test_shell_script_validates_and_renders_structured_input():
    script = ShellScript("bash")
    script.set_env("APP_HOME", "value with spaces")
    script.define_command("run-tool", ["tool", "argument with spaces"])

    assert script.render() == (
        "export APP_HOME='value with spaces'\n"
        "run-tool() {\n"
        "    command 'tool' 'argument with spaces' \"$@\"\n"
        "}"
    )

    with pytest.raises(ValueError):
        script.set_env("APP_HOME", "invalid\nvalue")
    with pytest.raises(TypeError):
        script.define_command("run-tool", "tool argument")


def test_command_stub_renders_platform_quoting(tmp_path):
    posix = CommandStub(tmp_path, "run-tool", system="linux")
    posix.write(["tool", "argument with spaces"])
    assert posix.path.read_text() == (
        "#!/bin/sh\nexec 'tool' 'argument with spaces' \"$@\"\n"
    )

    windows = CommandStub(tmp_path, "run-tool", system="windows")
    windows.write(["tool", 'argument "with quotes"'])
    assert windows.path.read_text() == (
        "@echo off\n\"tool\" \"argument \\\"with quotes\\\"\" %*\n"
        "exit /b %ERRORLEVEL%\n"
    )
