# -*- coding: utf-8 -*-
"""Golden-output + safety tests for linktools.system.shell.

Per-shell renderers are pinned by golden strings; bash additionally gets a
real round-trip execution test (the others aren't installed in this env, so
their structure is locked, not execution-verified, here). Safety tests
verify that hostile values stay literal text and that control chars /
malformed names / pre-joined command strings are rejected.
"""
import os
import subprocess
import sys

import pytest

from linktools.system.shell import (
    SUPPORTED_SHELLS, ShellScript, get_default_shell, get_shell,
)


# -- golden: bash ---------------------------------------------------------

def test_bash_set_env():
    assert ShellScript("bash").set_env("JAVA_VERSION", "17").render() == "export JAVA_VERSION='17'"


def test_bash_unset_env():
    assert ShellScript("bash").unset_env("FOO").render() == "unset FOO"


def test_bash_prepend_path():
    out = ShellScript("bash").prepend_path(["/a/bin", "/b/bin"]).render()
    assert out == "export PATH='/a/bin':'/b/bin':\"$PATH\""


def test_bash_append_path():
    out = ShellScript("bash").append_path(["/a/bin"]).render()
    assert out == "export PATH=\"$PATH\":'/a/bin'"


def test_bash_define_command():
    out = ShellScript("bash").define_command("java", ["/p/py", "-m", "lt", "tool", "java"]).render()
    assert out == (
        "java() {\n"
        "    command '/p/py' '-m' 'lt' 'tool' 'java' \"$@\"\n"
        "}"
    )


def test_bash_operations_emit_in_order_and_skip_empty():
    out = (ShellScript("bash")
           .set_env("A", "1")
           .prepend_path([])            # no-op -> skipped
           .define_command("java", ["/p/py", "java"])
           .render())
    assert out == "export A='1'\njava() {\n    command '/p/py' 'java' \"$@\"\n}"


# -- golden: zsh mirrors bash --------------------------------------------

def test_zsh_rendering_matches_bash_dialect():
    assert ShellScript("zsh").set_env("X", "1").render() == "export X='1'"
    assert ShellScript("zsh").shell == "zsh"


# -- golden: fish ---------------------------------------------------------

def test_fish_set_env():
    assert ShellScript("fish").set_env("JAVA_VERSION", "17").render() == "set -gx JAVA_VERSION '17'"


def test_fish_unset_env():
    assert ShellScript("fish").unset_env("FOO").render() == "set -e FOO"


def test_fish_prepend_path():
    out = ShellScript("fish").prepend_path(["/a/bin"]).render()
    assert out == "set -gx PATH '/a/bin' $PATH"


def test_fish_define_command():
    out = ShellScript("fish").define_command("java", ["/p/py", "-m", "lt"]).render()
    assert out == "function java\n    command '/p/py' '-m' 'lt' $argv\nend"


# -- golden: tcsh ---------------------------------------------------------

def test_tcsh_set_env():
    assert ShellScript("tcsh").set_env("JAVA_VERSION", "17").render() == "setenv JAVA_VERSION '17'"


def test_tcsh_unset_env():
    assert ShellScript("tcsh").unset_env("FOO").render() == "unsetenv FOO"


def test_tcsh_prepend_path():
    out = ShellScript("tcsh").prepend_path(["/a/bin"]).render()
    assert out == "setenv PATH '/a/bin':\"$PATH\""


def test_tcsh_define_command_forwards_args():
    out = ShellScript("tcsh").define_command("java", ["/p/py", "-m", "lt"]).render()
    # double-quoted body so \!* expands; args single-quoted inside
    assert out == 'alias java "\'/p/py\' \'-m\' \'lt\' \\!*"'


# -- golden: powershell ---------------------------------------------------

def test_powershell_set_env():
    assert ShellScript("powershell").set_env("JAVA_VERSION", "17").render() == "$env:JAVA_VERSION = '17'"


def test_powershell_unset_env():
    assert ShellScript("powershell").unset_env("FOO").render() == "Remove-Item Env:FOO"


def test_powershell_prepend_path():
    out = ShellScript("powershell").prepend_path([r"C:\a", r"C:\b"]).render()
    assert out == r"$env:PATH = ('C:\a' + ';' + 'C:\b' + ';' + $env:PATH)"


def test_powershell_define_command():
    out = ShellScript("powershell").define_command("java", [r"C:\py", "-m", "lt"]).render()
    assert out == (
        "function global:java {\n"
        r"    & 'C:\py' '-m' 'lt' @args" "\n"
        "}"
    )


# -- special characters stay literal -------------------------------------

@pytest.mark.parametrize("value", [
    "hello world",
    "a'b",
    'a"b',
    "$HOME",                       # must stay literal, not expand
    "$(touch /tmp/evil)",          # must not execute
    "a;b", "a&b", "a|b",
    "%PATH%",                      # not a windows var reference here
    "中文路径",
])
def test_bash_value_with_special_chars_is_single_quoted_literal(value):
    out = ShellScript("bash").set_env("V", value).render()
    # round-trip: sourcing the line in real bash must reproduce `value`
    assert _bash_eval(out + "; printf %s \"$V\"") == value


def test_define_command_argv_with_spaces_round_trips_in_bash():
    # greet prints its forwarded argv, proving spaces/quotes survive intact
    code = "import sys; import os; os.write(1, repr(sys.argv[1:]).encode())"
    script = ShellScript("bash").define_command("greet", [sys.executable, "-c", code]).render()
    out = _bash_eval(script + "\ngreet 'hello world' \"a'b\"\n")
    assert out == repr(["hello world", "a'b"])


# -- validation -----------------------------------------------------------

def test_rejects_invalid_env_name():
    with pytest.raises(ValueError):
        ShellScript("bash").set_env("BAD-NAME", "1")
    with pytest.raises(ValueError):
        ShellScript("bash").set_env("1ABC", "1")


def test_rejects_invalid_command_name():
    with pytest.raises(ValueError):
        ShellScript("bash").define_command("bad name", ["/p/py"])


def test_rejects_string_argv():
    with pytest.raises(TypeError):
        ShellScript("bash").define_command("java", "/p/py -m lt")


def test_rejects_newline_in_value():
    with pytest.raises(ValueError):
        ShellScript("bash").set_env("V", "line1\nline2")


def test_rejects_unsupported_shell():
    with pytest.raises(ValueError):
        ShellScript("csh")


def test_empty_render_is_empty_string():
    assert ShellScript("bash").render() == ""


def test_add_raw_appends_verbatim():
    out = ShellScript("bash").add_raw("# completion\ncomplete -c x").render()
    assert out == "# completion\ncomplete -c x"


# -- detection -----------------------------------------------------------

def test_get_default_shell_from_sHELL(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/zsh")
    assert get_default_shell(system="linux") == "zsh"


def test_get_default_shell_unsupported(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/unknownsh")
    with pytest.raises(ValueError):
        get_default_shell(system="linux")


def test_get_shell_returns_script_bound_to_detected_shell(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/bash")
    s = get_shell()
    assert isinstance(s, ShellScript)
    assert s.shell == "bash"


# -- helper: actually run bash -------------------------------------------

def _bash_eval(script: str, expect: bool = True):
    """Source `script` in real /bin/sh (bash-compatible) and return stdout.

    /bin/sh on this host parses POSIX single-quote + function syntax, so it
    validates the bash/zsh renderer's output faithfully. Returns None if
    `expect=False` and the shell exited non-zero (used to merely check
    parse-ability)."""
    proc = subprocess.run(
        ["/bin/sh"], input=script, capture_output=True, text=True,
    )
    if expect:
        assert proc.returncode == 0, f"sh failed: {proc.stderr}\nscript:\n{script}"
    elif proc.returncode != 0:
        return None
    return proc.stdout


def test_supported_shells_constant():
    assert SUPPORTED_SHELLS == ("bash", "zsh", "fish", "tcsh", "powershell")
