#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified shell-script generation (spec: "Linktools Shell 脚本能力统一重构").

Business code (``cli.env``, ``core``) describes *what* to execute -- env
vars, PATH entries, command names + argv lists -- as plain structured
values. Only this module turns that intent into shell-specific text:
quoting, escaping, ``$PATH``/``$env:PATH`` syntax, PATH separators and
``$@``/``$argv``/``@args`` forwarding all live here, never at a call site.

Public API: ``SUPPORTED_SHELLS``, ``get_default_shell``, ``get_shell``,
``ShellScript``. Per-shell ``_Renderer``s are private.
"""
import os
import re
from typing import TYPE_CHECKING

from .platform import get_system

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

__all__ = ["SUPPORTED_SHELLS", "ShellScript", "get_default_shell", "get_shell"]

SUPPORTED_SHELLS = ("bash", "zsh", "fish", "tcsh", "powershell")

# A value placed in an env-var / argv / PATH position must never carry a
# raw newline/CR -- it would silently split a generated statement across
# lines. Reject up front rather than emit a semantically broken script.
_BAD_CHARS_IN_VALUE = re.compile(r"[\r\n\x00]")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMMAND_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _check_env_name(name: str) -> None:
    if not isinstance(name, str) or not _ENV_NAME.match(name):
        raise ValueError("Invalid environment variable name: %r" % (name,))


def _check_command_name(name: str) -> None:
    if not isinstance(name, str) or not _COMMAND_NAME.match(name):
        raise ValueError("Invalid command name: %r" % (name,))


def _check_value(value: "Any") -> str:
    """Coerce a value to a string and reject control characters that would
    break a generated statement. ``$HOME`` / ``$(...)`` stay literal text --
    they are quoted, never executed."""
    text = value if isinstance(value, str) else str(value)
    if _BAD_CHARS_IN_VALUE.search(text):
        raise ValueError("Value contains a newline/NUL and cannot be rendered: %r" % (value,))
    return text


def _check_argv(argv: "Any") -> "list[str]":
    """argv must be a sequence of arguments, never a pre-joined command
    string -- a string would let an unquoted space/quote slip through."""
    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of arguments, not a command string")
    return [_check_value(a) for a in argv]


class _Renderer(object):
    """Per-shell text generation. Each method returns one rendered statement
    (possibly multi-line); ``ShellScript`` accumulates and joins them."""

    name = None  # type: ignore[assignment]
    path_sep = ":"
    path_var = "$PATH"

    def quote(self, value: str) -> str:
        raise NotImplementedError()

    def set_env(self, name: str, value: str) -> str:
        raise NotImplementedError()

    def unset_env(self, name: str) -> str:
        raise NotImplementedError()

    def prepend_path(self, paths: "list[str]") -> str:
        raise NotImplementedError()

    def append_path(self, paths: "list[str]") -> str:
        raise NotImplementedError()

    def define_command(self, name: str, argv: "list[str]") -> str:
        raise NotImplementedError()


class _PosixRenderer(_Renderer):
    """bash/zsh: single-quote with the ``'\\''`` close-reopen trick."""

    def quote(self, value: str) -> str:
        return "'" + value.replace("'", "'\\''") + "'"

    def set_env(self, name: str, value: str) -> str:
        return "export %s=%s" % (name, self.quote(value))

    def unset_env(self, name: str) -> str:
        return "unset %s" % (name,)

    def prepend_path(self, paths: "list[str]") -> str:
        joined = self.path_sep.join(self.quote(p) for p in paths)
        return 'export PATH=%s:"$PATH"' % (joined,)

    def append_path(self, paths: "list[str]") -> str:
        joined = self.path_sep.join(self.quote(p) for p in paths)
        return 'export PATH="$PATH":%s' % (joined,)

    def define_command(self, name: str, argv: "list[str]") -> str:
        body = " ".join(["command"] + [self.quote(a) for a in argv] + ['"$@"'])
        return "%s() {\n    %s\n}" % (name, body)


class _BashRenderer(_PosixRenderer):
    name = "bash"


class _ZshRenderer(_PosixRenderer):
    name = "zsh"


class _FishRenderer(_Renderer):
    """fish: single-quote, but inside single quotes ``\\`` and ``'`` are the
    only special sequences (escaped as ``\\\\`` / ``\\'``)."""

    name = "fish"

    def quote(self, value: str) -> str:
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    def set_env(self, name: str, value: str) -> str:
        return "set -gx %s %s" % (name, self.quote(value))

    def unset_env(self, name: str) -> str:
        return "set -e %s" % (name,)

    def prepend_path(self, paths: "list[str]") -> str:
        quoted = " ".join(self.quote(p) for p in paths)
        return "set -gx PATH %s $PATH" % (quoted,)

    def append_path(self, paths: "list[str]") -> str:
        quoted = " ".join(self.quote(p) for p in paths)
        return "set -gx PATH $PATH %s" % (quoted,)

    def define_command(self, name: str, argv: "list[str]") -> str:
        body = " ".join(["command"] + [self.quote(a) for a in argv] + ["$argv"])
        return "function %s\n    %s\nend" % (name, body)


class _TcshRenderer(_Renderer):
    """tcsh: single-quote using ``'"'"'`` adjacency (valid in tcsh, unlike
    bash's ``'\\'``). ``setenv`` for vars; aliases forward args via ``\\!*``."""

    name = "tcsh"

    def quote(self, value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"

    def set_env(self, name: str, value: str) -> str:
        return "setenv %s %s" % (name, self.quote(value))

    def unset_env(self, name: str) -> str:
        return "unsetenv %s" % (name,)

    def prepend_path(self, paths: "list[str]") -> str:
        joined = self.path_sep.join(self.quote(p) for p in paths)
        return 'setenv PATH %s:"$PATH"' % (joined,)

    def append_path(self, paths: "list[str]") -> str:
        joined = self.path_sep.join(self.quote(p) for p in paths)
        return 'setenv PATH "$PATH":%s' % (joined,)

    def define_command(self, name: str, argv: "list[str]") -> str:
        # The body is double-quoted so tcsh expands \!* (caller args); each
        # fixed arg is single-quoted inside. (v1: args are interpreter paths
        # / module names -- an arg containing a double quote is not supported.)
        body = " ".join(self.quote(a) for a in argv) + " \\!*"
        return 'alias %s "%s"' % (name, body)


class _PowerShellRenderer(_Renderer):
    """PowerShell: single-quote with ``''`` for an embedded quote; ``;`` PATH
    separator; ``$env:NAME`` vars; functions forward via ``@args``."""

    name = "powershell"
    path_sep = ";"
    path_var = "$env:PATH"

    def quote(self, value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def set_env(self, name: str, value: str) -> str:
        return "$env:%s = %s" % (name, self.quote(value))

    def unset_env(self, name: str) -> str:
        return "Remove-Item Env:%s" % (name,)

    def prepend_path(self, paths: "list[str]") -> str:
        parts = [self.quote(p) for p in paths]
        return "$env:PATH = (%s + ';' + $env:PATH)" % (" + ';' + ".join(parts))

    def append_path(self, paths: "list[str]") -> str:
        parts = [self.quote(p) for p in paths]
        return "$env:PATH = ($env:PATH + ';' + %s)" % (" + ';' + ".join(parts))

    def define_command(self, name: str, argv: "list[str]") -> str:
        body = " ".join(["&"] + [self.quote(a) for a in argv] + ["@args"])
        return "function global:%s {\n    %s\n}" % (name, body)


_RENDERERS = {
    "bash": _BashRenderer(),
    "zsh": _ZshRenderer(),
    "fish": _FishRenderer(),
    "tcsh": _TcshRenderer(),
    "powershell": _PowerShellRenderer(),
}


def get_default_shell(system: "str | None" = None) -> str:
    """Resolve the default shell name for ``system`` (auto-detected when
    ``None``), consulting ``$SHELL`` on POSIX. Raises ``ValueError`` if
    nothing supported is found."""
    import shutil

    system = system or get_system()
    if system == "windows":
        if shutil.which("powershell"):
            return "powershell"
        raise ValueError("No supported shell found, supported shells: %s"
                         % ", ".join(SUPPORTED_SHELLS))
    shell_env = os.environ.get("SHELL")
    if shell_env:
        name = os.path.basename(shell_env)
        if name in SUPPORTED_SHELLS:
            return name
    raise ValueError("Unsupported shell: %r, supported shells: %s"
                     % (shell_env, ", ".join(SUPPORTED_SHELLS)))


def get_shell(name: "str | None" = None, system: "str | None" = None) -> "ShellScript":
    """Return a :class:`ShellScript` for ``name`` (auto-detected when None)."""
    return ShellScript(name or get_default_shell(system=system))


class ShellScript(object):
    """Accumulates structured shell-script intent and renders it in one
    shell's dialect. Operations are emitted in the order they were added;
    empty renders are skipped. Call ``render()`` once at the end.

    Callers pass only plain values -- env-var names, string values, PATH
    entries, command names and argv lists. Quoting, escaping and dialect
    syntax are this class's job, never the caller's.
    """

    def __init__(self, shell: "str | None" = None):
        name = shell or get_default_shell()
        if name not in _RENDERERS:
            raise ValueError("Unsupported shell: %r, supported shells: %s"
                             % (name, ", ".join(SUPPORTED_SHELLS)))
        self._shell = name
        self._renderer = _RENDERERS[name]
        self._lines = []  # type: list[str]

    @property
    def shell(self) -> str:
        return self._shell

    def _append(self, text: "str | None") -> "ShellScript":
        if text:
            self._lines.append(text)
        return self

    def set_env(self, name: str, value: "Any") -> "ShellScript":
        _check_env_name(name)
        return self._append(self._renderer.set_env(name, _check_value(value)))

    def unset_env(self, name: str) -> "ShellScript":
        _check_env_name(name)
        return self._append(self._renderer.unset_env(name))

    def prepend_path(self, paths: "Iterable[str]") -> "ShellScript":
        paths = [_check_value(p) for p in paths]
        if not paths:
            return self
        return self._append(self._renderer.prepend_path(paths))

    def append_path(self, paths: "Iterable[str]") -> "ShellScript":
        paths = [_check_value(p) for p in paths]
        if not paths:
            return self
        return self._append(self._renderer.append_path(paths))

    def define_command(self, name: str, argv: "Any") -> "ShellScript":
        _check_command_name(name)
        args = _check_argv(argv)
        if not args:
            raise ValueError("define_command requires at least one argv element")
        return self._append(self._renderer.define_command(name, args))

    def add_raw(self, code: str) -> "ShellScript":
        """Add trusted, already-rendered shell code verbatim.

        Low-level escape hatch (e.g. for an externally-generated completion
        script). User input, config values and file paths must never go
        through here -- they belong in the structured methods above so they
        get quoted.
        """
        if not isinstance(code, str):
            raise TypeError("add_raw() expects a string of pre-rendered shell code")
        return self._append(code)

    def render(self) -> str:
        return "\n".join(self._lines)

    def __repr__(self) -> str:
        return "ShellScript<%s, %d statement(s)>" % (self._shell, len(self._lines))
