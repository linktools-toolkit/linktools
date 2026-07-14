#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executable wrapper-script generation (spec: "Linktools Shell 脚本能力统一重构").

``CommandStub`` writes the small executable wrapper that lets a command
(be it a managed tool or a CLI module) be invoked by name from a shell --
a POSIX ``/bin/sh`` script or a Windows ``.bat``. It takes an *argv list*,
never a pre-joined command string, so quoting is decided here, at the one
platform boundary, instead of guessed twice. Deliberately separate from
``system.shell`` (which renders code to be ``source``/``eval``'d): a
wrapper file and an init snippet are different concerns.
"""
import os
import pathlib
from typing import TYPE_CHECKING

from .platform import get_system

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from ..types import PathType

__all__ = ["CommandStub"]


def _quote_posix(value: str) -> str:
    """Single-quote with the ``'\\''`` close-reopen trick (POSIX sh)."""
    return "'" + value.replace("'", "'\\''") + "'"


def _quote_windows(value: str) -> str:
    """Double-quote for the Windows CRT/cmd argv parser, escaping an
    embedded ``"`` as ``\\"``. ``%`` is left intact: ``%*`` (appended
    separately) is what forwards caller args in a ``.bat``."""
    return '"' + value.replace('"', '\\"') + '"'


class CommandStub(object):
    """An executable wrapper script at ``<directory>/<name>`` (``.bat`` on
    Windows). ``write(argv)`` renders it atomically; ``remove()`` deletes it."""

    def __init__(self, directory: "PathType", name: str, system: "str | None" = None):
        self.system = system or get_system()
        self.name = "%s.bat" % name if self.system == "windows" else name
        self.path = pathlib.Path(directory, self.name)

    @property
    def exists(self) -> bool:
        """Whether the wrapper file currently exists."""
        return bool(self.path) and os.path.exists(self.path)

    def write(self, argv: "Sequence[str]") -> "pathlib.Path":
        """Atomically write (or overwrite) the wrapper for ``argv``.

        ``argv`` is a sequence of arguments -- the command and its fixed
        parameters; caller arguments are forwarded by the wrapper itself
        (``"$@"`` / ``%*``). A string is rejected: a pre-joined command line
        would let an unquoted space or quote slip straight through.
        """
        if isinstance(argv, (str, bytes)):
            raise TypeError("argv must be a sequence of arguments, not a command string")
        args = [str(a) for a in argv]
        if not args:
            raise ValueError("CommandStub.write requires at least one argv element")
        content = self._render(args)
        mode = 0o755 if self.system != "windows" else None
        self._atomic_write(self.path, content, mode=mode)
        return self.path

    def remove(self) -> None:
        """Remove the wrapper file if it exists (missing is not an error)."""
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass

    def _render(self, args: "list[str]") -> str:
        if self.system == "windows":
            body = " ".join(_quote_windows(a) for a in args)
            return "@echo off\n" + body + " %*\nexit /b %ERRORLEVEL%\n"
        body = " ".join(_quote_posix(a) for a in args)
        return "#!/bin/sh\nexec " + body + " \"$@\"\n"

    @staticmethod
    def _atomic_write(path: "pathlib.Path", content: str, mode: "int | None" = None) -> None:
        """Write ``content`` to a uniquely-named same-directory temp file, then
        ``os.replace`` it onto ``path`` -- concurrent ``write()``/``prepare()``
        runs never read a half-written wrapper, and no temp file is left
        behind. The temp name is unique per call (mkstemp), so concurrent
        writers in the same process never collide."""
        import tempfile

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + ".", suffix=".tmp",
        )
        temp = pathlib.Path(temp_name)
        try:
            with os.fdopen(fd, "w", newline="") as fp:
                fp.write(content)
                fp.flush()
                os.fsync(fp.fileno())
            if mode is not None:
                os.chmod(temp, mode)
            os.replace(temp, path)
        finally:
            try:
                if temp.exists():
                    os.remove(temp)
            except OSError:
                pass

    def __repr__(self) -> str:
        return "CommandStub<%s>" % (self.name,)
