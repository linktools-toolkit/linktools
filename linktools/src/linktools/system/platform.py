#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Platform/arch identification (spec §14.2 SYS-001)."""

import sys

_is_windows_like = _is_unix_like = False

try:
    import msvcrt  # noqa: F401
except ModuleNotFoundError:
    try:
        import pwd  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        _is_unix_like = True
else:
    _is_windows_like = True

_system = _machine = None

#  canonical architecture values + input aliases.
_ARCH_ALIASES = {
    "amd64": "x86_64",
    "aarch64": "arm64",
    "armv7l": "arm",
    "armv6l": "arm",
    "i386": "x86",
    "i686": "x86",
    "x86": "x86",
    "x86_64": "x86_64",
    "arm": "arm",
    "arm64": "arm64",
}


def get_system():
    """Return the normalised OS name (linux/darwin/windows)."""
    global _system
    if _system is None:
        import platform as _stdlib
        _system = _stdlib.system().lower()
    return _system


def get_machine():
    """Return the raw machine architecture (lowercased), e.g. ``aarch64``.

    Use :func:`normalize_arch` for the canonical value (``arm64``).
    """
    global _machine
    if _machine is None:
        import platform as _stdlib
        _machine = _stdlib.machine().lower()
    return _machine


def is_unix_like(system=None):
    if system:
        return normalize_platform(system) in ("darwin", "linux")
    return _is_unix_like


def is_windows(system=None):
    if system:
        return normalize_platform(system) == "windows"
    return _is_windows_like


def normalize_platform(value: str) -> str:
    """Normalise an OS name to lowercase (spec §14.2)."""
    return (value or "").strip().lower()


def normalize_arch(value: str) -> str:
    """Normalise an architecture alias to the canonical value (spec §14.2).

    ``amd64`` -> ``x86_64``, ``aarch64`` -> ``arm64``, ``armv7l`` -> ``arm`` ...
    Unknown values are returned lowercased unchanged.
    """
    canonical = _ARCH_ALIASES.get((value or "").strip().lower())
    return canonical if canonical is not None else (value or "").strip().lower()
