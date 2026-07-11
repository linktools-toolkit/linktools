#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Git capability detection.

Importable on every supported Python version without pulling in dulwich:
only tests whether it *could* be imported. ``linktools.git.repository`` (the
real dulwich-backed implementation) is imported lazily by ``linktools.git``
only after :func:`is_git_available` confirms it will succeed.
"""

import sys


def is_git_supported_python():
    return sys.version_info >= (3, 10)


def get_git_unavailable_reason():
    if not is_git_supported_python():
        return "Git repository operations require Python 3.10 or newer."

    try:
        import dulwich  # noqa: F401
    except ImportError:
        return "Git repository operations require the `linktools[git]` extra."

    return None


def is_git_available():
    return get_git_unavailable_reason() is None


def require_git_available(action="Git repository operation"):
    reason = get_git_unavailable_reason()

    if reason is not None:
        from linktools.errors import GitUnavailableError

        raise GitUnavailableError("%s is unavailable. %s" % (action, reason))
