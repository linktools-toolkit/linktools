#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Strict ``${ENV_NAME}`` expansion for MCP server config.

The runtime's general ``env:NAME`` resolver (``registry._utils.resolve_ref``) is
lenient (returns None on missing) and is intentionally left untouched -- other
config formats depend on that semantics. MCP server ``env``, by contrast, MUST
fail fast when a referenced variable is unset,
and only the ``${NAME}`` form is supported. This module is that strict resolver;
it is applied at MCP config load time so a missing secret aborts startup before
any connection is attempted, and the plaintext value never enters the
fingerprint/cache key (the connection manager already digests env)."""

import os
import re
from collections.abc import Mapping
from typing import Any

from ..errors import InvalidSpecError

# ${NAME} where NAME is a non-empty run of uppercase/digit/underscore (the POSIX
# env-var shape). Any other ${...} is rejected rather than silently passed
# through, so a typo like ${GITHUB_TOKEN} is caught even if the real var differs.
_ENV_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def expand_env_value(value: Any) -> Any:
    """Expand ``${NAME}`` references inside a string, recursively over
    containers. Raises InvalidSpecError if a referenced variable is unset or
    empty, or if a ``${...}`` does not match the POSIX env-var shape."""
    if isinstance(value, Mapping):
        return {key: expand_env_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [expand_env_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env_value(item) for item in value)
    if not isinstance(value, str):
        return value

    def _replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if not resolved:
            raise InvalidSpecError(f"environment variable ${{{name}}} is not set")
        return resolved

    # Reject any ${...} that is not a well-formed POSIX env ref so a malformed
    # reference is not silently left literal.
    for bad in re.findall(r"\$\{([^}]*)\}", value):
        if not _ENV_REF.fullmatch("${" + bad + "}"):
            raise InvalidSpecError(f"invalid env reference: ${{{bad}}}")
    return _ENV_REF.sub(_replace, value)


def expand_env_mapping(env: "Mapping[str, Any] | None") -> "dict[str, Any]":
    """Expand the ``env`` block of an MCP server spec.

    Returns a new plain dict; the input is never mutated. Keys are preserved
    verbatim (the connection manager digests values for the fingerprint)."""
    if not env:
        return {}
    if not isinstance(env, Mapping):
        raise InvalidSpecError("mcp env must be a mapping")
    return dict(expand_env_value(env))
