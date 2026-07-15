#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Error surface for ``linktools.ai_cli``.

The thin ``commands/ai`` shells import only ``linktools.ai_cli.*``, so the
exception classes they declare in ``known_errors`` (for exit-code mapping) are
re-exported here instead of pulled from ``linktools.ai`` directly.
``MissingConfigError`` is this layer's own signal that a required configuration
value could not be resolved."""

from linktools.ai.errors import (  # noqa: F401  (re-exported for command known_errors)
    InvalidRunTransitionError,
    RunConflictError,
    RunNotFoundError,
)
from linktools.ai.model.registry import (  # noqa: F401  (re-exported for command known_errors)
    ModelClientUnavailable,
    ModelOutputError,
    ModelTurnLimitExceeded,
)
from linktools.cli import CommandError


class MissingConfigError(CommandError):
    """A required configuration field had no explicit value, environment value,
    project setting, or cached value, and no interactive prompt was possible.

    Raised by the configuration resolver so a non-interactive invocation fails
    fast instead of silently falling back to a default."""
