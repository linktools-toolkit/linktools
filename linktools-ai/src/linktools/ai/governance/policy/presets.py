#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optional policy presets."""

from .command import DEFAULT_DENIED_COMMAND_PATTERNS, CommandRule
from .engine import PolicyEngine


def build_default_command_policy(
    *, denied_patterns: "tuple[str, ...]" = DEFAULT_DENIED_COMMAND_PATTERNS,
) -> PolicyEngine:
    """Build a policy with the standard command denylist."""
    return PolicyEngine(rules=(CommandRule(denied_patterns=denied_patterns),))


__all__ = ["build_default_command_policy"]
