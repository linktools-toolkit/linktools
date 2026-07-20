#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional policy presets. linktools-ai ships no default security policy --
callers opt in explicitly. ``build_default_command_policy`` recreates the
convenient command-denylist executor for deployments that want it."""

from typing import TYPE_CHECKING

from ...tool.executor import GovernedToolInvoker
from .command import DEFAULT_DENIED_COMMAND_PATTERNS, CommandRule
from .engine import PolicyEngine

if TYPE_CHECKING:
    from ...agent.approval import ApprovalStore


def build_default_command_policy(
    *,
    approval_store: "ApprovalStore | None" = None,
    denied_patterns: "tuple[str, ...]" = DEFAULT_DENIED_COMMAND_PATTERNS,
) -> GovernedToolInvoker:
    """A GovernedToolInvoker with a command denylist. A REQUIRE_APPROVAL decision
    raises RunPaused (the single approval path); wire an ``approval_store`` so
    the pause can be persisted. This is a convenience preset, not a Runtime
    default -- pass it explicitly via
    ``Runtime.build(tool_executor=build_default_command_policy(...))``."""
    return GovernedToolInvoker(
        policy=PolicyEngine(rules=(CommandRule(denied_patterns=denied_patterns),)),
        approval_store=approval_store,
    )


__all__ = ["build_default_command_policy"]
