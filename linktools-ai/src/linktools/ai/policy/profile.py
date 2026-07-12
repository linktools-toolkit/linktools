#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_default_policy_engine: assembles a PolicyEngine from a ToolRegistry so
the rich rules (Permission/Risk/Approval) actually enforce against real tool
declarations. Without this helper only CommandRule -- the one rule that needs
no metadata -- enforces, because nothing else consults the registry.

The helper awaits ``ToolRegistry.get_metadata_map()`` and hands the resulting
``{tool_name: ToolPolicyMetadata}`` mapping to each rich rule. Sensible defaults
are exposed as keyword arguments; callers wanting different policy can pass
overrides here, or build their own ``PolicyEngine`` and hand it to
``ToolExecutor(policy=...)`` directly."""

from typing import TYPE_CHECKING

from .approval import ApprovalRule
from .command import DEFAULT_DENIED_COMMAND_PATTERNS, CommandRule
from .engine import PolicyEngine
from .permission import PermissionRule
from .risk import RiskRule
from .rule import Permission, RiskLevel, SideEffectKind

if TYPE_CHECKING:
    from ..registry.tool import ToolRegistry


# Default allowed permission set: read/write/execute are the routine agent
# surface; NETWORK and ADMIN are not in the default allow-list, so any tool
# declaring them is denied by PermissionRule unless the caller overrides.
_DEFAULT_ALLOWED_PERMISSIONS: "frozenset[Permission]" = frozenset(
    {Permission.READ, Permission.WRITE, Permission.EXECUTE}
)


async def build_default_policy_engine(
    tool_registry: "ToolRegistry",
    *,
    allowed_permissions: "frozenset[Permission]" = _DEFAULT_ALLOWED_PERMISSIONS,
    max_risk: RiskLevel = RiskLevel.HIGH,
    approval_side_effect: SideEffectKind = SideEffectKind.DESTRUCTIVE,
    denied_command_patterns: "tuple[str, ...]" = DEFAULT_DENIED_COMMAND_PATTERNS,
) -> PolicyEngine:
    """Build a ``PolicyEngine`` whose rich rules consult the registry's tool
    metadata.

    Defaults:

    - allow READ/WRITE/EXECUTE (deny NETWORK/ADMIN)
    - deny risk strictly greater than HIGH (CRITICAL is denied)
    - require approval for DESTRUCTIVE side-effects (and anything ranked higher)
    - apply the standard command-deny blacklist via CommandRule

    ``CommandRule`` is included unconditionally because it consults only the
    request's ``command`` argument, never the registry -- omitting it would
    silently disable the terminal blacklist that today's default PolicyEngine
    enforces.
    """
    metadata = await tool_registry.get_metadata_map()
    return PolicyEngine(
        rules=(
            CommandRule(denied_patterns=denied_command_patterns),
            PermissionRule(allowed=allowed_permissions, tool_metadata=metadata),
            RiskRule(max_allowed=max_risk, tool_metadata=metadata),
            ApprovalRule(
                require_side_effect=approval_side_effect,
                tool_metadata=metadata,
            ),
        )
    )
