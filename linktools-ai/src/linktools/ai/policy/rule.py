#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Policy enums + ToolPolicyMetadata: the policy-relevant slice of a tool's
declaration (section 26.1), consumed by the rule modules (permission/risk/path/
network/approval). The full ToolSpec lands in registry/tool.py (Task 11)."""

from dataclasses import dataclass
from enum import Enum


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    ADMIN = "admin"


class RiskLevel(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SideEffectKind(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    NAMESPACE_MUTATING = "namespace_mutating"
    DESTRUCTIVE = "destructive"


class ApprovalMode(str, Enum):
    NEVER = "never"
    ON_RISK = "on_risk"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class ToolPolicyMetadata:
    permissions: "frozenset[Permission]"
    risk: RiskLevel
    side_effect: SideEffectKind
    approval: ApprovalMode
