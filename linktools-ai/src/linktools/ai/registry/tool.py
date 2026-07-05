#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolSpec (section 26.1 minimal) + ToolRegistry: loads tool declarations from
YAML via SpecLoader, caches per-revision, and exposes to_metadata_map() (the
bridge the policy rule modules -- PermissionRule/RiskRule/ApprovalRule -- consume)."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..errors import InvalidSpecError
from ..policy.rule import (
    ApprovalMode,
    Permission,
    RiskLevel,
    SideEffectKind,
    ToolPolicyMetadata,
)
from .parser import SpecLoader, parse_yaml_text


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str = ""
    permissions: "frozenset[Permission]" = field(
        default_factory=lambda: frozenset({Permission.READ})
    )
    risk: RiskLevel = RiskLevel.LOW
    side_effect: SideEffectKind = SideEffectKind.READ_ONLY
    approval: ApprovalMode = ApprovalMode.NEVER
    idempotent: bool = False
    timeout_seconds: "float | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


# Lookup tables keyed by the YAML-facing token (Permission/SideEffectKind/ApprovalMode
# values are lowercase strings; RiskLevel names are uppercase).
_PERMISSION_LOOKUP = {p.value: p for p in Permission}
_RISK_LOOKUP = {r.name: r for r in RiskLevel}
_SIDE_EFFECT_LOOKUP = {s.value: s for s in SideEffectKind}
_APPROVAL_LOOKUP = {a.value: a for a in ApprovalMode}


def _parse_permissions(items: Any) -> "frozenset[Permission]":
    # Omitting the key entirely means "default read-only permission"; an explicit
    # empty list means "no permissions granted".
    if items is None:
        return frozenset({Permission.READ})
    if not isinstance(items, (list, tuple)):
        raise InvalidSpecError("permissions must be a list")
    perms: "set[Permission]" = set()
    for item in items:
        key = item if isinstance(item, str) else str(item)
        if key not in _PERMISSION_LOOKUP:
            raise InvalidSpecError(f"unknown permission: {item!r}")
        perms.add(_PERMISSION_LOOKUP[key])
    return frozenset(perms)


def _parse_tool_spec(name: str, payload: "dict[str, Any]") -> ToolSpec:
    risk_key = str(payload.get("risk", "LOW")).upper()
    if risk_key not in _RISK_LOOKUP:
        raise InvalidSpecError(f"unknown risk level: {payload.get('risk')!r}")
    side_key = str(payload.get("side_effect", "read_only")).lower()
    if side_key not in _SIDE_EFFECT_LOOKUP:
        raise InvalidSpecError(f"unknown side_effect: {side_key!r}")
    approval_key = str(payload.get("approval", "never")).lower()
    if approval_key not in _APPROVAL_LOOKUP:
        raise InvalidSpecError(f"unknown approval mode: {approval_key!r}")
    timeout = payload.get("timeout_seconds")
    if timeout is not None and not isinstance(timeout, (int, float)):
        raise InvalidSpecError(f"timeout_seconds must be a number: {timeout!r}")
    return ToolSpec(
        name=name,
        description=str(payload.get("description", "")),
        permissions=_parse_permissions(payload.get("permissions")),
        risk=_RISK_LOOKUP[risk_key],
        side_effect=_SIDE_EFFECT_LOOKUP[side_key],
        approval=_APPROVAL_LOOKUP[approval_key],
        idempotent=bool(payload.get("idempotent", False)),
        timeout_seconds=float(timeout) if timeout is not None else None,
        metadata=dict(payload.get("metadata") or {}),
    )


class ToolRegistry:
    """Loads ToolSpecs from `{name}.yaml` files via a SpecLoader, revision-cached."""

    def __init__(self, loader: SpecLoader, *, suffix: str = ".yaml") -> None:
        self._loader = loader
        self._suffix = suffix
        self._cache: "dict[tuple[str, int], ToolSpec]" = {}
        self._cached_revision: "int | None" = None
        self._ids: "tuple[str, ...] | None" = None

    async def _ensure_fresh(self) -> None:
        revision = await self._loader.revision()
        if revision != self._cached_revision:
            self._cache.clear()
            self._ids = None
            self._cached_revision = revision

    async def list_ids(self) -> "tuple[str, ...]":
        await self._ensure_fresh()
        if self._ids is None:
            self._ids = await self._loader.list_ids(self._suffix)
        return self._ids

    async def get(self, tool_id: str) -> ToolSpec:
        await self._ensure_fresh()
        revision = self._cached_revision if self._cached_revision is not None else 0
        cache_key = (tool_id, revision)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        text = await self._loader.read(f"{tool_id}{self._suffix}")
        payload = parse_yaml_text(text, source=f"{tool_id}{self._suffix}")
        spec = _parse_tool_spec(tool_id, payload)
        self._cache[cache_key] = spec
        return spec

    async def to_metadata_map(self) -> "Mapping[str, ToolPolicyMetadata]":
        """Return {tool_name: ToolPolicyMetadata} for every loaded tool -- the bridge
        the PermissionRule/RiskRule/ApprovalRule consume."""
        ids = await self.list_ids()
        result: "dict[str, ToolPolicyMetadata]" = {}
        for tool_id in ids:
            spec = await self.get(tool_id)
            result[spec.name] = ToolPolicyMetadata(
                permissions=spec.permissions,
                risk=spec.risk,
                side_effect=spec.side_effect,
                approval=spec.approval,
            )
        return result
