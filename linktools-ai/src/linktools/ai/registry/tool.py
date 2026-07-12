#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolSpec + ToolRegistry: loads tool declarations from
YAML via SpecLoader, caches per-revision, and exposes get_metadata_map() (the
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
from .parser import SpecLoader, StrictConfigReader, parse_yaml_text


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str = ""
    enabled: bool = True
    permissions: "frozenset[Permission]" = field(
        default_factory=lambda: frozenset({Permission.READ})
    )
    risk: RiskLevel = RiskLevel.LOW
    side_effect: SideEffectKind = SideEffectKind.READ_ONLY
    approval: ApprovalMode = ApprovalMode.NEVER
    idempotent: "bool | None" = None
    timeout_seconds: "float | None" = None
    max_retries: "int | None" = None
    idempotency_strategy: "str | None" = None
    idempotency_key_field: "str | None" = None
    # bump when a tool's input contract changes shape so an idempotency
    # hash computed under the old schema is never mistaken for a match
    # against the new one (see tool/idempotency.py compute_request_hash).
    schema_version: str = "1"
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
        if not isinstance(item, str):
            raise InvalidSpecError(f"permission must be a string: {item!r}")
        key = item
        if key not in _PERMISSION_LOOKUP:
            raise InvalidSpecError(f"unknown permission: {item!r}")
        perms.add(_PERMISSION_LOOKUP[key])
    return frozenset(perms)


def _parse_tool_spec(name: str, payload: "dict[str, Any]") -> ToolSpec:
    allowed = {
        "description",
        "enabled",
        "permissions",
        "risk",
        "side_effect",
        "approval",
        "idempotent",
        "timeout_seconds",
        "max_retries",
        "schema_version",
        "idempotency_strategy",
        "idempotency_key_field",
        "metadata",
        "name",
    }
    reader = StrictConfigReader(payload, allowed=allowed, context=f"tool {name}")
    risk_raw = payload.get("risk", "LOW")
    if not isinstance(risk_raw, str):
        raise InvalidSpecError("risk must be a string")
    risk_key = risk_raw.upper()
    if risk_key not in _RISK_LOOKUP:
        raise InvalidSpecError(f"unknown risk level: {payload.get('risk')!r}")
    side_raw = payload.get("side_effect", "read_only")
    if not isinstance(side_raw, str):
        raise InvalidSpecError("side_effect must be a string")
    side_key = side_raw.lower()
    if side_key not in _SIDE_EFFECT_LOOKUP:
        raise InvalidSpecError(f"unknown side_effect: {side_key!r}")
    approval_raw = payload.get("approval", "never")
    if not isinstance(approval_raw, str):
        raise InvalidSpecError("approval must be a string")
    approval_key = approval_raw.lower()
    if approval_key not in _APPROVAL_LOOKUP:
        raise InvalidSpecError(f"unknown approval mode: {approval_key!r}")
    enabled = reader.bool("enabled", default=True)
    idempotent = reader.bool("idempotent")
    timeout = reader.positive_number("timeout_seconds")
    retries = reader.non_negative_int("max_retries")
    schema_version = payload.get("schema_version", "1")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise InvalidSpecError("schema_version must be non-empty")
    strategy = payload.get("idempotency_strategy")
    if strategy is not None and strategy not in ("exact_call", "business_key"):
        raise InvalidSpecError(
            "idempotency_strategy must be 'exact_call' or 'business_key'"
        )
    key_field = reader.optional_str("idempotency_key_field")
    if key_field is not None:
        key_field = key_field.strip()
        if not key_field:
            raise InvalidSpecError("idempotency_key_field must be non-empty")
    # Reuse the policy layer's rules so the registry rejects the same bad
    # combinations (e.g. business_key without idempotency_key_field) at load time
    # rather than letting them reach the first tool call.
    from ..tool.policy import IdempotencyStrategy, validate_idempotency_policy

    strategy_enum = IdempotencyStrategy(strategy) if strategy is not None else None
    try:
        validate_idempotency_policy(
            idempotent=idempotent,
            strategy=strategy_enum,
            key_field=key_field,
            effective=False,
        )
    except ValueError as exc:
        raise InvalidSpecError(str(exc)) from exc
    return ToolSpec(
        name=name,
        description=reader.optional_str("description") or "",
        enabled=enabled,
        permissions=_parse_permissions(payload.get("permissions")),
        risk=_RISK_LOOKUP[risk_key],
        side_effect=_SIDE_EFFECT_LOOKUP[side_key],
        approval=_APPROVAL_LOOKUP[approval_key],
        idempotent=idempotent,
        timeout_seconds=timeout,
        max_retries=retries,
        schema_version=schema_version,
        idempotency_strategy=strategy,
        idempotency_key_field=key_field,
        metadata=reader.mapping("metadata") or {},
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

    async def get_metadata_map(self) -> "Mapping[str, ToolPolicyMetadata]":
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
                idempotent=spec.idempotent,
                timeout_seconds=spec.timeout_seconds,
                schema_version=spec.schema_version,
                enabled=spec.enabled,
                max_retries=spec.max_retries,
                idempotency_strategy=spec.idempotency_strategy,
                idempotency_key_field=spec.idempotency_key_field,
                metadata=spec.metadata,
            )
        return result
