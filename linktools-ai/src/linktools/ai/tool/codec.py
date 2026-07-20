#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolSpecCodec: the CatalogCodec[ToolSpec] for the tool domain.

Owns the tool-specific parsing (moved here from registry/tool.py): a
``{name}.yaml`` item is parsed as YAML, strictly validated, and built into a
ToolSpec. Reuses the policy layer's validate_idempotency_policy so the codec
rejects the same bad combinations at load time."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping

from ..catalog import CatalogCodec
from ..catalog.parsing import (
    StrictConfigReader,
    parse_yaml_text,
)
from ..errors import InvalidSpecError
from ..governance.policy.rule import (
    ApprovalMode,
    Permission,
    RiskLevel,
    SideEffectKind,
)
from .models import ToolRef
from .policy import IdempotencyStrategy, validate_idempotency_policy
from .spec import ToolSpec


def parse_tool_refs(items: Any) -> "tuple[Any, ...] | None":
    """Build a tuple[ToolRef] from a list of tool declarations.

    Tool declarations are explicit mappings with string ``kind`` and ``name``;
    unknown fields are rejected and the names are normalized (stripped) so a
    stray space cannot silently turn into a different tool identity. Moved here
    from catalog/parsing -- it builds a tool-domain type (ToolRef), so it is
    tool-specific, not generic parser infra.
    """
    if items is None:
        # Distinguish "no tools key" (None -> runtime default) from "tools: []"
        # (empty tuple -> explicitly no tools) -- the three-state distinction.
        return None
    if not isinstance(items, (list, tuple)):
        raise InvalidSpecError("tools must be a list")
    refs: "list[Any]" = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise InvalidSpecError(f"tools[{index}]: invalid tool ref: {item!r}")
        item_reader = StrictConfigReader(
            item,
            allowed={"kind", "name", "config"},
            context=f"tools[{index}]",
        )
        kind = item_reader.required_str("kind").strip()
        if not kind:
            raise InvalidSpecError(f"tools[{index}]: kind must not be blank")
        name = item_reader.required_str("name").strip()
        if not name:
            raise InvalidSpecError(f"tools[{index}]: name must not be blank")
        config = item_reader.mapping("config") or {}
        refs.append(ToolRef(name=name, kind=kind, config=config))
    return tuple(refs)


def _reject_null(payload, name):
    """Return the value if present, None if missing, or raise on explicit null."""
    if name in payload and payload[name] is None:
        raise InvalidSpecError(f"tool policy: {name} must not be null")
    return payload.get(name)


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
    # Reuse the policy layer's rules so the codec rejects the same bad
    # combinations (e.g. business_key without idempotency_key_field) at load
    # time rather than letting them reach the first tool call.
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
        permissions=_parse_permissions(_reject_null(payload, "permissions")),
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


class ToolSpecCodec:
    """CatalogCodec[ToolSpec]: decode one ``{id}.yaml`` item's raw text into a
    ToolSpec. Strict; propagates the domain's rich errors."""

    def decode(self, item_id: str, raw: str) -> ToolSpec:
        source = f"{item_id}.yaml"
        payload = parse_yaml_text(raw, source=source)
        return _parse_tool_spec(item_id, payload)


__all__: "list[str]" = ["ToolSpecCodec", "_parse_tool_spec"]
