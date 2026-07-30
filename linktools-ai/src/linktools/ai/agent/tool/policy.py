#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool policy: runtime resolution (ResolvedToolPolicy / ToolPolicyResolver),
the spec codec (ToolSpecCodec), and the spec index (ToolSpecIndex)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from ..assembly.models import AgentFeatureRef
from ...errors import InvalidSpecError, ToolPolicyResolutionError
from ...governance.policy.rule import (
    ApprovalMode,
    Permission,
    RiskLevel,
    SideEffectKind,
    ToolPolicyMetadata,
)
from ...json import freeze_value
from ...spec import SpecCodec, SpecSource
from ...spec.index import SpecIndex
from ...spec.parsing import SpecLoader, StrictConfigReader, parse_yaml_text
from ...spec.source import SpecLoaderSource
from .models import ToolDescriptor, ToolSpec


# --- runtime resolution ------------------------------------------------------

class IdempotencyStrategy(str, Enum):
    EXACT_CALL = "exact_call"
    BUSINESS_KEY = "business_key"


def parse_idempotency_strategy(
    value: "str | IdempotencyStrategy | None",
) -> "IdempotencyStrategy | None":
    if value is None or isinstance(value, IdempotencyStrategy):
        return value
    try:
        return IdempotencyStrategy(value)
    except (TypeError, ValueError) as exc:
        from ...errors import ToolPolicyResolutionError

        raise ToolPolicyResolutionError(
            f"invalid idempotency strategy: {value!r}"
        ) from exc


def validate_idempotency_policy(
    *,
    idempotent: "bool | None",
    strategy: "IdempotencyStrategy | None",
    key_field: "str | None",
    effective: bool,
) -> None:
    """Single source of truth for the idempotency-field combination rules, shared
    by the registry parser and both policy dataclasses so the three layers cannot
    drift (a missing business_key key_field used to pass declaration and only
    surface on the first tool call).

    ``effective`` distinguishes the tri-state declaration layer (ResolvedToolPolicy)
    from the finalized concrete layer (EffectiveToolPolicy). EXACT_CALL is the
    legitimate default strategy of a non-idempotent finalized policy, so it is not
    treated as a contradictory strategy at the effective layer; an explicitly
    declared EXACT_CALL on an idempotent=False declaration still is."""
    if key_field is not None:
        key_field = key_field.strip()
    if strategy == IdempotencyStrategy.BUSINESS_KEY:
        if idempotent is False:
            raise ValueError("business_key requires idempotent=true")
        if effective and idempotent is not True:
            raise ValueError("effective business_key policy requires idempotent=true")
        if not key_field:
            raise ValueError("business_key requires idempotency_key_field")
    if key_field and strategy != IdempotencyStrategy.BUSINESS_KEY:
        raise ValueError("idempotency_key_field is only valid for business_key")
    if idempotent is False and strategy is not None:
        # idempotent=False + EXACT_CALL is the normal finalized non-idempotent
        # policy (EXACT_CALL is finalize_policy()'s fallback). At the declaration
        # layer an explicit EXACT_CALL still contradicts idempotent=False.
        if not (effective and strategy == IdempotencyStrategy.EXACT_CALL):
            raise ValueError("idempotency_strategy requires idempotent=true")


if TYPE_CHECKING:
    from ..execution.context import RunContext


@dataclass(frozen=True, slots=True)
class ResolvedToolPolicy:
    """One policy layer for a single tool invocation. Every field is
    tri-state (``None`` = not declared by this layer) so a layer that omits a
    field never overrides a more specific layer's explicit 0/False -- only
    finalize_policy() collapses the merged result to concrete values."""

    enabled: "bool | None" = None
    timeout_seconds: "float | None" = None
    max_retries: "int | None" = None
    idempotent: "bool | None" = None
    require_approval: "bool | None" = None
    risk: "str | None" = None
    schema_version: "str | None" = None
    idempotency_strategy: "IdempotencyStrategy | None" = None
    idempotency_key_field: "str | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0 or None, got {self.timeout_seconds}"
            )
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        strategy = parse_idempotency_strategy(self.idempotency_strategy)
        object.__setattr__(self, "idempotency_strategy", strategy)
        key_field = self.idempotency_key_field
        if key_field is not None:
            key_field = key_field.strip() or None
            object.__setattr__(self, "idempotency_key_field", key_field)
        if self.schema_version is not None:
            version = self.schema_version.strip()
            if not version:
                raise ValueError("schema_version must be non-empty")
            object.__setattr__(self, "schema_version", version)
        validate_idempotency_policy(
            idempotent=self.idempotent,
            strategy=strategy,
            key_field=key_field,
            effective=False,
        )
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class EffectiveToolPolicy:
    """The merged tri-state ResolvedToolPolicy collapsed to concrete values --
    what ToolExecutionService actually acts
    on. Produced only by finalize_policy()."""

    enabled: bool = True
    timeout_seconds: "float | None" = None
    max_retries: int = 0
    idempotent: bool = False
    require_approval: bool = False
    risk: str = "medium"
    schema_version: str = "1"
    idempotency_strategy: IdempotencyStrategy = IdempotencyStrategy.EXACT_CALL
    idempotency_key_field: "str | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        strategy = parse_idempotency_strategy(self.idempotency_strategy)
        if strategy is None:
            strategy = IdempotencyStrategy.EXACT_CALL
        key_field = self.idempotency_key_field
        if key_field is not None:
            key_field = key_field.strip() or None
            object.__setattr__(self, "idempotency_key_field", key_field)
        version = self.schema_version.strip()
        if not version:
            raise ValueError("schema_version must be non-empty")
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "idempotency_strategy", strategy)
        validate_idempotency_policy(
            idempotent=self.idempotent,
            strategy=strategy,
            key_field=key_field,
            effective=True,
        )
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """Everything the governance chain (pipeline, executor, middleware, events)
    needs for one tool call."""

    descriptor: ToolDescriptor
    arguments: "Mapping[str, Any]"
    run_context: "RunContext"
    policy: ResolvedToolPolicy
    call_id: str


@runtime_checkable
class ToolPolicyResolver(Protocol):
    """Resolves a tool descriptor + run context into a ResolvedToolPolicy. When
    absent, the runtime uses ResolvedToolPolicy() (defaults) + SecurityBaseline."""

    async def resolve(
        self,
        descriptor: ToolDescriptor,
        context: "RunContext",
    ) -> ResolvedToolPolicy: ...


# --- Policy merge ---


def merge_policies(
    descriptor_default: "ResolvedToolPolicy | None",
    baseline: "ResolvedToolPolicy | None",
    provider: "ResolvedToolPolicy | None" = None,
) -> ResolvedToolPolicy:
    """Merge tri-state policy layers into a (still tri-state) ResolvedToolPolicy.
    A field left undeclared (None) by every layer stays None -- only
    finalize_policy() applies a concrete default. Rules (only over layers that
    declared the field; an undeclared layer never contributes):
    - enabled: any declared False -> False; else True if any layer declared it,
      else None
    - timeout: smallest declared value, else None
    - max_retries: smallest declared value, else None
    - idempotent: True only if every layer that declared it declared True
      (and at least one did); any declared False -> False; else None
    - require_approval: any declared True -> True; else False if any layer
      declared it, else None
    - risk: highest declared level, else None
    - metadata: shallow-merged across layers (later layers win per key) so no
      layer's metadata is silently dropped
    """
    layers = [
        layer for layer in (descriptor_default, baseline, provider) if layer is not None
    ]
    if not layers:
        return ResolvedToolPolicy()

    def _merge_bool_any_true(values: "list[bool]") -> "bool | None":
        if not values:
            return None
        return True if any(values) else False

    declared_enabled = [layer.enabled for layer in layers if layer.enabled is not None]
    enabled = None
    if declared_enabled:
        enabled = False if any(v is False for v in declared_enabled) else True

    timeouts = [
        layer.timeout_seconds for layer in layers if layer.timeout_seconds is not None
    ]
    timeout = min(timeouts) if timeouts else None

    retries = [layer.max_retries for layer in layers if layer.max_retries is not None]
    max_retries = min(retries) if retries else None

    declared_idempotent = [
        layer.idempotent for layer in layers if layer.idempotent is not None
    ]
    idempotent = None
    if declared_idempotent:
        idempotent = all(declared_idempotent)

    declared_approval = [
        layer.require_approval for layer in layers if layer.require_approval is not None
    ]
    require_approval = _merge_bool_any_true(declared_approval)

    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    risks = [layer.risk for layer in layers if layer.risk]
    risk = max(risks, key=lambda r: risk_order.get(r, 2)) if risks else None

    # schema_version: explicit priority, NOT string sort. Layers are ordered
    # (descriptor, baseline, provider) -- the most specific layer (provider,
    # last) wins. Only the provider layer should set schema_version; baseline
    # never does. No max()/min() -- that would do lexicographic string sort
    # ("10" < "2") which is semantically wrong for versions.
    schema_version = None
    for layer in layers:
        if layer.schema_version:
            schema_version = layer.schema_version

    metadata: "dict[str, Any]" = {}
    for layer in layers:
        if layer.metadata:
            metadata.update(layer.metadata)

    return ResolvedToolPolicy(
        enabled=enabled,
        timeout_seconds=timeout,
        max_retries=max_retries,
        idempotent=idempotent,
        require_approval=require_approval,
        risk=risk,
        schema_version=schema_version,
        idempotency_strategy=next(
            (
                layer.idempotency_strategy
                for layer in reversed(layers)
                if layer.idempotency_strategy is not None
            ),
            None,
        ),
        idempotency_key_field=next(
            (
                layer.idempotency_key_field
                for layer in reversed(layers)
                if layer.idempotency_key_field is not None
            ),
            None,
        ),
        metadata=metadata,
    )


def finalize_policy(resolved: "ResolvedToolPolicy | None") -> EffectiveToolPolicy:
    """Collapse a (possibly partially tri-state) merged ResolvedToolPolicy into
    the concrete EffectiveToolPolicy the execution chain acts on. Undeclared
    fields get the safe default: enabled defaults open (True) since an absent
    policy layer must not silently disable every tool; idempotent/
    require_approval default closed (False) since "not declared" must never be
    read as an implicit safety opt-in."""
    if resolved is None:
        return EffectiveToolPolicy()
    return EffectiveToolPolicy(
        enabled=True if resolved.enabled is None else resolved.enabled,
        timeout_seconds=resolved.timeout_seconds,
        max_retries=0 if resolved.max_retries is None else resolved.max_retries,
        idempotent=bool(resolved.idempotent),
        require_approval=bool(resolved.require_approval),
        risk=resolved.risk or "medium",
        schema_version=resolved.schema_version or "1",
        idempotency_strategy=resolved.idempotency_strategy
        or IdempotencyStrategy.EXACT_CALL,
        idempotency_key_field=resolved.idempotency_key_field,
        metadata=resolved.metadata,
    )


class MetadataBackedPolicyProvider:
    """Wraps a ``get_metadata_map()`` provider and resolves a
    ToolDescriptor into a ResolvedToolPolicy by looking up the tool's metadata.
    Tools not in the metadata map get default policy (enabled, no approval).
    Provider errors fail closed."""

    def __init__(self, metadata_provider: Any) -> None:
        self._provider = metadata_provider

    async def resolve(
        self, descriptor: ToolDescriptor, context: "RunContext"
    ) -> ResolvedToolPolicy:
        # Fail closed: if the underlying metadata source is unavailable, raise
        # so ToolExecutionService can emit a degraded event and deny
        # the call -- never run a tool ungoverned because its policy couldn't
        # be resolved.
        try:
            metadata_map = await self._provider.get_metadata_map()
        except Exception as exc:
            raise ToolPolicyResolutionError(
                f"tool policy metadata source unavailable for {descriptor.name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        meta = metadata_map.get(descriptor.name)
        if meta is None:
            return ResolvedToolPolicy()
        risk_val = getattr(meta, "risk", None)
        if isinstance(risk_val, str):
            risk = risk_val.lower()
        elif risk_val is not None:
            risk = risk_val.name.lower()
        else:
            risk = "medium"
        approval = getattr(meta, "approval", None)
        from ...governance.policy.rule import ApprovalMode

        require_approval = approval not in (None, ApprovalMode.NEVER)
        side_effect = getattr(meta, "side_effect", None)
        idempotent = getattr(meta, "idempotent", None)
        timeout = getattr(meta, "timeout_seconds", None)
        meta_extra = dict(getattr(meta, "metadata", {}) or {})
        namespaced: "dict[str, Any]" = {
            "permissions": [str(p) for p in getattr(meta, "permissions", frozenset())],
            "side_effect": str(side_effect) if side_effect is not None else "read_only",
        }
        if meta_extra:
            namespaced["source_metadata"] = meta_extra
        return ResolvedToolPolicy(
            enabled=getattr(meta, "enabled", True),
            timeout_seconds=float(timeout) if timeout is not None else None,
            max_retries=getattr(meta, "max_retries", None),
            idempotent=idempotent,
            require_approval=require_approval,
            risk=risk,
            schema_version=str(getattr(meta, "schema_version", "1")) or None,
            idempotency_strategy=parse_idempotency_strategy(
                getattr(meta, "idempotency_strategy", None)
            ),
            idempotency_key_field=getattr(meta, "idempotency_key_field", None),
            metadata=namespaced,
        )


# --- spec codec ---------------------------------------------------------------

def parse_tool_refs(items: Any) -> "tuple[Any, ...] | None":
    """Build a tuple[AgentFeatureRef] from a list of tool declarations.

    Tool declarations are explicit mappings with string ``kind`` and ``name``;
    unknown fields are rejected and the names are normalized (stripped) so a
    stray space cannot silently turn into a different tool identity. It builds
    a tool-domain type, so it belongs here rather than in generic spec parsing.
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
        refs.append(AgentFeatureRef(name=name, kind=kind, config=config))
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
    """SpecCodec[ToolSpec]: decode one ``{id}.yaml`` item's raw text into a
    ToolSpec. Strict; propagates the domain's rich errors."""

    def decode(self, item_id: str, raw: str) -> ToolSpec:
        source = f"{item_id}.yaml"
        payload = parse_yaml_text(raw, source=source)
        return _parse_tool_spec(item_id, payload)


__all__: "list[str]" = ["ToolSpecCodec", "_parse_tool_spec"]


# --- spec index ---------------------------------------------------------------

class ToolSpecIndex(SpecIndex[ToolSpec]):

    def __init__(
        self,
        source: SpecSource,
        *,
        codec: "ToolSpecCodec | None" = None,
        suffix: str = ".yaml",
    ) -> None:
        super().__init__(
            source,
            codec or ToolSpecCodec(),
            suffix=suffix,
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".yaml"
    ) -> "ToolSpecIndex":
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def get_metadata_map(self) -> "Mapping[str, ToolPolicyMetadata]":
        """Return {tool_name: ToolPolicyMetadata} for every loaded tool -- the
        bridge the PermissionRule/RiskRule/ApprovalRule consume."""
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


__all__: "list[str]" = ["ToolSpecIndex"]
