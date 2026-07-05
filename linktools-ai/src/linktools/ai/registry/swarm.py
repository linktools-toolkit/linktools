#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmRegistry: resolves SwarmSpec from {name}.yaml files via SpecLoader,
revision-cached. Mirrors AgentRegistry — the loader exposes a revision() monotonic
clock; whenever it changes the per-(id, revision) cache and the id listing are
dropped so the next get() re-reads and re-parses the YAML."""

from decimal import Decimal
from typing import Any

from ..agent_runtime.spec import MiddlewareRef
from ..errors import InvalidSpecError, RegistryNotFoundError
from ..swarm_runtime.aggregation import AggregationMode, AggregationPolicy
from ..swarm_runtime.limits import DEFAULT_SWARM_LIMITS, SwarmLimits
from ..swarm_runtime.spec import (
    AgentRef,
    SwarmContextPolicy,
    SwarmSpec,
    SwarmStrategySpec,
)
from .agent import _parse_middleware_refs
from .parser import SpecLoader, parse_yaml_text


def _parse_agent_ref(item: Any, *, swarm_id: str, kind: str) -> AgentRef:
    """Build an AgentRef from a string or {agent_id, role?} mapping."""
    if isinstance(item, str):
        return AgentRef(agent_id=item)
    if isinstance(item, dict) and "agent_id" in item:
        role_raw = item.get("role")
        return AgentRef(
            agent_id=str(item["agent_id"]),
            role=str(role_raw) if role_raw is not None else None,
        )
    raise InvalidSpecError(
        f"swarm {swarm_id}: invalid {kind} ref: {item!r}"
    )


def parse_swarm_spec(swarm_id: str, payload: "dict[str, Any]") -> SwarmSpec:
    """Build a SwarmSpec from a parsed YAML dict."""
    name = payload.get("name") or swarm_id

    # agents — required, non-empty.
    agents_raw = payload.get("agents")
    if not agents_raw:
        raise InvalidSpecError(
            f"swarm {swarm_id}: 'agents' must be a non-empty list"
        )
    if not isinstance(agents_raw, (list, tuple)):
        raise InvalidSpecError(f"swarm {swarm_id}: 'agents' must be a list")
    agents = tuple(
        _parse_agent_ref(a, swarm_id=swarm_id, kind="agent") for a in agents_raw
    )

    # coordinator — required.
    coord_raw = payload.get("coordinator")
    if coord_raw is None:
        raise InvalidSpecError(f"swarm {swarm_id}: 'coordinator' is required")
    coordinator = _parse_agent_ref(coord_raw, swarm_id=swarm_id, kind="coordinator")

    # strategy — default coordinator_delegation.
    strat_raw = payload.get("strategy")
    if strat_raw is None:
        strategy = SwarmStrategySpec(kind="coordinator_delegation")
    elif isinstance(strat_raw, dict):
        kind = strat_raw.get("kind")
        if not kind:
            raise InvalidSpecError(
                f"swarm {swarm_id}: 'strategy.kind' is required"
            )
        strategy = SwarmStrategySpec(
            kind=str(kind),
            config=dict(strat_raw.get("config") or {}),
        )
    else:
        raise InvalidSpecError(f"swarm {swarm_id}: 'strategy' must be a mapping")

    # limits — fall back to DEFAULT_SWARM_LIMITS; missing fields inherit defaults.
    limits_raw = payload.get("limits")
    if limits_raw is None:
        limits = DEFAULT_SWARM_LIMITS
    elif isinstance(limits_raw, dict):
        def _int_field(key: str) -> "int | None":
            v = limits_raw.get(key, getattr(DEFAULT_SWARM_LIMITS, key))
            return None if v is None else int(v)

        def _decimal_field(key: str) -> "Decimal | None":
            v = limits_raw.get(key, getattr(DEFAULT_SWARM_LIMITS, key))
            return None if v is None else Decimal(str(v))

        def _float_field(key: str) -> "float | None":
            v = limits_raw.get(key, getattr(DEFAULT_SWARM_LIMITS, key))
            return None if v is None else float(v)

        limits = SwarmLimits(
            max_rounds=_int_field("max_rounds"),
            max_tasks=_int_field("max_tasks"),
            max_delegations=_int_field("max_delegations"),
            max_depth=_int_field("max_depth"),
            max_concurrency=_int_field("max_concurrency"),
            max_total_tokens=_int_field("max_total_tokens"),
            max_total_cost=_decimal_field("max_total_cost"),
            timeout_seconds=_float_field("timeout_seconds"),
        )
    else:
        raise InvalidSpecError(f"swarm {swarm_id}: 'limits' must be a mapping")

    # context_policy — default SwarmContextPolicy().
    cp_raw = payload.get("context_policy")
    if cp_raw is None:
        context_policy = SwarmContextPolicy()
    elif isinstance(cp_raw, dict):
        try:
            context_policy = SwarmContextPolicy(**cp_raw)
        except TypeError as exc:
            raise InvalidSpecError(
                f"swarm {swarm_id}: invalid 'context_policy': {exc}"
            ) from exc
    else:
        raise InvalidSpecError(
            f"swarm {swarm_id}: 'context_policy' must be a mapping"
        )

    # aggregation — string or {mode: ...}, default CONCAT.
    agg_raw = payload.get("aggregation")
    if agg_raw is None:
        aggregation = AggregationPolicy()
    elif isinstance(agg_raw, str):
        try:
            aggregation = AggregationPolicy(mode=AggregationMode(agg_raw))
        except ValueError as exc:
            raise InvalidSpecError(
                f"swarm {swarm_id}: unknown aggregation mode: {agg_raw!r}"
            ) from exc
    elif isinstance(agg_raw, dict):
        mode_raw = agg_raw.get("mode", AggregationMode.CONCAT.value)
        try:
            aggregation = AggregationPolicy(mode=AggregationMode(mode_raw))
        except ValueError as exc:
            raise InvalidSpecError(
                f"swarm {swarm_id}: unknown aggregation mode: {mode_raw!r}"
            ) from exc
    else:
        raise InvalidSpecError(
            f"swarm {swarm_id}: 'aggregation' must be a string or mapping"
        )

    # middleware — reuse the agent registry's helper.
    middleware: "tuple[MiddlewareRef, ...]" = _parse_middleware_refs(
        payload.get("middleware")
    )

    # metadata — optional mapping.
    metadata = dict(payload.get("metadata") or {})

    return SwarmSpec(
        id=swarm_id,
        name=str(name),
        agents=agents,
        coordinator=coordinator,
        strategy=strategy,
        limits=limits,
        context_policy=context_policy,
        aggregation=aggregation,
        middleware=middleware,
        metadata=metadata,
    )


class SwarmRegistry:
    """Loads SwarmSpecs from `{name}.yaml` files via a SpecLoader, revision-cached.

    Mirrors AgentRegistry: the loader exposes a revision() monotonic clock; whenever
    it changes the per-(id, revision) cache and the id listing are dropped so the
    next get() re-reads and re-parses the YAML.
    """

    def __init__(self, loader: SpecLoader, *, suffix: str = ".yaml") -> None:
        self._loader = loader
        self._suffix = suffix
        self._cache: "dict[tuple[str, int], SwarmSpec]" = {}
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

    async def get(self, swarm_id: str) -> SwarmSpec:
        await self._ensure_fresh()
        revision = self._cached_revision if self._cached_revision is not None else 0
        cache_key = (swarm_id, revision)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            text = await self._loader.read(f"{swarm_id}{self._suffix}")
        except RegistryNotFoundError:
            raise
        payload = parse_yaml_text(text, source=f"{swarm_id}{self._suffix}")
        spec = parse_swarm_spec(swarm_id, payload)
        self._cache[cache_key] = spec
        return spec
