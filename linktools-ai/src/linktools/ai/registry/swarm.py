#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmRegistry: resolves SwarmSpec from {name}.yaml files via SpecLoader,
revision-cached. Mirrors AgentRegistry — the loader exposes a revision() monotonic
clock; whenever it changes the per-(id, revision) cache and the id listing are
dropped so the next get() re-reads and re-parses the YAML."""

from typing import Any

from collections.abc import Mapping

from ..agent.spec import MiddlewareRef
from ..errors import InvalidSpecError, RegistryNotFoundError
from ..swarm.aggregation import AggregationMode, AggregationPolicy
from ..swarm.limits import DEFAULT_SWARM_LIMITS, SwarmLimits
from ..swarm.spec import (
    AgentRef,
    SwarmContextPolicy,
    SwarmSpec,
    SwarmStrategySpec,
)
from .agent import _parse_middleware_refs
from .parser import SpecLoader, StrictConfigReader, parse_yaml_text


def _parse_agent_ref(item: Any, *, swarm_id: str, kind: str) -> AgentRef:
    """Build an AgentRef from a string or {agent_id, role?} mapping. String and
    agent_id forms are stripped; unknown fields are rejected."""
    if isinstance(item, str):
        agent_id = item.strip()
        if not agent_id:
            raise InvalidSpecError(
                f"swarm {swarm_id}: {kind} agent_id must not be blank"
            )
        return AgentRef(agent_id=agent_id)
    if not isinstance(item, Mapping):
        raise InvalidSpecError(f"swarm {swarm_id}: invalid {kind} ref: {item!r}")
    item_reader = StrictConfigReader(
        item,
        allowed={"agent_id", "role"},
        context=f"swarm {swarm_id}: {kind}",
    )
    agent_id = item_reader.required_str("agent_id").strip()
    if not agent_id:
        raise InvalidSpecError(f"swarm {swarm_id}: {kind} agent_id must not be blank")
    role = item_reader.optional_str("role")
    if role is not None:
        role = role.strip()
        if not role:
            raise InvalidSpecError(f"swarm {swarm_id}: {kind} role must not be blank")
    return AgentRef(agent_id=agent_id, role=role)


def parse_swarm_spec(swarm_id: str, payload: "dict[str, Any]") -> SwarmSpec:
    """Build a SwarmSpec from a parsed YAML dict."""
    allowed = {
        "name",
        "agents",
        "coordinator",
        "strategy",
        "limits",
        "context_policy",
        "aggregation",
        "middleware",
        "metadata",
    }
    reader = StrictConfigReader(payload, allowed=allowed, context=f"swarm {swarm_id}")
    name = reader.optional_str("name") or swarm_id

    # agents — required, non-empty.
    agents_raw = payload.get("agents")
    if not agents_raw:
        raise InvalidSpecError(f"swarm {swarm_id}: 'agents' must be a non-empty list")
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
        if not isinstance(kind, str) or not kind.strip():
            raise InvalidSpecError(f"swarm {swarm_id}: 'strategy.kind' is required")
        strategy = SwarmStrategySpec(
            kind=kind,
            config=StrictConfigReader(
                strat_raw,
                allowed={"kind", "config"},
                context=f"swarm {swarm_id}.strategy",
            ).mapping("config")
            or {},
        )
    else:
        raise InvalidSpecError(f"swarm {swarm_id}: 'strategy' must be a mapping")

    # limits — fall back to DEFAULT_SWARM_LIMITS; missing fields inherit defaults.
    limits_raw = payload.get("limits")
    if limits_raw is None:
        limits = DEFAULT_SWARM_LIMITS
    elif isinstance(limits_raw, dict):
        limits_reader = StrictConfigReader(
            limits_raw,
            allowed={
                "max_rounds",
                "max_tasks",
                "max_delegations",
                "max_depth",
                "max_concurrency",
                "max_total_tokens",
                "max_total_cost",
                "timeout_seconds",
            },
            context=f"swarm {swarm_id}.limits",
        )
        limits = SwarmLimits(
            max_rounds=limits_reader.positive_int(
                "max_rounds", DEFAULT_SWARM_LIMITS.max_rounds
            ),
            max_tasks=limits_reader.positive_int(
                "max_tasks", DEFAULT_SWARM_LIMITS.max_tasks
            ),
            max_delegations=limits_reader.non_negative_int(
                "max_delegations", DEFAULT_SWARM_LIMITS.max_delegations
            ),
            max_depth=limits_reader.non_negative_int(
                "max_depth", DEFAULT_SWARM_LIMITS.max_depth
            ),
            max_concurrency=limits_reader.positive_int(
                "max_concurrency", DEFAULT_SWARM_LIMITS.max_concurrency
            ),
            max_total_tokens=limits_reader.positive_int("max_total_tokens"),
            max_total_cost=limits_reader.non_negative_decimal("max_total_cost"),
            timeout_seconds=limits_reader.positive_number(
                "timeout_seconds", DEFAULT_SWARM_LIMITS.timeout_seconds
            ),
        )
    else:
        raise InvalidSpecError(f"swarm {swarm_id}: 'limits' must be a mapping")

    # context_policy — default SwarmContextPolicy().
    cp_raw = reader.mapping("context_policy")
    if cp_raw is None:
        context_policy = SwarmContextPolicy()
    else:
        cp_reader = StrictConfigReader(
            cp_raw,
            allowed={
                "coordinator_reads_session",
                "worker_reads_session",
                "worker_reads_summary",
                "write_aggregate_to_session",
            },
            context=f"swarm {swarm_id}.context_policy",
        )
        context_policy = SwarmContextPolicy(
            coordinator_reads_session=cp_reader.bool("coordinator_reads_session", True),
            worker_reads_session=cp_reader.bool("worker_reads_session", False),
            worker_reads_summary=cp_reader.bool("worker_reads_summary", True),
            write_aggregate_to_session=cp_reader.bool(
                "write_aggregate_to_session", True
            ),
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
        agg_reader = StrictConfigReader(
            agg_raw,
            allowed={"mode"},
            context=f"swarm {swarm_id}.aggregation",
        )
        mode_raw = agg_reader.optional_str("mode") or AggregationMode.CONCAT.value
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
    metadata = reader.mapping("metadata") or {}

    return SwarmSpec(
        id=swarm_id,
        name=name,
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
