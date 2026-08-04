#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmSpecCodec: the SpecCodec[SwarmSpec] for the swarm domain.

Owns the swarm-specific parsing (moved here from registry/swarm.py): a
``{name}.yaml`` item is parsed as YAML, strictly validated, and built into a
SwarmSpec. Parse failures propagate the domain's existing errors
(InvalidSpecError / SpecParseError). SwarmSpec itself already lives in
swarm/spec.py."""


from dataclasses import dataclass
from typing import Any

from collections.abc import Mapping

from ...agent.codec import parse_middleware_refs
from ...agent.spec import MiddlewareRef
from ...spec import SpecCodec
from ...spec.parsing import (
    StrictConfigReader,
    parse_yaml_text,
    resolved_name,
)
from ...errors import InvalidSpecError
from .aggregation import AggregationMode, AggregationPolicy
from .limits import DEFAULT_SWARM_LIMITS, SwarmLimits
from .spec import (
    AgentRef,
    SwarmContextPolicy,
    SwarmSpec,
    SwarmStrategySpec,
)


@dataclass(frozen=True, slots=True)
class SwarmStrategyDefaults:
    limits: SwarmLimits
    aggregation: AggregationPolicy


def defaults_for_strategy(kind: str) -> SwarmStrategyDefaults:
    if kind == "task_graph":
        return SwarmStrategyDefaults(
            limits=SwarmLimits(
                max_rounds=1,
                max_tasks=DEFAULT_SWARM_LIMITS.max_tasks,
                max_delegations=0,
                max_depth=0,
                max_concurrency=DEFAULT_SWARM_LIMITS.max_concurrency,
                max_total_tokens=DEFAULT_SWARM_LIMITS.max_total_tokens,
                max_total_cost=DEFAULT_SWARM_LIMITS.max_total_cost,
                timeout_seconds=DEFAULT_SWARM_LIMITS.timeout_seconds,
            ),
            aggregation=AggregationPolicy(mode=AggregationMode.COLLECT),
        )
    return SwarmStrategyDefaults(
        limits=DEFAULT_SWARM_LIMITS,
        aggregation=AggregationPolicy(),
    )


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
    name = resolved_name(reader, swarm_id)

    # Fields parsed by value (payload.get) bypass the StrictConfigReader, so
    # distinguish a missing key from an explicit null here -- null is rejected.
    for _field in (
        "agents",
        "coordinator",
        "strategy",
        "limits",
        "aggregation",
        "middleware",
        "context_policy",
    ):
        if _field in payload and payload[_field] is None:
            raise InvalidSpecError(f"swarm {swarm_id}: '{_field}' must not be null")

    # agents — required, non-empty.
    agents_raw = payload.get("agents")
    if not agents_raw:
        raise InvalidSpecError(f"swarm {swarm_id}: 'agents' must be a non-empty list")
    if not isinstance(agents_raw, (list, tuple)):
        raise InvalidSpecError(f"swarm {swarm_id}: 'agents' must be a list")
    agents = tuple(
        _parse_agent_ref(a, swarm_id=swarm_id, kind="agent") for a in agents_raw
    )

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

    # coordinator — required for coordinator_delegation, forbidden for
    # task_graph (which has no coordinator agent).
    coord_raw = payload.get("coordinator")
    if strategy.kind == "task_graph":
        if coord_raw is not None:
            raise InvalidSpecError(
                f"swarm {swarm_id}: task_graph strategy has no coordinator"
            )
        coordinator = None
    else:
        if coord_raw is None:
            raise InvalidSpecError(f"swarm {swarm_id}: 'coordinator' is required")
        coordinator = _parse_agent_ref(coord_raw, swarm_id=swarm_id, kind="coordinator")

    defaults = defaults_for_strategy(strategy.kind)

    # limits — missing fields inherit strategy-specific defaults.
    limits_raw = payload.get("limits")
    if limits_raw is None:
        limits = defaults.limits
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
                "max_rounds", defaults.limits.max_rounds
            ),
            max_tasks=limits_reader.positive_int(
                "max_tasks", defaults.limits.max_tasks
            ),
            max_delegations=limits_reader.non_negative_int(
                "max_delegations", defaults.limits.max_delegations
            ),
            max_depth=limits_reader.non_negative_int(
                "max_depth", defaults.limits.max_depth
            ),
            max_concurrency=limits_reader.positive_int(
                "max_concurrency", defaults.limits.max_concurrency
            ),
            max_total_tokens=limits_reader.positive_int("max_total_tokens"),
            max_total_cost=limits_reader.non_negative_decimal("max_total_cost"),
            timeout_seconds=limits_reader.positive_number(
                "timeout_seconds", defaults.limits.timeout_seconds
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

    # aggregation — string or {mode: ...}, default strategy policy.
    agg_raw = payload.get("aggregation")
    if agg_raw is None:
        aggregation = defaults.aggregation
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
        mode_raw = agg_reader.optional_str("mode") or defaults.aggregation.mode.value
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

    # middleware — reuse the agent domain's helper.
    middleware: "tuple[MiddlewareRef, ...]" = parse_middleware_refs(
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


class SwarmSpecCodec:
    """SpecCodec[SwarmSpec]: decode one ``{id}.yaml`` item's raw text into a
    SwarmSpec. Strict; propagates the domain's rich errors."""

    def decode(self, item_id: str, raw: str) -> SwarmSpec:
        source = f"{item_id}.yaml"
        payload = parse_yaml_text(raw, source=source)
        return parse_swarm_spec(item_id, payload)


def encode_swarm_spec(spec: SwarmSpec) -> "JsonValue":
    """Canonical JSON encoding of a SwarmSpec for RunDefinition persistence. The
    swarm's agents/coordinator/middleware/limits/context_policy/aggregation are
    encoded so decode_swarm_spec rebuilds a semantically equal spec through the
    same constructors that validate it."""
    return {
        "schema": "swarm-spec.v1",
        "id": spec.id,
        "name": spec.name,
        "agents": [
            {"agent_id": a.agent_id, "role": a.role} for a in spec.agents
        ],
        "coordinator": (
            None
            if spec.coordinator is None
            else {"agent_id": spec.coordinator.agent_id, "role": spec.coordinator.role}
        ),
        "strategy": {"kind": spec.strategy.kind, "config": dict(spec.strategy.config)},
        "limits": _encode_limits(spec.limits),
        "context_policy": {
            "coordinator_reads_session": spec.context_policy.coordinator_reads_session,
            "worker_reads_session": spec.context_policy.worker_reads_session,
            "worker_reads_summary": spec.context_policy.worker_reads_summary,
            "write_aggregate_to_session": spec.context_policy.write_aggregate_to_session,
        },
        "aggregation": {"mode": spec.aggregation.mode.value},
        "middleware": [
            {"name": m.name, "config": dict(m.config)} for m in spec.middleware
        ],
        "metadata": dict(spec.metadata),
    }


def decode_swarm_spec(value: "JsonValue") -> SwarmSpec:
    data = value if isinstance(value, dict) else {}
    agents = tuple(
        AgentRef(agent_id=str(a["agent_id"]), role=a.get("role"))
        for a in data.get("agents", ())
    )
    coord_raw = data.get("coordinator")
    coordinator = (
        None
        if coord_raw is None
        else AgentRef(agent_id=str(coord_raw["agent_id"]), role=coord_raw.get("role"))
    )
    strat = data.get("strategy") or {}
    strategy = SwarmStrategySpec(kind=str(strat.get("kind", "")), config=dict(strat.get("config") or {}))
    strategy_kind = strategy.kind
    return SwarmSpec(
        id=str(data.get("id", "")),
        name=str(data.get("name", "")),
        agents=agents,
        coordinator=coordinator,
        strategy=strategy,
        limits=_decode_limits(data.get("limits"), strategy_kind),
        context_policy=SwarmContextPolicy(
            coordinator_reads_session=bool(
                (data.get("context_policy") or {}).get("coordinator_reads_session", True)
            ),
            worker_reads_session=bool(
                (data.get("context_policy") or {}).get("worker_reads_session", False)
            ),
            worker_reads_summary=bool(
                (data.get("context_policy") or {}).get("worker_reads_summary", True)
            ),
            write_aggregate_to_session=bool(
                (data.get("context_policy") or {}).get("write_aggregate_to_session", True)
            ),
        ),
        aggregation=AggregationPolicy(
            mode=AggregationMode(
                (data.get("aggregation") or {}).get(
                    "mode", defaults_for_strategy(strategy_kind).aggregation.mode.value
                )
            )
        ),
        middleware=tuple(
            MiddlewareRef(name=str(m["name"]), config=dict(m.get("config") or {}))
            for m in data.get("middleware", ())
        ),
        metadata=dict(data.get("metadata") or {}),
    )


def _encode_limits(limits: "SwarmLimits") -> "JsonValue":
    return {
        "max_rounds": limits.max_rounds,
        "max_tasks": limits.max_tasks,
        "max_delegations": limits.max_delegations,
        "max_depth": limits.max_depth,
        "max_concurrency": limits.max_concurrency,
        "max_total_tokens": limits.max_total_tokens,
        "max_total_cost": (
            None if limits.max_total_cost is None else format(limits.max_total_cost, "f")
        ),
        "timeout_seconds": limits.timeout_seconds,
    }


def _decode_limits(value: "JsonValue", strategy_kind: str = "") -> "SwarmLimits":
    from decimal import Decimal

    data = value if isinstance(value, dict) else {}
    defaults = defaults_for_strategy(strategy_kind).limits
    cost = data.get("max_total_cost")
    return SwarmLimits(
        max_rounds=int(data.get("max_rounds", defaults.max_rounds)),
        max_tasks=int(data.get("max_tasks", defaults.max_tasks)),
        max_delegations=int(data.get("max_delegations", defaults.max_delegations)),
        max_depth=int(data.get("max_depth", defaults.max_depth)),
        max_concurrency=int(data.get("max_concurrency", defaults.max_concurrency)),
        max_total_tokens=data.get("max_total_tokens"),
        max_total_cost=Decimal(cost) if cost is not None else None,
        timeout_seconds=data.get("timeout_seconds", defaults.timeout_seconds),
    )


__all__: "list[str]" = [
    "SwarmSpecCodec",
    "decode_swarm_spec",
    "encode_swarm_spec",
    "parse_swarm_spec",
    "SwarmStrategyDefaults",
    "defaults_for_strategy",
]
