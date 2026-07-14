#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunDefinitionSnapshot: the immutable record of what a Run was launched with
(spec + identity), persisted at run creation so resume can restore the EXACT
original definition instead of accepting a caller-supplied spec/identity (the
R-03 security gap). The snapshot carries a full serialized spec + a
canonical-JSON fingerprint; resume deserializes the spec, recomputes the
fingerprint, and rejects a mismatch."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Protocol, runtime_checkable

from ..json import canonical_json


def _serialize_output_schema(output_schema: Any) -> "str | None":
    if output_schema is None:
        return None
    if output_schema is str:
        return "str"
    if isinstance(output_schema, type):
        return f"{output_schema.__module__}:{output_schema.__qualname__}"
    return str(output_schema)


def _deserialize_output_schema(value: "str | None") -> Any:
    if value is None or value == "":
        return None
    if value == "str":
        return str
    if ":" in value:
        import importlib

        module_path, _, qualname = value.rpartition(":")
        mod = importlib.import_module(module_path)
        obj = mod
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    return None


def serialize_agent_spec(spec: Any) -> "dict[str, Any]":
    """Full JSON-safe serialization of an AgentSpec (all fields including tool
    configs) for both fingerprinting and round-trip reconstruction on resume."""
    return {
        "id": spec.id,
        "name": spec.name,
        "model": {
            "primary": spec.model.primary,
            "fallbacks": list(spec.model.fallbacks),
            "max_retries": spec.model.max_retries,
            "timeout_seconds": spec.model.timeout_seconds,
            "max_tokens": spec.model.max_tokens,
            "budget": (
                str(spec.model.budget) if spec.model.budget is not None else None
            ),
        },
        "instructions": spec.instructions.instructions,
        "sections": dict(spec.instructions.sections),
        "tools": (
            None
            if spec.tools is None
            else [
                {"kind": t.kind, "name": t.name, "config": dict(t.config)}
                for t in spec.tools
            ]
        ),
        "middleware": [
            {"name": m.name, "config": dict(m.config)} for m in spec.middleware
        ],
        "output_schema": _serialize_output_schema(spec.output_schema),
        "metadata": dict(spec.metadata),
    }


def deserialize_agent_spec(data: "dict[str, Any]") -> Any:
    """Reconstruct an AgentSpec from its serialized form (the round-trip pair of
    serialize_agent_spec). Imports output_schema by its saved path."""
    from ..agent.spec import AgentSpec, MiddlewareRef, PromptSpec, ToolRef
    from ..model.policy import ModelPolicy

    md = data["model"]
    budget = Decimal(md["budget"]) if md.get("budget") else None
    model = ModelPolicy(
        primary=md["primary"],
        fallbacks=tuple(md.get("fallbacks", [])),
        max_retries=md.get("max_retries", 0),
        timeout_seconds=md.get("timeout_seconds"),
        max_tokens=md.get("max_tokens"),
        budget=budget,
    )
    instructions = PromptSpec(
        instructions=data["instructions"],
        sections=data.get("sections", {}),
    )
    tools = (
        None
        if data.get("tools") is None
        else tuple(
            ToolRef(kind=t["kind"], name=t["name"], config=t.get("config", {}))
            for t in data["tools"]
        )
    )
    middleware = tuple(
        MiddlewareRef(name=m["name"], config=m.get("config", {}))
        for m in data.get("middleware", [])
    )
    return AgentSpec(
        id=data["id"],
        name=data["name"],
        model=model,
        instructions=instructions,
        tools=tools,
        middleware=middleware,
        output_schema=_deserialize_output_schema(data.get("output_schema")),
        metadata=data.get("metadata", {}),
    )


def spec_fingerprint(spec: Any) -> str:
    """SHA-256 of the canonical-JSON serialized spec. Resume recomputes this
    from the restored spec and rejects a mismatch with the snapshot."""
    return hashlib.sha256(
        canonical_json(serialize_agent_spec(spec)).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RunDefinitionSnapshot:
    run_id: str
    runnable_type: str
    runnable_id: str
    serialized_spec: "Mapping[str, Any]"
    spec_fingerprint: str
    user_id: "str | None"
    tenant_id: "str | None"
    workspace: "str | None"
    provider_revision: "str | None"
    created_at: datetime
    # ExecutionManifest: revisions of the bundle/policy/capabilities the run
    # was prepared against, so resume can detect environment drift. Deep-frozen
    # at preparation time; defaults to an empty frozen mapping.
    manifest: "Mapping[str, Any]" = field(default_factory=dict)


@runtime_checkable
class RunDefinitionStore(Protocol):
    async def create(self, snapshot: RunDefinitionSnapshot) -> None: ...

    async def get(self, run_id: str) -> "RunDefinitionSnapshot | None": ...


def serialize_swarm_spec(spec: Any) -> "dict[str, Any]":
    """Full JSON-safe serialization of a SwarmSpec for snapshot persistence."""

    return {
        "id": spec.id,
        "name": spec.name,
        "agents": [{"agent_id": a.agent_id, "role": a.role} for a in spec.agents],
        "coordinator": {
            "agent_id": spec.coordinator.agent_id,
            "role": spec.coordinator.role,
        },
        "strategy": {
            "kind": spec.strategy.kind,
            "config": {
                k: v
                for k, v in spec.strategy.config.items()
                if isinstance(v, (str, int, float, bool, type(None), list, dict))
            },
        },
        "limits": {
            "max_rounds": spec.limits.max_rounds,
            "max_tasks": spec.limits.max_tasks,
            "max_delegations": spec.limits.max_delegations,
            "max_depth": spec.limits.max_depth,
            "max_concurrency": spec.limits.max_concurrency,
            "max_total_tokens": spec.limits.max_total_tokens,
            "max_total_cost": str(spec.limits.max_total_cost)
            if spec.limits.max_total_cost
            else None,
            "timeout_seconds": spec.limits.timeout_seconds,
        },
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


def deserialize_swarm_spec(data: "dict[str, Any]") -> Any:
    """Reconstruct a SwarmSpec from its serialized form."""
    from decimal import Decimal

    from ..agent.spec import MiddlewareRef
    from ..swarm.aggregation import AggregationMode, AggregationPolicy
    from ..swarm.limits import SwarmLimits
    from ..swarm.models import AgentRef
    from ..swarm.spec import SwarmContextPolicy, SwarmSpec, SwarmStrategySpec

    limits_data = data["limits"]
    cost = limits_data.get("max_total_cost")
    return SwarmSpec(
        id=data["id"],
        name=data["name"],
        agents=tuple(
            AgentRef(agent_id=a["agent_id"], role=a.get("role")) for a in data["agents"]
        ),
        coordinator=AgentRef(
            agent_id=data["coordinator"]["agent_id"],
            role=data["coordinator"].get("role"),
        ),
        strategy=SwarmStrategySpec(
            kind=data["strategy"]["kind"], config=data["strategy"].get("config", {})
        ),
        limits=SwarmLimits(
            max_rounds=limits_data["max_rounds"],
            max_tasks=limits_data["max_tasks"],
            max_delegations=limits_data["max_delegations"],
            max_depth=limits_data["max_depth"],
            max_concurrency=limits_data["max_concurrency"],
            max_total_tokens=limits_data.get("max_total_tokens"),
            max_total_cost=Decimal(cost) if cost else None,
            timeout_seconds=limits_data.get("timeout_seconds"),
        ),
        context_policy=SwarmContextPolicy(**data["context_policy"]),
        aggregation=AggregationPolicy(
            mode=AggregationMode(data["aggregation"]["mode"])
        ),
        middleware=tuple(
            MiddlewareRef(name=m["name"], config=m.get("config", {}))
            for m in data.get("middleware", [])
        ),
        metadata=data.get("metadata", {}),
    )
