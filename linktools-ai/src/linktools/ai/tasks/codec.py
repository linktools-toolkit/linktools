#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical JSON round-trip codec for TaskPlan and TaskExecution.

``encode`` produces a JsonValue; ``decode`` reconstructs a semantically equal
value by routing through the frozen-dataclass constructors, so construction
validation (plan integrity, status invariants, usage bounds) is the single
shared authority for both paths."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import cast

from ..errors import InvalidSpecError
from ..json import JsonValue
from ..storage.coordination.lease import Lease
from .models import (
    DependencyFailurePolicy,
    TaskDependency,
    TaskExecution,
    TaskGraphNodePayload,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
)


def encode_plan(plan: TaskPlan) -> "JsonValue":
    return {
        "id": plan.id,
        "nodes": [_encode_node(node) for node in plan.nodes],
    }


def decode_plan(value: "JsonValue") -> TaskPlan:
    data = _as_mapping(value, "plan")
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list):
        raise InvalidSpecError("plan: 'nodes' must be a list")
    nodes = tuple(_decode_node(node, "plan") for node in raw_nodes)
    return TaskPlan(id=_str_field(data, "id", "plan"), nodes=nodes)


def encode_execution(execution: TaskExecution) -> "JsonValue":
    return {
        "id": execution.id,
        "plan_id": execution.plan_id,
        "node_id": execution.node_id,
        "status": execution.status.value,
        "lease": _encode_lease(execution.lease),
        "attempt": execution.attempt,
        "active_run_id": execution.active_run_id,
        "result": execution.result,
        "error": _encode_error(execution.error),
        "blocked_by": list(execution.blocked_by),
        "terminal_reason": execution.terminal_reason,
        "usage": _encode_usage(execution.usage),
        "created_at": _encode_dt(execution.created_at),
        "updated_at": _encode_dt(execution.updated_at),
    }


def decode_execution(value: "JsonValue") -> TaskExecution:
    data = _as_mapping(value, "execution")
    raw_status = _str_field(data, "status", "execution")
    try:
        status = TaskStatus(raw_status)
    except ValueError as exc:
        raise InvalidSpecError(f"execution: unknown status {raw_status!r}") from exc
    return TaskExecution(
        id=_str_field(data, "id", "execution"),
        plan_id=_str_field(data, "plan_id", "execution"),
        node_id=_str_field(data, "node_id", "execution"),
        status=status,
        lease=_decode_lease(data.get("lease"), "execution"),
        attempt=_int_field(data, "attempt", "execution"),
        active_run_id=_optional_str(data.get("active_run_id"), "execution.active_run_id"),
        result=data.get("result"),
        error=_decode_error(data.get("error")),
        blocked_by=_decode_str_tuple(data.get("blocked_by"), "execution.blocked_by"),
        terminal_reason=_optional_str(
            data.get("terminal_reason"), "execution.terminal_reason"
        ),
        usage=_decode_usage(data.get("usage"), "execution.usage"),
        created_at=_decode_dt(data.get("created_at"), "execution.created_at"),
        updated_at=_decode_dt(data.get("updated_at"), "execution.updated_at"),
    )


def _encode_node(node: TaskNode) -> "JsonValue":
    return {
        "id": node.id,
        "payload": _encode_payload(node.payload),
        "dependencies": [
            {"node_id": dep.node_id, "on_failure": dep.on_failure.value}
            for dep in node.dependencies
        ],
    }


def _decode_node(value: "JsonValue", context: str) -> TaskNode:
    data = _as_mapping(value, f"{context}.node")
    payload = _decode_payload(data.get("payload"), f"{context}.node.payload")
    raw_deps = data.get("dependencies")
    deps: "tuple[TaskDependency, ...]" = ()
    if raw_deps is not None:
        if not isinstance(raw_deps, list):
            raise InvalidSpecError(f"{context}.node: dependencies must be a list")
        deps = tuple(_decode_dependency(item, f"{context}.node") for item in raw_deps)
    return TaskNode(
        id=_str_field(data, "id", f"{context}.node"),
        payload=payload,
        dependencies=deps,
    )


def _decode_dependency(value: "JsonValue", context: str) -> TaskDependency:
    data = _as_mapping(value, f"{context}.dependency")
    node_id = _str_field(data, "node_id", f"{context}.dependency")
    raw_policy = data.get("on_failure", DependencyFailurePolicy.SKIP.value)
    try:
        policy = DependencyFailurePolicy(raw_policy)
    except ValueError as exc:
        raise InvalidSpecError(
            f"{context}.dependency: unknown on_failure {raw_policy!r}"
        ) from exc
    return TaskDependency(node_id=node_id, on_failure=policy)


def _encode_payload(payload: TaskGraphNodePayload) -> "JsonValue":
    return {
        "agent_id": payload.agent_id,
        "prompt": payload.prompt,
        "metadata": dict(payload.metadata),
    }


def _decode_payload(value: "JsonValue", context: str) -> TaskGraphNodePayload:
    data = _as_mapping(value, context)
    return TaskGraphNodePayload(
        agent_id=_str_field(data, "agent_id", context),
        prompt=_str_field(data, "prompt", context),
        metadata=dict(_as_mapping(data.get("metadata", {}), f"{context}.metadata")),
    )


def _encode_usage(usage: TaskUsage) -> "JsonValue":
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_cost": (
            None if usage.total_cost is None else format(usage.total_cost, "f")
        ),
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
    }


def _decode_usage(value: "JsonValue", context: str) -> TaskUsage:
    data = _as_mapping(value, context)
    raw_cost = data.get("total_cost")
    total_cost: "Decimal | None" = None
    if raw_cost is not None:
        if not isinstance(raw_cost, str):
            raise InvalidSpecError(f"{context}: total_cost must be a decimal string")
        total_cost = Decimal(raw_cost)
    return TaskUsage(
        input_tokens=_int_field(data, "input_tokens", context),
        output_tokens=_int_field(data, "output_tokens", context),
        total_cost=total_cost,
        cache_write_tokens=int(data.get("cache_write_tokens", 0)),
        cache_read_tokens=int(data.get("cache_read_tokens", 0)),
    )


def _encode_lease(lease: Lease) -> "JsonValue":
    return {
        "owner": lease.owner,
        "fence": lease.fence,
        "expires_at": _encode_dt(lease.expires_at) if lease.expires_at else None,
    }


def _decode_lease(value: "JsonValue | None", context: str) -> Lease:
    if value is None:
        return Lease()
    data = _as_mapping(value, f"{context}.lease")
    owner = _optional_str(data.get("owner"), f"{context}.lease.owner")
    expires_raw = data.get("expires_at")
    expires_at = _decode_dt(expires_raw, f"{context}.lease.expires_at") if expires_raw else None
    return Lease(owner=owner, fence=_int_field(data, "fence", f"{context}.lease"), expires_at=expires_at)


def _encode_error(error: "object") -> "JsonValue":
    if error is None:
        return None
    from ..execution.domain import RunError

    err = cast(RunError, error)
    return {
        "error_type": err.error_type,
        "message": err.message,
        "detail": err.detail,
    }


def _decode_error(value: "JsonValue | None"):
    if value is None:
        return None
    from ..execution.domain import RunError

    data = _as_mapping(value, "execution.error")
    return RunError(
        error_type=_str_field(data, "error_type", "execution.error"),
        message=_str_field(data, "message", "execution.error"),
        detail=data.get("detail"),
    )


def _encode_dt(dt: datetime) -> "str | None":
    if dt is None:
        return None
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()


def _decode_dt(value: "JsonValue", context: str) -> datetime:
    if not isinstance(value, str):
        raise InvalidSpecError(f"{context}: expected ISO datetime string")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidSpecError(f"{context}: invalid datetime {value!r}") from exc


def _as_mapping(value: "JsonValue", context: str) -> "dict[str, JsonValue]":
    if not isinstance(value, dict):
        raise InvalidSpecError(f"{context}: expected a mapping, got {type(value).__name__}")
    return value


def _str_field(data: "dict[str, JsonValue]", key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise InvalidSpecError(f"{context}: '{key}' must be a non-empty string")
    return value


def _int_field(data: "dict[str, JsonValue]", key: str, context: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidSpecError(f"{context}: '{key}' must be an integer")
    return value


def _optional_str(value: "JsonValue", context: str) -> "str | None":
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidSpecError(f"{context}: must be a string or null")
    return value


def _decode_str_tuple(value: "JsonValue", context: str) -> "tuple[str, ...]":
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidSpecError(f"{context}: must be a list")
    out: "list[str]" = []
    for item in value:
        if not isinstance(item, str):
            raise InvalidSpecError(f"{context}: items must be strings")
        out.append(item)
    return tuple(out)


__all__ = ["decode_execution", "decode_plan", "encode_execution", "encode_plan"]
