#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content-addressed request transport for Temporal workflows."""

import json
import re
from collections.abc import Mapping

from linktools.core import environ

from ..core import JsonValue, Principal, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..runtime import (
    ExecutionRequest,
    RuntimeObjectKeyFactory,
    put_runtime_object,
    read_runtime_object,
)
from ..runtime.state import RuntimeDomain
from ..storage import ObjectRef, ObjectStore
from ..task import TaskGraph, TaskGraphLimits, TaskGraphRequest, TaskNode
from .workflow import ExecutionWorkflowState

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TASK_FIELDS = frozenset({"version", "graph", "principal", "idempotency_key", "limits"})
_GRAPH_FIELDS = frozenset({"graph_id", "nodes"})
_NODE_FIELDS = frozenset({"node_id", "dependencies", "input", "budget_cost"})
_PRINCIPAL_FIELDS = frozenset({"principal_id", "tenant_id", "kind"})
_LIMIT_FIELDS = frozenset({"max_concurrency", "max_depth", "max_nodes", "max_budget"})
_EXECUTION_FIELDS = frozenset(
    {"version", "user_prompt", "principal", "idempotency_key", "memory_scope"}
)
_logger = environ.get_logger("ai.temporal.request")


async def put_task_request(
    store: ObjectStore,
    key_factory: RuntimeObjectKeyFactory,
    request: TaskGraphRequest,
) -> str:
    payload: dict[str, JsonValue] = {
        "version": 1,
        "graph": {
            "graph_id": request.graph.graph_id,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "dependencies": list(node.dependencies),
                    "input": node.input,
                    "budget_cost": node.budget_cost,
                }
                for node in request.graph.nodes
            ],
        },
        "principal": _principal_payload(request.principal),
        "idempotency_key": request.idempotency_key,
        "limits": {
            "max_concurrency": request.limits.max_concurrency,
            "max_depth": request.limits.max_depth,
            "max_nodes": request.limits.max_nodes,
            "max_budget": request.limits.max_budget,
        },
    }
    reference = await put_runtime_object(
        store,
        key_factory,
        RuntimeDomain.TASK,
        request.principal.tenant_id,
        canonical_json_bytes(payload),
    )
    _logger.debug("task request persisted: tenant=%s request_ref=%s", request.principal.tenant_id, reference.key)
    return reference.key


async def read_task_request(
    store: ObjectStore,
    key_factory: RuntimeObjectKeyFactory,
    *,
    tenant_id: str,
    request_ref: str,
) -> TaskGraphRequest:
    payload = await _read_payload(store, key_factory, tenant_id=tenant_id, request_ref=request_ref)
    try:
        value = _load_canonical(payload)
        request = _task_request_from_payload(value)
        if request.principal.tenant_id != tenant_id:
            raise ValueError("task request tenant does not match its object key")
        return request
    except AIError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


async def put_execution_request(
    store: ObjectStore,
    key_factory: RuntimeObjectKeyFactory,
    request: ExecutionRequest,
) -> str:
    payload: dict[str, JsonValue] = {
        "version": 1,
        "user_prompt": request.user_prompt,
        "principal": _principal_payload(request.principal),
        "idempotency_key": request.idempotency_key,
        "memory_scope": request.memory_scope,
    }
    reference = await put_runtime_object(
        store,
        key_factory,
        RuntimeDomain.TASK,
        request.principal.tenant_id,
        canonical_json_bytes(payload),
    )
    _logger.debug(
        "execution request persisted: tenant=%s request_ref=%s",
        request.principal.tenant_id,
        reference.key,
    )
    return reference.key


async def read_execution_request(
    store: ObjectStore,
    key_factory: RuntimeObjectKeyFactory,
    *,
    tenant_id: str,
    request_ref: str,
) -> ExecutionRequest:
    payload = await _read_payload(store, key_factory, tenant_id=tenant_id, request_ref=request_ref)
    try:
        value = _load_canonical(payload)
        request = _execution_request_from_payload(value)
        if request.principal.tenant_id != tenant_id:
            raise ValueError("execution request tenant does not match its object key")
        return request
    except AIError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


async def load_execution_request(
    store: ObjectStore,
    *,
    namespace: str,
    state: ExecutionWorkflowState,
) -> ExecutionRequest:
    if not isinstance(namespace, str) or not namespace.strip():
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    request = await read_execution_request(
        store,
        RuntimeObjectKeyFactory(namespace),
        tenant_id=state.tenant_id,
        request_ref=state.request_ref,
    )
    if request.principal.tenant_id != state.tenant_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _logger.debug(
        "execution request loaded: execution=%s request_ref=%s",
        state.execution_id,
        state.request_ref,
    )
    return request


def _principal_payload(principal: Principal) -> dict[str, str]:
    return {
        "principal_id": principal.principal_id,
        "tenant_id": principal.tenant_id,
        "kind": principal.kind,
    }


async def _read_payload(
    store: ObjectStore,
    key_factory: RuntimeObjectKeyFactory,
    *,
    tenant_id: str,
    request_ref: str,
) -> bytes:
    try:
        if not isinstance(request_ref, str):
            raise ValueError("request reference is invalid")
        digest = request_ref.rsplit("/", 1)[-1]
        if _DIGEST.fullmatch(digest) is None:
            raise ValueError("request reference digest is invalid")
        expected_key = key_factory.key(RuntimeDomain.TASK, tenant_id, digest)
        if expected_key != request_ref:
            raise ValueError("request reference key is invalid")
        stat = await store.stat(request_ref)
        if stat is None or stat.digest != digest:
            raise ValueError("request object is missing or corrupt")
        reference = ObjectRef(store.store_id, request_ref, stat.digest, stat.size)
        return await read_runtime_object(store, reference)
    except AIError as error:
        if error.code in {
            ErrorCode.REQUEST_FIELD_INVALID,
            ErrorCode.STORAGE_NOT_FOUND,
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _load_canonical(payload: bytes) -> dict[str, object]:
    value = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise ValueError("request payload is not canonical JSON")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON constant: {value}")


def _mapping(value: object, fields: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("request payload fields are invalid")
    return value


def _task_request_from_payload(value: Mapping[str, object]) -> TaskGraphRequest:
    payload = _mapping(value, _TASK_FIELDS)
    _require_version(payload["version"])
    principal = _principal_from_payload(payload["principal"])
    graph_value = _mapping(payload["graph"], _GRAPH_FIELDS)
    nodes_value = graph_value["nodes"]
    if not isinstance(nodes_value, list):
        raise ValueError("task graph nodes are invalid")
    nodes = tuple(_task_node_from_payload(item) for item in nodes_value)
    graph = TaskGraph(_require_string(graph_value["graph_id"]), nodes)
    limits_value = _mapping(payload["limits"], _LIMIT_FIELDS)
    limits = TaskGraphLimits(
        _require_positive_int(limits_value["max_concurrency"]),
        _require_positive_int(limits_value["max_depth"]),
        _require_positive_int(limits_value["max_nodes"]),
        _require_positive_int(limits_value["max_budget"]),
    )
    return TaskGraphRequest(
        graph,
        principal,
        _require_string(payload["idempotency_key"]),
        limits,
    )


def _task_node_from_payload(value: object) -> TaskNode:
    payload = _mapping(value, _NODE_FIELDS)
    dependencies = payload["dependencies"]
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        raise ValueError("task node dependencies are invalid")
    input_value = payload["input"]
    if not isinstance(input_value, dict):
        raise ValueError("task node input is invalid")
    return TaskNode(
        _require_string(payload["node_id"]),
        tuple(dependencies),
        input=input_value,
        budget_cost=_require_positive_int(payload["budget_cost"]),
    )


def _execution_request_from_payload(value: Mapping[str, object]) -> ExecutionRequest:
    payload = _mapping(value, _EXECUTION_FIELDS)
    _require_version(payload["version"])
    memory_scope = payload["memory_scope"]
    if memory_scope is not None and not isinstance(memory_scope, str):
        raise ValueError("execution memory scope is invalid")
    return ExecutionRequest(
        _require_string(payload["user_prompt"]),
        _principal_from_payload(payload["principal"]),
        _require_string(payload["idempotency_key"]),
        memory_scope,
    )


def _principal_from_payload(value: object) -> Principal:
    payload = _mapping(value, _PRINCIPAL_FIELDS)
    return Principal(
        _require_string(payload["principal_id"]),
        _require_string(payload["tenant_id"]),
        _require_string(payload["kind"]),
    )


def _require_version(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != 1:
        raise ValueError("request version is invalid")


def _require_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("request string field is invalid")
    return value


def _require_positive_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("request integer field is invalid")
    return value


__all__ = [
    "load_execution_request",
    "put_execution_request",
    "put_task_request",
    "read_execution_request",
    "read_task_request",
]
