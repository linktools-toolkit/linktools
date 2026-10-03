#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scoped storage identity and object closure for immutable input captures."""

from collections.abc import Iterator, Mapping
from datetime import datetime

from ...agent import AgentInputCaptureRef
from ...core import JsonValue, canonical_sha256
from ...errors import AIError, ErrorCode
from ...task import TaskDependencyCapture, TaskGraphCaptureRef, TaskGraphTemplateRef, TaskInvocationInputRef
from ._codec import decode_domain

_CAPTURE_TYPES = {
    "agent_input_capture_ref": (AgentInputCaptureRef, "agent"),
    "task_invocation_input_ref": (TaskInvocationInputRef, "task"),
    "task_graph_capture_ref": (TaskGraphCaptureRef, "graph"),
    "task_graph_template_ref": (TaskGraphTemplateRef, "template"),
}


def input_capture_key(namespace: str, tenant_id: str, kind: str, identity: str) -> str:
    digest = canonical_sha256({"namespace": namespace, "tenant_id": tenant_id, "kind": kind, "identity": identity})
    return f"v1/input-capture/{kind}/{digest}"


def input_capture_expiry_key(key: str) -> str:
    return "v1/input-capture/expired/" + canonical_sha256(key)


def input_capture_object_dependency(
    reference: AgentInputCaptureRef | TaskInvocationInputRef | TaskGraphCaptureRef | TaskGraphTemplateRef,
) -> tuple[str, str]:
    kind = next(kind for cls, kind in _CAPTURE_TYPES.values() if isinstance(reference, cls))
    return input_capture_key(reference.namespace, reference.tenant_id, kind, reference.capture_id), reference.digest


def iter_input_capture_dependencies(value: JsonValue, *, namespace: str, tenant_id: str,
                                    validate_scope: bool = True) -> Iterator[tuple[str, str]]:
    if isinstance(value, list):
        for item in value:
            yield from iter_input_capture_dependencies(item, namespace=namespace, tenant_id=tenant_id, validate_scope=validate_scope)
    elif isinstance(value, Mapping):
        wire_id = value.get("$dataclass")
        if isinstance(wire_id, str) and wire_id in _CAPTURE_TYPES:
            cls, _kind = _CAPTURE_TYPES[wire_id]
            reference = decode_domain(value, cls)
            if validate_scope and (reference.namespace != namespace or reference.tenant_id != tenant_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            yield input_capture_object_dependency(reference)
            return
        if wire_id == "task_dependency_capture":
            captured = decode_domain(value, TaskDependencyCapture)
            if captured.body_digest is not None:
                if validate_scope and (captured.source_ref.namespace != namespace or captured.source_ref.tenant_id != tenant_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                yield input_capture_key(captured.source_ref.namespace, captured.source_ref.tenant_id, "result", captured.body_digest), captured.body_digest
            return
        for item in value.values():
            yield from iter_input_capture_dependencies(item, namespace=namespace, tenant_id=tenant_id, validate_scope=validate_scope)


def validate_input_capture_payload(key: str, value: JsonValue, *, namespace: str, tenant_id: str) -> None:
    kind = key.split("/")[2]
    if kind in {"agent", "task", "graph", "template"}:
        if not isinstance(value, Mapping) or value.get("namespace") != namespace or value.get("tenant_id") != tenant_id or value.get("kind") != kind or value.get("version") != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif kind == "expired":
        if (not isinstance(value, Mapping) or set(value) != {"namespace", "tenant_id", "capture_key", "expired_at"}
                or value.get("namespace") != namespace or value.get("tenant_id") != tenant_id
                or not isinstance(value.get("capture_key"), str)
                or not value["capture_key"].startswith("v1/input-capture/")
                or value["capture_key"].startswith("v1/input-capture/expired/")
                or input_capture_expiry_key(value["capture_key"]) != key):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            if datetime.fromisoformat(value["expired_at"]).tzinfo is None:
                raise ValueError("capture expiry must be timezone-aware")
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    elif kind not in {"result", "invocation", "declaration"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = ["input_capture_key", "input_capture_expiry_key", "input_capture_object_dependency", "iter_input_capture_dependencies", "validate_input_capture_payload"]
