#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Custom wire projections use model defaults without inventing captured facts."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequest
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence import codec
from linktools.ai.persistence._model_client import prepare_client_requests, resolve_client_requests
from linktools.ai.persistence.contracts import ModelClientRequestRef, ModelInteractionRecord
from linktools.ai.task import TaskNodeView

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _interaction() -> ModelInteractionRecord:
    return ModelInteractionRecord(
        "run", 0, 1, "agent", None, {}, None, None, None,
        "RUNNING", None, None, None, _NOW, None,
    )


def _view() -> TaskNodeView:
    return TaskNodeView(
        "graph", "node", ("dependency",), TaskStatus.READY, "owner", 7, _NOW,
        None, None, None, "execution", _NOW, False, "node", {"label": "detail"},
    )


def _encode(value: object, persisted: bool):
    return codec._encode_domain(value, codec._CURRENT_CODEC, persisted=persisted)


def _decode(wire, target: type, persisted: bool):
    return codec._decode_domain(wire, target, codec._CURRENT_CODEC, persisted=persisted)


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("value,omitted,expected", (
    (ModelClientRequestRef("chat", None, "capture failed", record_http_bodies=False),
     ("capture_error", "response", "record_http_bodies", "diagnostics"), ModelClientRequestRef("chat", None)),
    (_interaction(), ("attachments", "client_requests", "stream_attempts"), _interaction()),
))
def test_model_wire_omissions_use_declared_defaults(persisted: bool, value, omitted, expected) -> None:
    wire = _encode(value, persisted)
    for field in omitted:
        wire["fields"].pop(field, None)
    assert _decode(wire, type(value), persisted) == expected
    assert not any(field in wire["fields"] for field in omitted)


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("value,field,invalid", (
    (ModelClientRequestRef("chat", None), "record_http_bodies", "false"),
    (_interaction(), "attachments", None),
    (_interaction(), "client_requests", {"$tuple": []}),
))
def test_model_explicit_invalid_values_do_not_use_defaults(persisted: bool, value, field: str, invalid) -> None:
    wire = _encode(value, persisted)
    wire["fields"][field] = invalid
    with pytest.raises(AIError) as caught:
        _decode(wire, type(value), persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("value,field", (
    (ModelClientRequestRef("chat", None), "body"),
    (_interaction(), "request_context"),
    (_view(), "result_digest"),
))
def test_projection_fields_without_defaults_remain_required(persisted: bool, value, field: str) -> None:
    wire = _encode(value, persisted)
    del wire["fields"][field]
    with pytest.raises(AIError) as caught:
        _decode(wire, type(value), persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert caught.value.safe_details["missing_fields"] == [field]


@pytest.mark.parametrize("persisted", (False, True))
def test_task_view_defaults_preserve_wire_projection_and_constructor_invariants(persisted: bool) -> None:
    view = _view()
    wire = _encode(view, persisted)
    projected = replace(view, dependencies=(), owner=None, fence=1, lease_expires_at=None) if persisted else view
    assert _decode(wire, TaskNodeView, persisted) == projected
    for name in ("execution_id", "next_attempt_at", "occupies_concurrency", "error_origin", "safe_error_details"):
        del wire["fields"][name]
    expected = TaskNodeView(
        projected.graph_id, projected.node_id, projected.dependencies, projected.status,
        projected.owner, projected.fence, projected.lease_expires_at,
        projected.result_digest, projected.error_code, projected.error_digest,
    )
    assert _decode(wire, TaskNodeView, persisted) == expected
    wire["fields"]["occupies_concurrency"] = True
    with pytest.raises(AIError) as caught:
        _decode(wire, TaskNodeView, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_persisted_task_view_keeps_ephemeral_lease_fields_out_of_restore() -> None:
    wire = _encode(_view(), True)
    for name in ("dependencies", "owner", "fence", "lease_expires_at"):
        wire["fields"][name] = "ignored projection data"
    assert _decode(wire, TaskNodeView, True) == replace(
        _view(), dependencies=(), owner=None, fence=1, lease_expires_at=None,
    )
    assert all(wire["fields"][name] == "ignored projection data" for name in (
        "dependencies", "owner", "fence", "lease_expires_at",
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted", (False, True))
async def test_request_default_policy_does_not_create_body_bytes_or_override_explicit_false(persisted: bool) -> None:
    original = ModelClientRequest("chat", None, record_http_bodies=False)
    reference, = await prepare_client_requests((original,), PersistenceDomain.TRACE)
    wire = _encode(reference, persisted)
    assert _decode(wire, ModelClientRequestRef, persisted).record_http_bodies is False
    del wire["fields"]["record_http_bodies"]
    restored = _decode(wire, ModelClientRequestRef, persisted)
    assert restored.record_http_bodies is True
    resolved, = await resolve_client_requests((restored,), PersistenceDomain.TRACE)
    assert resolved == replace(original, record_http_bodies=True)
    assert resolved.body is resolved.response is None

    with_body, = await prepare_client_requests((ModelClientRequest("chat", b"original"),), PersistenceDomain.TRACE)
    invalid = _encode(with_body, persisted)
    invalid["fields"]["record_http_bodies"] = False
    with pytest.raises(AIError) as caught:
        _decode(invalid, ModelClientRequestRef, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("field,invalid", (
    ("execution_id", 1), ("next_attempt_at", "not a timestamp"),
    ("occupies_concurrency", 0), ("error_origin", "unknown"), ("safe_error_details", None),
))
def test_task_view_explicit_invalid_defaulted_fields_still_fail(persisted: bool, field: str, invalid) -> None:
    wire = _encode(_view(), persisted)
    wire["fields"][field] = invalid
    with pytest.raises(AIError) as caught:
        _decode(wire, TaskNodeView, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("persisted", (False, True))
def test_waiting_task_view_cannot_default_away_its_execution_binding(persisted: bool) -> None:
    view = replace(_view(), status=TaskStatus.WAITING, owner=None, lease_expires_at=None, next_attempt_at=None)
    wire = _encode(view, persisted)
    del wire["fields"]["execution_id"]
    with pytest.raises(AIError) as caught:
        _decode(wire, TaskNodeView, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
