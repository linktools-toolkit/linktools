#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opaque resumable cursors for Runtime watch projections."""

import json
from collections.abc import Mapping

from ..core import HmacCursorSigner, canonical_sha256
from ..errors import AIError, ErrorCode
from ._cursor import decode_cursor as decode_runtime_cursor
from ._cursor import encode_cursor as encode_runtime_cursor
from ._runtime_identity import grant_key

_WATCH_CURSOR_VERSION = 1


def encode_execution_watch_cursor(
    namespace: str,
    tenant_id: str,
    execution_id: str,
    *,
    include_content: bool,
    sequences: Mapping[str, int],
) -> str:
    normalized = _execution_sequences(sequences)
    return encode_runtime_cursor(
        _signer(namespace, "execution-watch"),
        tenant_id=tenant_id,
        resource_kind="EXECUTION_WATCH",
        filter_digest=_filter_digest(
            execution_id,
            include_content=include_content,
            kind="execution",
        ),
        position=json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def decode_execution_watch_cursor(
    namespace: str,
    tenant_id: str,
    execution_id: str,
    cursor: str,
    *,
    include_content: bool,
) -> Mapping[str, int]:
    payload = decode_runtime_cursor(
        cursor,
        _signer(namespace, "execution-watch"),
        tenant_id=tenant_id,
        resource_kind="EXECUTION_WATCH",
        filter_digest=_filter_digest(
            execution_id,
            include_content=include_content,
            kind="execution",
        ),
    )
    if payload.revision != 0:
        raise AIError(ErrorCode.CURSOR_INVALID)
    try:
        value = json.loads(payload.position)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.CURSOR_INVALID)
    try:
        return _execution_sequences(value)
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error


def encode_graph_watch_cursor(
    namespace: str,
    tenant_id: str,
    graph_id: str,
    *,
    include_content: bool,
    graph_sequence: int,
    execution_sequences: Mapping[str, Mapping[str, int]],
) -> str:
    if (
        isinstance(graph_sequence, bool)
        or not isinstance(graph_sequence, int)
        or graph_sequence < 0
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    normalized = _graph_execution_sequences(execution_sequences)
    return encode_runtime_cursor(
        _signer(namespace, "task-graph-watch"),
        tenant_id=tenant_id,
        resource_kind="TASK_GRAPH_WATCH",
        filter_digest=_filter_digest(
            graph_id,
            include_content=include_content,
            kind="task_graph",
        ),
        position=json.dumps(
            {
                "graph_sequence": graph_sequence,
                "execution_sequences": normalized,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def decode_graph_watch_cursor(
    namespace: str,
    tenant_id: str,
    graph_id: str,
    cursor: str,
    *,
    include_content: bool,
) -> tuple[int, Mapping[str, Mapping[str, int]]]:
    payload = decode_runtime_cursor(
        cursor,
        _signer(namespace, "task-graph-watch"),
        tenant_id=tenant_id,
        resource_kind="TASK_GRAPH_WATCH",
        filter_digest=_filter_digest(
            graph_id,
            include_content=include_content,
            kind="task_graph",
        ),
    )
    if payload.revision != 0:
        raise AIError(ErrorCode.CURSOR_INVALID)
    try:
        value = json.loads(payload.position)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if not isinstance(value, Mapping) or set(value) != {
        "graph_sequence",
        "execution_sequences",
    }:
        raise AIError(ErrorCode.CURSOR_INVALID)
    sequence = value.get("graph_sequence")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    raw_execution = value.get("execution_sequences")
    if not isinstance(raw_execution, Mapping):
        raise AIError(ErrorCode.CURSOR_INVALID)
    try:
        execution = _graph_execution_sequences(raw_execution)
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    return sequence, execution


def _signer(namespace: str, purpose: str) -> HmacCursorSigner:
    return HmacCursorSigner(purpose, grant_key(namespace))


def _filter_digest(
    resource_id: str,
    *,
    include_content: bool,
    kind: str,
) -> str:
    if (
        not isinstance(resource_id, str)
        or not resource_id
        or not isinstance(include_content, bool)
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return canonical_sha256(
        {
            "version": _WATCH_CURSOR_VERSION,
            "kind": kind,
            "resource_id": resource_id,
            "include_content": include_content,
        }
    )


def _execution_sequences(value: Mapping[object, object]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result: dict[str, int] = {}
    for execution_id, sequence in value.items():
        if (
            not isinstance(execution_id, str)
            or not execution_id
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        result[execution_id] = sequence
    return result


def _graph_execution_sequences(
    value: Mapping[object, object],
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result: dict[str, dict[str, int]] = {}
    for node_id, raw_sequences in value.items():
        if (
            not isinstance(node_id, str)
            or not node_id
            or not isinstance(raw_sequences, Mapping)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        result[node_id] = _execution_sequences(raw_sequences)
    return result


__all__ = [
    "decode_execution_watch_cursor",
    "decode_graph_watch_cursor",
    "encode_execution_watch_cursor",
    "encode_graph_watch_cursor",
]
