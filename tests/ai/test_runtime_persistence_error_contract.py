#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence version errors must survive repository wrapper boundaries."""

import copy
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.agent import AgentCompiler
from linktools.ai.core import ExecutionEventType, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import (
    RuntimePayloadRef,
    TranscriptChunk,
    TranscriptOrigin,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
)
from linktools.ai.runtime.state._maintenance import RuntimeStorageInspection
from linktools.ai.runtime.state._migration import _migrate_legacy_binding
from linktools.ai.runtime.state._repositories import _domain_data
from linktools.ai.runtime.state._store import StoredFact, StoredRecord
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload
from linktools.ai.task import TaskNodeView, TaskStatus


def _future_schema(data: object) -> object:
    value = copy.deepcopy(data)
    value["value"]["payload"]["schema"] = 99
    return value


def _legacy_binding_payload(compiler: AgentCompiler) -> dict[str, object]:
    binding = compiler.bind(compiler.compile(AgentSpec("default")))
    payload = dict(binding.snapshot.to_payload())
    payload.pop("agent_digest")
    spec = dict(payload["agent_spec"])
    spec["metadata"] = {}
    payload["agent_spec"] = spec
    payload["binding_digest"] = "f" * 64
    return payload


def _compiler() -> AgentCompiler:
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        runtime_fingerprint="a" * 64,
    )


class _NoObjects:
    def object_store(self, domain: RuntimeDomain) -> object:
        raise AssertionError(f"unexpected object reference in {domain.value}")


def _inspection() -> RuntimeStorageInspection:
    return RuntimeStorageInspection(
        {},
        _NoObjects(),
        durable_domains=frozenset(),
    )


def _read_model_record(*, version: int = 1) -> StoredRecord:
    return StoredRecord(
        b"k" * 32,
        b"p" * 32,
        None,
        None,
        "execution_read_model",
        "execution",
        "COMPLETE",
        0,
        None,
        0,
        None,
        {
            "execution_id": "execution",
            "tenant_id": "tenant",
            "source_digest": "source",
            "model_version": version,
            "status": "COMPLETE",
            "trace_count": 0,
            "history_count": 0,
            "transcript_count": 0,
            "revision": 1,
        },
    )


def _record(kind: str, value: object) -> StoredRecord:
    return StoredRecord(
        hashlib.sha256(f"key:{kind}".encode()).digest(),
        hashlib.sha256(f"partition:{kind}".encode()).digest(),
        None,
        None,
        kind,
        kind,
        None,
        0,
        None,
        0,
        None,
        _domain_data(value),
    )


def _task_node() -> TaskNodeView:
    return TaskNodeView(
        "graph",
        "node",
        (),
        TaskStatus.PENDING,
        None,
        0,
        None,
        None,
        None,
        None,
    )


def _transcript_chunk_data() -> dict[str, object]:
    raw = b"[]"
    chunk = TranscriptChunk(
        "owner",
        0,
        1,
        TranscriptOrigin.RAW,
        "raw",
        hashlib.sha256(raw).hexdigest(),
        len(raw),
        RuntimePayloadRef(
            StoredPayload.inline_bytes(raw),
            RuntimeDomain.EXECUTION,
        ),
    )
    return encode_envelope(
        {
            "type": "transcript_chunk",
            "payload": _encode_persisted_domain(chunk),
        }
    )


def test_legacy_binding_malformed_known_field_remains_integrity_error() -> None:
    compiler = _compiler()
    payload = _legacy_binding_payload(compiler)
    payload["output_schema_fingerprint"] = "invalid"

    with pytest.raises(AIError) as raised:
        _migrate_legacy_binding(payload, compiler)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_legacy_binding_malformed_descriptor_remains_integrity_error() -> None:
    compiler = _compiler()
    payload = _legacy_binding_payload(compiler)
    payload["local_runtime_capability_descriptors"] = [None]

    with pytest.raises(AIError) as raised:
        _migrate_legacy_binding(payload, compiler)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_legacy_binding_future_version_remains_unsupported() -> None:
    compiler = _compiler()
    payload = _legacy_binding_payload(compiler)
    payload["version"] = 2

    with pytest.raises(AIError) as raised:
        _migrate_legacy_binding(payload, compiler)

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_persisted_explicit_null_schema_is_integrity_error() -> None:
    data = copy.deepcopy(_transcript_chunk_data())
    data["value"]["payload"]["schema"] = None

    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(data, TranscriptChunk)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_maintenance_accepts_current_raw_state_formats() -> None:
    inspection = _inspection()
    references: dict[int, set[str]] = {}
    event = StoredFact(
        b"s" * 32,
        1,
        b"o" * 32,
        ExecutionEventType.EXECUTION_CREATED.value,
        None,
        None,
        {"value": "event"},
    )
    read_model_fact = StoredFact(
        b"r" * 32,
        1,
        b"o" * 32,
        "execution_read_trace",
        None,
        None,
        {"items": [{"value": "trace"}]},
    )

    inspection._collect_references(
        RuntimeDomain.EXECUTION,
        (_read_model_record(),),
        (event, read_model_fact),
        (),
        references,
    )

    assert references == {}


def test_maintenance_accepts_lease_projected_records() -> None:
    now = datetime.now(timezone.utc)
    tool_operation = ToolOperationRecord(
        "operation",
        "tenant",
        "step",
        "call",
        "key",
        "tool",
        "arguments",
        "binding",
        True,
        ToolOperationStatus.PENDING,
        None,
        0,
        None,
        None,
        now,
        now,
    )

    references: dict[int, set[str]] = {}
    inspection = _inspection()
    inspection._collect_references(
        RuntimeDomain.TASK,
        (_record("task_node", _task_node()),),
        (),
        (),
        references,
    )
    inspection._collect_references(
        RuntimeDomain.RECOVERY,
        (_record("tool_operation", tool_operation),),
        (),
        (),
        references,
    )

    assert references == {}


def test_maintenance_preserves_future_lease_projected_schema() -> None:
    record = _record("task_node", _task_node())
    data = _future_schema(record.data)
    data["value"]["payload"]["fields"]["owner"] = None

    with pytest.raises(AIError) as raised:
        _inspection()._collect_references(
            RuntimeDomain.TASK,
            (replace(record, data=data),),
            (),
            (),
            {},
        )

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_maintenance_rejects_future_read_model_version() -> None:
    inspection = _inspection()
    with pytest.raises(AIError) as raised:
        inspection._collect_references(
            RuntimeDomain.EXECUTION,
            (_read_model_record(version=2),),
            (),
            (),
            {},
        )

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_maintenance_rejects_future_persisted_schema() -> None:
    fact = StoredFact(
        b"s" * 32,
        1,
        b"o" * 32,
        "transcript_chunk",
        None,
        "raw",
        _future_schema(_transcript_chunk_data()),
    )

    with pytest.raises(AIError) as raised:
        _inspection()._collect_references(
            RuntimeDomain.EXECUTION,
            (),
            (fact,),
            (),
            {},
        )

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.asyncio
async def test_transcript_decoders_preserve_future_schema_unsupported(
    tmp_path: Path,
) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="future-transcript-schema", tenant_id="tenant")
    try:
        history = state.steps.read_store(
            RuntimeDomain.EXECUTION
        ).transcript_repository

        head = history.empty_head("owner")
        head_record = history._new_head_record(head)
        with pytest.raises(AIError) as raised:
            history.decode_head(
                replace(head_record, data=_future_schema(head_record.data))
            )
        assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

        seek = TranscriptSeekRecord(
            "owner",
            TranscriptSeekDimension.MESSAGE,
            0,
            1,
            0,
            0,
            1,
        )
        seek_data = encode_envelope(
            {
                "type": "transcript_seek",
                "payload": _encode_persisted_domain(seek),
            }
        )
        with pytest.raises(AIError) as raised:
            history._decode_seek(
                replace(
                    head_record,
                    kind="transcript_seek",
                    data=_future_schema(seek_data),
                )
            )
        assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

        fact = StoredFact(
            b"s" * 32,
            1,
            b"o" * 32,
            "transcript_chunk",
            None,
            "raw",
            _future_schema(_transcript_chunk_data()),
        )
        with pytest.raises(AIError) as raised:
            history.decode_chunk(fact)
        assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    finally:
        await state.close()
