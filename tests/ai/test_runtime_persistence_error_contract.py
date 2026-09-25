#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence version errors must survive repository wrapper boundaries."""

import copy
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import (
    RuntimePayloadRef,
    SessionRecord,
    TranscriptChunk,
    TranscriptOrigin,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
)
from linktools.ai.runtime.state._store import StoredFact
from linktools.ai.storage import StoredPayload


def _future_schema(data: object) -> object:
    value = copy.deepcopy(dict(data))
    value["value"]["payload"]["schema"] = 99
    return value


def _precomposition_session_data() -> dict[str, object]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session = SessionRecord(
        session_id="session",
        owner_principal_id="owner",
        agent_id="agent",
        status=SessionStatus.OPEN,
        revision=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
    )
    payload = _encode_persisted_domain(session)
    payload["fields"].pop("agent_id")
    payload["fields"]["binding_digest"] = "b" * 64
    return encode_envelope({"type": "session_record", "payload": payload})


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


def test_current_session_wire_rejects_missing_agent_identity() -> None:
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_precomposition_session_data(), SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

def test_persisted_explicit_null_schema_is_integrity_error() -> None:
    data = copy.deepcopy(_transcript_chunk_data())
    data["value"]["payload"]["schema"] = None

    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(data, TranscriptChunk)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_transcript_decoders_preserve_future_schema_unsupported(
    tmp_path: Path,
) -> None:
    state = RuntimeStorage.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="future-transcript-schema", tenant_id="tenant")
    try:
        history = state.run_store.read_store(
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
        )
        seek_data = encode_envelope(
            {
                "type": "transcript_seek",
                "payload": _encode_persisted_domain(seek),
            }
        )
        with pytest.raises(AIError) as raised:
            _decode_enveloped_domain(_future_schema(seek_data), TranscriptSeekRecord)
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
