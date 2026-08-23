#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence version errors must survive repository wrapper boundaries."""

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.agent import AgentCompiler
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.runtime.state._codec import _encode_persisted_domain, encode_envelope
from linktools.ai.runtime.state._contracts import (
    RuntimePayloadRef,
    TranscriptChunk,
    TranscriptOrigin,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
)
from linktools.ai.runtime.state._migration import _migrate_legacy_binding
from linktools.ai.runtime.state._store import StoredFact
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload


def _future_schema(data: object) -> object:
    value = copy.deepcopy(data)
    value["value"]["payload"]["schema"] = 99
    return value


def test_legacy_binding_malformed_known_field_remains_integrity_error() -> None:
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        runtime_fingerprint="a" * 64,
    )
    binding = compiler.bind(compiler.compile(AgentSpec("default")))
    payload = dict(binding.snapshot.to_payload())
    payload.pop("agent_digest")
    spec = dict(payload["agent_spec"])
    spec["metadata"] = {}
    payload["agent_spec"] = spec
    payload["binding_digest"] = "f" * 64
    payload["output_schema_fingerprint"] = "invalid"

    with pytest.raises(AIError) as raised:
        _migrate_legacy_binding(payload, compiler)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


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
        chunk_data = encode_envelope(
            {
                "type": "transcript_chunk",
                "payload": _encode_persisted_domain(chunk),
            }
        )
        fact = StoredFact(
            b"s" * 32,
            1,
            b"o" * 32,
            "transcript_chunk",
            None,
            "raw",
            _future_schema(chunk_data),
        )
        with pytest.raises(AIError) as raised:
            history.decode_chunk(fact)
        assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    finally:
        await state.close()
