#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained bytes for one captured, unpublished Trace projection."""

import hashlib
import json
from dataclasses import replace
from types import MappingProxyType

from pydantic_ai.messages import ModelRequest, ModelResponse

from ..core import canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..model import ModelClientRequest, ModelClientResponse
from ._model_interaction import (
    StagedContextInline, StagedContextProjection, StagedContextSpan,
    StagedModelInteraction, StagedStreamAttemptRecord,
)
from ._step_archive import CapturedExecutionProjection
from ._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent
from ._transcript_staging import StagedTranscript
from .codec import _CURRENT_CODEC, _decode_domain, _encode_domain


# These temporary representations do not become supported durable record types.
_STAGED_TYPES = (
    ("checkpoint", AgentRunCheckpoint),
    ("context_inline", StagedContextInline),
    ("context_span", StagedContextSpan),
    ("context_projection", StagedContextProjection),
    ("interaction", StagedModelInteraction),
    ("stream_attempt", StagedStreamAttemptRecord),
    ("client_request", ModelClientRequest),
    ("client_response", ModelClientResponse),
)
_CODEC = replace(
    _CURRENT_CODEC,
    wire_ids=MappingProxyType({**_CURRENT_CODEC.wire_ids, **{
        target: "buffer_" + name for name, target in _STAGED_TYPES
    }}),
    domain_types=MappingProxyType({**_CURRENT_CODEC.domain_types, **{
        "buffer_" + name: target for name, target in _STAGED_TYPES
    }}),
)


def encode_projection_payload(projection: CapturedExecutionProjection) -> bytes:
    """Freeze values and all new content bytes without accessing a source store."""
    transcript = projection.transcript
    checkpoints = tuple(replace(
        value, messages=list(value.messages),
        context_messages=None if value.context_messages is None else list(value.context_messages),
    ) for value in projection.checkpoints)
    values = {
        "run": projection.run,
        "events": projection.events,
        "checkpoints": checkpoints,
        "interactions": projection.interactions,
        "transcript": None if transcript is None else (
            tuple(transcript.messages), transcript.pending, transcript.pending_keys,
        ),
        "local_messages": tuple(projection.local_messages),
        "payloads": projection.payloads,
        "positions": (
            projection.base_event_offset, projection.target_event_offset,
            projection.base_checkpoint_offset, projection.target_checkpoint_offset,
            projection.base_interaction_offset, projection.target_interaction_offset,
            projection.base_message_index, projection.target_message_index,
        ),
        "producer_generation": projection.producer_generation,
    }
    return canonical_json_bytes({
        "format": "trace-projection-v1",
        "projection": {name: _encode_domain(value, _CODEC) for name, value in values.items()},
    })


def decode_projection_payload(content: bytes) -> CapturedExecutionProjection:
    """Validate and restore a projection without consulting a staging mirror."""
    try:
        envelope = json.loads(content)
        if not isinstance(envelope, dict) or envelope.get("format") != "trace-projection-v1":
            raise ValueError("unsupported Trace projection payload")
        values = envelope["projection"]
        if not isinstance(values, dict):
            raise ValueError("invalid Trace projection payload")
        run = _decode_domain(values["run"], AgentRunRecord, _CODEC)
        events = _decode_domain(values["events"], tuple[StepEvent, ...], _CODEC)
        checkpoints = _decode_domain(values["checkpoints"], tuple[AgentRunCheckpoint, ...], _CODEC)
        interactions = _decode_domain(values["interactions"], tuple[StagedModelInteraction, ...], _CODEC)
        transcript = _decode_domain(
            values["transcript"], tuple[tuple[ModelRequest | ModelResponse, ...], ModelRequest | ModelResponse | None, tuple[str, ...]] | None, _CODEC,
        )
        local_messages = _decode_domain(values["local_messages"], tuple[ModelRequest | ModelResponse, ...], _CODEC)
        payloads = _decode_domain(values["payloads"], tuple[tuple[str, bytes], ...], _CODEC)
        positions = _decode_domain(values["positions"], tuple[int, int, int, int, int, int, int, int], _CODEC)
        generation = _decode_domain(values["producer_generation"], int | None, _CODEC)
        if (any(base < 0 or target < base for base, target in zip(positions[::2], positions[1::2]))
                or generation is not None and generation < 0
                or len(events) != positions[1] - positions[0]
                or len(checkpoints) != positions[3] - positions[2]
                or any(value.agent_run_id != run.agent_run_id for value in (*events, *checkpoints, *interactions))
                or len({digest for digest, _ in payloads}) != len(payloads)
                or any(hashlib.sha256(data).hexdigest() != digest for digest, data in payloads)):
            raise ValueError("invalid Trace projection identity or frontier")
        contents = dict(payloads)
        for interaction in interactions:
            for digest, size in interaction.payload_dependencies():
                if digest not in contents or size is not None and len(contents[digest]) != size:
                    raise ValueError("missing or invalid Trace projection dependency")
        if transcript is not None and len(transcript[0]) != positions[7] - positions[6]:
            raise ValueError("invalid Trace transcript frontier")
        return CapturedExecutionProjection(
            run, events, checkpoints, positions[0], positions[2], positions[1], positions[3],
            interactions, positions[4], positions[5],
            None if transcript is None else StagedTranscript(*transcript),
            positions[6], positions[7], generation, payloads, local_messages,
        )
    except AIError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
