#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Immutable staging snapshots preserve range and consumer acknowledgement semantics."""

import gc

import pytest

from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.persistence._transcript_staging import MessageArchive
from linktools.ai.persistence._step_archive import StagingAgentRunStore, _ProjectionOffset
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord


def test_checkpoint_staging_bodies_shared_and_capture_survives_trim(monkeypatch):
    staging = StagingAgentRunStore()
    staging.register_agent_run_local(AgentRunRecord(agent_run_id="run"))
    messages = []
    for index in range(4):
        messages.append(ModelRequest(parts=[UserPromptPart(str(index))]))
        staging.save_checkpoint_local(AgentRunCheckpoint("run", index, list(messages)))
    archive = staging._message_archives["run"]
    assert len(archive._leaves) == 4
    assert len(archive._branches) < 4 * 8
    from linktools.ai.persistence import _transcript_staging
    decoded = []
    decode = _transcript_staging.decode_model_messages

    def observe(payload):
        decoded.append(payload)
        return decode(payload)

    monkeypatch.setattr(_transcript_staging, "decode_model_messages", observe)
    captured = staging.capture_projection_local("run", _ProjectionOffset())
    assert decoded == []
    assert captured is not None and len(captured.checkpoints) == 4
    staging.trim_checkpoints_local("run", checkpoint_offset=4)
    assert staging.checkpoint_count_local("run") == 4
    assert len(staging.capture_projection_local("run", _ProjectionOffset(checkpoints=4)).checkpoints) == 0
    messages[0].parts[0].content = "caller mutation"
    first = captured.checkpoints[0]
    assert len(decoded) == 1
    assert first.messages[0].parts[0].content == "0"
    first.messages[0].parts[0].content = "reader mutation"
    assert captured.checkpoints[0].messages[0].parts[0].content == "0"
    staging.save_checkpoint_local(AgentRunCheckpoint("run", 4, messages))
    suffix = staging.capture_projection_local("run", _ProjectionOffset(checkpoints=4))
    assert suffix.target_checkpoint_offset == 5 and len(suffix.checkpoints) == 1
    staging.release_agent_run_local("run")
    assert captured.checkpoints[-1].messages[-1].parts[0].content == "3"
    del captured
    gc.collect()
    assert len(archive._branches) == 3
    assert len(archive._leaves) == 4


def test_message_staging_releases_unreferenced_context_and_returns_independent_values():
    archive = MessageArchive()
    first = archive.snapshot([ModelRequest(parts=[UserPromptPart("first")])])
    later = archive.snapshot([ModelRequest(parts=[UserPromptPart("later")])])
    assert len(archive._leaves) == 2
    del first
    gc.collect()
    assert len(archive._leaves) == 1
    assert later[0].parts[0].content == "later"


def test_trace_staging_waits_for_its_acknowledged_frontier():
    from linktools.ai.persistence._plan import PersistenceDomain
    from linktools.ai.persistence._step_archive import InMemoryStepArchive
    from linktools.ai.persistence.runs import RuntimeAgentRunStore

    staging = StagingAgentRunStore()
    staging.register_agent_run_local(AgentRunRecord(agent_run_id="run"))
    owner = RuntimeAgentRunStore(
        staging, conversation_archive=InMemoryStepArchive(PersistenceDomain.CONVERSATION),
        execution_archive=InMemoryStepArchive(PersistenceDomain.TRACE),
    )
    for index in range(4):
        staging.save_checkpoint_local(AgentRunCheckpoint(
            "run", index, [ModelRequest(parts=[UserPromptPart(str(index))])],
        ))
    owner.trace_persistence.offsets["run"] = _ProjectionOffset(checkpoints=0)
    owner._trim_checkpoint_staging_locked("run")
    assert len(staging.list_checkpoints_local("run")) == 4
    owner.trace_persistence.offsets["run"].checkpoints = 2
    owner._trim_checkpoint_staging_locked("run")
    assert [value.step_index for value in staging.list_checkpoints_local("run")] == [2, 3]
    owner.trace_persistence.offsets["run"].checkpoints = 3
    owner._trim_checkpoint_staging_locked("run")
    assert [value.step_index for value in staging.list_checkpoints_local("run")] == [3]
    owner.trace_persistence.offsets["run"].checkpoints = 4
    owner._trim_checkpoint_staging_locked("run")
    assert staging.list_checkpoints_local("run") == []
    assert staging.latest_checkpoint_local("run").step_index == 3


def test_message_staging_slice_does_not_pin_excluded_prefix():
    archive = MessageArchive()
    original = archive.snapshot([ModelRequest(parts=[UserPromptPart(str(i))]) for i in range(4)])
    tail = archive.snapshot(original[-1:])
    del original
    gc.collect()
    assert len(archive._leaves) == 1
    assert len(archive._branches) == 0
    assert len(tail) == 1 and tail[0].parts[0].content == "3"


def test_message_staging_sequential_read_does_not_repeat_index_traversal(monkeypatch):
    from linktools.ai.persistence._transcript_staging import MessageSnapshot

    snapshot = MessageArchive().snapshot([ModelRequest(parts=[UserPromptPart(str(i))]) for i in range(4)])

    def no_index(*args):
        raise AssertionError("sequential read must traverse the tree once")

    monkeypatch.setattr(MessageSnapshot, "__getitem__", no_index)
    assert [message.parts[0].content for message in snapshot] == [str(i) for i in range(4)]


@pytest.mark.asyncio
async def test_memory_archive_counts_and_ranges_do_not_decode_whole_checkpoint(monkeypatch):
    from linktools.ai.persistence import _transcript_staging
    from linktools.ai.persistence._plan import PersistenceDomain
    from linktools.ai.persistence._step_archive import InMemoryStepArchive

    archive = InMemoryStepArchive(PersistenceDomain.CONVERSATION)
    archive.register_agent_run_local(AgentRunRecord(agent_run_id="run"))
    archive.save_checkpoint_local(AgentRunCheckpoint(
        "run", 0, [ModelRequest(parts=[UserPromptPart(str(i))]) for i in range(4)],
    ))
    decoded = []
    original = _transcript_staging.decode_model_messages

    def observe(payload):
        decoded.append(payload)
        return original(payload)

    monkeypatch.setattr(_transcript_staging, "decode_model_messages", observe)
    assert await archive.transcript_message_count("run") == 4
    assert decoded == []
    page = [message async for message in archive.iter_message_range(agent_run_id="run", start=1, end=3)]
    assert [message.parts[0].content for message in page] == ["1", "2"]
    assert len(decoded) == 2


def test_frozen_message_digest_matches_durable_codec_and_concatenation():
    import hashlib
    from pydantic_ai.messages import ModelResponse, TextPart, ToolReturnPart
    from linktools.ai.model.messages import encode_model_messages

    values = [
        ModelRequest(parts=[UserPromptPart("你好，世界")]),
        ModelResponse(parts=[TextPart("answer")]),
        ModelRequest(parts=[ToolReturnPart("tool", {"value": [1, None, "é"]}, "call")]),
    ]
    archive = MessageArchive()
    snapshot = archive.concatenate(archive.snapshot(values[:1]), archive.snapshot(values[1:]))
    for sequence in (snapshot, snapshot[:2], snapshot[1:], snapshot[:0]):
        raw = encode_model_messages(tuple(sequence))
        assert sequence.encoded_digest() == (hashlib.sha256(raw).hexdigest(), len(raw))
        assert tuple(sequence.encoded_payloads()) == tuple(encode_model_messages((message,)) for message in sequence)
    values[0].parts[0].content = "changed"
    assert snapshot[0].parts[0].content == "你好，世界"
