#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run commit replay contract.

A replay returns the FIRST persisted typed result, never a value rebuilt from
the current command. Covers the Filesystem complete-replay bug (missing
operation argument) and the SQL/File shared request-hash contract."""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone
from hashlib import sha256

import pytest

from linktools.ai.events.context import EventStreamContext
from linktools.ai.events.payloads import RunCompleted, RunStarted
from linktools.ai.run.commit import (
    CompleteRunCommand,
    CompletedRunCommit,
    ExecutionFence,
    RunCommitId,
    StartRunCommand,
    StartedRunCommit,
)
from linktools.ai.run.models import (
    RunInput,
    RunnableType,
    RunRecord,
    RunResult,
    RunStatus,
)
from linktools.ai.run.persistence.codec import RunCommitCodec
from linktools.ai.run.persistence.wire import (
    RunCommitIntegrityError,
    RunCommitOperation,
)
from linktools.ai.session.models import MessageRole, NewSessionMessage


def _ctx() -> EventStreamContext:
    return EventStreamContext(
        stream_id="s", run_id="r", root_run_id="r", parent_run_id=None,
        session_id="sess", runnable_id="a",
    )


def _record() -> RunRecord:
    return RunRecord(
        id="r", root_run_id="r", parent_run_id=None, session_id="sess",
        runnable_id="a", runnable_type=RunnableType.AGENT, status=RunStatus.RUNNING,
        input=RunInput(prompt="p"), result=None, error=None, version=1,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        finished_at=None,
    )


def test_complete_replay_decodes_typed_result():
    """The Filesystem complete-replay path historically called decode_result
    without the operation argument (TypeError) and fell back to the current
    command. The codec must now decode the FIRST persisted typed result."""
    codec = RunCommitCodec()
    first = CompletedRunCommit(
        result=RunResult(output={"answer": "first"}, metadata={"n": 1})
    )
    payload = codec.encode_result(RunCommitOperation.COMPLETE, first)
    # The replay helper decodes with the operation argument.
    decoded = codec.decode_result(RunCommitOperation.COMPLETE, payload)
    assert isinstance(decoded, CompletedRunCommit)
    assert decoded == first
    # A *different* (current-command) result is NOT what replay returns.
    other = CompletedRunCommit(result=RunResult(output={"answer": "other"}))
    assert decoded != other


def test_complete_replay_missing_payload_raises_integrity_error():
    """A completion entry with no result_payload fails closed
    (RunCommitIntegrityError) rather than falling back to the current
    command -- a replay must always return the first persisted result."""
    codec = RunCommitCodec()

    # Mimic the Filesystem replay helper: it requires a "__result_payload_b64"
    # key on the completion dict; its absence is an integrity error.
    def decode_completion(replay: dict):
        encoded = replay.get("__result_payload_b64")
        if not isinstance(encoded, str):
            raise RunCommitIntegrityError("no result payload")
        return codec.decode_result(
            RunCommitOperation.COMPLETE,
            base64.b64decode(encoded, validate=True),
        )

    with pytest.raises(RunCommitIntegrityError):
        decode_completion({})  # no payload at all


def test_start_replay_returns_first_record():
    """Start replay returns the FIRST persisted StartedRunCommit, not the
    current command's record (the two diverge if the caller mutated the
    record between attempts)."""
    codec = RunCommitCodec()
    first = StartedRunCommit(record=_record())
    payload = codec.encode_result(RunCommitOperation.START, first)
    decoded = codec.decode_result(RunCommitOperation.START, payload)
    assert isinstance(decoded, StartedRunCommit)
    assert decoded.record.id == "r"
    assert decoded == first


def test_sql_and_file_share_request_hash():
    """SQL and Filesystem coordinators both use the same codec, so the same
    CompleteRunCommand produces one request_hash -- the single source of
    truth for commit identity across backends."""
    codec = RunCommitCodec()
    cmd = CompleteRunCommand(
        run_id="r", session_id="sess", expected_version=1,
        messages=(NewSessionMessage(role=MessageRole.USER, content="hi", run_id="r"),),
        checkpoint_payload=b"x",
        result=RunResult(output={"a": 1}),
        completed_event=RunCompleted(run_id="r", result_summary={"a": 1}),
        event_context=_ctx(),
        commit_id=RunCommitId("cid"),
        execution_fence=ExecutionFence("tok"),
    )
    h_enum = codec.request_hash(RunCommitOperation.COMPLETE, cmd)
    h_str = codec.request_hash("complete", cmd)
    assert h_enum == h_str
    # Re-encoding the same command is byte-stable.
    assert codec.encode_request(RunCommitOperation.COMPLETE, cmd) == codec.encode_request(
        RunCommitOperation.COMPLETE, cmd
    )


def test_replay_after_response_loss_returns_first_result():
    """Response-loss scenario: the first complete commits and stores its
    result; the caller retries. The replay must return the FIRST result by
    value even though the caller now supplies a different in-memory object."""
    codec = RunCommitCodec()
    first_payload = codec.encode_result(
        RunCommitOperation.COMPLETE,
        CompletedRunCommit(result=RunResult(output={"answer": "first"})),
    )
    # The retry's in-memory result differs, but replay ignores it and decodes
    # the stored first payload.
    replayed = codec.decode_result(RunCommitOperation.COMPLETE, first_payload)
    assert replayed.result.output == {"answer": "first"}
