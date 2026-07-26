#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run commit wire codec contract.

The codec is a typed wire protocol: every request and result round-trips
back into an equal typed domain object, bytes/datetime/tuple/event survive,
and identical logical values produce byte-identical wire bytes across
PYTHONHASHSEED (so SQL and Filesystem agree on request_hash)."""

from __future__ import annotations

import asyncio
import base64
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256

import pytest

from linktools.ai.events.context import EventStreamContext
from linktools.ai.events.payloads import (
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunPaused,
    RunResumed,
    RunStarted,
)
from linktools.ai.run.commit import (
    AcknowledgeCancelRunCommand,
    ApprovalRequestData,
    CancelledRunCommit,
    CancellingRunCommit,
    CompleteRunCommand,
    CompletedRunCommit,
    ExecutionFence,
    FailRunCommand,
    FailedRunCommit,
    PauseRunCommand,
    PausedRunCommit,
    RequestCancelRunCommand,
    ResumeRunCommand,
    ResumedRunCommit,
    RunCommitId,
    StartRunCommand,
    StartedRunCommit,
)
from linktools.ai.run.models import (
    RunErrorInfo,
    RunInput,
    RunnableType,
    RunRecord,
    RunResult,
    RunStatus,
)
from linktools.ai.run.persistence.codec import RunCommitCodec
from linktools.ai.run.persistence.wire import (
    RunCommitCodecError,
    RunCommitIntegrityError,
    RunCommitOperation,
)
from linktools.ai.session.models import MessageRole, NewSessionMessage


def _ctx() -> EventStreamContext:
    return EventStreamContext(
        stream_id="stream-1",
        run_id="run-1",
        root_run_id="run-1",
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
    )


def _record(*, status: RunStatus = RunStatus.RUNNING) -> RunRecord:
    return RunRecord(
        id="run-1",
        root_run_id="run-1",
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        status=status,
        input=RunInput(prompt="hello"),
        result=None,
        error=None,
        version=1,
        created_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        started_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        finished_at=None,
    )


def _messages() -> "tuple[NewSessionMessage, ...]":
    return (
        NewSessionMessage(
            role=MessageRole.USER,
            content="hi",
            run_id="run-1",
            metadata={"k": "v"},
        ),
        NewSessionMessage(
            role=MessageRole.ASSISTANT,
            content="hello",
            run_id="run-1",
        ),
    )


def _approval() -> ApprovalRequestData:
    return ApprovalRequestData(
        approval_id="appr-1",
        tool_name="t",
        reason="r",
        arguments={"a": 1},
        tenant_id="tenant-1",
        tool_call_id="tc-1",
        binding={"descriptor_fingerprint": "fp"},
    )


def _start_command() -> StartRunCommand:
    return StartRunCommand(
        record=_record(),
        started_event=RunStarted(run_id="run-1", runnable_id="agent-1"),
        event_context=_ctx(),
        commit_id=RunCommitId("cid-start"),
    )


def _pause_command() -> PauseRunCommand:
    return PauseRunCommand(
        run_id="run-1",
        expected_version=1,
        approval_request=_approval(),
        checkpoint_payload=b"\x00\x01\x02",
        paused_event=RunPaused(run_id="run-1", reason="need-approval"),
        event_context=_ctx(),
        commit_id=RunCommitId("cid-pause"),
        execution_fence=ExecutionFence("tok"),
        messages=_messages(),
    )


def _resume_command() -> ResumeRunCommand:
    return ResumeRunCommand(
        run_id="run-1",
        expected_version=2,
        approval_id="appr-1",
        resumed_event=RunResumed(run_id="run-1"),
        event_context=_ctx(),
        commit_id=RunCommitId("cid-resume"),
    )


def _complete_command() -> CompleteRunCommand:
    return CompleteRunCommand(
        run_id="run-1",
        session_id="session-1",
        expected_version=3,
        messages=_messages(),
        checkpoint_payload=b"\xff\xfe",
        result=RunResult(output={"answer": 42}, metadata={"m": 1}),
        completed_event=RunCompleted(
            run_id="run-1", result_summary={"answer": 42}
        ),
        event_context=_ctx(),
        commit_id=RunCommitId("cid-complete"),
        execution_fence=ExecutionFence("tok"),
    )


def _fail_command() -> FailRunCommand:
    return FailRunCommand(
        run_id="run-1",
        expected_version=2,
        error=RunErrorInfo(error_type="Boom", message="bad"),
        failed_event=RunFailed(
            run_id="run-1", error_type="Boom", message="bad"
        ),
        event_context=_ctx(),
        commit_id=RunCommitId("cid-fail"),
        execution_fence=ExecutionFence("tok"),
    )


def _request_cancel_command() -> RequestCancelRunCommand:
    return RequestCancelRunCommand(
        run_id="run-1",
        expected_version=3,
        requested_by="user",
        reason="done",
        event_context=_ctx(),
        commit_id=RunCommitId("cid-req"),
    )


def _ack_cancel_command() -> AcknowledgeCancelRunCommand:
    return AcknowledgeCancelRunCommand(
        run_id="run-1",
        expected_version=4,
        cancelled_event=RunCancelled(run_id="run-1", reason="done"),
        event_context=_ctx(),
        commit_id=RunCommitId("cid-ack"),
        execution_fence=ExecutionFence("tok"),
    )


REQUESTS = {
    RunCommitOperation.START: _start_command(),
    RunCommitOperation.PAUSE: _pause_command(),
    RunCommitOperation.RESUME: _resume_command(),
    RunCommitOperation.COMPLETE: _complete_command(),
    RunCommitOperation.FAIL: _fail_command(),
    RunCommitOperation.REQUEST_CANCEL: _request_cancel_command(),
    RunCommitOperation.ACKNOWLEDGE_CANCEL: _ack_cancel_command(),
}

RESULTS = {
    RunCommitOperation.START: StartedRunCommit(record=_record()),
    RunCommitOperation.PAUSE: PausedRunCommit(approval_id="appr-1", checkpoint_id="cp-1"),
    RunCommitOperation.RESUME: ResumedRunCommit(run_id="run-1"),
    RunCommitOperation.COMPLETE: CompletedRunCommit(
        result=RunResult(output={"answer": 42}, metadata={"m": 1})
    ),
    RunCommitOperation.FAIL: FailedRunCommit(run_id="run-1"),
    RunCommitOperation.REQUEST_CANCEL: CancellingRunCommit(run_id="run-1"),
    RunCommitOperation.ACKNOWLEDGE_CANCEL: CancelledRunCommit(run_id="run-1"),
}


@pytest.mark.parametrize("op", list(REQUESTS))
def test_request_round_trip(op: RunCommitOperation):
    codec = RunCommitCodec()
    command = REQUESTS[op]
    decoded = codec.decode_request(op, codec.encode_request(op, command))
    assert decoded == command
    assert type(decoded) is type(command)


@pytest.mark.parametrize("op", list(RESULTS))
def test_result_round_trip(op: RunCommitOperation):
    codec = RunCommitCodec()
    result = RESULTS[op]
    decoded = codec.decode_result(op, codec.encode_result(op, result))
    assert decoded == result
    assert type(decoded) is type(result)


def test_bytes_round_trip():
    codec = RunCommitCodec()
    command = _complete_command()
    assert command.checkpoint_payload == b"\xff\xfe"
    decoded = codec.decode_request(
        RunCommitOperation.COMPLETE, codec.encode_request(RunCommitOperation.COMPLETE, command)
    )
    assert decoded.checkpoint_payload == b"\xff\xfe"


def test_utc_datetime_round_trip():
    codec = RunCommitCodec()
    started = StartedRunCommit(record=_record())
    decoded = codec.decode_result(
        RunCommitOperation.START, codec.encode_result(RunCommitOperation.START, started)
    )
    assert decoded.record.created_at.tzinfo is not None
    assert decoded.record.created_at == started.record.created_at


def test_message_tuple_order_preserved():
    codec = RunCommitCodec()
    command = _pause_command()
    decoded = codec.decode_request(
        RunCommitOperation.PAUSE, codec.encode_request(RunCommitOperation.PAUSE, command)
    )
    assert isinstance(decoded.messages, tuple)
    assert [m.content for m in decoded.messages] == [m.content for m in command.messages]


def test_event_round_trip():
    codec = RunCommitCodec()
    command = _complete_command()
    decoded = codec.decode_request(
        RunCommitOperation.COMPLETE, codec.encode_request(RunCommitOperation.COMPLETE, command)
    )
    assert decoded.completed_event == command.completed_event


def test_unknown_schema_version_rejected():
    codec = RunCommitCodec()
    payload = codec.encode_result(RunCommitOperation.FAIL, RESULTS[RunCommitOperation.FAIL])
    import json
    envelope = json.loads(payload)
    envelope["schema_version"] = 999
    bad = json.dumps(envelope).encode()
    with pytest.raises(RunCommitCodecError):
        codec.decode_result(RunCommitOperation.FAIL, bad)


def test_unknown_operation_rejected():
    codec = RunCommitCodec()
    import json
    envelope = json.loads(codec.encode_result(RunCommitOperation.FAIL, RESULTS[RunCommitOperation.FAIL]))
    envelope["operation"] = "no_such_op"
    bad = json.dumps(envelope).encode()
    with pytest.raises(RunCommitCodecError):
        codec.decode_result(RunCommitOperation.FAIL, bad)


def test_mismatched_operation_rejected():
    codec = RunCommitCodec()
    # Encode a FAIL result but ask the codec to decode it as COMPLETE.
    payload = codec.encode_result(RunCommitOperation.FAIL, RESULTS[RunCommitOperation.FAIL])
    with pytest.raises(RunCommitCodecError):
        codec.decode_result(RunCommitOperation.COMPLETE, payload)


def test_malformed_base64_rejected():
    codec = RunCommitCodec()
    import json
    envelope = json.loads(
        codec.encode_request(RunCommitOperation.COMPLETE, _complete_command())
    )
    envelope["payload"]["checkpoint_payload_b64"] = "!!!"
    with pytest.raises(RunCommitCodecError):
        codec.decode_request(
            RunCommitOperation.COMPLETE,
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode(),
        )


def test_malformed_json_rejected():
    codec = RunCommitCodec()
    with pytest.raises(RunCommitCodecError):
        codec.decode_result(RunCommitOperation.FAIL, b"{not json")


def test_request_hash_stable_across_hashseed():
    """The same logical command must produce identical wire bytes (and thus
    identical request_hash) regardless of PYTHONHASHSEED. Verified in two
    subprocesses with different seeds so the seed is actually different."""
    import os
    script = (
        "import sys\n"
        "from datetime import datetime, timezone\n"
        "from linktools.ai.events.context import EventStreamContext\n"
        "from linktools.ai.events.payloads import RunCompleted\n"
        "from linktools.ai.run.commit import CompleteRunCommand, ExecutionFence, RunCommitId\n"
        "from linktools.ai.run.models import RunInput, RunnableType, RunRecord, RunResult, RunStatus\n"
        "from linktools.ai.run.persistence.codec import RunCommitCodec\n"
        "from linktools.ai.session.models import MessageRole, NewSessionMessage\n"
        "ctx = EventStreamContext(stream_id='s', run_id='r', root_run_id='r', parent_run_id=None, session_id='sess', runnable_id='a')\n"
        "rec = RunRecord(id='r', root_run_id='r', parent_run_id=None, session_id='sess', runnable_id='a', runnable_type=RunnableType.AGENT, status=RunStatus.RUNNING, input=RunInput(prompt='p'), result=None, error=None, version=1, created_at=datetime(2026,1,1,tzinfo=timezone.utc), started_at=datetime(2026,1,1,tzinfo=timezone.utc), finished_at=None)\n"
        "msgs = (NewSessionMessage(role=MessageRole.USER, content='hi', run_id='r'),)\n"
        "cmd = CompleteRunCommand(run_id='r', session_id='sess', expected_version=1, messages=msgs, checkpoint_payload=b'x', result=RunResult(output={'a': 1}), completed_event=RunCompleted(run_id='r', result_summary={'a': 1}), event_context=ctx, commit_id=RunCommitId('cid'), execution_fence=ExecutionFence('tok'))\n"
        "sys.stdout.write(RunCommitCodec().request_hash('complete', cmd).hex())\n"
    )
    env0 = {**os.environ, "PYTHONHASHSEED": "0"}
    env1 = {**os.environ, "PYTHONHASHSEED": "1"}
    r0 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env0)
    r1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env1)
    assert r0.returncode == 0, f"seed0 failed: {r0.stderr}"
    assert r1.returncode == 0, f"seed1 failed: {r1.stderr}"
    assert r0.stdout.strip(), "seed0 produced no hash"
    assert r0.stdout.strip() == r1.stdout.strip(), "request_hash differs across PYTHONHASHSEED"


def test_request_hash_sql_and_file_identical():
    """The codec is shared, so SQL and Filesystem derive the same hash from
    the same command (the single source of truth for request identity)."""
    codec = RunCommitCodec()
    command = _complete_command()
    a = codec.request_hash(RunCommitOperation.COMPLETE, command)
    b = codec.request_hash("complete", command)
    assert a == b
    assert isinstance(a, bytes) and len(a) == 32
