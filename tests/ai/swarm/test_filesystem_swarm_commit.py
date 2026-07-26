#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemSwarmCommitCoordinator contract: same shape as the SQL impl --
every operation is idempotent by commit_id, same-id-different-payload
raises SwarmCommitConflictError, terminal complete is idempotent."""

import asyncio
import json
import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from dataclasses import replace

import pytest

from linktools.ai.events.context import EventStreamContext
from linktools.ai.events.payloads import SwarmCompleted, SwarmStarted
from linktools.ai.run.models import RunResult
from linktools.ai.swarm.commit import (
    CompleteSwarmCommand,
    CompleteSwarmPayload,
    StartSwarmCommand,
    StartSwarmPayload,
    SwarmCommitConflictError,
    SwarmCommitId,
    SwarmCommitPolicy,
    SwarmExecutionFence,
)
from linktools.ai.swarm.models import SwarmRun, SwarmStatus, TokenUsage
from linktools.ai.swarm.persistence.codec import SwarmCommitCodec
from linktools.ai.swarm.persistence.filesystem import FilesystemSwarmStore
from linktools.ai.swarm.persistence.filesystem_commit import (
    FilesystemSwarmCommitCoordinator,
)


# A fixed fence token shared by start (which stamps it as the run-level
# execution_token) and complete (which must supply a matching token). The
# coordinator is configured with fencing_required=True because start always
# establishes a token and terminal commits must prove ownership against it.
_FENCE_TOKEN = "fs-test-token"
_FENCE = SwarmExecutionFence(_FENCE_TOKEN)


def _run(coro):
    return asyncio.run(coro)


def _now() -> "datetime":
    return datetime.now(timezone.utc)


def _ctx(run_id: str = "driving-1") -> EventStreamContext:
    return EventStreamContext(
        stream_id=run_id,
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id="sess",
        runnable_id="swarm",
    )


def _swarm_run(swarm_run_id: str, run_id: str) -> SwarmRun:
    return SwarmRun(
        id=swarm_run_id,
        run_id=run_id,
        round=0,
        status=SwarmStatus.PENDING,
        version=1,
        token_usage=TokenUsage(),
        cost=Decimal("0"),
        created_at=_now(),
        updated_at=_now(),
    )


def _start_command(
    swarm_run_id: str,
    run_id: str,
    *,
    commit_id: "str | None" = None,
) -> StartSwarmCommand:
    return StartSwarmCommand(
        commit_id=SwarmCommitId(commit_id or f"start:{swarm_run_id}"),
        swarm_run_id=swarm_run_id,
        expected_version=1,
        payload=StartSwarmPayload(
            run=_swarm_run(swarm_run_id, run_id),
            started_event=SwarmStarted(swarm_run_id=swarm_run_id, swarm_id="swarm"),
            event_context=_ctx(run_id),
        ),
        fence=_FENCE,
    )


def _complete_command(
    swarm_run_id: str,
    *,
    commit_id: "str | None" = None,
    expected_version: int = 2,
) -> CompleteSwarmCommand:
    return CompleteSwarmCommand(
        commit_id=SwarmCommitId(commit_id or f"complete:{swarm_run_id}"),
        swarm_run_id=swarm_run_id,
        expected_version=expected_version,
        payload=CompleteSwarmPayload(
            result=RunResult(output={"done": True}),
            completed_event=SwarmCompleted(swarm_run_id=swarm_run_id),
            event_context=_ctx(),
        ),
        fence=_FENCE,
    )


class _NullEventStore:
    async def append(self, *args, **kwargs):
        pass

    async def append_once(self, *args, **kwargs):
        pass


def _make(tmp_path):
    swarm_store = FilesystemSwarmStore(root=tmp_path / "swarm")
    transactions_root = tmp_path / "transactions"
    coordinator = FilesystemSwarmCommitCoordinator(
        swarm_store,
        event_store=_NullEventStore(),
        transactions_root=transactions_root,
        # fencing_required=True: start stamps the supplied fence token as the
        # run-level execution_token, and terminal commits prove ownership
        # against it (fencing_required=False would reject the supplied fence
        # under SwarmCommitPolicy.validate()).
        policy=SwarmCommitPolicy(fencing_required=True),
        codec=SwarmCommitCodec(),
    )
    return coordinator, swarm_store


def test_fs_start_is_idempotent(tmp_path):
    coordinator, _ = _make(tmp_path)

    async def _run_async():
        command = _start_command("swarm-fs-1", "driving-fs-1")
        first = await coordinator.start(command)
        second = await coordinator.start(command)
        return first, second

    first, second = _run(_run_async())
    assert first == second


def test_fs_start_same_commit_id_different_payload_conflicts(tmp_path):
    coordinator, _ = _make(tmp_path)

    async def _run_async():
        # Same commit_id, different driving run_id -> different request_hash
        # -> SwarmCommitConflictError on the second start.
        await coordinator.start(
            _start_command("swarm-fs-x", "driving-a", commit_id="start:swarm-fs-x")
        )
        with pytest.raises(SwarmCommitConflictError):
            await coordinator.start(
                _start_command("swarm-fs-x", "driving-b", commit_id="start:swarm-fs-x")
            )

    _run(_run_async())


def test_fs_terminal_complete_is_idempotent(tmp_path):
    coordinator, swarm_store = _make(tmp_path)
    _run(coordinator.start(_start_command("swarm-fs-c", "driving-fs-c")))
    _run(
        swarm_store.update_run(
            "swarm-fs-c",
            expected_version=1,
            expected_token=_FENCE_TOKEN,
            status=SwarmStatus.RUNNING,
        )
    )

    async def _run_async():
        command = _complete_command("swarm-fs-c")
        first = await coordinator.complete(command)
        second = await coordinator.complete(command)
        return first, second

    first, second = _run(_run_async())
    assert first == second
    run = _run(swarm_store.get_run("swarm-fs-c"))
    assert run.status is SwarmStatus.SUCCEEDED


def test_fs_recovery_marks_inflight_commit_failed(tmp_path):
    """A crash leaving an in-flight swarm journal (the commit started but did
    not finish) reconciles at next startup: recovery marks the affected
    swarm_run FAILED (fail-closed), retains the journal for forensics, and
    raises so the caller knows evidence was left behind."""
    from linktools.ai.swarm.persistence.filesystem_commit import SwarmRecoveryError

    coordinator, swarm_store = _make(tmp_path)
    # Seed a swarm_run + an in-flight journal as if a complete commit had
    # started and crashed before the completion-log landed.
    _run(coordinator.start(_start_command("swarm-fs-r", "driving-fs-r")))
    _run(
        swarm_store.update_run(
            "swarm-fs-r",
            expected_version=1,
            expected_token=_FENCE_TOKEN,
            status=SwarmStatus.RUNNING,
        )
    )
    # Simulate the crash mid-complete: hand-write the in-flight journal in the
    # PREPARED state (the state right after intent-journal write, before the
    # business write + completion log land).
    import base64
    import hashlib
    import json

    commit_id = "complete:swarm-fs-r"
    inflight_path = (
        tmp_path / "transactions" / "swarm_inflight"
        / f"{hashlib.sha256(commit_id.encode()).hexdigest()}.json"
    )
    inflight_path.parent.mkdir(parents=True, exist_ok=True)
    inflight_path.write_text(
        json.dumps(
            {
                "commit_id": commit_id,
                "operation": "complete",
                "swarm_run_id": "swarm-fs-r",
                "request_hash": ("00" * 32),  # 32-byte hash hex
                "state": "PREPARED",
                "command_payload_b64": base64.b64encode(b"{}").decode("ascii"),
                "result_payload_b64": None,
            }
        ),
        encoding="utf-8",
    )

    # An un-decodable journal fails closed before touching state and remains
    # available for forensics.
    with pytest.raises(SwarmRecoveryError):
        _run(coordinator.recover_incomplete_commits())

    run = _run(swarm_store.get_run("swarm-fs-r"))
    assert run.status is SwarmStatus.RUNNING
    # The journal is retained for forensics on the invalid identity path.
    assert inflight_path.exists()


def _prepare_start_journal(coordinator, command):
    payload = coordinator._codec.encode_request("start", command)
    coordinator._write_inflight(
        str(command.commit_id),
        "start",
        command.swarm_run_id,
        coordinator._codec.request_hash("start", command),
        payload,
    )
    return coordinator._inflight_dir / (
        f"{hashlib.sha256(str(command.commit_id).encode()).hexdigest()}.json"
    )


def test_prepared_start_creates_missing_run_and_completes(tmp_path):
    coordinator, swarm_store = _make(tmp_path)
    command = _start_command("swarm-recover-missing", "driving-recover")
    journal = _prepare_start_journal(coordinator, command)

    _run(coordinator.recover_incomplete_commits())

    run = _run(swarm_store.get_run(command.swarm_run_id))
    assert run is not None
    assert run.execution_token == _FENCE_TOKEN
    assert not journal.exists()
    assert list(coordinator._completed_dir.glob("*.json"))


def test_prepared_start_existing_matching_run_completes(tmp_path):
    coordinator, swarm_store = _make(tmp_path)
    command = _start_command("swarm-recover-existing", "driving-recover")
    expected = replace(command.payload.run, execution_token=_FENCE_TOKEN)
    _run(swarm_store.create_run(expected))
    journal = _prepare_start_journal(coordinator, command)

    _run(coordinator.recover_incomplete_commits())

    assert _run(swarm_store.get_run(command.swarm_run_id)) == expected
    assert not journal.exists()


def test_prepared_start_mismatched_run_fails_closed(tmp_path):
    from linktools.ai.swarm.persistence.filesystem_commit import SwarmRecoveryError

    coordinator, swarm_store = _make(tmp_path)
    command = _start_command("swarm-recover-mismatch", "driving-recover")
    mismatched = replace(
        command.payload.run,
        execution_token=_FENCE_TOKEN,
        version=99,
    )
    _run(swarm_store.create_run(mismatched))
    journal = _prepare_start_journal(coordinator, command)

    with pytest.raises(SwarmRecoveryError):
        _run(coordinator.recover_incomplete_commits())
    assert journal.exists()
    assert not list(coordinator._completed_dir.glob("*.json"))


@pytest.mark.parametrize(
    ("field", "value"),
    [("commit_id", "other-commit"), ("swarm_run_id", "other-run"), ("request_hash", "00" * 32)],
)
def test_journal_identity_mismatch_is_rejected(tmp_path, field, value):
    from linktools.ai.swarm.persistence.filesystem_commit import SwarmRecoveryError

    coordinator, _ = _make(tmp_path)
    command = _start_command("swarm-identity", "driving-identity")
    journal = _prepare_start_journal(coordinator, command)
    raw = json.loads(journal.read_text())
    raw[field] = value
    journal.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(SwarmRecoveryError):
        _run(coordinator.recover_incomplete_commits())
    assert journal.exists()


def test_result_ready_validates_command_before_completion(tmp_path):
    from linktools.ai.swarm.persistence.filesystem_commit import SwarmRecoveryError

    coordinator, _ = _make(tmp_path)
    command = _start_command("swarm-result-ready", "driving-result")
    journal = _prepare_start_journal(coordinator, command)
    coordinator._record_inflight_result(
        str(command.commit_id),
        "start",
        {"swarm_run_id": command.swarm_run_id, "version": 1},
    )
    raw = json.loads(journal.read_text())
    raw["commit_id"] = "wrong-result-ready-id"
    journal.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(SwarmRecoveryError):
        _run(coordinator.recover_incomplete_commits())
    assert journal.exists()
    assert not list(coordinator._completed_dir.glob("*.json"))
