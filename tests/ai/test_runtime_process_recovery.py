#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh processes recover orphaned execution and prepared terminal handoff."""

from pathlib import Path

import pytest

from linktools.ai.errors import ErrorCode

from ._runtime_process_test_helpers import _RuntimeProcess


@pytest.mark.parametrize("phase", (
    "active", pytest.param("admitted", marks=pytest.mark.merge), "pending",
))
def test_sql_orphan_waits_for_explicit_recovery(tmp_path: Path, phase: str) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase=phase)
    remote = None
    try:
        owner.crash()
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase=phase, execution_id=owner.initial["execution_id"],
        )
        assert remote.initial["generation"] == owner.initial["generation"]
        assert remote.initial["model_calls"] == 0
        if phase == "pending":
            assert owner.initial["status"] == "PENDING_START"
            rejected = remote.request("try_recover_keyed")
            assert rejected["error_code"] == ErrorCode.EXECUTION_NOT_READY.value
            assert rejected["details"]["phase"] == "execution_recover_unprepared_start"
            assert remote.request("snapshot")["model_calls"] == 0
            # An unprepared reservation resumes through the same start identity.
            replay = remote.request("replay_start")
            assert replay["execution_id"] == owner.initial["execution_id"]
        else:
            remote.request("recover")
        resumed = remote.request("await_model")
        assert resumed["generation"] > owner.initial["generation"]
        assert resumed["model_calls"] == 1
        assert remote.request("finish")["status"] == "SUCCEEDED"
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_explicit_recovery_finishes_orphaned_cancel_without_restarting_model(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db")
    remote = None
    try:
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", execution_id=owner.initial["execution_id"],
        )
        # This scenario starts after the owner's pending observations are durable.
        # A projection racing cancellation has its own explicit-CAS retry contract.
        flushed = owner.request("flush_trace")
        assert flushed["status"] == "STARTED"
        assert flushed["generation"] == owner.initial["generation"]
        assert flushed["model_calls"] == 1
        cancelled = remote.request("cancel")
        assert cancelled["status"] == "CANCELLING"
        assert cancelled["cancelled"] is False
        assert remote.request("cancel_events") == {"requested": 1}
        owner.crash()
        recovered = remote.request("recover")
        assert recovered["status"] == "CANCELLED"
        assert remote.request("cancel_events") == {"requested": 1}
        assert remote.request("result") == {
            "status": "CANCELLED", "output": None, "model_calls": 0,
        }
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_sql_startup_completes_a_prepared_terminal_handoff(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase="handoff")
    remote = None
    try:
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase="handoff", execution_id=owner.initial["execution_id"],
        )
        assert owner.process.is_alive()
        assert remote.initial["status"] == "SUCCEEDED"
        assert remote.initial["generation"] == owner.initial["generation"]
        assert remote.request("result") == {
            "status": "SUCCEEDED", "output": {"text": "owner answer"}, "model_calls": 0,
        }
        assert owner.request("finish")["status"] == "SUCCEEDED"
        owner.close(check=True)
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


@pytest.mark.parametrize("phase", ("local_sqlite",))
def test_exclusive_execution_route_preserves_automatic_crash_recovery(tmp_path: Path, phase: str) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase=phase)
    remote = None
    try:
        owner.crash()
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase=phase, execution_id=owner.initial["execution_id"],
        )
        assert remote.initial["generation"] > owner.initial["generation"]
        assert remote.request("await_model")["model_calls"] == 1
        assert remote.request("finish")["status"] == "SUCCEEDED"
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_split_local_sqlite_execution_rejects_a_second_live_process(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase="local_sqlite")
    try:
        with pytest.raises(AssertionError) as rejected:
            _RuntimeProcess(tmp_path / "runtime.db", phase="local_sqlite",
                            execution_id=owner.initial["execution_id"])
        failure = rejected.value.args[0]
        assert failure["error"] == "AIError"
        assert failure["code"] == ErrorCode.STORAGE_CONFLICT.value
        assert owner.process.is_alive()
        current = owner.request("await_model")
        assert current["generation"] == owner.initial["generation"]
        assert current["model_calls"] == 1
        assert owner.request("finish")["status"] == "SUCCEEDED"
        owner.close(check=True)
    finally:
        owner.close()


def test_reserved_orphan_resumes_only_after_explicit_start_replay(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase="reserved")
    remote = None
    try:
        assert owner.initial["status"] == "PENDING_START"
        assert owner.initial["model_calls"] == 0
        owner.crash()
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase="reserved", execution_id=owner.initial["execution_id"],
        )
        assert remote.initial["status"] == "PENDING_START"
        assert remote.initial["model_calls"] == 0
        replay = remote.request("replay_start")
        assert replay["execution_id"] == owner.initial["execution_id"]
        assert remote.request("await_model")["model_calls"] == 1
        assert remote.request("finish")["status"] == "SUCCEEDED"
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()
